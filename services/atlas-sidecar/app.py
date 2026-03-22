"""
Atlas CLI Sidecar Service
Provenance collection for the ML Provenance Pipeline using atlas-cli.

Wraps the IntelLabs Atlas CLI (https://github.com/IntelLabs/atlas-cli)
and exposes an HTTP API so pipeline services can call it at the start
of each pipeline step to register C2PA / OMS provenance manifests for
every artifact produced (datasets, pipeline outputs, models).

Architecture:
  - Each pipeline service calls POST /collect/{type} when a step starts.
  - The sidecar downloads the artifact from S3 to a temp file (atlas-cli
    needs a local path to compute the content hash).
  - atlas-cli writes C2PA manifests to a shared filesystem volume.
  - Manifests are linked across stages to form the provenance chain.

API:
  POST /collect/dataset   - register raw/tokenized dataset (Stage 1 & 2)
  POST /collect/pipeline  - register pipeline-step provenance (Stage 2)
  POST /collect/model     - register trained model (Stage 3)
  POST /link              - link two existing manifests
  GET  /manifests         - list all collected manifests
  GET  /export/{id}       - export full provenance graph as JSON
  GET  /verify/{id}       - verify manifest integrity
  GET  /signing-key       - return the public key PEM
  GET  /health            - liveness probe
"""

import json
import logging
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Atlas CLI Sidecar",
    description="ML Pipeline provenance collection via IntelLabs Atlas CLI",
    version="1.0.0",
)

# ── Config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT   = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET     = os.getenv("S3_BUCKET", "ml-provenance")
AWS_KEY       = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET    = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MANIFESTS_DIR = Path(os.getenv("MANIFESTS_DIR", "/manifests/atlas"))
KEY_PATH      = MANIFESTS_DIR / "signing-key.pem"
AUTHOR        = os.getenv("ATLAS_AUTHOR", "ml-pipeline")

# In-memory manifest registry (s3_uri → manifest record)
_registry: dict = {}
_lock = threading.Lock()


# ── S3 helper ─────────────────────────────────────────────────────────────────
def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


# ── Atlas CLI helpers ─────────────────────────────────────────────────────────
def _atlas_flags() -> list[str]:
    """Storage + signing flags for dataset/model subcommands."""
    return [
        "--storage-type=local",
        f"--storage-url={MANIFESTS_DIR}",
        f"--key={KEY_PATH}",
    ]


def _run_atlas(*args: str, timeout: int = 120) -> tuple[str, str, int]:
    cmd = ["atlas-cli"] + list(args)
    log.info("atlas-cli: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        log.info("stdout: %s", r.stdout.strip())
    if r.stderr:
        log.info("stderr: %s", r.stderr.strip())
    return r.stdout, r.stderr, r.returncode


def _parse_manifest_id(text: str) -> Optional[str]:
    """Extract a manifest ID from atlas-cli output (URN, UUID, or integer)."""
    # C2PA URN: urn:c2pa:<uuid>[:<extra>]
    m = re.search(r"urn:c2pa:[0-9a-f-]{36}(?::[^\s\"']+)?", text, re.IGNORECASE)
    if m:
        return m.group(0)
    # Plain UUID fallback
    m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text, re.IGNORECASE)
    if m:
        return m.group(0)
    # Integer ID (e.g. pipeline generate-provenance: "Manifest stored successfully with ID: 0")
    m = re.search(r"\bID:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _latest_manifest_id_from_index() -> Optional[str]:
    """Read the most recently written manifest ID from atlas's index file."""
    for name in ("manifest_index.json", "index.json"):
        idx_path = MANIFESTS_DIR / name
        if idx_path.exists():
            try:
                with open(idx_path) as f:
                    data = json.load(f)
                # Could be a list of IDs or a dict
                if isinstance(data, list) and data:
                    entry = data[-1]
                    return entry if isinstance(entry, str) else entry.get("id") or entry.get("manifest_id")
                if isinstance(data, dict):
                    keys = list(data.keys())
                    return keys[-1] if keys else None
            except Exception:
                pass
    return None


def _download_s3_to_temp(s3_uri: str) -> Path:
    """Stream an S3 artifact to a temp file inside MANIFESTS_DIR. Returns path."""
    key = s3_uri.replace(f"s3://{S3_BUCKET}/", "")
    suffix = Path(key.split("/")[-1]).suffix or ".bin"
    tmp = MANIFESTS_DIR / f"tmp_{uuid.uuid4().hex}{suffix}"
    log.info("Downloading %s → %s", s3_uri, tmp.name)
    get_s3().download_file(S3_BUCKET, key, str(tmp))
    return tmp


def _register(s3_uri: str, manifest_id: str, stage: str, kind: str, ingredient: str):
    with _lock:
        _registry[s3_uri] = {
            "manifest_id": manifest_id,
            "stage": stage,
            "type": kind,
            "ingredient_name": ingredient,
        }
    _persist_registry()


def _persist_registry():
    with _lock:
        data = dict(_registry)
    path = MANIFESTS_DIR / "manifest_registry.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_registry():
    path = MANIFESTS_DIR / "manifest_registry.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        with _lock:
            _registry.update(data)
        log.info("Loaded %d manifest entries from registry", len(data))


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate RSA-4096 signing key if not present
    if not KEY_PATH.exists():
        log.info("Generating RSA-4096 signing key → %s", KEY_PATH)
        r = subprocess.run(
            ["openssl", "genrsa", "-out", str(KEY_PATH), "4096"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            log.error("Key generation failed: %s", r.stderr)
        else:
            KEY_PATH.chmod(0o600)
            log.info("Signing key ready")

    # Verify atlas-cli binary
    stdout, stderr, rc = _run_atlas("--version")
    if rc == 0:
        log.info("atlas-cli ready: %s", stdout.strip())
    else:
        log.error("atlas-cli not available: %s", stderr)

    _load_registry()


# ── Request models ────────────────────────────────────────────────────────────
class DatasetCollectRequest(BaseModel):
    stage: str                          # "data-ingestion" | "preprocessing"
    artifact_s3_uri: str                # s3://bucket/key
    ingredient_name: str
    author: str = AUTHOR
    linked_manifest_ids: list[str] = [] # previously registered manifests to link
    metadata: dict = {}


class PipelineCollectRequest(BaseModel):
    stage: str                          # "preprocessing"
    input_s3_uris: list[str]            # upstream artifacts
    output_s3_uri: str                  # artifact produced by this step
    ingredient_name: str
    author: str = AUTHOR
    build_script: str = ""              # description of the transformation
    metadata: dict = {}


class ModelCollectRequest(BaseModel):
    stage: str                          # "fine-tuning"
    artifact_s3_uri: str
    ingredient_name: str
    author: str = AUTHOR
    linked_dataset_manifest_ids: list[str] = []
    metadata: dict = {}


class LinkRequest(BaseModel):
    source_manifest_id: str
    target_manifest_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    stdout, _, rc = _run_atlas("--version")
    return {
        "status": "ok",
        "service": "atlas-sidecar",
        "atlas_cli": stdout.strip() if rc == 0 else "unavailable",
        "key_exists": KEY_PATH.exists(),
        "manifests_dir": str(MANIFESTS_DIR),
        "registered_artifacts": len(_registry),
    }


@app.get("/signing-key")
def signing_key():
    """Return the public key PEM for out-of-band signature verification."""
    if not KEY_PATH.exists():
        raise HTTPException(404, "Signing key not yet generated")
    r = subprocess.run(
        ["openssl", "rsa", "-in", str(KEY_PATH), "-pubout"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise HTTPException(500, f"Could not extract public key: {r.stderr}")
    return {"public_key_pem": r.stdout}


@app.post("/collect/dataset")
def collect_dataset(req: DatasetCollectRequest):
    """
    Register a dataset artifact with Atlas CLI.
    Used by Stage 1 (raw data) and Stage 2 (tokenized data).
    Downloads the S3 artifact to a temp file so atlas-cli can compute the hash.
    """
    tmp = None
    try:
        tmp = _download_s3_to_temp(req.artifact_s3_uri)

        stdout, stderr, rc = _run_atlas(
            "dataset", "create",
            f"--name={req.ingredient_name}",
            f"--paths={tmp}",
            f"--ingredient-names={req.ingredient_name}",
            f"--author-org={req.author}",
            *_atlas_flags(),
        )

        manifest_id = _parse_manifest_id(stdout + stderr) or _latest_manifest_id_from_index()
        if rc != 0 and not manifest_id:
            raise HTTPException(500, f"atlas-cli dataset create failed (rc={rc}): {stderr.strip()}")

        # Link to upstream manifests if provided
        for linked_id in req.linked_manifest_ids:
            _run_atlas("manifest", "link",
                       f"--source-id={manifest_id}",
                       f"--target-id={linked_id}",
                       "--storage-type=filesystem",
                       f"--storage-url={MANIFESTS_DIR}")

        _register(req.artifact_s3_uri, manifest_id, req.stage, "dataset", req.ingredient_name)

        return {
            "manifest_id": manifest_id,
            "stage": req.stage,
            "type": "dataset",
            "artifact_s3_uri": req.artifact_s3_uri,
            "linked_manifest_ids": req.linked_manifest_ids,
            "atlas_stdout": stdout.strip(),
        }
    finally:
        if tmp and tmp.exists():
            tmp.unlink(missing_ok=True)


@app.post("/collect/pipeline")
def collect_pipeline(req: PipelineCollectRequest):
    """
    Register a pipeline-step provenance manifest.
    Uses atlas-cli pipeline generate-provenance to create SLSA provenance
    documenting the transformation from input → output artifacts.
    """
    input_tmps: list[Path] = []
    output_tmp = None
    try:
        # Download all inputs
        for uri in req.input_s3_uris:
            input_tmps.append(_download_s3_to_temp(uri))
        # Download output
        output_tmp = _download_s3_to_temp(req.output_s3_uri)

        inputs_arg  = ",".join(str(p) for p in input_tmps)
        outputs_arg = str(output_tmp)

        # Write a minimal build-script description to a temp file
        script_path = MANIFESTS_DIR / f"build_script_{uuid.uuid4().hex}.txt"
        script_path.write_text(req.build_script or f"Pipeline stage: {req.stage}")

        stdout, stderr, rc = _run_atlas(
            "pipeline", "generate-provenance",
            f"--pipeline={script_path}",
            f"--inputs={inputs_arg}",
            f"--products={outputs_arg}",
            "--storage-type=local-fs",
            f"--storage-url={MANIFESTS_DIR}",
            f"--key={KEY_PATH}",
        )
        script_path.unlink(missing_ok=True)

        manifest_id = _parse_manifest_id(stdout + stderr) or _latest_manifest_id_from_index()
        if rc != 0 and not manifest_id:
            raise HTTPException(500, f"atlas-cli pipeline generate-provenance failed (rc={rc}): {stderr.strip()}")

        _register(req.output_s3_uri, manifest_id, req.stage, "pipeline", req.ingredient_name)

        return {
            "manifest_id": manifest_id,
            "stage": req.stage,
            "type": "pipeline",
            "input_s3_uris": req.input_s3_uris,
            "output_s3_uri": req.output_s3_uri,
            "atlas_stdout": stdout.strip(),
        }
    finally:
        for p in input_tmps:
            p.unlink(missing_ok=True)
        if output_tmp and output_tmp.exists():
            output_tmp.unlink(missing_ok=True)


@app.post("/collect/model")
def collect_model(req: ModelCollectRequest):
    """
    Register a model artifact with Atlas CLI.
    Used by Stage 3 (fine-tuned BERT).
    After creating the model manifest, links it to each dataset manifest.
    """
    tmp = None
    try:
        tmp = _download_s3_to_temp(req.artifact_s3_uri)

        stdout, stderr, rc = _run_atlas(
            "model", "create",
            f"--name={req.ingredient_name}",
            f"--paths={tmp}",
            f"--ingredient-names={req.ingredient_name}",
            f"--author-org={req.author}",
            *_atlas_flags(),
        )

        manifest_id = _parse_manifest_id(stdout + stderr) or _latest_manifest_id_from_index()
        if rc != 0 and not manifest_id:
            raise HTTPException(500, f"atlas-cli model create failed (rc={rc}): {stderr.strip()}")

        # Link model → each upstream dataset manifest
        for ds_id in req.linked_dataset_manifest_ids:
            lout, lerr, lrc = _run_atlas(
                "model", "link-dataset",
                f"--model-id={manifest_id}",
                f"--dataset-id={ds_id}",
                "--storage-type=filesystem",
                f"--storage-url={MANIFESTS_DIR}",
            )
            if lrc != 0:
                log.warning("link-dataset failed for %s → %s: %s", manifest_id, ds_id, lerr.strip())

        _register(req.artifact_s3_uri, manifest_id, req.stage, "model", req.ingredient_name)

        return {
            "manifest_id": manifest_id,
            "stage": req.stage,
            "type": "model",
            "artifact_s3_uri": req.artifact_s3_uri,
            "linked_dataset_manifest_ids": req.linked_dataset_manifest_ids,
            "atlas_stdout": stdout.strip(),
        }
    finally:
        if tmp and tmp.exists():
            tmp.unlink(missing_ok=True)


@app.post("/link")
def link_manifests(req: LinkRequest):
    """Explicitly link two existing manifests (source → target)."""
    stdout, stderr, rc = _run_atlas(
        "manifest", "link",
        f"--source-id={req.source_manifest_id}",
        f"--target-id={req.target_manifest_id}",
        "--storage-type=filesystem",
        f"--storage-url={MANIFESTS_DIR}",
    )
    if rc != 0:
        raise HTTPException(500, f"manifest link failed: {stderr.strip()}")
    return {"linked": True, "source": req.source_manifest_id, "target": req.target_manifest_id}


@app.get("/manifests")
def list_manifests(stage: Optional[str] = None):
    """List all collected manifests, optionally filtered by pipeline stage."""
    with _lock:
        entries = list(_registry.items())
    if stage:
        entries = [(uri, rec) for uri, rec in entries if rec.get("stage") == stage]
    return {
        "count": len(entries),
        "manifests": [
            {"s3_uri": uri, **rec}
            for uri, rec in entries
        ],
    }


@app.get("/export/{manifest_id:path}")
def export_manifest(manifest_id: str, depth: int = 5):
    """Export the full provenance graph for a manifest ID as JSON."""
    stdout, stderr, rc = _run_atlas(
        "manifest", "export",
        f"--id={manifest_id}",
        f"--depth={depth}",
        "--output-format=json",
        "--storage-type=filesystem",
        f"--storage-url={MANIFESTS_DIR}",
    )
    if rc != 0:
        raise HTTPException(500, f"manifest export failed: {stderr.strip()}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"raw_output": stdout.strip(), "stderr": stderr.strip()}


@app.get("/verify/{manifest_id:path}")
def verify_manifest(manifest_id: str):
    """Verify the cryptographic integrity of a manifest."""
    stdout, stderr, rc = _run_atlas(
        "manifest", "validate",
        "--storage-type=filesystem",
        f"--storage-url={MANIFESTS_DIR}",
    )
    return {
        "manifest_id": manifest_id,
        "valid": rc == 0,
        "output": stdout.strip(),
        "errors": stderr.strip() if rc != 0 else None,
    }


@app.get("/registry")
def registry():
    """Return the full in-memory manifest registry (s3_uri → manifest_id map)."""
    with _lock:
        return dict(_registry)
