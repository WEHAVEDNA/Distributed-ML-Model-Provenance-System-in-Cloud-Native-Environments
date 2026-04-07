"""
Atlas CLI Sidecar Service

Wraps the IntelLabs Atlas CLI and exposes an HTTP API for collecting
pipeline-scoped provenance manifests. Artifacts are registered with a
standard pipeline ID so multiple independent pipelines can reuse the same
sidecar service without sharing lineage state.
"""

import json
import logging
import os
import re
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Atlas CLI Sidecar",
    description="Pipeline provenance collection via IntelLabs Atlas CLI",
    version="1.1.0",
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET = os.getenv("S3_BUCKET", "ml-provenance")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MANIFESTS_DIR = Path(os.getenv("MANIFESTS_DIR", "/manifests/atlas"))
KEY_PATH = MANIFESTS_DIR / "signing-key.pem"
AUTHOR = os.getenv("ATLAS_AUTHOR", "ml-pipeline")
DEFAULT_PIPELINE_ID = os.getenv("PIPELINE_ID", "default")
DEFAULT_PIPELINE_STAGES = [
    stage.strip() for stage in os.getenv(
        "PIPELINE_STAGES",
        "data-ingestion,preprocessing,fine-tuning",
    ).split(",")
    if stage.strip()
]
DEPLOYMENT_MODE = os.getenv(
    "DEPLOYMENT_MODE",
    "kubernetes" if os.getenv("KUBERNETES_SERVICE_HOST") else "local",
)
POD_NAME = os.getenv("POD_NAME")
POD_NAMESPACE = os.getenv("POD_NAMESPACE")
NODE_NAME = os.getenv("NODE_NAME")

_registry: dict = {}
_lock = threading.Lock()
_atlas_cli_version: Optional[str] = None


def _normalize_pipeline_id(pipeline_id: Optional[str]) -> str:
    candidate = (pipeline_id or DEFAULT_PIPELINE_ID).strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", candidate).strip("._-")
    if not normalized:
        raise ValueError("pipeline_id must contain at least one alphanumeric character")
    return normalized


def _default_stage_order(stage: str) -> int:
    try:
        return (DEFAULT_PIPELINE_STAGES.index(stage) + 1) * 10
    except ValueError:
        return 10_000


def _resolve_pipeline_id_or_400(pipeline_id: Optional[str]) -> str:
    try:
        return _normalize_pipeline_id(pipeline_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


def _atlas_flags() -> list[str]:
    return [
        "--storage-type=local-fs",
        f"--storage-url={MANIFESTS_DIR}",
        f"--key={KEY_PATH}",
    ]


def _run_atlas(*args: str, timeout: int = 120) -> tuple[str, str, int]:
    cmd = ["atlas-cli"] + list(args)
    log.info("atlas-cli: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        log.info("stdout: %s", result.stdout.strip())
    if result.stderr:
        log.info("stderr: %s", result.stderr.strip())
    return result.stdout, result.stderr, result.returncode


def _is_resolvable_manifest_id(value: Optional[str]) -> bool:
    if not isinstance(value, str):
        return False

    candidate = value.strip()
    if not candidate or candidate.isdigit():
        return False

    if candidate.lower().startswith("urn:c2pa:"):
        return True

    return bool(
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            candidate,
            re.IGNORECASE,
        )
    )


def _parse_manifest_id(text: str) -> Optional[str]:
    urn_match = re.search(r"urn:c2pa:[0-9a-f-]{36}(?::[^\s\"']+)?", text, re.IGNORECASE)
    if urn_match:
        return urn_match.group(0)

    uuid_match = re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        text,
        re.IGNORECASE,
    )
    if uuid_match:
        return uuid_match.group(0)

    return None


def _manifest_ids_from_index() -> list[str]:
    manifest_ids: list[str] = []
    for name in ("manifest_index.json", "index.json"):
        index_path = MANIFESTS_DIR / name
        if not index_path.exists():
            continue

        try:
            with open(index_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue

        if isinstance(data, list) and data:
            for entry in data:
                candidate = entry if isinstance(entry, str) else entry.get("id") or entry.get("manifest_id")
                if _is_resolvable_manifest_id(candidate):
                    manifest_ids.append(candidate)
        if isinstance(data, dict):
            for candidate in data.keys():
                if _is_resolvable_manifest_id(candidate):
                    manifest_ids.append(candidate)

    return manifest_ids


def _latest_manifest_id_from_index() -> Optional[str]:
    manifest_ids = _manifest_ids_from_index()
    return manifest_ids[-1] if manifest_ids else None


def _resolve_manifest_id_from_command(text: str, previous_index_ids: list[str]) -> Optional[str]:
    manifest_id = _parse_manifest_id(text)
    if manifest_id:
        return manifest_id

    previous = set(previous_index_ids)
    for candidate in reversed(_manifest_ids_from_index()):
        if candidate not in previous:
            return candidate

    return None


def _download_s3_to_temp(s3_uri: str) -> Path:
    key = s3_uri.replace(f"s3://{S3_BUCKET}/", "")
    suffix = Path(key.split("/")[-1]).suffix or ".bin"
    tmp_path = MANIFESTS_DIR / f"tmp_{uuid.uuid4().hex}{suffix}"
    log.info("Downloading %s -> %s", s3_uri, tmp_path.name)
    get_s3().download_file(S3_BUCKET, key, str(tmp_path))
    return tmp_path


def _persist_registry():
    with _lock:
        data = dict(_registry)
    registry_path = MANIFESTS_DIR / "manifest_registry.json"
    with open(registry_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _load_registry():
    registry_path = MANIFESTS_DIR / "manifest_registry.json"
    if not registry_path.exists():
        return

    with open(registry_path, encoding="utf-8") as handle:
        data = json.load(handle)
    with _lock:
        _registry.update(data)
    log.info("Loaded %d manifest entries from registry", len(data))


def _registry_key_for_record(
    *,
    artifact_uri: str,
    kind: str,
    pipeline_id: str,
    stage: str,
    stage_order: int,
) -> str:
    if kind != "pipeline":
        return artifact_uri
    return f"pipeline::{pipeline_id}::{stage_order:05d}::{stage}::{artifact_uri}"


def _artifact_uri_for_record(registry_key: str, record: dict) -> str:
    return record.get("artifact_uri") or registry_key


def _record_tracking_id(registry_key: str, record: dict) -> str:
    return record.get("tracking_id") or registry_key


def _register(
    s3_uri: str,
    manifest_id: Optional[str],
    pipeline_id: str,
    stage: str,
    stage_order: int,
    kind: str,
    ingredient_name: str,
    metadata: dict,
    input_s3_uris: Optional[list[str]] = None,
    linked_manifest_ids: Optional[list[str]] = None,
):
    registry_key = _registry_key_for_record(
        artifact_uri=s3_uri,
        kind=kind,
        pipeline_id=pipeline_id,
        stage=stage,
        stage_order=stage_order,
    )
    record = {
        "tracking_id": registry_key,
        "artifact_uri": s3_uri,
        "manifest_id": manifest_id,
        "pipeline_id": pipeline_id,
        "stage": stage,
        "stage_order": stage_order,
        "type": kind,
        "ingredient_name": ingredient_name,
        "input_artifact_uris": list(input_s3_uris or []),
        "linked_manifest_ids": list(linked_manifest_ids or []),
        "metadata": metadata,
        "registered_at": datetime.utcnow().isoformat() + "Z",
    }
    with _lock:
        _registry[registry_key] = record
    _persist_registry()
    return record


def _filtered_registry(pipeline_id: Optional[str] = None, stage: Optional[str] = None) -> dict:
    with _lock:
        data = dict(_registry)

    if pipeline_id is not None:
        resolved_pipeline_id = _normalize_pipeline_id(pipeline_id)
        data = {
            uri: record
            for uri, record in data.items()
            if record.get("pipeline_id", _normalize_pipeline_id(DEFAULT_PIPELINE_ID)) == resolved_pipeline_id
        }
    if stage is not None:
        data = {
            uri: record
            for uri, record in data.items()
            if record.get("stage") == stage
        }
    return data


def _ordered_entries_for_pipeline(pipeline_id: str) -> list[tuple[str, dict]]:
    entries = list(_filtered_registry(pipeline_id=pipeline_id).items())
    return sorted(
        entries,
        key=lambda item: (
            item[1].get("stage_order", _default_stage_order(item[1].get("stage", ""))),
            item[1].get("registered_at", ""),
            item[0],
        ),
    )


def _pipeline_stage_names(entries: list[tuple[str, dict]]) -> list[str]:
    seen = set()
    ordered_names = []
    for _, record in entries:
        stage = record.get("stage")
        if stage and stage not in seen:
            seen.add(stage)
            ordered_names.append(stage)
    return ordered_names


def _expected_stage_names() -> list[str]:
    return list(DEFAULT_PIPELINE_STAGES)


@app.on_event("startup")
async def on_startup():
    global _atlas_cli_version

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    if not KEY_PATH.exists():
        log.info("Generating RSA-4096 signing key -> %s", KEY_PATH)
        result = subprocess.run(
            ["openssl", "genrsa", "-out", str(KEY_PATH), "4096"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.error("Key generation failed: %s", result.stderr)
        else:
            KEY_PATH.chmod(0o600)
            log.info("Signing key ready")

    stdout, stderr, rc = _run_atlas("--version")
    if rc == 0:
        _atlas_cli_version = stdout.strip()
        log.info("atlas-cli ready: %s", _atlas_cli_version)
    else:
        _atlas_cli_version = None
        log.error("atlas-cli not available: %s", stderr)

    _load_registry()


class DatasetCollectRequest(BaseModel):
    pipeline_id: str = DEFAULT_PIPELINE_ID
    stage: str
    stage_order: int = 0
    artifact_s3_uri: str
    ingredient_name: str
    author: str = AUTHOR
    linked_manifest_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class PipelineCollectRequest(BaseModel):
    pipeline_id: str = DEFAULT_PIPELINE_ID
    stage: str
    stage_order: int = 0
    input_s3_uris: list[str] = Field(default_factory=list)
    linked_manifest_ids: list[str] = Field(default_factory=list)
    output_s3_uri: str
    ingredient_name: str
    author: str = AUTHOR
    build_script: str = ""
    metadata: dict = Field(default_factory=dict)


class ModelCollectRequest(BaseModel):
    pipeline_id: str = DEFAULT_PIPELINE_ID
    stage: str
    stage_order: int = 0
    artifact_s3_uri: str
    ingredient_name: str
    author: str = AUTHOR
    linked_dataset_manifest_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class LinkRequest(BaseModel):
    source_manifest_id: str
    target_manifest_id: str


def _linked_manifest_ids_for_uris(s3_uris: list[str]) -> list[str]:
    with _lock:
        records = list(_registry.items())

    manifest_ids: list[str] = []
    seen: set[str] = set()
    for uri in s3_uris:
        for registry_key, record in records:
            if _artifact_uri_for_record(registry_key, record) != uri:
                continue
            manifest_id = record.get("manifest_id")
            if _is_resolvable_manifest_id(manifest_id) and manifest_id not in seen:
                seen.add(manifest_id)
                manifest_ids.append(manifest_id)
    return manifest_ids


def _collect_pipeline_step_manifest(
    *,
    output_tmp: Path,
    ingredient_name: str,
    author: str,
    linked_manifest_ids: list[str],
) -> tuple[Optional[str], str]:
    previous_index_ids = _manifest_ids_from_index()
    pipeline_step_name = f"{ingredient_name} pipeline step"
    stdout, stderr, rc = _run_atlas(
        "dataset",
        "create",
        f"--name={pipeline_step_name}",
        f"--paths={output_tmp}",
        f"--ingredient-names={pipeline_step_name}",
        f"--author-org={author}",
        *_atlas_flags(),
    )
    manifest_id = _resolve_manifest_id_from_command(stdout + stderr, previous_index_ids)
    if rc != 0 and not manifest_id:
        raise HTTPException(
            status_code=500,
            detail=f"atlas-cli pipeline step manifest failed (rc={rc}): {stderr.strip()}",
        )

    for linked_id in linked_manifest_ids:
        if not manifest_id:
            break
        _run_atlas(
            "manifest",
            "link",
            f"--source={manifest_id}",
            f"--target={linked_id}",
            "--storage-type=local-fs",
            f"--storage-url={MANIFESTS_DIR}",
        )

    return manifest_id, stdout.strip()


@app.get("/health")
def health():
    with _lock:
        known_pipelines = sorted({record.get("pipeline_id", _normalize_pipeline_id(DEFAULT_PIPELINE_ID)) for record in _registry.values()})
    return {
        "status": "ok",
        "service": "atlas-sidecar",
        "atlas_cli": _atlas_cli_version if _atlas_cli_version is not None else "unavailable",
        "key_exists": KEY_PATH.exists(),
        "manifests_dir": str(MANIFESTS_DIR),
        "registered_artifacts": len(_registry),
        "default_pipeline_id": _normalize_pipeline_id(DEFAULT_PIPELINE_ID),
        "known_pipelines": known_pipelines,
        "deployment_mode": DEPLOYMENT_MODE,
        "pod_name": POD_NAME,
        "pod_namespace": POD_NAMESPACE,
        "node_name": NODE_NAME,
    }


@app.get("/signing-key")
def signing_key():
    if not KEY_PATH.exists():
        raise HTTPException(status_code=404, detail="Signing key not yet generated")

    result = subprocess.run(
        ["openssl", "rsa", "-in", str(KEY_PATH), "-pubout"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Could not extract public key: {result.stderr}")
    return {"public_key_pem": result.stdout}


@app.post("/collect/dataset")
def collect_dataset(req: DatasetCollectRequest):
    tmp_path = None
    pipeline_id = _resolve_pipeline_id_or_400(req.pipeline_id)
    stage_order = req.stage_order or _default_stage_order(req.stage)
    try:
        tmp_path = _download_s3_to_temp(req.artifact_s3_uri)
        previous_index_ids = _manifest_ids_from_index()

        stdout, stderr, rc = _run_atlas(
            "dataset",
            "create",
            f"--name={req.ingredient_name}",
            f"--paths={tmp_path}",
            f"--ingredient-names={req.ingredient_name}",
            f"--author-org={req.author}",
            *_atlas_flags(),
        )

        manifest_id = _resolve_manifest_id_from_command(stdout + stderr, previous_index_ids)
        if rc != 0 and not manifest_id:
            raise HTTPException(status_code=500, detail=f"atlas-cli dataset create failed (rc={rc}): {stderr.strip()}")

        for linked_id in req.linked_manifest_ids:
            if not manifest_id:
                break
            _run_atlas(
                "manifest",
                "link",
                f"--source={manifest_id}",
                f"--target={linked_id}",
                "--storage-type=local-fs",
                f"--storage-url={MANIFESTS_DIR}",
            )

        record = _register(
            req.artifact_s3_uri,
            manifest_id,
            pipeline_id,
            req.stage,
            stage_order,
            "dataset",
            req.ingredient_name,
            req.metadata,
            linked_manifest_ids=req.linked_manifest_ids,
        )

        return {
            "tracking_id": record["tracking_id"],
            "manifest_id": manifest_id,
            "pipeline_id": pipeline_id,
            "stage": req.stage,
            "stage_order": stage_order,
            "type": "dataset",
            "artifact_s3_uri": req.artifact_s3_uri,
            "linked_manifest_ids": req.linked_manifest_ids,
            "atlas_stdout": stdout.strip(),
        }
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/collect/pipeline")
def collect_pipeline(req: PipelineCollectRequest):
    input_tmps: list[Path] = []
    output_tmp = None
    script_path = None
    pipeline_id = _resolve_pipeline_id_or_400(req.pipeline_id)
    stage_order = req.stage_order or _default_stage_order(req.stage)
    try:
        for uri in req.input_s3_uris:
            input_tmps.append(_download_s3_to_temp(uri))
        output_tmp = _download_s3_to_temp(req.output_s3_uri)
        previous_index_ids = _manifest_ids_from_index()

        script_path = MANIFESTS_DIR / f"build_script_{uuid.uuid4().hex}.txt"
        script_path.write_text(req.build_script or f"Pipeline stage: {req.stage}", encoding="utf-8")

        stdout, stderr, rc = _run_atlas(
            "pipeline",
            "generate-provenance",
            f"--pipeline={script_path}",
            f"--inputs={','.join(str(path) for path in input_tmps)}",
            f"--products={output_tmp}",
            "--storage-type=local-fs",
            f"--storage-url={MANIFESTS_DIR}",
            f"--key={KEY_PATH}",
        )

        manifest_id = _resolve_manifest_id_from_command(stdout + stderr, previous_index_ids)
        if rc != 0 and not manifest_id:
            raise HTTPException(
                status_code=500,
                detail=f"atlas-cli pipeline generate-provenance failed (rc={rc}): {stderr.strip()}",
            )

        linked_manifest_ids = req.linked_manifest_ids or _linked_manifest_ids_for_uris(req.input_s3_uris)
        pipeline_step_stdout = ""
        if not manifest_id:
            log.info(
                "atlas-cli pipeline generate-provenance did not return a resolvable manifest ID; "
                "creating an explicit pipeline-step manifest instead"
            )
            manifest_id, pipeline_step_stdout = _collect_pipeline_step_manifest(
                output_tmp=output_tmp,
                ingredient_name=req.ingredient_name,
                author=req.author,
                linked_manifest_ids=linked_manifest_ids,
            )

        record = _register(
            req.output_s3_uri,
            manifest_id,
            pipeline_id,
            req.stage,
            stage_order,
            "pipeline",
            req.ingredient_name,
            req.metadata,
            input_s3_uris=req.input_s3_uris,
            linked_manifest_ids=linked_manifest_ids,
        )

        return {
            "tracking_id": record["tracking_id"],
            "manifest_id": manifest_id,
            "pipeline_id": pipeline_id,
            "stage": req.stage,
            "stage_order": stage_order,
            "type": "pipeline",
            "input_s3_uris": req.input_s3_uris,
            "linked_manifest_ids": linked_manifest_ids,
            "output_s3_uri": req.output_s3_uri,
            "atlas_stdout": "\n".join(part for part in (stdout.strip(), pipeline_step_stdout) if part),
        }
    finally:
        for path in input_tmps:
            path.unlink(missing_ok=True)
        if output_tmp and output_tmp.exists():
            output_tmp.unlink(missing_ok=True)
        if script_path and script_path.exists():
            script_path.unlink(missing_ok=True)


@app.post("/collect/model")
def collect_model(req: ModelCollectRequest):
    tmp_path = None
    pipeline_id = _resolve_pipeline_id_or_400(req.pipeline_id)
    stage_order = req.stage_order or _default_stage_order(req.stage)
    try:
        tmp_path = _download_s3_to_temp(req.artifact_s3_uri)
        previous_index_ids = _manifest_ids_from_index()

        stdout, stderr, rc = _run_atlas(
            "model",
            "create",
            f"--name={req.ingredient_name}",
            f"--paths={tmp_path}",
            f"--ingredient-names={req.ingredient_name}",
            f"--author-org={req.author}",
            *_atlas_flags(),
            timeout=600,
        )

        manifest_id = _resolve_manifest_id_from_command(stdout + stderr, previous_index_ids)
        if rc != 0 and not manifest_id:
            raise HTTPException(status_code=500, detail=f"atlas-cli model create failed (rc={rc}): {stderr.strip()}")

        for dataset_manifest_id in req.linked_dataset_manifest_ids:
            if not manifest_id:
                break
            if str(dataset_manifest_id).strip().isdigit():
                log.warning(
                    "Skipping manifest link: id %r is an integer, not a resolvable manifest URN",
                    dataset_manifest_id,
                )
                continue

            _, link_stderr, link_rc = _run_atlas(
                "manifest",
                "link",
                f"--source={manifest_id}",
                f"--target={dataset_manifest_id}",
                "--storage-type=local-fs",
                f"--storage-url={MANIFESTS_DIR}",
            )
            if link_rc != 0:
                log.warning(
                    "manifest link failed for %s -> %s: %s",
                    manifest_id,
                    dataset_manifest_id,
                    link_stderr.strip(),
                )

        record = _register(
            req.artifact_s3_uri,
            manifest_id,
            pipeline_id,
            req.stage,
            stage_order,
            "model",
            req.ingredient_name,
            req.metadata,
            linked_manifest_ids=req.linked_dataset_manifest_ids,
        )

        return {
            "tracking_id": record["tracking_id"],
            "manifest_id": manifest_id,
            "pipeline_id": pipeline_id,
            "stage": req.stage,
            "stage_order": stage_order,
            "type": "model",
            "artifact_s3_uri": req.artifact_s3_uri,
            "linked_dataset_manifest_ids": req.linked_dataset_manifest_ids,
            "atlas_stdout": stdout.strip(),
        }
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/link")
def link_manifests(req: LinkRequest):
    stdout, stderr, rc = _run_atlas(
        "manifest",
        "link",
        f"--source={req.source_manifest_id}",
        f"--target={req.target_manifest_id}",
        "--storage-type=local-fs",
        f"--storage-url={MANIFESTS_DIR}",
    )
    if rc != 0:
        raise HTTPException(status_code=500, detail=f"manifest link failed: {stderr.strip()}")
    return {
        "linked": True,
        "source": req.source_manifest_id,
        "target": req.target_manifest_id,
        "atlas_stdout": stdout.strip(),
    }


@app.get("/lineage")
def lineage(pipeline_id: Optional[str] = None):
    resolved_pipeline_id = _resolve_pipeline_id_or_400(pipeline_id)
    entries = _ordered_entries_for_pipeline(resolved_pipeline_id)
    stages_complete = _pipeline_stage_names(entries)
    expected_stages = _expected_stage_names()

    chain = [
        {
            "pipeline_id": resolved_pipeline_id,
            "stage": record.get("stage"),
            "stage_order": record.get("stage_order", _default_stage_order(record.get("stage", ""))),
            "artifact_uri": _artifact_uri_for_record(uri, record),
            "tracking_id": _record_tracking_id(uri, record),
            "manifest_id": record.get("manifest_id"),
            "has_manifest": _is_resolvable_manifest_id(record.get("manifest_id")),
            "type": record.get("type"),
            "ingredient_name": record.get("ingredient_name"),
            "input_artifact_uris": record.get("input_artifact_uris", []),
            "linked_manifest_ids": record.get("linked_manifest_ids", []),
        }
        for uri, record in entries
    ]

    return {
        "pipeline_id": resolved_pipeline_id,
        "chain": chain,
        "stages_complete": stages_complete,
        "expected_stages": expected_stages,
        "chain_complete": bool(expected_stages) and all(stage in stages_complete for stage in expected_stages),
        "total_manifests": len(entries),
    }


@app.get("/pipeline/status")
def pipeline_status(pipeline_id: Optional[str] = None):
    resolved_pipeline_id = _resolve_pipeline_id_or_400(pipeline_id)
    entries = _ordered_entries_for_pipeline(resolved_pipeline_id)
    expected_stages = _expected_stage_names()
    discovered_stages = _pipeline_stage_names(entries)
    all_stage_names = expected_stages + [stage for stage in discovered_stages if stage not in expected_stages]

    stages = {}
    for stage in all_stage_names:
        stage_records = [record for _, record in entries if record.get("stage") == stage]
        stages[stage] = {
            "done": len(stage_records) > 0,
            "artifact_count": len(stage_records),
            "tracking_ids": [record.get("tracking_id") for record in stage_records],
            "manifest_count": sum(1 for record in stage_records if _is_resolvable_manifest_id(record.get("manifest_id"))),
            "manifest_ids": [record.get("manifest_id") for record in stage_records if record.get("manifest_id")],
            "stage_order": stage_records[0].get("stage_order", _default_stage_order(stage)) if stage_records else _default_stage_order(stage),
        }

    return {
        "pipeline_id": resolved_pipeline_id,
        "expected_stages": expected_stages,
        "stages": stages,
        "chain_complete": bool(expected_stages) and all(stages[stage]["done"] for stage in expected_stages),
    }


@app.get("/pipelines")
def list_pipelines():
    with _lock:
        entries = list(_registry.items())

    pipelines = {}
    for _, record in entries:
        pipeline_id = record.get("pipeline_id", _normalize_pipeline_id(DEFAULT_PIPELINE_ID))
        pipeline_summary = pipelines.setdefault(
            pipeline_id,
            {
                "artifact_count": 0,
                "stages": [],
            },
        )
        pipeline_summary["artifact_count"] += 1
        stage = record.get("stage")
        if stage and stage not in pipeline_summary["stages"]:
            pipeline_summary["stages"].append(stage)

    return {
        "default_pipeline_id": _normalize_pipeline_id(DEFAULT_PIPELINE_ID),
        "pipelines": pipelines,
    }


@app.get("/manifests")
def list_manifests(pipeline_id: Optional[str] = None, stage: Optional[str] = None):
    if pipeline_id is not None:
        pipeline_id = _resolve_pipeline_id_or_400(pipeline_id)
    entries = list(_filtered_registry(pipeline_id=pipeline_id, stage=stage).items())
    return {
        "count": len(entries),
        "manifests": [
            {
                "registry_key": uri,
                "tracking_id": _record_tracking_id(uri, record),
                "s3_uri": _artifact_uri_for_record(uri, record),
                "has_manifest": _is_resolvable_manifest_id(record.get("manifest_id")),
                **record,
            }
            for uri, record in entries
        ],
    }


@app.get("/export/{manifest_id:path}")
def export_manifest(manifest_id: str, depth: int = 5):
    stdout, stderr, rc = _run_atlas(
        "manifest",
        "export",
        f"--id={manifest_id}",
        f"--max-depth={depth}",
        "--encoding=json",
        "--storage-type=local-fs",
        f"--storage-url={MANIFESTS_DIR}",
    )
    if rc != 0:
        raise HTTPException(status_code=500, detail=f"manifest export failed: {stderr.strip()}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"raw_output": stdout.strip(), "stderr": stderr.strip()}


@app.get("/verify/{manifest_id:path}")
def verify_manifest(manifest_id: str):
    stdout, stderr, rc = _run_atlas(
        "manifest",
        "validate",
        f"--id={manifest_id}",
        "--storage-type=local-fs",
        f"--storage-url={MANIFESTS_DIR}",
    )
    return {
        "manifest_id": manifest_id,
        "valid": rc == 0,
        "output": stdout.strip(),
        "errors": stderr.strip() if rc != 0 else None,
    }


@app.get("/registry")
def registry(pipeline_id: Optional[str] = None, stage: Optional[str] = None):
    if pipeline_id is not None:
        pipeline_id = _resolve_pipeline_id_or_400(pipeline_id)
    return _filtered_registry(pipeline_id=pipeline_id, stage=stage)
