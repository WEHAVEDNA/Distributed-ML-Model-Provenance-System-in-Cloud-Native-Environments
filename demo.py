#!/usr/bin/env python3
"""
ML Provenance Pipeline – Demo Script
BU EC528 Spring 2026 / Intel Labs

Runs the full three-stage pipeline and displays the provenance chain.

Usage:
    python demo.py                      # 100 samples, skip training
    python demo.py --samples 200        # 200 samples
    python demo.py --train              # also run fine-tuning (slow on CPU)
    python demo.py --samples 500 --train

Prerequisites:
    docker compose up --build           # start all services first
    pip install requests                # requests must be installed in your env
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# ── Service URLs ──────────────────────────────────────────────────────────────
INGEST_URL   = "http://localhost:8001"
PREPROC_URL  = "http://localhost:8002"
FINETUNE_URL = "http://localhost:8003"
SIDECAR_URL  = "http://localhost:8004"

POLL_INTERVAL = 3   # seconds

# ── Terminal colours ──────────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def h1(msg: str):  print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'═'*60}{RESET}")
def h2(msg: str):  print(f"\n{BOLD}── {msg} ──{RESET}")
def ok(msg: str):  print(f"{GREEN}  ✓{RESET}  {msg}")
def info(msg: str): print(f"     {msg}")
def warn(msg: str): print(f"{YELLOW}  ⚠{RESET}  {msg}")


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _request(method: str, url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode() if payload else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"HTTP {e.code} from {url}: {body[:300]}", file=sys.stderr)
        raise
    except urllib.error.URLError as e:
        print(f"Cannot reach {url}: {e.reason}", file=sys.stderr)
        raise


def get(url: str, timeout: int = 10) -> dict:
    return _request("GET", url, timeout=timeout)


def post(url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    return _request("POST", url, payload=payload, timeout=timeout)


def wait_for_job(base_url: str, job_id: str, label: str, timeout_s: int = 1800) -> dict:
    """Poll /jobs/{job_id} until completed or failed, printing progress."""
    deadline = time.time() + timeout_s
    elapsed = 0
    while time.time() < deadline:
        data = get(f"{base_url}/jobs/{job_id}")
        status = data.get("status", "unknown")
        if status == "completed":
            ok(f"{label} completed  ({elapsed}s)")
            return data
        if status == "failed":
            print(f"\n  ERROR: {label} failed — {data.get('error')}", file=sys.stderr)
            sys.exit(1)
        # Show epoch info for training jobs
        extra = ""
        if "epoch" in data and "losses" in data:
            ep = data["epoch"]
            losses = data["losses"]
            if losses:
                extra = f"  epoch={ep}  loss={losses[-1]:.4f}"
        print(f"  {label}: {status}{extra}  ({elapsed}s)\r", end="", flush=True)
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    print(f"\n  ERROR: {label} timed out after {timeout_s}s", file=sys.stderr)
    sys.exit(1)


# ── Main demo ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ML Provenance Pipeline demo")
    parser.add_argument("--samples", type=int, default=100,
                        help="IMDB samples to ingest (default 100, use 500+ for good accuracy)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--train", action="store_true",
                        help="Also run fine-tuning (slow on CPU, ~30 min with 500 samples)")
    args = parser.parse_args()

    h1("ML Provenance Pipeline Demo")
    info(f"samples={args.samples}  split={args.split}  run_training={args.train}")

    # ── Health checks ─────────────────────────────────────────────────────────
    h2("Service health")
    services = [
        ("data-ingestion", INGEST_URL),
        ("preprocessing",  PREPROC_URL),
        ("fine-tuning",    FINETUNE_URL),
        ("atlas-sidecar",  SIDECAR_URL),
    ]
    for name, url in services:
        try:
            body = get(f"{url}/health", timeout=5)
            ok(f"{name:20s}  {body.get('status', '?')}")
            if name == "atlas-sidecar":
                atlas_ver = body.get("atlas_cli", "unavailable")
                key_ok    = body.get("key_exists", False)
                info(f"atlas-cli: {atlas_ver}   signing-key: {'✓' if key_ok else '✗'}")
        except Exception as exc:
            print(f"\n  ERROR: {name} is not reachable ({exc})", file=sys.stderr)
            print("  Run:  docker compose up --build", file=sys.stderr)
            sys.exit(1)

    # ── Stage 1: Data Ingestion ───────────────────────────────────────────────
    h2("Stage 1 – Data Ingestion")
    info(f"Downloading IMDB ({args.samples} samples, split={args.split}) → MinIO")
    resp = post(f"{INGEST_URL}/ingest?split={args.split}&num_samples={args.samples}")
    job_id = resp["job_id"]
    info(f"Job ID: {job_id}")
    ingest_result = wait_for_job(INGEST_URL, job_id, "Data Ingestion", timeout_s=600)

    info(f"samples : {ingest_result.get('num_samples')}")
    info(f"sha256  : {ingest_result.get('sha256', '?')[:16]}…")
    info(f"s3_uri  : {ingest_result.get('s3_uri')}")
    if ingest_result.get("manifest_id"):
        ok(f"Atlas manifest: {ingest_result['manifest_id']}")
    else:
        warn("No Atlas manifest (sidecar may not have registered this artifact yet)")

    # ── Stage 2: Preprocessing ────────────────────────────────────────────────
    h2("Stage 2 – Preprocessing (BERT tokenization)")
    info(f"Tokenizing with bert-base-uncased (max_length=128)")
    resp = post(f"{PREPROC_URL}/preprocess?split={args.split}")
    job_id = resp["job_id"]
    info(f"Job ID: {job_id}")
    preproc_result = wait_for_job(PREPROC_URL, job_id, "Preprocessing", timeout_s=600)

    info(f"samples    : {preproc_result.get('num_samples')}")
    info(f"max_length : {preproc_result.get('max_length')}")
    info(f"sha256     : {preproc_result.get('sha256', '?')[:16]}…")
    info(f"s3_uri     : {preproc_result.get('s3_uri')}")
    if preproc_result.get("manifest_id"):
        ok(f"Atlas manifest: {preproc_result['manifest_id']}")
    else:
        warn("No Atlas manifest from preprocessing")

    # ── Stage 3: Fine-Tuning (optional) ──────────────────────────────────────
    train_result = None
    if args.train:
        h2("Stage 3 – Fine-Tuning (BERT on IMDB)")
        warn("This can take 10–60 min on CPU depending on sample count.")
        info(f"Starting fine-tuning job (split={args.split})")
        resp = post(f"{FINETUNE_URL}/train?split={args.split}")
        job_id = resp["job_id"]
        info(f"Job ID: {job_id}   device: {resp.get('device')}")
        train_result = wait_for_job(FINETUNE_URL, job_id, "Fine-Tuning", timeout_s=7200)

        info(f"epochs      : {train_result.get('epochs')}")
        info(f"epoch_losses: {train_result.get('epoch_losses')}")
        info(f"sha256      : {train_result.get('sha256', '?')[:16]}…")
        info(f"model_uri   : {train_result.get('s3_uri')}")
        if train_result.get("manifest_id"):
            ok(f"Atlas manifest: {train_result['manifest_id']}")
        else:
            warn("No Atlas manifest from fine-tuning")
    else:
        warn("Skipping fine-tuning (pass --train to include it)")

    # ── Provenance chain ──────────────────────────────────────────────────────
    h2("Provenance chain  (GET :8004/lineage)")
    try:
        lineage = get(f"{SIDECAR_URL}/lineage", timeout=10)
        chain   = lineage.get("chain", [])
        if chain:
            for entry in chain:
                stage = entry.get("stage", "?")
                mid   = entry.get("manifest_id", "none")
                kind  = entry.get("type", "?")
                uri   = entry.get("artifact_uri", "")
                print(f"  {BOLD}{stage:20s}{RESET}  {kind:10s}  manifest={mid}")
                info(f"  └─ {uri}")
        else:
            warn("No manifests in the provenance chain yet.")

        complete = lineage.get("chain_complete", False)
        stages_done = lineage.get("stages_complete", [])
        info(f"\n  Stages with provenance: {', '.join(stages_done) or 'none'}")
        if complete:
            ok("Full provenance chain complete  (raw → tokenized → model)")
        else:
            warn(f"Chain incomplete — run with --train and --samples 200+ for full chain")
    except Exception as exc:
        warn(f"Could not fetch lineage: {exc}")

    # ── Pipeline status ───────────────────────────────────────────────────────
    h2("Pipeline status  (GET :8004/pipeline/status)")
    try:
        status = get(f"{SIDECAR_URL}/pipeline/status", timeout=10)
        for stage, info_dict in status.get("stages", {}).items():
            done  = "✓" if info_dict["done"] else "✗"
            count = info_dict["artifact_count"]
            print(f"  {done}  {stage:25s}  {count} artifact(s)")
    except Exception as exc:
        warn(f"Could not fetch pipeline status: {exc}")

    # ── Inference smoke test (only if model exists) ───────────────────────────
    if args.train and train_result:
        h2("Inference smoke test  (POST :8003/predict)")
        samples = [
            ("A truly spectacular film. Moved me to tears.", "positive"),
            ("Complete garbage. Poorly directed and acted.", "negative"),
            ("An okay film, nothing special but watchable.", None),
        ]
        for text, expected in samples:
            try:
                result = post(f"{FINETUNE_URL}/predict", {"text": text}, timeout=30)
                label  = result["label"]
                conf   = result["confidence"]
                icon   = "✓" if expected is None or label == expected else "✗"
                print(f"  {icon}  [{label:8s} {conf:.3f}]  {text[:60]}")
            except Exception as exc:
                warn(f"Predict failed: {exc}")

    # ── Useful endpoints summary ───────────────────────────────────────────────
    h2("Explore further")
    print(f"  Sidecar registry : {SIDECAR_URL}/registry")
    print(f"  Pipeline status  : {SIDECAR_URL}/pipeline/status")
    print(f"  Provenance chain : {SIDECAR_URL}/lineage")
    print(f"  Interactive docs : http://localhost:8001/docs  (and 8002, 8003, 8004)")
    print()


if __name__ == "__main__":
    main()
