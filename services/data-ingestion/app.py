"""
Data Ingestion Service
Stage 1 of the ML Provenance Pipeline

Downloads the IMDB text dataset from HuggingFace and stores raw data
to S3-compatible storage (MinIO locally, AWS S3 in cloud).

API:
  GET  /health        - liveness probe
  POST /ingest        - download dataset and upload to S3
  GET  /status        - check if raw data exists in S3
"""

import os
import json
import logging
import hashlib
import datetime
from typing import Optional

import urllib.request

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Data Ingestion Service",
    description="ML Pipeline Stage 1 – Downloads IMDB dataset and stores to S3",
    version="1.0.0",
)

# ── Config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT   = os.getenv("S3_ENDPOINT_URL")          # None = real AWS
S3_BUCKET     = os.getenv("S3_BUCKET", "ml-provenance")
AWS_KEY       = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET    = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DATASET_NAME      = os.getenv("DATASET_NAME", "imdb")
ATLAS_SIDECAR_URL = os.getenv("ATLAS_SIDECAR_URL", "")

# In-memory job state
_jobs: dict = {}


# ── S3 helper ─────────────────────────────────────────────────────────────────
def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


# ── Atlas Sidecar helper ──────────────────────────────────────────────────────
def _notify_sidecar(endpoint: str, payload: dict) -> Optional[dict]:
    """Fire-and-forget call to atlas-sidecar. Pipeline continues on failure."""
    if not ATLAS_SIDECAR_URL:
        return None
    url = f"{ATLAS_SIDECAR_URL}/collect/{endpoint}"
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        log.warning("Atlas sidecar notification failed (%s): %s", url, exc)
        return None


def ensure_bucket(s3_client):
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
    except ClientError:
        s3_client.create_bucket(Bucket=S3_BUCKET)
        log.info("Created bucket %s", S3_BUCKET)


# ── Background ingestion task ─────────────────────────────────────────────────
def _do_ingest(job_id: str, split: str, num_samples: int):
    _jobs[job_id] = {"status": "running", "split": split}
    try:
        log.info("[%s] Loading %s dataset split=%s samples=%d", job_id, DATASET_NAME, split, num_samples)
        dataset = load_dataset(DATASET_NAME, split=split)
        # Shuffle before slicing: IMDB is ordered (all negatives first, then
        # all positives), so an unshuffled head is entirely one class.
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

        records = [{"text": item["text"], "label": item["label"]} for item in dataset]

        # Compute checksum for provenance
        payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest()

        s3 = get_s3()
        ensure_bucket(s3)

        data_key = f"raw/{split}_data.json"
        meta_key = f"raw/{split}_meta.json"

        s3.put_object(Bucket=S3_BUCKET, Key=data_key, Body=payload, ContentType="application/json")

        meta = {
            "dataset": DATASET_NAME,
            "split": split,
            "num_samples": len(records),
            "sha256": checksum,
            "ingested_at": datetime.datetime.utcnow().isoformat() + "Z",
            "s3_uri": f"s3://{S3_BUCKET}/{data_key}",
        }
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=meta_key,
            Body=json.dumps(meta, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        _jobs[job_id] = {
            "status": "completed",
            "split": split,
            "num_samples": len(records),
            "sha256": checksum,
            "s3_uri": f"s3://{S3_BUCKET}/{data_key}",
        }
        log.info("[%s] Ingestion complete. %d samples → %s", job_id, len(records), data_key)

        # ── Notify Atlas sidecar (called at completion of this pipeline step) ─
        sidecar_resp = _notify_sidecar("dataset", {
            "stage": "data-ingestion",
            "artifact_s3_uri": f"s3://{S3_BUCKET}/{data_key}",
            "ingredient_name": f"IMDB {split} Dataset",
            "author": "data-ingestion-service",
            "metadata": meta,
        })
        if sidecar_resp:
            _jobs[job_id]["manifest_id"] = sidecar_resp.get("manifest_id")
            log.info("[%s] Atlas manifest: %s", job_id, sidecar_resp.get("manifest_id"))

    except Exception as exc:
        log.exception("[%s] Ingestion failed", job_id)
        _jobs[job_id] = {"status": "failed", "error": str(exc)}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "data-ingestion"}


@app.post("/ingest")
def ingest(
    background_tasks: BackgroundTasks,
    split: str = "train",
    num_samples: int = 500,
):
    """
    Start an asynchronous data ingestion job.

    - **split**: HuggingFace dataset split (`train` or `test`)
    - **num_samples**: number of samples to ingest (default 500 for local dev)
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_do_ingest, job_id, split, num_samples)
    _jobs[job_id] = {"status": "queued", "split": split}
    log.info("Queued ingestion job %s", job_id)
    return {"job_id": job_id, "status": "queued", "split": split, "num_samples": num_samples}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/status")
def data_status(split: str = "train"):
    """Check whether raw data for a split already exists in S3."""
    try:
        s3 = get_s3()
        key = f"raw/{split}_meta.json"
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        meta = json.loads(obj["Body"].read())
        return {"available": True, "meta": meta}
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"available": False}
        raise HTTPException(status_code=500, detail=str(e))
