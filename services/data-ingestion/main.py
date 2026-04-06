"""
Data Ingestion Service  –  port 8001
--------------------------------------
POST /ingest?split=train&num_samples=500  → start job, return {job_id}
GET  /jobs/{job_id}                       → status / result
GET  /health                              → {"status":"ok"}
"""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from datasets import load_dataset
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ingestion] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Data Ingestion Service")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT",  "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS   = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
BUCKET         = os.environ.get("ARTIFACT_BUCKET",  "atlas-artifacts")
DATASET_NAME   = os.environ.get("DATASET_NAME",     "imdb")

# In-memory job store
jobs: dict[str, dict] = {}


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS,
        aws_secret_access_key=MINIO_SECRET,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(s3):
    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if BUCKET not in existing:
        s3.create_bucket(Bucket=BUCKET)
        log.info(f"Created bucket: {BUCKET}")


def run_ingest(job_id: str, split: str, num_samples: int):
    jobs[job_id]["status"] = "running"
    try:
        s3 = s3_client()
        ensure_bucket(s3)

        log.info(f"[{job_id}] Loading {DATASET_NAME} split={split} num_samples={num_samples}")
        ds = load_dataset(DATASET_NAME, split=f"{split}[:{num_samples}]", trust_remote_code=True)
        log.info(f"[{job_id}] Loaded {len(ds)} examples")

        lines = [json.dumps({"text": row["text"], "label": row["label"]}) for row in ds]
        data  = "\n".join(lines).encode("utf-8")
        sha   = hashlib.sha256(data).hexdigest()
        key   = f"datasets/raw/{DATASET_NAME}_{split}_{num_samples}.jsonl"

        s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType="application/x-ndjson")
        log.info(f"[{job_id}] Uploaded s3://{BUCKET}/{key} ({len(data):,} bytes)")

        jobs[job_id].update({
            "status":      "completed",
            "num_samples": len(ds),
            "sha256":      sha,
            "s3_uri":      f"s3://{BUCKET}/{key}",
            "s3_key":      key,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

    except Exception as exc:
        log.exception(f"[{job_id}] Ingestion failed")
        jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "service": "data-ingestion"}


@app.post("/ingest")
def ingest(
    split:       str = Query(default="train"),
    num_samples: int = Query(default=500),
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":     job_id,
        "status":     "pending",
        "split":      split,
        "num_samples": num_samples,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    threading.Thread(target=run_ingest, args=(job_id, split, num_samples), daemon=True).start()
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs")
def list_jobs():
    return list(jobs.values())
