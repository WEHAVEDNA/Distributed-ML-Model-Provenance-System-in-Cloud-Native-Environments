"""
Preprocessing Service  –  port 8002
--------------------------------------
POST /preprocess?split=train   → start job, return {job_id}
GET  /jobs/{job_id}            → status / result
GET  /health                   → {"status":"ok"}
"""

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from fastapi import FastAPI, HTTPException, Query
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [preprocessing] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Preprocessing Service")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT",  "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS   = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
BUCKET         = os.environ.get("ARTIFACT_BUCKET",  "atlas-artifacts")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME",   "bert-base-uncased")
MAX_LENGTH     = int(os.environ.get("MAX_LENGTH",    "128"))
DATASET_NAME   = os.environ.get("DATASET_NAME",     "imdb")

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


def latest_raw_key(s3, split: str) -> str:
    """Find the most recently uploaded raw dataset key for this split."""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"datasets/raw/{DATASET_NAME}_{split}_")
    objects = resp.get("Contents", [])
    if not objects:
        raise RuntimeError(f"No raw dataset found for split={split}. Run ingestion first.")
    return sorted(objects, key=lambda o: o["LastModified"], reverse=True)[0]["Key"]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip('"\'').strip()


def run_preprocess(job_id: str, split: str):
    jobs[job_id]["status"] = "running"
    try:
        s3 = s3_client()

        raw_key = latest_raw_key(s3, split)
        log.info(f"[{job_id}] Reading s3://{BUCKET}/{raw_key}")
        obj     = s3.get_object(Bucket=BUCKET, Key=raw_key)
        records = [json.loads(l) for l in obj["Body"].read().decode().splitlines() if l.strip()]
        log.info(f"[{job_id}] Read {len(records)} records")

        log.info(f"[{job_id}] Loading tokenizer {TOKENIZER_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        processed = []
        for r in records:
            text = clean(r["text"])
            if not text:
                continue
            enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True,
                            padding="max_length", return_attention_mask=True)
            processed.append({
                "text":           text,
                "label":          int(r["label"]),
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            })
        log.info(f"[{job_id}] Tokenised {len(processed)} records")

        data  = "\n".join(json.dumps(r) for r in processed).encode("utf-8")
        sha   = hashlib.sha256(data).hexdigest()
        key   = f"datasets/processed/{DATASET_NAME}_{split}_tokenized.jsonl"

        s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType="application/x-ndjson")
        log.info(f"[{job_id}] Uploaded s3://{BUCKET}/{key} ({len(data):,} bytes)")

        jobs[job_id].update({
            "status":       "completed",
            "num_samples":  len(processed),
            "max_length":   MAX_LENGTH,
            "tokenizer":    TOKENIZER_NAME,
            "sha256":       sha,
            "s3_uri":       f"s3://{BUCKET}/{key}",
            "s3_key":       key,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

    except Exception as exc:
        log.exception(f"[{job_id}] Preprocessing failed")
        jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}


@app.post("/preprocess")
def preprocess(split: str = Query(default="train")):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":     job_id,
        "status":     "pending",
        "split":      split,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    threading.Thread(target=run_preprocess, args=(job_id, split), daemon=True).start()
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs")
def list_jobs():
    return list(jobs.values())
