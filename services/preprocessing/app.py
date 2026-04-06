"""
Preprocessing Service
Stage 2 of the ML Provenance Pipeline

Reads raw IMDB JSON from S3, tokenizes with BertTokenizer, and stores
the tokenized tensors back to S3 for the fine-tuning stage.

API:
  GET  /health          - liveness probe
  POST /preprocess      - tokenize raw data and upload to S3
  GET  /status          - check if preprocessed data exists
"""

import os
import json
import logging
import hashlib
import datetime
import uuid
from typing import Optional

import urllib.request

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, BackgroundTasks
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Preprocessing Service",
    description="ML Pipeline Stage 2 – BERT tokenization of raw IMDB data",
    version="1.0.0",
)

# ── Config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT  = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET    = os.getenv("S3_BUCKET", "ml-provenance")
AWS_KEY      = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET   = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MAX_LENGTH   = int(os.getenv("MAX_SEQ_LENGTH", "128"))
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", "64"))   # tokenize in chunks to limit RAM
BERT_MODEL        = os.getenv("BERT_MODEL", "bert-base-uncased")
ATLAS_SIDECAR_URL = os.getenv("ATLAS_SIDECAR_URL", "")

_jobs: dict = {}
_last_manifest_id: Optional[str] = None


# ── Atlas Sidecar helper ──────────────────────────────────────────────────────
def _notify_sidecar(endpoint: str, payload: dict):
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


# ── S3 helper ─────────────────────────────────────────────────────────────────
def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


# ── Background preprocessing task ────────────────────────────────────────────
def _do_preprocess(job_id: str, split: str):
    _jobs[job_id] = {"status": "running", "split": split}
    try:
        s3 = get_s3()

        # ── 1. Load raw data ──────────────────────────────────────────────────
        raw_key = f"raw/{split}_data.json"
        log.info("[%s] Fetching %s from S3", job_id, raw_key)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=raw_key)
        records = json.loads(obj["Body"].read())

        texts  = [r["text"]  for r in records]
        labels = [r["label"] for r in records]

        # ── 2. Tokenize in chunks ─────────────────────────────────────────────
        log.info("[%s] Loading tokenizer %s", job_id, BERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

        all_input_ids      = []
        all_attention_mask = []
        all_token_type_ids = []

        for i in range(0, len(texts), CHUNK_SIZE):
            chunk = texts[i : i + CHUNK_SIZE]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            all_input_ids.extend(enc["input_ids"].tolist())
            all_attention_mask.extend(enc["attention_mask"].tolist())
            all_token_type_ids.extend(enc["token_type_ids"].tolist())
            log.info("[%s] Tokenized %d/%d", job_id, min(i + CHUNK_SIZE, len(texts)), len(texts))

        preprocessed = {
            "input_ids":      all_input_ids,
            "attention_mask": all_attention_mask,
            "token_type_ids": all_token_type_ids,
            "labels":         labels,
        }

        # ── 3. Upload to S3 ───────────────────────────────────────────────────
        payload  = json.dumps(preprocessed).encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest()

        data_key = f"preprocessed/{split}_tokenized.json"
        meta_key = f"preprocessed/{split}_meta.json"

        s3.put_object(Bucket=S3_BUCKET, Key=data_key, Body=payload, ContentType="application/json")

        meta = {
            "bert_model":   BERT_MODEL,
            "split":        split,
            "num_samples":  len(labels),
            "max_length":   MAX_LENGTH,
            "sha256":       checksum,
            "processed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "s3_uri":       f"s3://{S3_BUCKET}/{data_key}",
            "source_uri":   f"s3://{S3_BUCKET}/{raw_key}",
        }
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=meta_key,
            Body=json.dumps(meta, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        _jobs[job_id] = {
            "status":      "completed",
            "split":       split,
            "num_samples": len(labels),
            "max_length":  MAX_LENGTH,
            "sha256":      checksum,
            "s3_uri":      f"s3://{S3_BUCKET}/{data_key}",
        }
        log.info("[%s] Preprocessing complete. %d samples → %s", job_id, len(labels), data_key)

        # ── Notify Atlas sidecar with pipeline-step provenance ─────────────────
        global _last_manifest_id
        _notify_sidecar("pipeline", {
            "stage": "preprocessing",
            "input_s3_uris": [f"s3://{S3_BUCKET}/{raw_key}"],
            "output_s3_uri": f"s3://{S3_BUCKET}/{data_key}",
            "ingredient_name": f"IMDB {split} Tokenized",
            "author": "preprocessing-service",
            "build_script": (
                f"BertTokenizer.from_pretrained('{BERT_MODEL}') "
                f"max_length={MAX_LENGTH} truncation=True padding=max_length"
            ),
            "metadata": meta,
        })

        # Also register the tokenized output as a dataset artifact to get a
        # proper C2PA URN (pipeline generate-provenance uses integer IDs).
        dataset_resp = _notify_sidecar("dataset", {
            "stage": "preprocessing",
            "artifact_s3_uri": f"s3://{S3_BUCKET}/{data_key}",
            "ingredient_name": f"IMDB {split} Tokenized",
            "author": "preprocessing-service",
            "metadata": meta,
        })
        if dataset_resp:
            manifest_id = dataset_resp.get("manifest_id")
            _jobs[job_id]["manifest_id"] = manifest_id
            _last_manifest_id = manifest_id
            log.info("[%s] Atlas manifest: %s", job_id, manifest_id)

    except Exception as exc:
        log.exception("[%s] Preprocessing failed", job_id)
        _jobs[job_id] = {"status": "failed", "error": str(exc)}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}


@app.post("/preprocess")
def preprocess(background_tasks: BackgroundTasks, split: str = "train"):
    """
    Tokenize raw data for the given dataset split with BertTokenizer.

    - **split**: dataset split to process (`train` or `test`)
    """
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_do_preprocess, job_id, split)
    _jobs[job_id] = {"status": "queued", "split": split}
    log.info("Queued preprocessing job %s for split=%s", job_id, split)
    return {"job_id": job_id, "status": "queued", "split": split}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/provenance")
def provenance():
    """Return the most recent Atlas manifest ID registered by this service."""
    return {
        "service": "preprocessing",
        "manifest_id": _last_manifest_id,
        "sidecar_url": ATLAS_SIDECAR_URL or None,
    }


@app.get("/status")
def data_status(split: str = "train"):
    """Check whether tokenized data for a split exists in S3."""
    try:
        s3 = get_s3()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"preprocessed/{split}_meta.json")
        meta = json.loads(obj["Body"].read())
        return {"available": True, "meta": meta}
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"available": False}
        raise HTTPException(status_code=500, detail=str(e))
