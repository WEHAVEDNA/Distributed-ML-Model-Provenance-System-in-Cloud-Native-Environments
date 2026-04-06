"""
Preprocessing Service
Stage 2 of the ML Provenance Pipeline

Reads raw records from S3, tokenizes them, and stores the tokenized dataset
back to S3 under a pipeline-specific namespace.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import uuid
from typing import Optional

import urllib.request

import boto3
from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, FastAPI, HTTPException
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Preprocessing Service",
    description="ML Pipeline Stage 2 - Tokenizes pipeline-scoped raw data",
    version="1.1.0",
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET = os.getenv("S3_BUCKET", "ml-provenance")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MAX_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "128"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "64"))
BERT_MODEL = os.getenv("BERT_MODEL", "bert-base-uncased")
ATLAS_SIDECAR_URL = os.getenv("ATLAS_SIDECAR_URL", "")
PIPELINE_ROOT_PREFIX = os.getenv("PIPELINE_ROOT_PREFIX", "pipelines")
PIPELINE_STAGE = "preprocessing"
PIPELINE_STAGE_ORDER = 20
DEFAULT_PIPELINE_ID = os.getenv("PIPELINE_ID", "default")

_jobs: dict = {}
_last_manifest_ids: dict[str, str] = {}


def _normalize_pipeline_id(pipeline_id: Optional[str]) -> str:
    candidate = (pipeline_id or DEFAULT_PIPELINE_ID).strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", candidate).strip("._-")
    if not normalized:
        raise ValueError("pipeline_id must contain at least one alphanumeric character")
    return normalized


def _resolve_pipeline_id(pipeline_id: Optional[str]) -> str:
    try:
        return _normalize_pipeline_id(pipeline_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _pipeline_prefix(pipeline_id: str) -> str:
    return f"{PIPELINE_ROOT_PREFIX}/{pipeline_id}"


def _raw_data_key(pipeline_id: str, split: str) -> str:
    return f"{_pipeline_prefix(pipeline_id)}/raw/{split}_data.json"


def _preprocessed_data_key(pipeline_id: str, split: str) -> str:
    return f"{_pipeline_prefix(pipeline_id)}/preprocessed/{split}_tokenized.json"


def _preprocessed_meta_key(pipeline_id: str, split: str) -> str:
    return f"{_pipeline_prefix(pipeline_id)}/preprocessed/{split}_meta.json"


def _s3_uri(key: str) -> str:
    return f"s3://{S3_BUCKET}/{key}"


def _is_cancel_requested(job_id: str) -> bool:
    return bool(_jobs.get(job_id, {}).get("cancel_requested"))


def _cancel_job_state(
    job_id: str,
    *,
    split: str,
    pipeline_id: str,
    num_samples: int = 0,
    tokenized_samples: int = 0,
):
    _jobs[job_id] = {
        "status": "cancelled",
        "split": split,
        "pipeline_id": pipeline_id,
        "num_samples": num_samples,
        "tokenized_samples": tokenized_samples,
        "cancel_requested": True,
    }
    log.info("[%s] Preprocessing cancelled", job_id)


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


def _notify_sidecar(endpoint: str, payload: dict):
    if not ATLAS_SIDECAR_URL:
        return None
    url = f"{ATLAS_SIDECAR_URL}/collect/{endpoint}"
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        log.warning("Atlas sidecar notification failed (%s): %s", url, exc)
        return None


def _lookup_manifest_ids(pipeline_id: str, s3_uris: list[str]) -> list[str]:
    if not ATLAS_SIDECAR_URL or not s3_uris:
        return []
    try:
        registry_url = f"{ATLAS_SIDECAR_URL}/registry?pipeline_id={pipeline_id}"
        req = urllib.request.Request(registry_url, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            registry = json.loads(resp.read())
    except Exception as exc:
        log.warning("Could not query sidecar registry for pipeline=%s: %s", pipeline_id, exc)
        return []

    manifest_ids = []
    for uri in s3_uris:
        manifest_id = registry.get(uri, {}).get("manifest_id")
        if manifest_id:
            manifest_ids.append(manifest_id)
    return manifest_ids


def _do_preprocess(job_id: str, split: str, pipeline_id: str):
    existing_job = _jobs.get(job_id, {})
    if existing_job.get("cancel_requested"):
        _cancel_job_state(job_id, split=split, pipeline_id=pipeline_id)
        return

    _jobs[job_id] = {
        "status": "running",
        "split": split,
        "pipeline_id": pipeline_id,
        "cancel_requested": existing_job.get("cancel_requested", False),
    }
    try:
        s3 = get_s3()

        raw_key = _raw_data_key(pipeline_id, split)
        log.info("[%s] Fetching %s from S3", job_id, raw_key)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=raw_key)
        records = json.loads(obj["Body"].read())

        if _is_cancel_requested(job_id):
            _cancel_job_state(job_id, split=split, pipeline_id=pipeline_id, num_samples=len(records))
            return

        texts = [record["text"] for record in records]
        labels = [record["label"] for record in records]

        log.info("[%s] Loading tokenizer %s", job_id, BERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        tokenized_samples = 0

        for index in range(0, len(texts), CHUNK_SIZE):
            if _is_cancel_requested(job_id):
                _cancel_job_state(
                    job_id,
                    split=split,
                    pipeline_id=pipeline_id,
                    num_samples=len(labels),
                    tokenized_samples=tokenized_samples,
                )
                return

            chunk = texts[index:index + CHUNK_SIZE]
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
            tokenized_samples = min(index + CHUNK_SIZE, len(texts))
            _jobs[job_id]["tokenized_samples"] = tokenized_samples
            log.info("[%s] Tokenized %d/%d", job_id, tokenized_samples, len(texts))

        preprocessed = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "token_type_ids": all_token_type_ids,
            "labels": labels,
        }

        payload = json.dumps(preprocessed).encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest()

        data_key = _preprocessed_data_key(pipeline_id, split)
        meta_key = _preprocessed_meta_key(pipeline_id, split)

        if _is_cancel_requested(job_id):
            _cancel_job_state(
                job_id,
                split=split,
                pipeline_id=pipeline_id,
                num_samples=len(labels),
                tokenized_samples=tokenized_samples,
            )
            return

        s3.put_object(Bucket=S3_BUCKET, Key=data_key, Body=payload, ContentType="application/json")

        meta = {
            "pipeline_id": pipeline_id,
            "stage": PIPELINE_STAGE,
            "stage_order": PIPELINE_STAGE_ORDER,
            "bert_model": BERT_MODEL,
            "split": split,
            "num_samples": len(labels),
            "max_length": MAX_LENGTH,
            "sha256": checksum,
            "processed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "s3_uri": _s3_uri(data_key),
            "source_uri": _s3_uri(raw_key),
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
            "pipeline_id": pipeline_id,
            "num_samples": len(labels),
            "tokenized_samples": len(labels),
            "max_length": MAX_LENGTH,
            "sha256": checksum,
            "s3_uri": _s3_uri(data_key),
        }
        log.info("[%s] Preprocessing complete. %d samples -> %s", job_id, len(labels), data_key)

        input_uri = _s3_uri(raw_key)
        _notify_sidecar(
            "pipeline",
            {
                "pipeline_id": pipeline_id,
                "stage": PIPELINE_STAGE,
                "stage_order": PIPELINE_STAGE_ORDER,
                "input_s3_uris": [input_uri],
                "output_s3_uri": _s3_uri(data_key),
                "ingredient_name": f"{pipeline_id} {split} tokenized dataset",
                "author": "preprocessing-service",
                "build_script": (
                    f"BertTokenizer.from_pretrained('{BERT_MODEL}') "
                    f"max_length={MAX_LENGTH} truncation=True padding=max_length"
                ),
                "metadata": meta,
            },
        )

        linked_manifest_ids = _lookup_manifest_ids(pipeline_id, [input_uri])
        dataset_resp = _notify_sidecar(
            "dataset",
            {
                "pipeline_id": pipeline_id,
                "stage": PIPELINE_STAGE,
                "stage_order": PIPELINE_STAGE_ORDER,
                "artifact_s3_uri": _s3_uri(data_key),
                "ingredient_name": f"{pipeline_id} {split} tokenized dataset",
                "author": "preprocessing-service",
                "linked_manifest_ids": linked_manifest_ids,
                "metadata": meta,
            },
        )
        if dataset_resp:
            manifest_id = dataset_resp.get("manifest_id")
            _jobs[job_id]["manifest_id"] = manifest_id
            if manifest_id:
                _last_manifest_ids[pipeline_id] = manifest_id
            log.info("[%s] Atlas manifest: %s", job_id, manifest_id)

    except Exception as exc:
        log.exception("[%s] Preprocessing failed", job_id)
        _jobs[job_id] = {
            "status": "failed",
            "pipeline_id": pipeline_id,
            "split": split,
            "error": str(exc),
        }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "preprocessing",
        "default_pipeline_id": _normalize_pipeline_id(DEFAULT_PIPELINE_ID),
        "pipeline_root_prefix": PIPELINE_ROOT_PREFIX,
    }


@app.post("/preprocess")
def preprocess(
    background_tasks: BackgroundTasks,
    split: str = "train",
    pipeline_id: Optional[str] = None,
):
    resolved_pipeline_id = _resolve_pipeline_id(pipeline_id)
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_do_preprocess, job_id, split, resolved_pipeline_id)
    _jobs[job_id] = {
        "status": "queued",
        "split": split,
        "pipeline_id": resolved_pipeline_id,
        "cancel_requested": False,
    }
    log.info("Queued preprocessing job %s for pipeline=%s", job_id, resolved_pipeline_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "split": split,
        "pipeline_id": resolved_pipeline_id,
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    status = job.get("status")
    if status in {"completed", "failed", "cancelled"}:
        return job

    updated_job = {**job, "status": "cancel_requested", "cancel_requested": True}
    _jobs[job_id] = updated_job
    log.info("[%s] Cancellation requested", job_id)
    return updated_job


@app.get("/provenance")
def provenance(pipeline_id: Optional[str] = None):
    resolved_pipeline_id = _resolve_pipeline_id(pipeline_id)
    return {
        "service": "preprocessing",
        "pipeline_id": resolved_pipeline_id,
        "manifest_id": _last_manifest_ids.get(resolved_pipeline_id),
        "sidecar_url": ATLAS_SIDECAR_URL or None,
    }


@app.get("/status")
def data_status(split: str = "train", pipeline_id: Optional[str] = None):
    resolved_pipeline_id = _resolve_pipeline_id(pipeline_id)
    try:
        s3 = get_s3()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=_preprocessed_meta_key(resolved_pipeline_id, split))
        meta = json.loads(obj["Body"].read())
        return {"available": True, "pipeline_id": resolved_pipeline_id, "meta": meta}
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"available": False, "pipeline_id": resolved_pipeline_id}
        raise HTTPException(status_code=500, detail=str(exc))
