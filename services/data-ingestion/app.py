"""
Data Ingestion Service
Stage 1 of the ML Provenance Pipeline

Downloads the configured text dataset from HuggingFace and stores raw data
to S3-compatible storage. Artifacts are namespaced by pipeline ID so multiple
pipelines can reuse the same service without colliding in storage or provenance.
"""

import datetime
import hashlib
import json
import logging
import os
import re
from typing import Optional

import urllib.request

import boto3
from botocore.exceptions import ClientError
from datasets import load_dataset
from fastapi import BackgroundTasks, FastAPI, HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Data Ingestion Service",
    description="ML Pipeline Stage 1 - Downloads a dataset split and stores it in S3",
    version="1.1.0",
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET = os.getenv("S3_BUCKET", "ml-provenance")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DATASET_NAME = os.getenv("DATASET_NAME", "imdb")
ATLAS_SIDECAR_URL = os.getenv("ATLAS_SIDECAR_URL", "")
PIPELINE_ROOT_PREFIX = os.getenv("PIPELINE_ROOT_PREFIX", "pipelines")
PIPELINE_STAGE = "data-ingestion"
PIPELINE_STAGE_ORDER = 10
DEFAULT_PIPELINE_ID = os.getenv("PIPELINE_ID", "default")
DEPLOYMENT_MODE = os.getenv(
    "DEPLOYMENT_MODE",
    "kubernetes" if os.getenv("KUBERNETES_SERVICE_HOST") else "local",
)
POD_NAME = os.getenv("POD_NAME")
POD_NAMESPACE = os.getenv("POD_NAMESPACE")
NODE_NAME = os.getenv("NODE_NAME")

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


def _raw_meta_key(pipeline_id: str, split: str) -> str:
    return f"{_pipeline_prefix(pipeline_id)}/raw/{split}_meta.json"


def _s3_uri(key: str) -> str:
    return f"s3://{S3_BUCKET}/{key}"


def _is_cancel_requested(job_id: str) -> bool:
    return bool(_jobs.get(job_id, {}).get("cancel_requested"))


def _cancel_job_state(
    job_id: str,
    *,
    split: str,
    pipeline_id: str,
    num_samples: int,
):
    _jobs[job_id] = {
        "status": "cancelled",
        "split": split,
        "pipeline_id": pipeline_id,
        "num_samples": num_samples,
        "cancel_requested": True,
    }
    log.info("[%s] Ingestion cancelled", job_id)


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


def _notify_sidecar(endpoint: str, payload: dict) -> Optional[dict]:
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


def ensure_bucket(s3_client):
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
    except ClientError:
        s3_client.create_bucket(Bucket=S3_BUCKET)
        log.info("Created bucket %s", S3_BUCKET)


def _do_ingest(job_id: str, split: str, num_samples: int, pipeline_id: str):  # noqa: C901
    existing_job = _jobs.get(job_id, {})
    if existing_job.get("cancel_requested"):
        _cancel_job_state(job_id, split=split, pipeline_id=pipeline_id, num_samples=num_samples)
        return

    _jobs[job_id] = {
        "status": "running",
        "split": split,
        "pipeline_id": pipeline_id,
        "num_samples": num_samples,
        "cancel_requested": existing_job.get("cancel_requested", False),
    }
    try:
        log.info(
            "[%s] Loading dataset=%s split=%s samples=%d pipeline=%s",
            job_id,
            DATASET_NAME,
            split,
            num_samples,
            pipeline_id,
        )
        dataset = load_dataset(DATASET_NAME, split=split)

        if _is_cancel_requested(job_id):
            _cancel_job_state(job_id, split=split, pipeline_id=pipeline_id, num_samples=num_samples)
            return

        half = num_samples // 2
        neg = dataset.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(half))
        pos = dataset.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(half))
        import datasets as hf_datasets

        dataset = hf_datasets.concatenate_datasets([neg, pos]).shuffle(seed=42)
        records = [{"text": item["text"], "label": item["label"]} for item in dataset]

        if _is_cancel_requested(job_id):
            _cancel_job_state(job_id, split=split, pipeline_id=pipeline_id, num_samples=len(records))
            return

        payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest()

        s3 = get_s3()
        ensure_bucket(s3)

        data_key = _raw_data_key(pipeline_id, split)
        meta_key = _raw_meta_key(pipeline_id, split)

        if _is_cancel_requested(job_id):
            _cancel_job_state(job_id, split=split, pipeline_id=pipeline_id, num_samples=len(records))
            return

        s3.put_object(Bucket=S3_BUCKET, Key=data_key, Body=payload, ContentType="application/json")

        meta = {
            "pipeline_id": pipeline_id,
            "stage": PIPELINE_STAGE,
            "stage_order": PIPELINE_STAGE_ORDER,
            "dataset": DATASET_NAME,
            "split": split,
            "num_samples": len(records),
            "sha256": checksum,
            "ingested_at": datetime.datetime.utcnow().isoformat() + "Z",
            "s3_uri": _s3_uri(data_key),
        }
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=meta_key,
            Body=json.dumps(meta, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        job_state = {
            "status": "completed",
            "split": split,
            "pipeline_id": pipeline_id,
            "num_samples": len(records),
            "sha256": checksum,
            "s3_uri": _s3_uri(data_key),
        }
        _jobs[job_id] = {**job_state, "status": "finalizing"}
        log.info("[%s] Ingestion complete. %d samples -> %s", job_id, len(records), data_key)

        sidecar_resp = _notify_sidecar(
            "dataset",
            {
                "pipeline_id": pipeline_id,
                "stage": PIPELINE_STAGE,
                "stage_order": PIPELINE_STAGE_ORDER,
                "artifact_s3_uri": _s3_uri(data_key),
                "ingredient_name": f"{DATASET_NAME} {split} Dataset",
                "author": "data-ingestion-service",
                "metadata": meta,
            },
        )
        if sidecar_resp:
            manifest_id = sidecar_resp.get("manifest_id")
            _jobs[job_id]["manifest_id"] = manifest_id
            if manifest_id:
                _last_manifest_ids[pipeline_id] = manifest_id
            log.info("[%s] Atlas manifest: %s", job_id, manifest_id)

        _jobs[job_id] = {**_jobs[job_id], **job_state}

    except Exception as exc:
        log.exception("[%s] Ingestion failed", job_id)
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
        "service": "data-ingestion",
        "default_pipeline_id": _normalize_pipeline_id(DEFAULT_PIPELINE_ID),
        "pipeline_root_prefix": PIPELINE_ROOT_PREFIX,
        "deployment_mode": DEPLOYMENT_MODE,
        "pod_name": POD_NAME,
        "pod_namespace": POD_NAMESPACE,
        "node_name": NODE_NAME,
    }


@app.post("/ingest")
def ingest(
    background_tasks: BackgroundTasks,
    split: str = "train",
    num_samples: int = 500,
    pipeline_id: Optional[str] = None,
):
    if num_samples < 2:
        raise HTTPException(status_code=400, detail="num_samples must be at least 2")

    resolved_pipeline_id = _resolve_pipeline_id(pipeline_id)

    import uuid

    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_do_ingest, job_id, split, num_samples, resolved_pipeline_id)
    _jobs[job_id] = {
        "status": "queued",
        "split": split,
        "pipeline_id": resolved_pipeline_id,
        "num_samples": num_samples,
        "cancel_requested": False,
    }
    log.info("Queued ingestion job %s for pipeline=%s", job_id, resolved_pipeline_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "split": split,
        "num_samples": num_samples,
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
        "service": "data-ingestion",
        "pipeline_id": resolved_pipeline_id,
        "manifest_id": _last_manifest_ids.get(resolved_pipeline_id),
        "sidecar_url": ATLAS_SIDECAR_URL or None,
    }


@app.get("/status")
def data_status(split: str = "train", pipeline_id: Optional[str] = None):
    resolved_pipeline_id = _resolve_pipeline_id(pipeline_id)
    try:
        s3 = get_s3()
        key = _raw_meta_key(resolved_pipeline_id, split)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        meta = json.loads(obj["Body"].read())
        return {"available": True, "pipeline_id": resolved_pipeline_id, "meta": meta}
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"available": False, "pipeline_id": resolved_pipeline_id}
        raise HTTPException(status_code=500, detail=str(exc))
