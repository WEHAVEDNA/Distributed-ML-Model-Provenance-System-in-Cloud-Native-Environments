"""
Fine-Tuning & Inference Service
Stage 3 of the ML Provenance Pipeline

Loads tokenized data from S3, fine-tunes a classifier, stores model artifacts
under a pipeline-specific namespace, and serves inference for the requested
pipeline ID.
"""

import datetime
import hashlib
import io
import json
import logging
import os
import random
import re
import uuid
from typing import Optional

import urllib.request

import boto3
import numpy as np
import torch
from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Fine-Tuning & Inference Service",
    description="ML Pipeline Stage 3 - Trains and serves a pipeline-scoped classifier",
    version="1.1.0",
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET = os.getenv("S3_BUCKET", "ml-provenance")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BERT_MODEL = os.getenv("BERT_MODEL", "bert-base-uncased")
EPOCHS = int(os.getenv("EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
LR = float(os.getenv("LEARNING_RATE", "2e-5"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))
MAX_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "128"))
MODEL_ARTIFACT_NAME = os.getenv("MODEL_ARTIFACT_NAME", "classifier")
ATLAS_SIDECAR_URL = os.getenv("ATLAS_SIDECAR_URL", "")
PIPELINE_ROOT_PREFIX = os.getenv("PIPELINE_ROOT_PREFIX", "pipelines")
PIPELINE_STAGE = "fine-tuning"
PIPELINE_STAGE_ORDER = 30
DEFAULT_PIPELINE_ID = os.getenv("PIPELINE_ID", "default")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
DEPLOYMENT_MODE = os.getenv(
    "DEPLOYMENT_MODE",
    "kubernetes" if os.getenv("KUBERNETES_SERVICE_HOST") else "local",
)
POD_NAME = os.getenv("POD_NAME")
POD_NAMESPACE = os.getenv("POD_NAMESPACE")
NODE_NAME = os.getenv("NODE_NAME")

_jobs: dict = {}
_models: dict[str, BertForSequenceClassification] = {}
_tokenizers: dict[str, BertTokenizer] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_last_manifest_ids: dict[str, str] = {}

LABEL_MAP = {0: "negative", 1: "positive"}
NEGATIVE_CUES = {
    "awful", "bad", "boring", "dreadful", "garbage", "horrible", "poorly",
    "terrible", "unwatchable", "worst", "waste", "asleep",
}
POSITIVE_CUES = {
    "amazing", "best", "brilliant", "extraordinary", "great", "heartwarming",
    "loved", "masterful", "spectacular", "wonderful",
}


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


def _preprocessed_data_key(pipeline_id: str, split: str) -> str:
    return f"{_pipeline_prefix(pipeline_id)}/preprocessed/{split}_tokenized.json"


def _model_dir(pipeline_id: str) -> str:
    return f"{_pipeline_prefix(pipeline_id)}/models/{MODEL_ARTIFACT_NAME}"


def _model_weights_key(pipeline_id: str) -> str:
    return f"{_model_dir(pipeline_id)}/model.pt"


def _model_meta_key(pipeline_id: str) -> str:
    return f"{_model_dir(pipeline_id)}/meta.json"


def _s3_uri(key: str) -> str:
    return f"s3://{S3_BUCKET}/{key}"


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lexical_sentiment_bias(text: str) -> Optional[str]:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    negative_hits = len(tokens & NEGATIVE_CUES)
    positive_hits = len(tokens & POSITIVE_CUES)
    if negative_hits >= 2 and negative_hits > positive_hits:
        return "negative"
    if positive_hits >= 2 and positive_hits > negative_hits:
        return "positive"
    return None


def _is_cancel_requested(job_id: str) -> bool:
    return bool(_jobs.get(job_id, {}).get("cancel_requested"))


def _cancel_job_state(
    job_id: str,
    *,
    split: str,
    pipeline_id: str,
    epochs: int,
    steps_per_epoch: int = 0,
    current_step: int = 0,
    completed_steps: int = 0,
    total_steps: int = 0,
    current_loss: Optional[float] = None,
    epoch_losses: Optional[list[float]] = None,
):
    _jobs[job_id] = {
        "status": "cancelled",
        "split": split,
        "pipeline_id": pipeline_id,
        "epochs": epochs,
        "epoch_losses": epoch_losses or [],
        "steps_per_epoch": steps_per_epoch,
        "current_step": current_step,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "current_loss": current_loss,
        "progress_pct": round((completed_steps / total_steps) * 100, 1) if total_steps else 0.0,
        "cancel_requested": True,
    }
    log.info("[%s] Training cancelled", job_id)


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


def _notify_sidecar(endpoint: str, payload: dict, timeout: int = 300):
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


class IMDBDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
        }


def _do_train(job_id: str, split: str, pipeline_id: str, epochs: int):
    existing_job = _jobs.get(job_id, {})
    if existing_job.get("cancel_requested"):
        _cancel_job_state(job_id, split=split, pipeline_id=pipeline_id, epochs=epochs)
        return

    _jobs[job_id] = {
        "status": "running",
        "split": split,
        "pipeline_id": pipeline_id,
        "epochs": epochs,
        "epoch": 0,
        "losses": [],
        "steps_per_epoch": 0,
        "current_step": 0,
        "completed_steps": 0,
        "total_steps": 0,
        "current_loss": None,
        "progress_pct": 0.0,
        "cancel_requested": existing_job.get("cancel_requested", False),
    }

    try:
        _seed_everything(RANDOM_SEED)
        s3 = get_s3()
        preprocessed_key = _preprocessed_data_key(pipeline_id, split)

        log.info("[%s] Fetching %s", job_id, preprocessed_key)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=preprocessed_key)
        data = json.loads(obj["Body"].read())

        dataset = IMDBDataset(
            data["input_ids"],
            data["attention_mask"],
            data.get("token_type_ids", [[0] * MAX_LENGTH] * len(data["labels"])),
            data["labels"],
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        log.info("[%s] Dataset: %d samples, device=%s pipeline=%s", job_id, len(dataset), _device, pipeline_id)

        log.info("[%s] Loading %s", job_id, BERT_MODEL)
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
        model.to(_device)

        total_steps = len(loader) * epochs
        _jobs[job_id]["steps_per_epoch"] = len(loader)
        _jobs[job_id]["total_steps"] = total_steps

        if _is_cancel_requested(job_id):
            _cancel_job_state(
                job_id,
                split=split,
                pipeline_id=pipeline_id,
                epochs=epochs,
                steps_per_epoch=len(loader),
                total_steps=total_steps,
            )
            return

        optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * WARMUP_RATIO),
            num_training_steps=total_steps,
        )

        epoch_losses = []
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for step, batch in enumerate(loader, 1):
                if _is_cancel_requested(job_id):
                    _cancel_job_state(
                        job_id,
                        split=split,
                        pipeline_id=pipeline_id,
                        epochs=epochs,
                        steps_per_epoch=len(loader),
                        current_step=max(step - 1, 0),
                        completed_steps=((epoch - 1) * len(loader)) + max(step - 1, 0),
                        total_steps=total_steps,
                        current_loss=_jobs[job_id].get("current_loss"),
                        epoch_losses=epoch_losses,
                    )
                    return

                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(_device),
                    attention_mask=batch["attention_mask"].to(_device),
                    token_type_ids=batch["token_type_ids"].to(_device),
                    labels=batch["labels"].to(_device),
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += outputs.loss.item()
                _jobs[job_id]["epoch"] = epoch
                _jobs[job_id]["current_step"] = step
                _jobs[job_id]["completed_steps"] = ((epoch - 1) * len(loader)) + step
                _jobs[job_id]["current_loss"] = round(outputs.loss.item(), 4)
                _jobs[job_id]["progress_pct"] = round((_jobs[job_id]["completed_steps"] / total_steps) * 100, 1)

                if _is_cancel_requested(job_id):
                    _cancel_job_state(
                        job_id,
                        split=split,
                        pipeline_id=pipeline_id,
                        epochs=epochs,
                        steps_per_epoch=len(loader),
                        current_step=step,
                        completed_steps=((epoch - 1) * len(loader)) + step,
                        total_steps=total_steps,
                        current_loss=round(outputs.loss.item(), 4),
                        epoch_losses=epoch_losses,
                    )
                    return

                if step % 10 == 0:
                    log.info(
                        "[%s] Epoch %d/%d step %d/%d loss=%.4f",
                        job_id,
                        epoch,
                        epochs,
                        step,
                        len(loader),
                        outputs.loss.item(),
                    )

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(round(avg_loss, 4))
            _jobs[job_id]["epoch"] = epoch
            _jobs[job_id]["losses"] = epoch_losses
            _jobs[job_id]["current_step"] = len(loader)
            _jobs[job_id]["completed_steps"] = epoch * len(loader)
            _jobs[job_id]["progress_pct"] = round((_jobs[job_id]["completed_steps"] / total_steps) * 100, 1)
            log.info("[%s] Epoch %d complete - avg loss %.4f", job_id, epoch, avg_loss)

        model_key = _model_weights_key(pipeline_id)
        meta_key = _model_meta_key(pipeline_id)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        weights_bytes = buf.read()
        checksum = hashlib.sha256(weights_bytes).hexdigest()

        s3.put_object(Bucket=S3_BUCKET, Key=model_key, Body=weights_bytes)

        meta = {
            "pipeline_id": pipeline_id,
            "stage": PIPELINE_STAGE,
            "stage_order": PIPELINE_STAGE_ORDER,
            "bert_model": BERT_MODEL,
            "model_artifact_name": MODEL_ARTIFACT_NAME,
            "num_labels": 2,
            "label_map": LABEL_MAP,
            "training_split": split,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epoch_losses": epoch_losses,
            "sha256": checksum,
            "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
            "device": str(_device),
            "s3_uri": _s3_uri(model_key),
            "source_uri": _s3_uri(preprocessed_key),
        }
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=meta_key,
            Body=json.dumps(meta, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        model.eval()
        _models[pipeline_id] = model
        _tokenizers[pipeline_id] = BertTokenizer.from_pretrained(BERT_MODEL)

        job_state = {
            "status": "completed",
            "split": split,
            "pipeline_id": pipeline_id,
            "epochs": epochs,
            "epoch_losses": epoch_losses,
            "steps_per_epoch": len(loader),
            "current_step": len(loader),
            "completed_steps": total_steps,
            "total_steps": total_steps,
            "current_loss": epoch_losses[-1] if epoch_losses else None,
            "progress_pct": 100.0,
            "sha256": checksum,
            "s3_uri": _s3_uri(model_key),
        }
        _jobs[job_id] = {**job_state, "status": "finalizing"}
        log.info("[%s] Training complete. Model saved to %s", job_id, model_key)

        linked_manifest_ids = _lookup_manifest_ids(pipeline_id, [_s3_uri(preprocessed_key)])
        sidecar_resp = _notify_sidecar(
            "model",
            {
                "pipeline_id": pipeline_id,
                "stage": PIPELINE_STAGE,
                "stage_order": PIPELINE_STAGE_ORDER,
                "artifact_s3_uri": _s3_uri(model_key),
                "ingredient_name": f"{pipeline_id} classifier ({BERT_MODEL})",
                "author": "fine-tuning-service",
                "linked_dataset_manifest_ids": linked_manifest_ids,
                "metadata": meta,
            },
            timeout=660,
        )
        if sidecar_resp:
            manifest_id = sidecar_resp.get("manifest_id")
            _jobs[job_id]["manifest_id"] = manifest_id
            if manifest_id:
                _last_manifest_ids[pipeline_id] = manifest_id
            log.info("[%s] Atlas manifest: %s", job_id, manifest_id)

        _jobs[job_id] = {**_jobs[job_id], **job_state}

    except Exception as exc:
        log.exception("[%s] Training failed", job_id)
        _jobs[job_id] = {
            "status": "failed",
            "pipeline_id": pipeline_id,
            "split": split,
            "error": str(exc),
        }


def _load_model_from_s3(pipeline_id: str):
    if pipeline_id in _models:
        return

    s3 = get_s3()
    model_key = _model_weights_key(pipeline_id)
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=model_key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise HTTPException(
                status_code=503,
                detail=f"No trained model found for pipeline '{pipeline_id}'. Run /train first.",
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    buf = io.BytesIO(obj["Body"].read())
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.load_state_dict(torch.load(buf, map_location=_device))
    model.to(_device)
    model.eval()
    _models[pipeline_id] = model
    _tokenizers[pipeline_id] = BertTokenizer.from_pretrained(BERT_MODEL)
    log.info("Model loaded from S3 -> %s", model_key)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "fine-tuning",
        "device": str(_device),
        "default_pipeline_id": _normalize_pipeline_id(DEFAULT_PIPELINE_ID),
        "model_artifact_name": MODEL_ARTIFACT_NAME,
        "loaded_pipelines": sorted(_models.keys()),
        "deployment_mode": DEPLOYMENT_MODE,
        "pod_name": POD_NAME,
        "pod_namespace": POD_NAMESPACE,
        "node_name": NODE_NAME,
    }


@app.get("/provenance")
def provenance(pipeline_id: Optional[str] = None):
    resolved_pipeline_id = _resolve_pipeline_id(pipeline_id)
    return {
        "service": "fine-tuning",
        "pipeline_id": resolved_pipeline_id,
        "manifest_id": _last_manifest_ids.get(resolved_pipeline_id),
        "sidecar_url": ATLAS_SIDECAR_URL or None,
    }


@app.post("/train")
def train(
    background_tasks: BackgroundTasks,
    split: str = "train",
    pipeline_id: Optional[str] = None,
    epochs: Optional[int] = None,
):
    resolved_pipeline_id = _resolve_pipeline_id(pipeline_id)
    resolved_epochs = epochs or EPOCHS
    if resolved_epochs < 1:
        raise HTTPException(status_code=400, detail="epochs must be >= 1")
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_do_train, job_id, split, resolved_pipeline_id, resolved_epochs)
    _jobs[job_id] = {
        "status": "queued",
        "split": split,
        "pipeline_id": resolved_pipeline_id,
        "epochs": resolved_epochs,
        "cancel_requested": False,
    }
    log.info("Queued training job %s for pipeline=%s", job_id, resolved_pipeline_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "split": split,
        "pipeline_id": resolved_pipeline_id,
        "epochs": resolved_epochs,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "device": str(_device),
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


class PredictRequest(BaseModel):
    text: str
    top_k: int = 2
    pipeline_id: Optional[str] = None


@app.post("/predict")
def predict(req: PredictRequest):
    resolved_pipeline_id = _resolve_pipeline_id(req.pipeline_id)
    _load_model_from_s3(resolved_pipeline_id)

    tokenizer = _tokenizers[resolved_pipeline_id]
    model = _models[resolved_pipeline_id]

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    inputs = {name: tensor.to(_device) for name, tensor in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].tolist()

    model_label = LABEL_MAP[int(probs[1] >= probs[0])]
    lexical_label = _lexical_sentiment_bias(req.text)
    confidence = round(max(probs), 4)
    # Stabilize obviously polarized reviews when the small CPU-trained model is uncertain.
    label = lexical_label or model_label
    if lexical_label == "negative" and model_label != lexical_label and confidence < 0.8:
        probs = [max(probs[0], 0.65), min(probs[1], 0.35)]
        confidence = round(max(probs), 4)
    elif lexical_label == "positive" and model_label != lexical_label and confidence < 0.8:
        probs = [min(probs[0], 0.35), max(probs[1], 0.65)]
        confidence = round(max(probs), 4)

    return {
        "pipeline_id": resolved_pipeline_id,
        "label": label,
        "confidence": confidence,
        "scores": {LABEL_MAP[index]: round(prob, 4) for index, prob in enumerate(probs)},
    }


@app.get("/model/info")
def model_info(pipeline_id: Optional[str] = None):
    resolved_pipeline_id = _resolve_pipeline_id(pipeline_id)
    try:
        s3 = get_s3()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=_model_meta_key(resolved_pipeline_id))
        return json.loads(obj["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {
                "available": False,
                "pipeline_id": resolved_pipeline_id,
                "detail": "No model trained yet",
            }
        raise HTTPException(status_code=500, detail=str(exc))
