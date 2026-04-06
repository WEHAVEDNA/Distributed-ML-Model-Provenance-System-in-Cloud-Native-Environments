"""
Fine-Tuning & Inference Service
Stage 3 of the ML Provenance Pipeline

Loads BERT-tokenized IMDB data from S3, fine-tunes bert-base-uncased for
binary sentiment classification, saves the model back to S3, then serves
real-time inference via a /predict endpoint.

API:
  GET  /health        - liveness probe
  POST /train         - start fine-tuning job (async)
  GET  /jobs/{id}     - check job status / training metrics
  POST /predict       - run inference on free text
  GET  /model/info    - info about the loaded model
"""

import io
import os
import json
import logging
import hashlib
import datetime
import uuid
import urllib.request
from typing import Optional

import boto3
import torch
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Fine-Tuning & Inference Service",
    description="ML Pipeline Stage 3 – Fine-tunes BERT on IMDB and serves predictions",
    version="1.0.0",
)

# ── Config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT  = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET    = os.getenv("S3_BUCKET", "ml-provenance")
AWS_KEY      = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET   = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BERT_MODEL   = os.getenv("BERT_MODEL", "bert-base-uncased")
EPOCHS       = int(os.getenv("EPOCHS", "2"))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "16"))
LR           = float(os.getenv("LEARNING_RATE", "2e-5"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))
MAX_LENGTH   = int(os.getenv("MAX_SEQ_LENGTH", "128"))
MODEL_KEY         = "models/bert-imdb/model.pt"
META_KEY          = "models/bert-imdb/meta.json"
ATLAS_SIDECAR_URL = os.getenv("ATLAS_SIDECAR_URL", "")

# In-memory state
_jobs:               dict = {}
_model:              Optional[BertForSequenceClassification] = None
_tokenizer:          Optional[BertTokenizer] = None
_device:             torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_last_manifest_id:   Optional[str] = None

LABEL_MAP = {0: "negative", 1: "positive"}


# ── Atlas Sidecar helper ──────────────────────────────────────────────────────
def _notify_sidecar(endpoint: str, payload: dict, timeout: int = 300):
    if not ATLAS_SIDECAR_URL:
        return None
    url = f"{ATLAS_SIDECAR_URL}/collect/{endpoint}"
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids      = torch.tensor(input_ids,      dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        self.labels         = torch.tensor(labels,         dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels":         self.labels[idx],
        }


# ── Background fine-tuning task ───────────────────────────────────────────────
def _do_train(job_id: str, split: str):
    global _model, _tokenizer
    _jobs[job_id] = {"status": "running", "split": split, "epoch": 0, "losses": []}

    try:
        s3 = get_s3()

        # ── 1. Load preprocessed data from S3 ────────────────────────────────
        log.info("[%s] Fetching preprocessed/%s_tokenized.json", job_id, split)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"preprocessed/{split}_tokenized.json")
        data = json.loads(obj["Body"].read())

        dataset = IMDBDataset(
            data["input_ids"],
            data["attention_mask"],
            data.get("token_type_ids", [[0] * MAX_LENGTH] * len(data["labels"])),
            data["labels"],
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        log.info("[%s] Dataset: %d samples, device: %s", job_id, len(dataset), _device)

        # ── 2. Initialise model ───────────────────────────────────────────────
        log.info("[%s] Loading %s", job_id, BERT_MODEL)
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
        model.to(_device)

        total_steps = len(loader) * EPOCHS
        optimizer   = AdamW(model.parameters(), lr=LR, eps=1e-8)
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * WARMUP_RATIO),
            num_training_steps=total_steps,
        )

        # ── 3. Training loop ──────────────────────────────────────────────────
        epoch_losses = []
        model.train()
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0.0
            for step, batch in enumerate(loader, 1):
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

                if step % 10 == 0:
                    log.info("[%s] Epoch %d/%d  step %d/%d  loss=%.4f",
                             job_id, epoch, EPOCHS, step, len(loader), outputs.loss.item())

            avg = epoch_loss / len(loader)
            epoch_losses.append(round(avg, 4))
            _jobs[job_id]["epoch"]  = epoch
            _jobs[job_id]["losses"] = epoch_losses
            log.info("[%s] Epoch %d complete — avg loss %.4f", job_id, epoch, avg)

        # ── 4. Save model to S3 ───────────────────────────────────────────────
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        weights_bytes = buf.read()
        checksum = hashlib.sha256(weights_bytes).hexdigest()

        s3.put_object(Bucket=S3_BUCKET, Key=MODEL_KEY, Body=weights_bytes)

        meta = {
            "bert_model":    BERT_MODEL,
            "num_labels":    2,
            "label_map":     LABEL_MAP,
            "training_split": split,
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "learning_rate": LR,
            "epoch_losses":  epoch_losses,
            "sha256":        checksum,
            "trained_at":    datetime.datetime.utcnow().isoformat() + "Z",
            "device":        str(_device),
            "s3_uri":        f"s3://{S3_BUCKET}/{MODEL_KEY}",
        }
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=META_KEY,
            Body=json.dumps(meta, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        # ── 5. Cache model in memory for inference ────────────────────────────
        model.eval()
        _model     = model
        _tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

        _jobs[job_id] = {
            "status":        "completed",
            "split":         split,
            "epochs":        EPOCHS,
            "epoch_losses":  epoch_losses,
            "sha256":        checksum,
            "s3_uri":        f"s3://{S3_BUCKET}/{MODEL_KEY}",
        }
        log.info("[%s] Training complete. Model saved to %s", job_id, MODEL_KEY)

        # ── Notify Atlas sidecar (model + link to upstream tokenized dataset) ──
        # First look up the tokenized dataset's manifest_id from the sidecar
        dataset_manifest_ids = []
        if ATLAS_SIDECAR_URL:
            try:
                reg_req = urllib.request.Request(
                    f"{ATLAS_SIDECAR_URL}/registry", method="GET"
                )
                with urllib.request.urlopen(reg_req, timeout=10) as resp:
                    registry = json.loads(resp.read())
                tokenized_uri = f"s3://{S3_BUCKET}/preprocessed/{split}_tokenized.json"
                if tokenized_uri in registry:
                    dataset_manifest_ids = [registry[tokenized_uri]["manifest_id"]]
            except Exception as exc:
                log.warning("Could not fetch sidecar registry: %s", exc)

        global _last_manifest_id
        sidecar_resp = _notify_sidecar("model", {
            "stage": "fine-tuning",
            "artifact_s3_uri": f"s3://{S3_BUCKET}/{MODEL_KEY}",
            "ingredient_name": f"BERT IMDB Sentiment Classifier ({BERT_MODEL})",
            "author": "fine-tuning-service",
            "linked_dataset_manifest_ids": dataset_manifest_ids,
            "metadata": meta,
        }, timeout=660)
        if sidecar_resp:
            manifest_id = sidecar_resp.get("manifest_id")
            _jobs[job_id]["manifest_id"] = manifest_id
            _last_manifest_id = manifest_id
            log.info("[%s] Atlas manifest: %s", job_id, manifest_id)

    except Exception as exc:
        log.exception("[%s] Training failed", job_id)
        _jobs[job_id] = {"status": "failed", "error": str(exc)}


# ── Load model from S3 (on-demand) ────────────────────────────────────────────
def _load_model_from_s3():
    global _model, _tokenizer
    if _model is not None:
        return
    s3 = get_s3()
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    except ClientError as e:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model found in S3 ({MODEL_KEY}). Run /train first.",
        )
    buf = io.BytesIO(obj["Body"].read())
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.load_state_dict(torch.load(buf, map_location=_device))
    model.to(_device)
    model.eval()
    _model     = model
    _tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    log.info("Model loaded from S3 → %s", MODEL_KEY)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "service":      "fine-tuning",
        "device":       str(_device),
        "model_loaded": _model is not None,
    }


@app.get("/provenance")
def provenance():
    """Return the most recent Atlas manifest ID registered by this service."""
    return {
        "service": "fine-tuning",
        "manifest_id": _last_manifest_id,
        "sidecar_url": ATLAS_SIDECAR_URL or None,
    }


@app.post("/train")
def train(background_tasks: BackgroundTasks, split: str = "train"):
    """
    Fine-tune BERT on preprocessed IMDB data (runs in background).

    Prerequisites: run /ingest and /preprocess first.
    """
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_do_train, job_id, split)
    _jobs[job_id] = {"status": "queued", "split": split}
    log.info("Queued training job %s", job_id)
    return {
        "job_id":     job_id,
        "status":     "queued",
        "split":      split,
        "epochs":     EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr":         LR,
        "device":     str(_device),
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


class PredictRequest(BaseModel):
    text: str
    top_k: int = 2  # return scores for all labels


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Sentiment prediction for free text using the fine-tuned BERT model.

    Returns label (`positive` / `negative`), confidence, and per-class scores.
    """
    _load_model_from_s3()

    inputs = _tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[0].tolist()

    label = LABEL_MAP[int(probs[1] >= probs[0])]
    return {
        "label":      label,
        "confidence": round(max(probs), 4),
        "scores":     {LABEL_MAP[i]: round(p, 4) for i, p in enumerate(probs)},
    }


@app.get("/model/info")
def model_info():
    """Return metadata for the saved model in S3."""
    try:
        s3  = get_s3()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=META_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"available": False, "detail": "No model trained yet"}
        raise HTTPException(status_code=500, detail=str(e))
