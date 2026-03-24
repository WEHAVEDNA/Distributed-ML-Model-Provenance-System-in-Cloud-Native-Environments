"""
Fine-tuning Service  –  port 8003
--------------------------------------
POST /train?split=train        → start training job, return {job_id, device}
GET  /jobs/{job_id}            → status / result (epochs, losses, sha256, s3_uri)
POST /predict                  → {"text": "..."} → {label, confidence}
GET  /model/info               → current loaded model metadata
GET  /health                   → {"status":"ok"}
"""

import hashlib
import json
import logging
import os
import tarfile
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [finetuning] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Fine-tuning Service")

MINIO_ENDPOINT  = os.environ.get("MINIO_ENDPOINT",   "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS    = os.environ.get("MINIO_ACCESS_KEY",  "minioadmin")
MINIO_SECRET    = os.environ.get("MINIO_SECRET_KEY",  "minioadmin")
BUCKET          = os.environ.get("ARTIFACT_BUCKET",   "atlas-artifacts")
BASE_MODEL      = os.environ.get("BASE_MODEL",         "bert-base-uncased")
NUM_EPOCHS      = int(os.environ.get("NUM_EPOCHS",     "2"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE",     "16"))
LEARNING_RATE   = float(os.environ.get("LEARNING_RATE","2e-5"))
DATASET_NAME    = os.environ.get("DATASET_NAME",       "imdb")
STATUS_INTERVAL = int(os.environ.get("STATUS_INTERVAL","10"))   # seconds between progress logs
NUM_LABELS      = 2

jobs: dict[str, dict] = {}

_model      = None
_tokenizer  = None
_model_meta = {}
_model_lock = threading.Lock()
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS,
        aws_secret_access_key=MINIO_SECRET,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def latest_processed_key(s3, split: str) -> str:
    resp    = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"datasets/processed/{DATASET_NAME}_{split}_")
    objects = resp.get("Contents", [])
    if not objects:
        raise RuntimeError(f"No processed dataset found for split={split}. Run preprocessing first.")
    return sorted(objects, key=lambda o: o["LastModified"], reverse=True)[0]["Key"]


class TextDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "input_ids":      torch.tensor(r["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(r["label"],          dtype=torch.long),
        }


def _status_reporter(job_id: str, stop_event: threading.Event):
    """Background thread — logs a progress summary every STATUS_INTERVAL seconds."""
    while not stop_event.wait(STATUS_INTERVAL):
        job = jobs.get(job_id, {})
        if job.get("status") not in ("running",):
            break
        epoch       = job.get("current_epoch", 0)
        total_epochs= job.get("total_epochs", NUM_EPOCHS)
        step        = job.get("current_step", 0)
        total_steps = job.get("steps_per_epoch", "?")
        loss        = job.get("current_loss")
        acc         = job.get("current_acc")
        elapsed     = job.get("elapsed_s", 0)
        eta         = job.get("eta_s")

        loss_str    = f"loss={loss:.4f}" if loss is not None else "loss=—"
        acc_str     = f"acc={acc:.3f}"  if acc  is not None else "acc=—"
        eta_str     = f"eta={eta:.0f}s" if eta  is not None else ""

        log.info(
            f"[{job_id}] ⏳  epoch {epoch}/{total_epochs}  "
            f"step {step}/{total_steps}  "
            f"{loss_str}  {acc_str}  "
            f"elapsed={elapsed:.0f}s  {eta_str}"
        )


def train_model(job_id: str, split: str):
    global _model, _tokenizer, _model_meta

    start_time = time.time()
    stop_reporter = threading.Event()

    jobs[job_id].update({
        "status":          "running",
        "total_epochs":    NUM_EPOCHS,
        "current_epoch":   0,
        "current_step":    0,
        "steps_per_epoch": "?",
        "current_loss":    None,
        "current_acc":     None,
        "elapsed_s":       0,
        "eta_s":           None,
        "started_at":      datetime.now(timezone.utc).isoformat(),
    })

    # Start the background status reporter
    reporter = threading.Thread(
        target=_status_reporter, args=(job_id, stop_reporter), daemon=True
    )
    reporter.start()

    try:
        s3 = s3_client()

        input_key = latest_processed_key(s3, split)
        log.info(f"[{job_id}] Reading s3://{BUCKET}/{input_key}")
        obj     = s3.get_object(Bucket=BUCKET, Key=input_key)
        records = [json.loads(l) for l in obj["Body"].read().decode().splitlines() if l.strip()]
        log.info(f"[{job_id}] Loaded {len(records)} records, device={DEVICE}")

        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=NUM_LABELS
        ).to(DEVICE)

        dl            = DataLoader(TextDataset(records), batch_size=BATCH_SIZE, shuffle=True)
        total_batches = len(dl) * NUM_EPOCHS
        optimizer     = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler     = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_batches // 10),
            num_training_steps=total_batches,
        )

        jobs[job_id]["steps_per_epoch"] = len(dl)
        log.info(f"[{job_id}] Training: {len(records)} samples, "
                 f"{len(dl)} batches/epoch, {NUM_EPOCHS} epochs")

        epoch_losses   = []
        global_step    = 0
        step_times     = []

        model.train()
        for epoch in range(1, NUM_EPOCHS + 1):
            jobs[job_id]["current_epoch"] = epoch
            total_loss, correct, total    = 0.0, 0, 0

            for step, batch in enumerate(dl, 1):
                step_start = time.time()

                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss  += out.loss.item()
                preds        = out.logits.argmax(dim=-1)
                correct      += (preds == labels).sum().item()
                total        += labels.size(0)
                global_step  += 1

                # Track step timing for ETA
                step_times.append(time.time() - step_start)
                if len(step_times) > 50:
                    step_times.pop(0)

                elapsed       = time.time() - start_time
                avg_step_time = sum(step_times) / len(step_times)
                remaining     = (total_batches - global_step) * avg_step_time

                # Update job state so the reporter always has fresh numbers
                jobs[job_id].update({
                    "current_step": step,
                    "current_loss": round(total_loss / step, 4),
                    "current_acc":  round(correct / total, 4),
                    "elapsed_s":    round(elapsed, 1),
                    "eta_s":        round(remaining, 0),
                })

                # Also log at every 20 steps so logs show progress without the reporter
                if step % 20 == 0 or step == len(dl):
                    log.info(
                        f"[{job_id}]   epoch {epoch}/{NUM_EPOCHS}  "
                        f"step {step}/{len(dl)}  "
                        f"loss={total_loss/step:.4f}  "
                        f"acc={correct/total:.3f}"
                    )

            avg_loss = total_loss / len(dl)
            epoch_losses.append(round(avg_loss, 4))
            log.info(
                f"[{job_id}] ✓ Epoch {epoch}/{NUM_EPOCHS} complete — "
                f"avg_loss={avg_loss:.4f}  acc={correct/total:.3f}  "
                f"elapsed={time.time()-start_time:.0f}s"
            )

        # ── Save and upload ────────────────────────────────────────────────────
        log.info(f"[{job_id}] Saving model ...")
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            model.save_pretrained(str(model_dir))
            AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(str(model_dir))

            tar_path = Path(tmp) / "model.tar.gz"
            with tarfile.open(str(tar_path), "w:gz") as tar:
                tar.add(str(model_dir), arcname="model")

            tar_bytes = tar_path.read_bytes()
            sha       = hashlib.sha256(tar_bytes).hexdigest()
            key       = f"models/finetuned/{DATASET_NAME}_{split}_model.tar.gz"

            s3.put_object(Bucket=BUCKET, Key=key, Body=tar_bytes, ContentType="application/gzip")
            log.info(f"[{job_id}] Uploaded model → s3://{BUCKET}/{key}  ({len(tar_bytes):,} bytes)")

        total_elapsed = round(time.time() - start_time, 1)

        # Cache model in memory for inference
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        with _model_lock:
            _model      = model.eval()
            _tokenizer  = tokenizer
            _model_meta = {
                "base_model":   BASE_MODEL,
                "num_epochs":   NUM_EPOCHS,
                "epoch_losses": epoch_losses,
                "sha256":       sha,
                "s3_uri":       f"s3://{BUCKET}/{key}",
                "trained_at":   datetime.now(timezone.utc).isoformat(),
                "num_records":  len(records),
                "elapsed_s":    total_elapsed,
            }

        jobs[job_id].update({
            "status":       "completed",
            "epochs":       NUM_EPOCHS,
            "epoch_losses": epoch_losses,
            "sha256":       sha,
            "s3_uri":       f"s3://{BUCKET}/{key}",
            "elapsed_s":    total_elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

        log.info(
            f"[{job_id}] ✅ Training complete — "
            f"{NUM_EPOCHS} epochs, losses={epoch_losses}, "
            f"total time={total_elapsed}s"
        )

    except Exception as exc:
        log.exception(f"[{job_id}] Training failed")
        jobs[job_id].update({"status": "failed", "error": str(exc)})
    finally:
        stop_reporter.set()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "fine-tuning", "device": str(DEVICE)}


@app.post("/train")
def train(split: str = Query(default="train")):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":     job_id,
        "status":     "pending",
        "split":      split,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    threading.Thread(target=train_model, args=(job_id, split), daemon=True).start()
    return {"job_id": job_id, "status": "pending", "device": str(DEVICE)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs")
def list_jobs():
    return list(jobs.values())


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: PredictRequest):
    with _model_lock:
        if _model is None or _tokenizer is None:
            raise HTTPException(status_code=503, detail="No model loaded. Run /train first.")
        model     = _model
        tokenizer = _tokenizer

    enc = tokenizer(req.text, return_tensors="pt", truncation=True,
                    padding=True, max_length=128)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
    probs     = torch.softmax(logits, dim=-1)[0]
    pred      = probs.argmax().item()
    label_map = {0: "negative", 1: "positive"}

    return {
        "label":      label_map.get(pred, str(pred)),
        "confidence": round(probs[pred].item(), 4),
        "scores":     {label_map[i]: round(p.item(), 4) for i, p in enumerate(probs)},
    }


@app.get("/model/info")
def model_info():
    with _model_lock:
        if not _model_meta:
            raise HTTPException(status_code=404, detail="No model loaded yet.")
        return _model_meta