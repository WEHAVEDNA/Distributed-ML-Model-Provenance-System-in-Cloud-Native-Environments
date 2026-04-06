# Distributed ML Model Provenance System in Cloud-Native Environments

**BU EC528 Spring 2026 — Intel Labs Capstone Project**
Mentors: Marcela Melara, Marcin Spoczynski (Intel Labs)

---

## Overview

This project extends the [IntelLabs Atlas CLI](https://github.com/IntelLabs/atlas-cli) into a cloud-native distributed provenance system by implementing a three-stage ML pipeline — data ingestion, preprocessing, and fine-tuning — where every artifact produced is automatically tracked with cryptographically signed C2PA provenance manifests via an Atlas CLI sidecar service.

Each pipeline stage runs as an independent containerised service. Artifacts flow through S3-compatible object storage (MinIO locally). The Atlas sidecar intercepts each completed stage and calls `atlas-cli` to generate a manifest that records what was produced, how, and by whom — forming a linked provenance chain from raw data to trained model.

---

## Architecture

```
  ┌──────────────────────────────────────────────────────────┐
  │                     ML Pipeline                          │
  │                                                          │
  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
  │  │data-ingestion│ ─>│ preprocessing│─ >│ fine-tuning  │  │
  │  │   :8001      │   │   :8002      │   │   :8003      │  │
  │  │              │   │              │   │              │  │
  │  │ HuggingFace  │   │BertTokenizer │   │bert-base-    │  │
  │  │ IMDB dataset │   │max_length=128│   │uncased+AdamW │  │
  │  └──────┬───────┘   └──────┬───────┘   └───────┬──────┘  │
  │         └──────────────────┴───────────────────┘         │
  │                        boto3 (S3)                        │
  │                            │                             │
  │                   ┌────────▼────────┐                    │
  │                   │  MinIO / S3     │                    │
  │                   │ ml-provenance   │                    │
  │                   │ raw/            │                    │
  │                   │ preprocessed/   │                    │
  │                   │ models/         │                    │
  │                   └────────┬────────┘                    │
  │                            │ notify for each stage       │
  │                   ┌────────▼────────┐                    │
  │                   │  atlas-sidecar  │                    │
  │                   │    :8004        │                    │
  │                   │                 │                    │
  │                   │  atlas-cli      │                    │
  │                   │  C2PA manifests │                    │
  │                   │  RSA-4096 signed│                    │
  │                   └─────────────────┘                    │
  └──────────────────────────────────────────────────────────┘
```

### Provenance chain

```
raw/train_data.json
    │  dataset manifest  (atlas-cli dataset create)
    ▼
preprocessed/train_tokenized.json
    │  pipeline manifest (atlas-cli pipeline generate-provenance)
    ▼
models/bert-imdb/model.pt
       model manifest    (atlas-cli model create)
       linked back to upstream dataset manifests
```

---

## Services

| Service | Port | Responsibility |
|---------|------|----------------|
| **data-ingestion** | 8001 | Downloads IMDB from HuggingFace, shuffles for class balance, stores `{text, label}` JSON to S3 |
| **preprocessing** | 8002 | Tokenizes with `BertTokenizer` (`bert-base-uncased`, max_length=128), stores tensor arrays to S3 |
| **fine-tuning** | 8003 | Fine-tunes BERT for binary sentiment classification with AdamW + linear warmup; serves `/predict` |
| **atlas-sidecar** | 8004 | Wraps Atlas CLI via HTTP API; generates signed C2PA manifests for every artifact after each stage |

All services are FastAPI with async background jobs. Storage is MinIO (local) or AWS S3 (cloud) — switching requires only changing the `S3_ENDPOINT_URL` environment variable.

---

## Stack

| Layer | Technology |
|-------|-----------|
| ML model | `bert-base-uncased` (HuggingFace Transformers) |
| Dataset | IMDB sentiment (HuggingFace `datasets`) |
| Services | Python 3.11, FastAPI, uvicorn |
| Storage | MinIO (S3-compatible), boto3 |
| Provenance | [IntelLabs Atlas CLI](https://github.com/IntelLabs/atlas-cli) — C2PA / OMS manifests, RSA-4096 signing |
| Orchestration | Docker Compose (local), Kubernetes (deployment) |
| Testing | pytest, requests, pytest-timeout |

---

## Quick Start

**Prerequisites:** Docker Desktop, Python 3.11+

```powershell
# 1. Start all services (first build ~15 min — compiles atlas-cli from Rust source)
docker compose up --build

# 2. Run the demo (new terminal)
python demo.py --samples 200           # ingest + preprocess + show provenance chain
python demo.py --samples 500 --train   # full pipeline including fine-tuning (slow)

# 3. Run tests
pytest tests/ -v -m smoke             # fast schema/health checks only
pytest tests/ -v -m "wired" --samples 200   # provenance chain after pipeline
pytest tests/ -v -m slow --samples 500      # full suite including fine-tuning
```

See [SETUP.md](SETUP.md) for step-by-step recreation instructions.

---

## API Reference

### Pipeline services

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `:8001/health` | Liveness |
| POST | `:8001/ingest?split=train&num_samples=500` | Start ingestion job |
| GET | `:8001/jobs/{id}` | Poll job status |
| GET | `:8001/status?split=train` | Check if raw data exists in S3 |
| POST | `:8002/preprocess?split=train` | Start tokenization job |
| GET | `:8002/jobs/{id}` | Poll job status |
| POST | `:8003/train?split=train` | Start fine-tuning job |
| GET | `:8003/jobs/{id}` | Poll job status (includes epoch + losses) |
| POST | `:8003/predict` | `{"text": "..."}` → label + confidence |
| GET | `:8003/model/info` | Trained model metadata |

### Atlas sidecar

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `:8004/health` | Liveness + cached atlas-cli version + key status |
| POST | `:8004/collect/dataset` | Register dataset artifact, create C2PA manifest |
| POST | `:8004/collect/pipeline` | Register pipeline-step SLSA provenance |
| POST | `:8004/collect/model` | Register model artifact, link to dataset manifests |
| GET | `:8004/lineage` | Ordered provenance chain: raw data → tokenized → model |
| GET | `:8004/pipeline/status` | Per-stage manifest counts and completion flags |
| GET | `:8004/registry` | All tracked artifacts: `s3_uri → {manifest_id, stage, type}` |
| GET | `:8004/export/{manifest_id}` | Export full provenance graph as JSON |
| GET | `:8004/verify/{manifest_id}` | Verify manifest cryptographic integrity |
| GET | `:8004/signing-key` | RSA public key PEM |

### Per-service provenance

Each pipeline service exposes `GET /provenance` returning the manifest ID of the last artifact it registered:

| Service | Endpoint |
|---------|----------|
| data-ingestion | `:8001/provenance` |
| preprocessing | `:8002/provenance` |
| fine-tuning | `:8003/provenance` |

Interactive docs: `http://localhost:800{1,2,3,4}/docs`

---

## Tests

```powershell
pytest tests/ -v -m smoke                       # fast health + schema checks only
pytest tests/ -v                                # all non-slow tests
pytest tests/ -v -m "wired" --samples 200       # provenance chain after pipeline run
pytest tests/ -v -m slow --samples 500          # full suite including fine-tuning
pytest tests/test_atlas_sidecar.py -v           # sidecar direct collect tests
pytest tests/test_provenance_chain.py -v        # lineage/pipeline-status tests
```

| File | What it covers |
|------|---------------|
| `test_health.py` | All services reachable, /docs reachable |
| `test_data_ingestion.py` | Job lifecycle, SHA-256, S3 URI, idempotency |
| `test_preprocessing.py` | Token counts, max_length, source linkage |
| `test_fine_tuning.py` | Loss progression, model metadata, predict schema, sentiment accuracy |
| `test_pipeline_e2e.py` | Full provenance chain, SHA-256 uniqueness, end-to-end sentiment |
| `test_atlas_sidecar.py` | Direct collect calls, registry updates, manifest export/verify |
| `test_provenance_chain.py` | /lineage and /pipeline/status schema + wired chain correctness |

---

## Project Goals

| Requirement | Status |
|-------------|--------|
| Data ingestion service | Done — IMDB via HuggingFace, S3 storage |
| Preprocessing service | Done — BertTokenizer, chunked processing |
| Fine-tuning service | Done — BERT sentiment classifier, `/predict` endpoint |
| Kubernetes deployment | Done — `k8s/` manifests (namespace, MinIO, configmap, 3 services, atlas-sidecar) |
| Sidecar container pattern | Done — Atlas CLI wrapped as HTTP service, called automatically after each stage |
| C2PA / OMS manifests | Done — `dataset create`, `pipeline generate-provenance`, `model create` |
| Cryptographic signing | Done — RSA-4096 key on sidecar startup, signs all manifests |
| Artifact sharing via S3 | Done — MinIO; boto3 `S3_ENDPOINT_URL` switches to real AWS S3 |
| Provenance query API | Done — `/registry`, `/export`, `/verify` |
| Provenance verification (stretch) | Done — `atlas-cli manifest validate` via `/verify/{id}` |
| Atlas CLI testing script (stretch) | Done — `test_atlas_sidecar.py` direct + wired test modes |
| Confidential computing (stretch) | Not implemented |
| Multi-cloud (stretch) | Not implemented |
