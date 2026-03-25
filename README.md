# Distributed ML Model Provenance System in Cloud-Native Environments

**BU EC528 · Spring 2026**  
Intel Labs mentored project — extending [Atlas CLI](https://github.com/IntelLabs/atlas-cli) into a cloud-native distributed provenance system.

Demo 1 presentation: https://docs.google.com/presentation/d/1z5usLUe9r3ex-ovyxyOwZE1G7A_tVfEOVQuWLKoFlhg/edit?usp=sharing

Demo 2 presentation: https://docs.google.com/presentation/d/1WfeDlM877Tw3Msrw_t-0FO3Swb3vk92SMZvVm914wR0/edit?usp=sharing

Demo 2 video: https://drive.google.com/file/d/1iPYX0nW1rNNxc22FYdzfefnvu_PNE1oM/view?usp=sharing

---

## Overview

This project implements a distributed ML pipeline across three containerised HTTP services, with cryptographic provenance tracking for every artifact produced. Each stage — data ingestion, preprocessing, and fine-tuning — runs as an independent FastAPI service, shares artifacts through MinIO (S3-compatible storage), and is designed to report provenance to an Atlas CLI sidecar.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ml-pipeline network                     │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │ data-         │   │ preprocessing│   │   fine-tuning    │ │
│  │ ingestion     │──▶│   :8002      │──▶│      :8003       │ │
│  │   :8001       │   │              │   │                  │ │
│  └──────┬────────┘   └──────┬───────┘   └────────┬─────────┘ │
│         │                   │                    │           │
│         └───────────────────┴────────────────────┘           │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │      MinIO      │                       │
│                    │  (S3 storage)   │                       │
│                    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Services

| Service | Port | Responsibility |
|---|---|---|
| `data-ingestion` | 8001 | Downloads IMDB dataset from HuggingFace, uploads raw JSON to MinIO |
| `preprocessing` | 8002 | Tokenises raw text with BERT tokenizer, uploads processed tensors to MinIO |
| `fine-tuning` | 8003 | Fine-tunes `bert-base-uncased`, serves real-time inference |
| `minio` | internal | S3-compatible artifact store shared across all services |

Each service exposes an async job API — `POST` to start a job, `GET /jobs/{id}` to poll status. All data flows through MinIO; services are stateless and independently restartable.

---

## Directory structure

```
Distributed-ML-Model-Provenance-System-in-Cloud-Native-Environments/
│
├── docker-compose.yml          # Local dev stack (all services + MinIO)
├── .env                        # Sets COMPOSE_PROJECT_NAME=atlas
├── README.md
│
├── services/
│   ├── data-ingestion/
│   │   ├── app.py              # FastAPI — /ingest /jobs /health /status
│   │   └── Dockerfile
│   ├── preprocessing/
│   │   ├── app.py              # FastAPI — /preprocess /jobs /health /status
│   │   └── Dockerfile
│   └── fine-tuning/
│       ├── app.py              # FastAPI — /train /predict /model/info /jobs /health
│       └── Dockerfile
│
├── scripts/
│   ├── build_and_push.sh       # Build images (--kind / --minikube / --registry)
│   └── run_pipeline.sh         # End-to-end pipeline runner (docker-compose or --k8s)
│
└── k8s/
    └── services.yaml           # Kubernetes Deployments + Services for all three stages
```

---

## Quickstart — Docker Compose

### Prerequisites

- Docker Desktop running
- Create .env file in project root directory with contents 'COMPOSE_PROJECT_NAME=atlas'

### Run

```bash
# Start all services in the background
docker compose up -d

# Run the full pipeline with 50 samples
./scripts/run_pipeline.sh --samples 50
```

Expected output:

```
[INFO]  All services healthy.
════════════════════════════════════════════════════
 ML Provenance Pipeline
 samples=50  split=train
════════════════════════════════════════════════════

[INFO]  Stage 1 – Data Ingestion (IMDB, 200 samples)
[INFO]  Data Ingestion: completed ✓
  samples : 50
  sha256  : a3f9c2...
  s3_uri  : s3://ml-provenance/raw/train_data.json

[INFO]  Stage 2 – Preprocessing (BERT tokenizer, max_length=128)
[INFO]  Preprocessing: completed ✓

[INFO]  Stage 3 – Fine-Tuning bert-base-uncased on IMDB
[INFO]  Fine-Tuning: completed ✓

[INFO]  Smoke test – Inference
  This movie was absolutely fantastic! The acting was superb.  → negative (0.6930)
  What a terrible waste of time. I fell asleep after 20 minute → negative (0.6624)
  An okay film, nothing special but watchable enough.          → negative (0.6786)

[INFO]  Pipeline complete ✓
```

### Useful commands

```bash
# Confirm everything is healthy
docker compose ps

# Stream logs from all services
docker compose logs -f

# Check a specific service
docker compose logs -f fine-tuning

# Health checks
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# Check whether data already exists in S3
curl http://localhost:8001/status
curl http://localhost:8002/status

# Run inference against a trained model
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This film was a masterpiece."}'

# View trained model metadata
curl http://localhost:8003/model/info

# Tear down
docker compose down
```

### Configuration

All parameters are set as environment variables in `docker-compose.yml`:

| Variable | Default | Used by |
|---|---|---|
| `MINIO_ENDPOINT` | `http://minio:9000` | all |
| `ARTIFACT_BUCKET` | `ml-provenance` | all |
| `MINIO_ACCESS_KEY` | `minioadmin` | all |
| `MINIO_SECRET_KEY` | `minioadmin` | all |
| `DATASET_NAME` | `imdb` | data-ingestion |
| `BERT_MODEL` | `bert-base-uncased` | preprocessing, fine-tuning |
| `MAX_SEQ_LENGTH` | `128` | preprocessing, fine-tuning |
| `EPOCHS` | `2` | fine-tuning |
| `BATCH_SIZE` | `16` | fine-tuning |
| `LEARNING_RATE` | `2e-5` | fine-tuning |
| `ATLAS_SIDECAR_URL` | `` (disabled) | all — set to `http://atlas-sidecar:8004` to enable Step 3 |

---

## API reference

### data-ingestion (:8001)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/ingest?split=train&num_samples=500` | Start ingestion job |
| `GET` | `/jobs/{job_id}` | Poll job status and results |
| `GET` | `/status?split=train` | Check if raw data exists in S3 |

### preprocessing (:8002)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/preprocess?split=train` | Start tokenisation job |
| `GET` | `/jobs/{job_id}` | Poll job status and results |
| `GET` | `/status?split=train` | Check if tokenised data exists in S3 |

### fine-tuning (:8003)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/train?split=train` | Start fine-tuning job |
| `GET` | `/jobs/{job_id}` | Poll job status, losses, and metrics |
| `POST` | `/predict` | `{"text": "..."}` → `{label, confidence, scores}` |
| `GET` | `/model/info` | Metadata for the trained model in S3 |

---

## Project spec progress

| Step | Description | Status |
|---|---|---|
| 1 | Design ML lifecycle with three stages | Done |
| 2 | Implement stages as containerised services with Kubernetes deployment | Done |
| 3 | Atlas CLI sidecar for automatic provenance collection at each stage | In progress |
| 4 | Kubernetes security best practices, Helm charts, service mesh | In progress |

---

## References

- [Atlas CLI](https://github.com/IntelLabs/atlas-cli) — Intel Labs ML provenance tooling
- [C2PA](https://c2pa.org/) — Content authenticity standard
- [OpenSSF Model Signing](https://github.com/sigstore/model-transparency) — OMS specification
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) — ML pipeline orchestration
- [Atlas paper](https://arxiv.org/abs/2502.19567) — Atlas: A Framework for ML Lifecycle Provenance & Transparency
