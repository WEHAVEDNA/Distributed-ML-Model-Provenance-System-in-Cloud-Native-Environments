# Distributed ML Model Provenance System in Cloud-Native Environments

**BU EC528 Spring 2026**
Mentors: Marcela Melara, Marcin Spoczynski (Intel Labs)

Extending [Atlas CLI](https://github.com/IntelLabs/atlas-cli) into a cloud-native distributed provenance system.

Demo 1 presentation: https://docs.google.com/presentation/d/1z5usLUe9r3ex-ovyxyOwZE1G7A_tVfEOVQuWLKoFlhg/edit?usp=sharing

Demo 2 presentation: https://docs.google.com/presentation/d/1WfeDlM877Tw3Msrw_t-0FO3Swb3vk92SMZvVm914wR0/edit?usp=sharing

Demo 2 video: https://drive.google.com/file/d/1iPYX0nW1rNNxc22FYdzfefnvu_PNE1oM/view?usp=sharing

Demo 3 presentation: https://docs.google.com/presentation/d/1WfeDlM877Tw3Msrw_t-0FO3Swb3vk92SMZvVm914wR0/edit?usp=sharing

Demo 3 video: https://drive.google.com/file/d/1BLxXcpIp_xUrE9GHFw9CGTNGA9couwYI/view?usp=sharing

---

## Overview

This project extends the [IntelLabs Atlas CLI](https://github.com/IntelLabs/atlas-cli) into a cloud-native distributed provenance system by implementing a three-stage ML pipeline — data ingestion, preprocessing, and fine-tuning — where every artifact produced is automatically tracked with cryptographically signed C2PA provenance manifests via an Atlas CLI sidecar service.

Each pipeline stage runs as an independent containerised service. Artifacts flow through S3-compatible object storage (MinIO locally). The Atlas sidecar intercepts each completed stage and calls `atlas-cli` to generate a manifest that records what was produced, how, and by whom — forming a linked provenance chain from raw data to trained model.

Pipelines are now detached from the service instances through a shared `pipeline_id` contract. The same running services can host multiple independent pipelines by storing artifacts under `pipelines/<pipeline_id>/...` and querying provenance with that same ID.

---

### Provenance chain

```
pipelines/default/raw/train_data.json
    │  dataset manifest  (atlas-cli dataset create)
    ▼
pipelines/default/preprocessed/train_tokenized.json
    │  pipeline manifest (atlas-cli pipeline generate-provenance)
    ▼
pipelines/default/models/classifier/model.pt
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

All services are FastAPI with async background jobs. Storage is MinIO (local) or AWS S3 (cloud) — switching requires only changing the `S3_ENDPOINT_URL` environment variable. The default pipeline ID is `default`, but each job/status/provenance call can override it with `pipeline_id=<name>`.

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

Choose one local runtime:

### Option A — Docker Compose

```powershell
# 1. Start the local stack and keep this terminal open
#    first build is the slowest because atlas-sidecar compiles atlas-cli from Rust
docker compose --progress plain up --build
```

### Option B — Kubernetes on Docker Desktop

```bat
scripts\setup_k8s_docker_desktop.bat
```

That batched Windows wrapper builds the images and deploys the Kubernetes stack.
In Docker Desktop mode it now clears the existing Kubernetes service deployments and waits for their old pods to disappear first, so each startup comes up on fresh pods instead of reusing leftovers from the previous run.
After deployment it keeps the same terminal attached to a live Kubernetes pod-status stream instead of opening a second terminal automatically.

Then start the localhost port-forwards manually in the terminal where you want them to run:

```powershell
pwsh scripts\port_forward.ps1
```

If you only want the deploy step without the live watcher:

```bat
scripts\setup_k8s_docker_desktop.bat --no-watch
```

### Common next steps

```powershell
# 1. In a new terminal, optionally verify all four /health endpoints return "ok"
Invoke-RestMethod http://localhost:8001/health
Invoke-RestMethod http://localhost:8002/health
Invoke-RestMethod http://localhost:8003/health
Invoke-RestMethod http://localhost:8004/health

# 2. Run the demo in a separate terminal
#    default demo: ingestion + preprocessing only
python demo.py --samples 200 --pipeline-id demo-200

#    full demo: ingestion + preprocessing + 1-epoch training + inference smoke test
python demo.py --samples 500 --train --pipeline-id demo-500

#    full demo with a custom inference text
python demo.py --samples 500 --train --pipeline-id demo-500 --predict-text "Loved it."

#    full demo with a longer 3-epoch training run
python demo.py --samples 500 --train --train-epochs 3 --pipeline-id demo-500

#    only Stage 1: ingestion
python demo.py --stage ingest --samples 200 --pipeline-id stage-demo

#    only Stage 2: preprocessing
#    expects raw data for stage-demo to already exist
python demo.py --stage preprocess --pipeline-id stage-demo

#    only Stage 3: training + inference
#    expects preprocessed data for stage-demo to already exist
python demo.py --stage train --pipeline-id stage-demo --predict-text "Loved it."

# 3. Optionally verify provenance after a full run
Invoke-RestMethod "http://localhost:8004/lineage?pipeline_id=demo-500"
Invoke-RestMethod "http://localhost:8004/pipeline/status?pipeline_id=demo-500"
Invoke-RestMethod "http://localhost:8004/registry?pipeline_id=demo-500"

# 4. Optionally run tests
pytest tests/ -v -m smoke                  # fast schema/health checks only
pytest tests/ -v -m "wired" --samples 200  # provenance chain after pipeline
pytest tests/ -v -m slow --samples 500     # full suite including fine-tuning
```

See [SETUP.md](SETUP.md) for step-by-step recreation instructions.

Notes:
- Keep the selected backend running while you use the demo, scripts, or tests: either the `docker compose up --build` terminal or the Kubernetes watch terminal started by `scripts\setup_k8s_docker_desktop.bat`, plus a separate terminal running `pwsh scripts\port_forward.ps1` for localhost access.
- Do not leave both local backends active on the same ports at the same time. Stop Compose before using the Kubernetes port-forwards, and stop the Kubernetes port-forwards before switching back to Compose.
- `demo.py` now waits for service health automatically before starting the pipeline.
- When the backend is Kubernetes, `demo.py` now prints `Runtime backend: kubernetes` and includes each service pod and node in the health section so localhost traffic is clearly identified as a port-forward into the cluster.
- `demo.py` is self-contained. You do not need to run the test suite before running the demo.
- `demo.py --train` now defaults to `1` epoch to keep the demo shorter. Use `--train-epochs 3` for the longer full training run.
- `demo.py --stage pipeline` runs ingestion + preprocessing only.
- `demo.py --stage full` runs ingestion + preprocessing + training.
- `demo.py --stage ingest`, `--stage preprocess`, and `--stage train` run only that individual stage.
- `--train` is still supported as a legacy alias for `--stage full`.
- Pressing `Ctrl+C` during demo job polling now sends a cancellation request to the active service job instead of only stopping the local script.
- All three job-based services support `cancel_requested` / `cancelled` job states. If cancellation arrives very late, a job may still race to `completed`.
- `demo.py` and the service scripts all accept `pipeline_id`; use a fresh one when you want isolated artifacts and provenance.
- `split=validation` only works after you have ingested that split first.

## Deployment Modes

Docker Compose is the current day-to-day development path. Kubernetes is included as the next deployment target, not the required local workflow.

Use Docker Compose when:
- you are iterating locally
- you want the fastest setup
- you want the test suite and demo scripts to work exactly as documented

Use Kubernetes when:
- you want to validate the service split in a cluster
- you want MinIO + the pipeline services behind cluster networking
- you are preparing to move the stack to EKS or another managed cluster

### Kubernetes Migration Path

Docker Desktop is the simplest local cluster target because the cluster can use images built by the local Docker daemon directly.

The repository keeps Docker as the source-of-truth local environment, while the `k8s/` directory mirrors the same services for staged migration.

For Windows, the batched Kubernetes setup is:

```bat
scripts\setup_k8s_docker_desktop.bat
```

That wrapper builds the images, deploys the Kubernetes stack, then streams `kubectl get pods -w` in the same terminal.

```powershell
# 0. Enable Kubernetes in Docker Desktop and switch context once
kubectl config use-context docker-desktop

# 1. Build local images into Docker Desktop's image store
pwsh scripts/build_images.ps1 -DockerDesktop

# 2. Apply the stack and patch the four local service deployments
pwsh scripts/deploy_k8s.ps1 -DockerDesktop

# or do both in one step
pwsh scripts/deploy_k8s.ps1 -BuildImages -DockerDesktop

# 3. Start the localhost port-forwards in the current terminal
pwsh scripts/port_forward.ps1
```

Notes:
- `k8s/kustomization.yaml` lets you deploy the full stack with `kubectl apply -k k8s`.
- `pwsh scripts/deploy_k8s.ps1 -DockerDesktop` now enforces the `docker-desktop` context, applies the Kubernetes manifests with the four local service deployments already set to `imagePullPolicy: Never`, and waits for `minio-init` to complete.
- In Docker Desktop mode, the deploy script deletes the existing `minio`, `atlas-sidecar`, `data-ingestion`, `preprocessing`, and `fine-tuning` deployments plus the `minio-init` job before reapplying manifests, then waits for the old pods to terminate. That gives you a clean local-cluster start without stale pods lingering or a second rollout caused by post-apply patching.
- `scripts\setup_k8s_docker_desktop.bat` is the one-shot Windows Kubernetes deploy entrypoint. It now streams `kubectl get pods -w` in the same terminal after deployment. Start port-forwards manually with `pwsh scripts\port_forward.ps1`, stop them later with `scripts\stop_k8s_port_forward.bat`, and use `scripts\setup_k8s_docker_desktop.bat --no-watch` if you only want the deploy step.
- The Kubernetes config now matches the Compose environment contract for pipeline IDs, root prefixes, model artifact naming, and sidecar stage ordering.
- The HuggingFace cache PVC defaults to `ReadWriteOnce` for local-cluster compatibility. If you move to a multi-node cluster and want a truly shared cache, switch that PVC to a storage class that supports `ReadWriteMany`.
- `scripts/build_and_push.sh` now builds all four service images, including `atlas-sidecar`, so the Kubernetes manifests and Docker Compose use the same container set.
- If you want Minikube or kind instead, the existing `-Minikube` and `-Kind` script paths still work.

### Stopping And Switching

Stop Docker Compose:

```powershell
docker compose down
```

Stop Docker Compose and remove local volumes:

```powershell
docker compose down -v
```

Stop Kubernetes localhost port-forwards only:

```bat
scripts\stop_k8s_port_forward.bat
```

Remove the Kubernetes stack from the Docker Desktop cluster:

```powershell
kubectl delete -k k8s
```

Switch from Docker Compose to Kubernetes:

```powershell
docker compose down
scripts\setup_k8s_docker_desktop.bat
```

Switch from Kubernetes back to Docker Compose:

```powershell
scripts\stop_k8s_port_forward.bat
kubectl delete -k k8s
docker compose --progress plain up --build
```

---

## API Reference

### Pipeline services

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `:8001/health` | Liveness + deployment mode and runtime location metadata |
| POST | `:8001/ingest?split=train&num_samples=500&pipeline_id=default` | Start ingestion job |
| GET | `:8001/jobs/{id}` | Poll job status |
| POST | `:8001/jobs/{id}/cancel` | Request cancellation for an ingestion job |
| GET | `:8001/status?split=train&pipeline_id=default` | Check if raw data exists in S3 |
| POST | `:8002/preprocess?split=train&pipeline_id=default` | Start tokenization job |
| GET | `:8002/jobs/{id}` | Poll job status |
| POST | `:8002/jobs/{id}/cancel` | Request cancellation for a preprocessing job |
| POST | `:8003/train?split=train&pipeline_id=default` | Start fine-tuning job |
| GET | `:8003/jobs/{id}` | Poll job status (includes epoch, loss, and progress fields while training) |
| POST | `:8003/jobs/{id}/cancel` | Request cancellation for a fine-tuning job |
| POST | `:8003/predict` | `{"text": "...", "pipeline_id": "default"}` → label + confidence |
| GET | `:8003/model/info?pipeline_id=default` | Trained model metadata |

### Atlas sidecar

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `:8004/health` | Liveness + cached atlas-cli version + key status + deployment mode and runtime location metadata |
| POST | `:8004/collect/dataset` | Register dataset artifact, create C2PA manifest |
| POST | `:8004/collect/pipeline` | Register pipeline-step SLSA provenance |
| POST | `:8004/collect/model` | Register model artifact, link to dataset manifests |
| GET | `:8004/lineage?pipeline_id=default` | Ordered sidecar-tracked pipeline history, including pipeline-step records |
| GET | `:8004/pipeline/status?pipeline_id=default` | Per-stage tracked-record counts, manifest counts, and completion flags |
| GET | `:8004/registry?pipeline_id=default` | Tracked artifacts for one pipeline |
| GET | `:8004/pipelines` | Summary of all known pipeline IDs in the registry |
| GET | `:8004/export/{manifest_id}` | Export full provenance graph as JSON |
| GET | `:8004/verify/{manifest_id}` | Verify manifest cryptographic integrity |
| GET | `:8004/signing-key` | RSA public key PEM |

### Per-service provenance

Each pipeline service exposes `GET /provenance?pipeline_id=...` returning the manifest ID of the last artifact it registered for that pipeline:

| Service | Endpoint |
|---------|----------|
| data-ingestion | `:8001/provenance` |
| preprocessing | `:8002/provenance` |
| fine-tuning | `:8003/provenance` |

`/lineage` entries always include a sidecar `tracking_id`. Exportable Atlas/C2PA objects also include `manifest_id`. When `atlas-cli pipeline generate-provenance` does not return a resolvable URN, the sidecar creates an explicit pipeline-step manifest and links it to the upstream inputs.

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
| `test_connection_handling.py` | Startup waits, transient disconnect retry, timeout handling |
| `test_data_ingestion.py` | Job lifecycle, SHA-256, S3 URI, idempotency |
| `test_preprocessing.py` | Token counts, max_length, source linkage |
| `test_fine_tuning.py` | Loss progression, model metadata, predict schema, sentiment accuracy |
| `test_pipeline_e2e.py` | Full provenance chain, SHA-256 uniqueness, end-to-end sentiment |
| `test_atlas_sidecar.py` | Direct collect calls, registry updates, manifest export/verify |
| `test_provenance_chain.py` | /lineage and /pipeline/status schema + wired chain correctness |
| `test_demo_cancellation.py` | Demo-side remote cancellation requests on local interrupt |

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
