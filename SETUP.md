# ML Provenance Pipeline – Test Recreation Guide

BU EC528 Spring 2026 · Intel Labs Atlas CLI

---

## Prerequisites

- **Docker Desktop** — `winget install Docker.DockerDesktop`, then open a new terminal
- **Python 3.11+** — `winget install Python.Python.3.11`
- **Test dependencies** — `pip install pytest requests pytest-timeout`

---

## Step 1 — Start the Stack

Choose one local runtime.

### Option A — Docker Compose

```bat
docker compose --progress plain up --build
```

Keep that terminal open. Wait until all four services are healthy. The first build is the slowest because `atlas-sidecar` compiles `atlas-cli` from Rust source; later starts are much faster because Docker reuses the cached layers.

### Option B — Kubernetes on Docker Desktop

```bat
scripts\setup_k8s_docker_desktop.bat
```

That batched Windows wrapper builds the images and deploys the cluster stack.
In Docker Desktop mode it also clears the previous Kubernetes service deployments and waits for their pods to disappear first, so each startup comes up on fresh pods.
After deployment it keeps the current terminal attached to a live Kubernetes pod-status stream instead of opening another terminal automatically.

Then start the port-forwards manually in the terminal where you want them to run:

```powershell
pwsh scripts\port_forward.ps1
```

If you only want the deploy step without the live watcher:

```bat
scripts\setup_k8s_docker_desktop.bat --no-watch
```

### Verify

```powershell
Invoke-RestMethod http://localhost:8001/health   # data-ingestion
Invoke-RestMethod http://localhost:8002/health   # preprocessing
Invoke-RestMethod http://localhost:8003/health   # fine-tuning
Invoke-RestMethod http://localhost:8004/health   # atlas-sidecar
```

Each should return `{ "status": "ok" }`.

Do not leave both local backends active on the same localhost ports at the same time. Stop Compose before using the Kubernetes port-forwards, and stop the Kubernetes port-forwards before switching back to Compose.

Docker Compose is the default local setup. The Kubernetes manifests are available for staged migration, but they are optional for normal development.

### Optional: local Kubernetes deployment

If you want to validate the cluster deployment without replacing the Docker workflow:

```bat
scripts\setup_k8s_docker_desktop.bat
```

That batched Windows wrapper builds the images, deploys the cluster stack, and then streams `kubectl get pods -w` in the same terminal.

```powershell
# Enable Docker Desktop Kubernetes once, then switch context
kubectl config use-context docker-desktop

# Build local images into Docker Desktop's image store
pwsh scripts/build_images.ps1 -DockerDesktop

# Apply the stack and patch the four local service deployments
pwsh scripts/deploy_k8s.ps1 -DockerDesktop

# or do both in one step
pwsh scripts/deploy_k8s.ps1 -BuildImages -DockerDesktop

# Reuse the existing localhost-based scripts by port-forwarding
pwsh scripts\port_forward.ps1
```

`pwsh scripts/deploy_k8s.ps1 -DockerDesktop` now enforces the `docker-desktop` context, deletes the existing local Kubernetes workloads first, waits for their old pods to terminate, applies the manifests with the four local service deployments already set to `imagePullPolicy: Never`, and waits for `minio-init` to complete. Because the deployments are created with the correct policy up front, the script no longer triggers an immediate second rollout on fresh pods.

Use `scripts\stop_k8s_port_forward.bat` later to stop the port-forwards. Use `scripts\setup_k8s_docker_desktop.bat --no-watch` if you want the deployment without the live watcher.

If you prefer minikube or kind, the existing `pwsh scripts/build_images.ps1 -Minikube` and `-Kind` paths still work.

The Kubernetes path is intended as a migration target. Keep Compose as the baseline until you are ready to swap the operational default.

### Stop and switch

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

## Step 2 — Run the Demo

Open a new terminal while your chosen backend is still running:

```powershell
python demo.py --samples 200 --pipeline-id demo-200
python demo.py --samples 500 --train --pipeline-id demo-500
python demo.py --samples 500 --train --pipeline-id demo-500 --predict-text "Loved it."
python demo.py --samples 500 --train --train-epochs 3 --pipeline-id demo-500
python demo.py --stage ingest --samples 200 --pipeline-id stage-demo
python demo.py --stage preprocess --pipeline-id stage-demo
python demo.py --stage train --pipeline-id stage-demo --predict-text "Loved it."
```

What the demo does:
- waits for all four `/health` endpoints to become healthy
- runs ingestion, preprocessing, and optionally training
- shows the sidecar lineage and pipeline status for the chosen `pipeline_id`
- runs a small inference smoke test when `--train` is enabled
- prints the deployment mode during health checks; on Kubernetes it explicitly shows `Runtime backend: kubernetes` together with the service pod and node names

Use a fresh `pipeline_id` when you want a clean provenance chain without reusing older artifacts.
The demo is standalone. You do not need to run the tests before running `demo.py`.
When `--train` is enabled, the demo uses `1` epoch by default so the walkthrough finishes faster. Use `--train-epochs 3` if you want the longer full training run.
Use `--stage pipeline` for ingestion + preprocessing only, `--stage full` for the complete pipeline, or `--stage ingest|preprocess|train` to run only one stage. `--train` remains available as a legacy alias for `--stage full`.
If you press `Ctrl+C` while the demo is waiting on ingestion, preprocessing, or training, it now sends a cancellation request to the active service job before exiting.
Cancellation is best-effort. A job that is already finishing may still report `completed` instead of `cancelled`.

---

## Step 3 — Verify Provenance Collection

After a full demo run such as:

```powershell
python demo.py --samples 500 --train --pipeline-id demo-500
```

verify that provenance was collected and linked properly:

```powershell
$pipeline = "demo-500"

Invoke-RestMethod "http://localhost:8004/lineage?pipeline_id=$pipeline"
Invoke-RestMethod "http://localhost:8004/pipeline/status?pipeline_id=$pipeline"
Invoke-RestMethod "http://localhost:8004/registry?pipeline_id=$pipeline"

Invoke-RestMethod "http://localhost:8001/provenance?pipeline_id=$pipeline"
Invoke-RestMethod "http://localhost:8002/provenance?pipeline_id=$pipeline"
Invoke-RestMethod "http://localhost:8003/provenance?pipeline_id=$pipeline"

Invoke-RestMethod "http://localhost:8003/model/info?pipeline_id=$pipeline"
```

Expected success signals:
- `/lineage` shows `chain_complete: true`
- `/lineage.chain` contains `data-ingestion`, `preprocessing`, and `fine-tuning`
- every lineage entry has a sidecar `tracking_id`
- some pipeline-step entries may be `tracked-only` if Atlas does not return a resolvable manifest URN
- dataset/model lineage entries have manifest IDs; pipeline-step entries may be tracked without an exportable manifest ID
- `/pipeline/status` marks all stages done
- `/registry` contains the pipeline-scoped raw, tokenized, and model artifact URIs
- the three service `/provenance` endpoints all return manifest IDs for the same pipeline
- `/model/info` returns the trained model metadata for that pipeline

You can also export and verify a specific manifest:

```powershell
$lineage = Invoke-RestMethod "http://localhost:8004/lineage?pipeline_id=demo-500"
$manifest = $lineage.chain[0].manifest_id
$encoded = [uri]::EscapeDataString($manifest)

Invoke-RestMethod "http://localhost:8004/export/$encoded"
Invoke-RestMethod "http://localhost:8004/verify/$encoded"
```

---

## Step 4 — Run the Pipeline Scripts

If you want the stages individually or through the PowerShell wrapper, use the provided scripts:

```powershell
scripts\run_pipeline.bat 500
# or directly:
pwsh scripts\pipeline.ps1 -Samples 500 -PipelineId default
```

Or run each stage individually:

```powershell
scripts\01_ingest.bat 500
scripts\02_preprocess.bat
scripts\03_train.bat
```

Training on CPU is the slowest stage. Wait for the final completed status before running inference or the `slow` tests.

---

## Step 5 — Run the Tests

```bat
:: Fast health/schema checks only
pytest tests/ -v -m smoke

:: All non-slow tests
pytest tests/ -v

:: Provenance chain tests after running the pipeline
pytest tests/ -v -m wired --samples 200

:: Full suite including fine-tuning and sentiment checks
pytest tests/ -v -m slow --samples 500
```

### Test files

| File | What it tests |
|------|--------------|
| `test_connection_handling.py` | transient disconnect retry, startup waits, timeout behavior |
| `test_data_ingestion.py` | Ingestion job lifecycle, SHA-256, S3 URI, idempotency |
| `test_preprocessing.py` | Token counts, max_length, source linkage |
| `test_fine_tuning.py` | Loss progression, model metadata, `/predict` schema |
| `test_pipeline_e2e.py` | Full provenance chain, SHA-256 uniqueness, sentiment accuracy |
| `test_atlas_sidecar.py` | Direct sidecar calls, manifest collect/export/verify, wired pipeline checks |
| `test_demo_cancellation.py` | demo interrupt behavior and remote cancellation requests |

### Sidecar test modes

The sidecar tests have two modes controlled by markers:

```powershell
# Direct — calls the sidecar manually, works regardless of ATLAS_SIDECAR_URL wiring
pytest tests/test_atlas_sidecar.py -v

# Wired — verifies the pipeline services auto-called the sidecar
# Run the pipeline first, then:
pytest tests/test_atlas_sidecar.py -v -m wired --samples 200
```

If wired tests fail with "not in registry", restart the containers with `docker compose up -d --force-recreate`, then re-run the pipeline before re-running the test.

### Notes

- Sentiment accuracy tests (`test_inference_label_correct`, etc.) **auto-skip** when `--samples < 500`. This is expected — BERT needs at least 500 samples and 3 epochs to classify both classes reliably.
- `--samples 100` runs everything except sentiment tests, in about 1–2 minutes total.
- `split=validation` is not available unless you ingest that split first. The default demo and scripts use `train`.
- All job-based services now expose `POST /jobs/{id}/cancel`. Tests and the demo treat `cancelled` as a terminal job state.

---

## Recreate from Scratch

```bat
:: Tear down all containers and data
docker compose down -v

:: Rebuild and restart
docker compose --progress plain up --build

:: Re-run pipeline and tests (new terminal)
python demo.py --samples 200 --pipeline-id recreate-demo
scripts\run_pipeline.bat 500
pytest tests/ -v -m slow --samples 500
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `docker: not found` | Open a new terminal after installing Docker Desktop |
| Container stuck `unhealthy` | Check logs: `docker compose logs <service>` |
| atlas-sidecar never becomes healthy | Rust build may have timed out — retry `docker compose up --build` |
| All predictions return `negative` | Re-run `01_ingest.bat` (shuffle fix already applied), then preprocess and train again |
| Sentiment tests skip at `--samples 500` | Confirm you passed `--samples 500` and training completed with `status: completed` |
| `Ctrl+C` stopped the demo but work kept running earlier | Rebuild the stack; current images propagate local demo cancellation to the running service job |
