# ML Provenance Pipeline – Test Recreation Guide

BU EC528 Spring 2026 · Intel Labs Atlas CLI

---

## Prerequisites

- **Docker Desktop** — `winget install Docker.DockerDesktop`, then open a new terminal
- **Python 3.11+** — `winget install Python.Python.3.11`
- **Test dependencies** — `pip install pytest requests pytest-timeout`

---

## Step 1 — Start the Stack

```bat
docker compose up --build
```

Wait until all four services are healthy. First build takes **15–20 minutes** (atlas-sidecar compiles atlas-cli from Rust source). Subsequent starts use cache and take under a minute.

Verify:
```powershell
Invoke-RestMethod http://localhost:8001/health   # data-ingestion
Invoke-RestMethod http://localhost:8002/health   # preprocessing
Invoke-RestMethod http://localhost:8003/health   # fine-tuning
Invoke-RestMethod http://localhost:8004/health   # atlas-sidecar
```

Each should return `{ "status": "ok" }`.

---

## Step 2 — Run the Pipeline

Use the provided script to run all three stages in sequence:

```powershell
scripts\run_pipeline.bat 500
# or directly:
pwsh scripts\pipeline.ps1 -Samples 500
```

Or run each stage individually:

```powershell
scripts\01_ingest.bat 500
scripts\02_preprocess.bat
scripts\03_train.bat
```

Training on CPU takes **5–10 minutes** for 500 samples. Wait for `Done.` before running tests.

---

## Step 3 — Run the Tests

```bat
:: Fast tests only (no training required)
pytest tests/ -v

:: Full test suite including training and sentiment accuracy
pytest tests/ -v -m slow --samples 500
```

### Test files

| File | What it tests |
|------|--------------|
| `test_data_ingestion.py` | Ingestion job lifecycle, SHA-256, S3 URI, idempotency |
| `test_preprocessing.py` | Token counts, max_length, source linkage |
| `test_fine_tuning.py` | Loss progression, model metadata, `/predict` schema |
| `test_pipeline_e2e.py` | Full provenance chain, SHA-256 uniqueness, sentiment accuracy |
| `test_atlas_sidecar.py` | Direct sidecar calls, manifest collect/export/verify, wired pipeline checks |

### Sidecar test modes

The sidecar tests have two modes controlled by markers:

```powershell
# Direct — calls the sidecar manually, works regardless of ATLAS_SIDECAR_URL wiring
pytest tests/test_atlas_sidecar.py -v

# Wired — verifies the pipeline services auto-called the sidecar
# Requires containers restarted with: docker compose up -d --force-recreate
pytest tests/test_atlas_sidecar.py -v -m wired --samples 500
```

If wired tests fail with "not in registry": the containers were started before `ATLAS_SIDECAR_URL` was added to docker-compose.yaml. Fix: `docker compose up -d --force-recreate`, then re-run the pipeline.

### Notes

- Sentiment accuracy tests (`test_inference_label_correct`, etc.) **auto-skip** when `--samples < 500`. This is expected — BERT needs at least 500 samples and 3 epochs to classify both classes reliably.
- `--samples 100` runs everything except sentiment tests, in about 1–2 minutes total.

---

## Recreate from Scratch

```bat
:: Tear down all containers and data
docker compose down -v

:: Rebuild and restart
docker compose up --build

:: Re-run pipeline and tests (new terminal)
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
