"""
Shared pytest fixtures and helpers for ML pipeline integration tests.

Services must be running before executing these tests:
  docker compose up          (or run individual services locally)

Run with:
  pip install pytest requests
  pytest tests/ -v
  pytest tests/ -v -m smoke           # health checks only
  pytest tests/ -v -m integration     # full pipeline (slow)
"""

import time
import pytest
import requests

# ── Service base URLs ──────────────────────────────────────────────────────────
INGEST_URL   = "http://localhost:8001"
PREPROC_URL  = "http://localhost:8002"
FINETUNE_URL = "http://localhost:8003"
SIDECAR_URL  = "http://localhost:8004"

POLL_INTERVAL = 3    # seconds between job-status polls
INGEST_TIMEOUT   = 300   # 5 min  – dataset download
PREPROC_TIMEOUT  = 300   # 5 min  – tokenisation
TRAIN_TIMEOUT    = 3600  # 60 min – BERT fine-tuning on CPU


# ── Helpers ────────────────────────────────────────────────────────────────────
def wait_for_job(base_url: str, job_id: str, timeout: int) -> dict:
    """Poll /jobs/{job_id} until status is 'completed' or 'failed'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{base_url}/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] == "completed":
            return data
        if data["status"] == "failed":
            pytest.fail(f"Job {job_id} failed: {data.get('error')}")
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Job {job_id} timed out after {timeout}s")


# ── Session-scoped fixtures (run once per test session) ────────────────────────
@pytest.fixture(scope="session")
def ingest_job(request):
    """
    Run a small ingestion job (100 samples) and return the completed job dict.
    Cached for the whole test session so downstream fixtures reuse the data.
    """
    num_samples = request.config.getoption("--samples", default=100)
    resp = requests.post(
        f"{INGEST_URL}/ingest",
        params={"split": "train", "num_samples": int(num_samples)},
        timeout=10,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    return wait_for_job(INGEST_URL, job_id, INGEST_TIMEOUT)


@pytest.fixture(scope="session")
def preprocess_job(ingest_job):
    """Tokenise the ingested data. Depends on ingest_job completing first."""
    resp = requests.post(
        f"{PREPROC_URL}/preprocess",
        params={"split": "train"},
        timeout=10,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    return wait_for_job(PREPROC_URL, job_id, PREPROC_TIMEOUT)


@pytest.fixture(scope="session")
def train_job(preprocess_job):
    """Fine-tune BERT. This is slow on CPU – mark tests that need it @slow."""
    resp = requests.post(
        f"{FINETUNE_URL}/train",
        params={"split": "train"},
        timeout=10,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    return wait_for_job(FINETUNE_URL, job_id, TRAIN_TIMEOUT)


# ── CLI option: --samples ──────────────────────────────────────────────────────
def pytest_addoption(parser):
    parser.addoption(
        "--samples",
        action="store",
        default=100,
        help="Number of IMDB samples to ingest (default: 100)",
    )


# ── Minimum-sample guard ───────────────────────────────────────────────────────
# BERT needs at least ~200 balanced samples to generalise well enough for
# sentiment tests (100 positive + 100 negative).  Tests that use this fixture
# are skipped automatically when the sample count is below threshold instead of
# failing with wrong/random predictions.
MIN_SAMPLES_FOR_SENTIMENT = 200

@pytest.fixture(scope="session")
def require_sufficient_samples(request):
    n = int(request.config.getoption("--samples", default=100))
    if n < MIN_SAMPLES_FOR_SENTIMENT:
        pytest.skip(
            f"Sentiment accuracy tests require --samples >= {MIN_SAMPLES_FOR_SENTIMENT} "
            f"(got {n}). BERT cannot generalise on fewer samples."
        )
