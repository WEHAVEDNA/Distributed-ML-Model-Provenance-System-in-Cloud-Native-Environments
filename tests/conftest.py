"""
Shared pytest fixtures and helpers for ML pipeline integration tests.

Services must be running before executing these tests:
  docker compose up          (or run individual services locally)

Run with:
  pip install pytest requests
  pytest tests/ -v
  pytest tests/ -v -m smoke           # health checks only
  pytest tests/ -v -m wired --samples 200
  pytest tests/ -v -m slow --samples 500
"""

import time
import pytest
import requests

# ── Service base URLs ──────────────────────────────────────────────────────────
INGEST_URL   = "http://localhost:8001"
PREPROC_URL  = "http://localhost:8002"
FINETUNE_URL = "http://localhost:8003"
SIDECAR_URL  = "http://localhost:8004"
PIPELINE_ID = "default"
PIPELINE_PREFIX = f"s3://ml-provenance/pipelines/{PIPELINE_ID}"
MODEL_ARTIFACT_NAME = "classifier"

POLL_INTERVAL = 3    # seconds between job-status polls
INGEST_TIMEOUT   = 300   # 5 min  – dataset download
PREPROC_TIMEOUT  = 300   # 5 min  – tokenisation
TRAIN_TIMEOUT    = 3600  # 60 min – BERT fine-tuning on CPU
HEALTHCHECK_TIMEOUT = 180
TRANSIENT_HTTP_ATTEMPTS = 8
TRANSIENT_HTTP_DELAY = 1.0
TRANSIENT_HTTP_MAX_ELAPSED = 30

HEALTH_ENDPOINTS = (
    ("data-ingestion", f"{INGEST_URL}/health"),
    ("preprocessing", f"{PREPROC_URL}/health"),
    ("fine-tuning", f"{FINETUNE_URL}/health"),
    ("atlas-sidecar", f"{SIDECAR_URL}/health"),
)

RETRIABLE_REQUEST_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

_ORIGINAL_SESSION_REQUEST = requests.sessions.Session.request


def _request_with_transient_retry(self, method, url, **kwargs):
    method_upper = method.upper()
    attempts = TRANSIENT_HTTP_ATTEMPTS if method_upper in {"GET", "HEAD", "OPTIONS"} else 1
    deadline = time.time() + TRANSIENT_HTTP_MAX_ELAPSED
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return _ORIGINAL_SESSION_REQUEST(self, method, url, **kwargs)
        except RETRIABLE_REQUEST_EXCEPTIONS as exc:
            last_exc = exc
            if attempt >= attempts or time.time() >= deadline:
                raise
            remaining = deadline - time.time()
            if remaining <= 0:
                raise
            time.sleep(min(min(TRANSIENT_HTTP_DELAY * attempt, 5), remaining))
    raise last_exc


requests.sessions.Session.request = _request_with_transient_retry


def raw_data_uri(split: str = "train", pipeline_id: str = PIPELINE_ID) -> str:
    return f"s3://ml-provenance/pipelines/{pipeline_id}/raw/{split}_data.json"


def preprocessed_data_uri(split: str = "train", pipeline_id: str = PIPELINE_ID) -> str:
    return f"s3://ml-provenance/pipelines/{pipeline_id}/preprocessed/{split}_tokenized.json"


def model_uri(pipeline_id: str = PIPELINE_ID, artifact_name: str = MODEL_ARTIFACT_NAME) -> str:
    return f"s3://ml-provenance/pipelines/{pipeline_id}/models/{artifact_name}/model.pt"


# ── Helpers ────────────────────────────────────────────────────────────────────
def wait_for_job(base_url: str, job_id: str, timeout: int) -> dict:
    """Poll /jobs/{job_id} until status is 'completed' or 'failed'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/jobs/{job_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            time.sleep(POLL_INTERVAL)
            continue

        if data["status"] == "completed":
            return data
        if data["status"] == "cancelled":
            return data
        if data["status"] == "failed":
            pytest.fail(f"Job {job_id} failed: {data.get('error')}")
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Job {job_id} timed out after {timeout}s")


@pytest.fixture(scope="session", autouse=True)
def ensure_services_ready(request):
    """Wait for all service health endpoints before running integration tests."""
    items = getattr(request.session, "items", [])
    if items and all(item.get_closest_marker("unit") for item in items):
        return

    deadline = time.time() + HEALTHCHECK_TIMEOUT
    while time.time() < deadline:
        all_ready = True
        for _, url in HEALTH_ENDPOINTS:
            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                if resp.json().get("status") != "ok":
                    all_ready = False
                    break
            except requests.RequestException:
                all_ready = False
                break
        if all_ready:
            return
        time.sleep(2)
    pytest.exit(
        f"Services did not become healthy within {HEALTHCHECK_TIMEOUT}s. "
        "Start Docker Compose and wait for all /health endpoints to return ok.",
        returncode=1,
    )


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
        params={"split": "train", "num_samples": int(num_samples), "pipeline_id": PIPELINE_ID},
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
        params={"split": "train", "pipeline_id": PIPELINE_ID},
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
        params={"split": "train", "pipeline_id": PIPELINE_ID},
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
MIN_SAMPLES_FOR_SENTIMENT = 500

@pytest.fixture(scope="session")
def require_sufficient_samples(request):
    n = int(request.config.getoption("--samples", default=100))
    if n < MIN_SAMPLES_FOR_SENTIMENT:
        pytest.skip(
            f"Sentiment accuracy tests require --samples >= {MIN_SAMPLES_FOR_SENTIMENT} "
            f"(got {n}). BERT cannot generalise on fewer samples."
        )
