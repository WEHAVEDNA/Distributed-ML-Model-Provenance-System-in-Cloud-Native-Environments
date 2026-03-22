"""
Tests for the Data Ingestion Service (Stage 1).

Covers:
  - Job submission and async polling
  - Artifact provenance metadata (sha256, s3_uri, timestamps)
  - Idempotency: re-ingestion overwrites cleanly
  - Invalid input handling
  - /status endpoint before and after ingestion

  pytest tests/test_data_ingestion.py -v
"""

import pytest
import requests
from conftest import INGEST_URL, wait_for_job, INGEST_TIMEOUT


class TestIngestionJobLifecycle:
    def test_ingest_returns_job_id(self):
        r = requests.post(f"{INGEST_URL}/ingest", params={"split": "train", "num_samples": 50}, timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running")
        assert body["split"] == "train"
        assert body["num_samples"] == 50

    def test_ingest_job_completes(self, ingest_job):
        assert ingest_job["status"] == "completed"

    def test_ingest_job_sample_count(self, ingest_job, request):
        """Returned sample count must be positive and not exceed the requested amount."""
        requested = int(request.config.getoption("--samples", default=100))
        assert 0 < ingest_job["num_samples"] <= requested

    def test_ingest_job_has_sha256(self, ingest_job):
        sha = ingest_job["sha256"]
        assert len(sha) == 64              # hex SHA-256
        assert all(c in "0123456789abcdef" for c in sha)

    def test_ingest_job_has_s3_uri(self, ingest_job):
        uri = ingest_job["s3_uri"]
        assert uri.startswith("s3://ml-provenance/raw/")
        assert uri.endswith("_data.json")

    @pytest.mark.timeout(10)
    def test_unknown_job_id_returns_404(self):
        r = requests.get(f"{INGEST_URL}/jobs/doesnotexist", timeout=5)
        assert r.status_code == 404

    def test_ingest_test_split(self):
        """The test split should also be ingestible."""
        r = requests.post(
            f"{INGEST_URL}/ingest",
            params={"split": "test", "num_samples": 30},
            timeout=10,
        )
        assert r.status_code == 200
        job = wait_for_job(INGEST_URL, r.json()["job_id"], INGEST_TIMEOUT)
        assert job["status"] == "completed"
        assert job["num_samples"] <= 30


class TestIngestionIdempotency:
    def test_re_ingest_succeeds(self, ingest_job):
        """Calling /ingest again must succeed (S3 put_object overwrites)."""
        r = requests.post(
            f"{INGEST_URL}/ingest",
            params={"split": "train", "num_samples": 50},
            timeout=10,
        )
        assert r.status_code == 200
        job = wait_for_job(INGEST_URL, r.json()["job_id"], INGEST_TIMEOUT)
        assert job["status"] == "completed"

    def test_re_ingest_produces_different_sha_for_different_size(self, ingest_job):
        """50 samples should produce a different checksum than 100 samples."""
        r = requests.post(
            f"{INGEST_URL}/ingest",
            params={"split": "train", "num_samples": 50},
            timeout=10,
        )
        job = wait_for_job(INGEST_URL, r.json()["job_id"], INGEST_TIMEOUT)
        # different sample count → different payload → different sha256
        assert job["sha256"] != ingest_job["sha256"]


class TestIngestionStatus:
    def test_status_available_after_ingest(self, ingest_job):
        r = requests.get(f"{INGEST_URL}/status", params={"split": "train"}, timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        meta = body["meta"]
        assert meta["dataset"] == "imdb"
        assert meta["split"] == "train"
        assert "sha256" in meta
        assert "ingested_at" in meta
        assert meta["ingested_at"].endswith("Z")   # UTC ISO-8601

    def test_status_unavailable_for_unknown_split(self):
        r = requests.get(f"{INGEST_URL}/status", params={"split": "validation"}, timeout=10)
        assert r.status_code == 200
        assert r.json()["available"] is False
