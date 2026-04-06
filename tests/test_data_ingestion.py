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
from conftest import INGEST_TIMEOUT, INGEST_URL, PIPELINE_ID, raw_data_uri, wait_for_job


class TestIngestionJobLifecycle:
    def test_ingest_returns_job_id(self):
        r = requests.post(
            f"{INGEST_URL}/ingest",
            params={"split": "train", "num_samples": 50, "pipeline_id": PIPELINE_ID},
            timeout=10,
        )
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running")
        assert body["split"] == "train"
        assert body["num_samples"] == 50
        assert body["pipeline_id"] == PIPELINE_ID

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
        assert uri == raw_data_uri()

    @pytest.mark.timeout(10)
    def test_unknown_job_id_returns_404(self):
        r = requests.get(f"{INGEST_URL}/jobs/doesnotexist", timeout=5)
        assert r.status_code == 404

    @pytest.mark.timeout(10)
    def test_unknown_job_id_cancel_returns_404(self):
        r = requests.post(f"{INGEST_URL}/jobs/doesnotexist/cancel", timeout=5)
        assert r.status_code == 404

    def test_ingest_test_split(self):
        """The test split should also be ingestible."""
        r = requests.post(
            f"{INGEST_URL}/ingest",
            params={"split": "test", "num_samples": 30, "pipeline_id": PIPELINE_ID},
            timeout=10,
        )
        assert r.status_code == 200
        job = wait_for_job(INGEST_URL, r.json()["job_id"], INGEST_TIMEOUT)
        assert job["status"] == "completed"
        assert job["num_samples"] <= 30

    def test_ingest_job_can_be_cancelled(self):
        r = requests.post(
            f"{INGEST_URL}/ingest",
            params={"split": "train", "num_samples": 8, "pipeline_id": "cancel-ingest"},
            timeout=10,
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        cancel = requests.post(f"{INGEST_URL}/jobs/{job_id}/cancel", timeout=10)
        assert cancel.status_code == 200
        assert cancel.json()["status"] in {"cancel_requested", "cancelled", "completed"}

        final = wait_for_job(INGEST_URL, job_id, INGEST_TIMEOUT)
        assert final["status"] in {"cancelled", "completed"}


class TestIngestionIdempotency:
    def test_re_ingest_succeeds(self, ingest_job):
        """Calling /ingest again must succeed (S3 put_object overwrites)."""
        r = requests.post(
            f"{INGEST_URL}/ingest",
            params={"split": "train", "num_samples": 50, "pipeline_id": PIPELINE_ID},
            timeout=10,
        )
        assert r.status_code == 200
        job = wait_for_job(INGEST_URL, r.json()["job_id"], INGEST_TIMEOUT)
        assert job["status"] == "completed"

    def test_re_ingest_produces_different_sha_for_different_size(self, ingest_job):
        """50 samples should produce a different checksum than 100 samples."""
        r = requests.post(
            f"{INGEST_URL}/ingest",
            params={"split": "train", "num_samples": 50, "pipeline_id": PIPELINE_ID},
            timeout=10,
        )
        job = wait_for_job(INGEST_URL, r.json()["job_id"], INGEST_TIMEOUT)
        # different sample count → different payload → different sha256
        assert job["sha256"] != ingest_job["sha256"]


class TestIngestionStatus:
    def test_status_available_after_ingest(self, ingest_job):
        r = requests.get(
            f"{INGEST_URL}/status",
            params={"split": "train", "pipeline_id": PIPELINE_ID},
            timeout=10,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["pipeline_id"] == PIPELINE_ID
        meta = body["meta"]
        assert meta["pipeline_id"] == PIPELINE_ID
        assert meta["dataset"] == "imdb"
        assert meta["split"] == "train"
        assert "sha256" in meta
        assert "ingested_at" in meta
        assert meta["ingested_at"].endswith("Z")   # UTC ISO-8601

    def test_status_unavailable_for_unknown_split(self):
        r = requests.get(
            f"{INGEST_URL}/status",
            params={"split": "validation", "pipeline_id": PIPELINE_ID},
            timeout=10,
        )
        assert r.status_code == 200
        assert r.json()["available"] is False
