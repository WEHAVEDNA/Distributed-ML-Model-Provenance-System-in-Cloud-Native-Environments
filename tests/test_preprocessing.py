"""
Tests for the Preprocessing Service (Stage 2).

Covers:
  - Requires Stage 1 data to be present (depends on ingest_job fixture)
  - Tokenization output shape and content
  - Provenance linkage (source_uri matches ingestion output)
  - /status endpoint
  - Edge cases: missing source data

  pytest tests/test_preprocessing.py -v
"""

import pytest
import requests
from conftest import PREPROC_URL, wait_for_job, PREPROC_TIMEOUT


class TestPreprocessingJobLifecycle:
    def test_preprocess_returns_job_id(self, ingest_job):
        r = requests.post(
            f"{PREPROC_URL}/preprocess",
            params={"split": "train"},
            timeout=10,
        )
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running")
        assert body["split"] == "train"

    def test_preprocess_job_completes(self, preprocess_job):
        assert preprocess_job["status"] == "completed"

    def test_preprocess_sample_count_matches_ingestion(self, ingest_job, preprocess_job):
        # Idempotency tests may re-ingest with fewer samples and overwrite S3
        # before the preprocess_job fixture runs, so we assert <= not ==.
        assert preprocess_job["num_samples"] > 0
        assert preprocess_job["num_samples"] <= ingest_job["num_samples"]

    def test_preprocess_max_length(self, preprocess_job):
        """max_length must match the configured MAX_SEQ_LENGTH (default 128)."""
        assert preprocess_job["max_length"] == 128

    def test_preprocess_has_sha256(self, preprocess_job):
        sha = preprocess_job["sha256"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_preprocess_s3_uri(self, preprocess_job):
        uri = preprocess_job["s3_uri"]
        assert uri.startswith("s3://ml-provenance/preprocessed/")
        assert uri.endswith("_tokenized.json")

    def test_unknown_job_id_returns_404(self):
        r = requests.get(f"{PREPROC_URL}/jobs/doesnotexist", timeout=5)
        assert r.status_code == 404


class TestPreprocessingProvenanceMetadata:
    def test_status_includes_source_uri(self, preprocess_job):
        r = requests.get(f"{PREPROC_URL}/status", params={"split": "train"}, timeout=10)
        assert r.status_code == 200
        meta = r.json()["meta"]
        # source_uri must point back to the raw data (stage 1 output)
        assert "source_uri" in meta
        assert "raw/train_data.json" in meta["source_uri"]

    def test_status_includes_bert_model(self, preprocess_job):
        r = requests.get(f"{PREPROC_URL}/status", params={"split": "train"}, timeout=10)
        meta = r.json()["meta"]
        assert meta["bert_model"] == "bert-base-uncased"

    def test_status_includes_timestamp(self, preprocess_job):
        r = requests.get(f"{PREPROC_URL}/status", params={"split": "train"}, timeout=10)
        meta = r.json()["meta"]
        assert "processed_at" in meta
        assert meta["processed_at"].endswith("Z")

    def test_status_unavailable_before_data(self):
        """A split that was never ingested should return available=False."""
        r = requests.get(f"{PREPROC_URL}/status", params={"split": "validation"}, timeout=10)
        assert r.status_code == 200
        assert r.json()["available"] is False


class TestPreprocessingWithoutIngestion:
    def test_preprocess_missing_split_fails(self):
        """
        Preprocessing a split that has no raw data in S3 must result in
        a failed job, not an unhandled 500 from the endpoint itself.
        """
        r = requests.post(
            f"{PREPROC_URL}/preprocess",
            params={"split": "validation"},
            timeout=10,
        )
        assert r.status_code == 200   # endpoint accepts the job
        job_id = r.json()["job_id"]

        import time
        deadline = time.time() + 60
        while time.time() < deadline:
            status = requests.get(f"{PREPROC_URL}/jobs/{job_id}", timeout=5).json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(3)

        assert status["status"] == "failed"
        assert "error" in status
