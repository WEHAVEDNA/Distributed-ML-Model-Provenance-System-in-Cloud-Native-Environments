"""
Smoke tests – verify all three services are reachable and healthy.
These are fast and have no side effects.

  pytest tests/test_health.py -v
"""

import pytest
import requests
from conftest import INGEST_URL, PREPROC_URL, FINETUNE_URL

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(10)]


class TestDataIngestionHealth:
    def test_health_status_ok(self):
        r = requests.get(f"{INGEST_URL}/health", timeout=5)
        assert r.status_code == 200

    def test_health_body(self):
        body = requests.get(f"{INGEST_URL}/health", timeout=5).json()
        assert body["status"] == "ok"
        assert body["service"] == "data-ingestion"

    def test_docs_reachable(self):
        """FastAPI auto-generates /docs; confirms the app loaded correctly."""
        r = requests.get(f"{INGEST_URL}/docs", timeout=5)
        assert r.status_code == 200


class TestPreprocessingHealth:
    def test_health_status_ok(self):
        r = requests.get(f"{PREPROC_URL}/health", timeout=5)
        assert r.status_code == 200

    def test_health_body(self):
        body = requests.get(f"{PREPROC_URL}/health", timeout=5).json()
        assert body["status"] == "ok"
        assert body["service"] == "preprocessing"

    def test_docs_reachable(self):
        r = requests.get(f"{PREPROC_URL}/docs", timeout=5)
        assert r.status_code == 200


class TestFineTuningHealth:
    def test_health_status_ok(self):
        r = requests.get(f"{FINETUNE_URL}/health", timeout=5)
        assert r.status_code == 200

    def test_health_body(self):
        body = requests.get(f"{FINETUNE_URL}/health", timeout=5).json()
        assert body["status"] == "ok"
        assert body["service"] == "fine-tuning"
        assert "device" in body
        assert "model_loaded" in body

    def test_docs_reachable(self):
        r = requests.get(f"{FINETUNE_URL}/docs", timeout=5)
        assert r.status_code == 200
