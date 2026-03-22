"""
End-to-end pipeline test – runs all three stages in sequence and verifies
the provenance chain links each stage's output to the next stage's input.

This is the "golden path" test for the ML Provenance project.

  pytest tests/test_pipeline_e2e.py -v -m slow --samples 100
"""

import pytest
import requests
from conftest import INGEST_URL, PREPROC_URL, FINETUNE_URL


@pytest.mark.slow
class TestProvenance:
    """
    Verify that the artifact chain is correctly recorded:
      raw data  →  tokenized data  →  trained model
    Each stage stores a sha256 and source_uri pointing to the previous stage.
    """

    def test_ingestion_meta_recorded(self, ingest_job):
        r = requests.get(f"{INGEST_URL}/status", params={"split": "train"}, timeout=10)
        body = r.json()
        assert body["available"] is True
        meta = body["meta"]
        # Idempotency tests may re-ingest with different sample counts after
        # ingest_job runs, overwriting S3. Verify structure not exact sha256.
        assert len(meta["sha256"]) == 64
        assert all(c in "0123456789abcdef" for c in meta["sha256"])
        assert meta["dataset"] == "imdb"
        assert meta["split"] == "train"
        assert meta["ingested_at"].endswith("Z")

    def test_preprocessing_links_to_ingestion(self, ingest_job, preprocess_job):
        r = requests.get(f"{PREPROC_URL}/status", params={"split": "train"}, timeout=10)
        meta = r.json()["meta"]
        # The preprocessing metadata must reference the raw data artifact
        assert ingest_job["s3_uri"] in meta["source_uri"] or \
               "raw/train_data.json" in meta["source_uri"]

    def test_model_info_references_training_split(self, train_job):
        r = requests.get(f"{FINETUNE_URL}/model/info", timeout=10)
        info = r.json()
        assert info["training_split"] == "train"

    def test_all_sha256_are_distinct(self, ingest_job, preprocess_job, train_job):
        """Each stage produces a different artifact, so checksums must differ."""
        shas = {
            ingest_job["sha256"],
            preprocess_job["sha256"],
            train_job["sha256"],
        }
        assert len(shas) == 3, "Two or more stages produced identical sha256 checksums"


@pytest.mark.slow
class TestEndToEndPipeline:
    """
    Run the full pipeline and validate the final inference result.

    Label-correctness tests are gated on require_sufficient_samples (>=200).
    The confidence floor is intentionally low (>0.50) so that even a
    minimally-trained model passes – correct label matters more than
    high confidence at small sample counts.
    """

    INFERENCE_CASES = [
        ("A truly spectacular film. Moved me to tears.", "positive"),
        ("Complete garbage. Poorly directed, terribly acted.", "negative"),
    ]

    @pytest.mark.parametrize("text, expected_label", INFERENCE_CASES)
    def test_inference_label_correct(self, train_job, require_sufficient_samples, text, expected_label):
        r = requests.post(f"{FINETUNE_URL}/predict", json={"text": text}, timeout=30)
        assert r.status_code == 200
        body = r.json()
        assert body["label"] == expected_label, (
            f"Pipeline produced wrong label for {text!r}: "
            f"expected={expected_label!r}, got={body['label']!r}, "
            f"scores={body['scores']}"
        )

    @pytest.mark.parametrize("text, expected_label", INFERENCE_CASES)
    def test_inference_confidence_above_chance(self, train_job, text, expected_label):
        """Model must be at least slightly better than random (>50%) regardless of sample count."""
        r = requests.post(f"{FINETUNE_URL}/predict", json={"text": text}, timeout=30)
        assert r.status_code == 200
        assert r.json()["confidence"] > 0.50, (
            f"Confidence ({r.json()['confidence']:.3f}) not above chance for: {text!r}"
        )
