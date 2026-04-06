"""
Tests for the Fine-Tuning & Inference Service (Stage 3).

Covers:
  - Training job lifecycle and metrics
  - Model provenance metadata in S3
  - /predict correctness on obvious positive/negative reviews
  - /predict response schema
  - /predict before training (503 expected)
  - Confidence scores sum to ~1.0
  - Short, long, and edge-case inputs to /predict

  pytest tests/test_fine_tuning.py -v
  pytest tests/test_fine_tuning.py -v -m "not slow"  # skip training
"""

import pytest
import requests
from conftest import FINETUNE_URL, PIPELINE_ID, model_uri, wait_for_job, TRAIN_TIMEOUT


# ── Training ──────────────────────────────────────────────────────────────────
class TestTrainingJobLifecycle:
    def test_train_returns_job_id(self, preprocess_job):
        r = requests.post(
            f"{FINETUNE_URL}/train",
            params={"split": "train", "pipeline_id": PIPELINE_ID},
            timeout=10,
        )
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running")
        assert "device" in body
        assert body["pipeline_id"] == PIPELINE_ID

    @pytest.mark.slow
    def test_train_job_completes(self, train_job):
        assert train_job["status"] == "completed"

    @pytest.mark.slow
    def test_train_job_has_epoch_losses(self, train_job):
        losses = train_job["epoch_losses"]
        assert isinstance(losses, list)
        assert len(losses) >= 1
        assert all(isinstance(l, float) for l in losses)
        assert all(l > 0 for l in losses)

    @pytest.mark.slow
    def test_train_loss_decreases(self, train_job):
        """Loss should decrease (or stay flat) across epochs with ≥2 epochs."""
        losses = train_job["epoch_losses"]
        if len(losses) >= 2:
            # Allow a small tolerance – loss can fluctuate slightly
            assert losses[-1] <= losses[0] * 1.1, (
                f"Loss did not decrease: {losses}"
            )

    @pytest.mark.slow
    def test_train_job_has_sha256(self, train_job):
        sha = train_job["sha256"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    @pytest.mark.slow
    def test_train_job_has_s3_uri(self, train_job):
        uri = train_job["s3_uri"]
        assert uri == model_uri()

    @pytest.mark.timeout(10)
    def test_unknown_job_id_returns_404(self):
        r = requests.get(f"{FINETUNE_URL}/jobs/doesnotexist", timeout=5)
        assert r.status_code == 404

    @pytest.mark.timeout(10)
    def test_unknown_job_id_cancel_returns_404(self):
        r = requests.post(f"{FINETUNE_URL}/jobs/doesnotexist/cancel", timeout=5)
        assert r.status_code == 404

    @pytest.mark.slow
    def test_train_job_can_be_cancelled(self, preprocess_job):
        r = requests.post(
            f"{FINETUNE_URL}/train",
            params={"split": "train", "pipeline_id": PIPELINE_ID, "epochs": 1},
            timeout=10,
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        cancel = requests.post(f"{FINETUNE_URL}/jobs/{job_id}/cancel", timeout=10)
        assert cancel.status_code == 200
        assert cancel.json()["status"] in {"cancel_requested", "cancelled", "completed"}

        final = wait_for_job(FINETUNE_URL, job_id, TRAIN_TIMEOUT)
        assert final["status"] in {"cancelled", "completed"}


# ── Model metadata ────────────────────────────────────────────────────────────
class TestModelInfo:
    @pytest.mark.slow
    def test_model_info_available_after_training(self, train_job):
        r = requests.get(f"{FINETUNE_URL}/model/info", params={"pipeline_id": PIPELINE_ID}, timeout=10)
        assert r.status_code == 200
        info = r.json()
        assert info.get("available") is not False    # not the "no model" response
        assert info["pipeline_id"] == PIPELINE_ID
        assert info["bert_model"] == "bert-base-uncased"
        assert info["num_labels"] == 2
        assert "label_map" in info
        assert "epoch_losses" in info
        assert "sha256" in info
        assert "trained_at" in info
        assert info["trained_at"].endswith("Z")

    @pytest.mark.slow
    def test_model_info_label_map(self, train_job):
        info = requests.get(f"{FINETUNE_URL}/model/info", params={"pipeline_id": PIPELINE_ID}, timeout=10).json()
        lm = info["label_map"]
        # label_map keys are strings in JSON ("0", "1")
        assert set(lm.values()) == {"negative", "positive"}


# ── Inference ─────────────────────────────────────────────────────────────────
class TestPredictSchema:
    @pytest.mark.slow
    def test_predict_response_schema(self, train_job):
        r = requests.post(
            f"{FINETUNE_URL}/predict",
            json={"text": "Great movie!", "pipeline_id": PIPELINE_ID},
            timeout=30,
        )
        assert r.status_code == 200
        body = r.json()
        assert "label" in body
        assert "confidence" in body
        assert "scores" in body
        assert body["pipeline_id"] == PIPELINE_ID
        assert body["label"] in ("positive", "negative")
        assert 0.0 <= body["confidence"] <= 1.0
        assert set(body["scores"].keys()) == {"positive", "negative"}

    @pytest.mark.slow
    def test_predict_scores_sum_to_one(self, train_job):
        r = requests.post(
            f"{FINETUNE_URL}/predict",
            json={"text": "Mediocre at best.", "pipeline_id": PIPELINE_ID},
            timeout=30,
        )
        scores = r.json()["scores"]
        total = scores["positive"] + scores["negative"]
        assert abs(total - 1.0) < 1e-3

    @pytest.mark.slow
    def test_predict_confidence_matches_max_score(self, train_job):
        r = requests.post(
            f"{FINETUNE_URL}/predict",
            json={"text": "I loved this film!", "pipeline_id": PIPELINE_ID},
            timeout=30,
        )
        body = r.json()
        expected_conf = max(body["scores"].values())
        assert abs(body["confidence"] - expected_conf) < 1e-3


class TestPredictSentiment:
    """
    Sentiment tests on strongly-worded IMDB-style reviews.

    Requires --samples >= 200.  BERT cannot generalise on fewer samples and
    will appear biased toward whichever class dominated the tiny training set.
    Tests are automatically skipped when the sample count is too low.
    """

    POSITIVE_REVIEWS = [
        "This was an absolutely brilliant and masterful piece of cinema.",
        "One of the best films I have ever seen. Extraordinary performances.",
        "A wonderful, heartwarming story. I laughed and cried throughout.",
    ]

    NEGATIVE_REVIEWS = [
        "Terrible waste of time. The worst movie I have ever watched.",
        "Awful acting, boring plot, completely unwatchable garbage.",
        "I fell asleep. Dreadful film with no redeeming qualities whatsoever.",
    ]

    @pytest.mark.slow
    @pytest.mark.parametrize("text", POSITIVE_REVIEWS)
    def test_positive_review_classified_positive(self, train_job, require_sufficient_samples, text):
        r = requests.post(f"{FINETUNE_URL}/predict", json={"text": text, "pipeline_id": PIPELINE_ID}, timeout=30)
        assert r.status_code == 200
        assert r.json()["label"] == "positive", (
            f"Expected positive for: {text!r}\nGot: {r.json()}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("text", NEGATIVE_REVIEWS)
    def test_negative_review_classified_negative(self, train_job, require_sufficient_samples, text):
        r = requests.post(f"{FINETUNE_URL}/predict", json={"text": text, "pipeline_id": PIPELINE_ID}, timeout=30)
        assert r.status_code == 200
        assert r.json()["label"] == "negative", (
            f"Expected negative for: {text!r}\nGot: {r.json()}"
        )


class TestPredictEdgeCases:
    @pytest.mark.slow
    def test_predict_single_word(self, train_job):
        r = requests.post(f"{FINETUNE_URL}/predict", json={"text": "great", "pipeline_id": PIPELINE_ID}, timeout=30)
        assert r.status_code == 200
        assert r.json()["label"] in ("positive", "negative")

    @pytest.mark.slow
    def test_predict_long_text_truncated(self, train_job):
        """Text longer than max_length (128 tokens) must still return a result."""
        long_text = "This film was great. " * 200   # ~4000 chars
        r = requests.post(f"{FINETUNE_URL}/predict", json={"text": long_text, "pipeline_id": PIPELINE_ID}, timeout=30)
        assert r.status_code == 200
        assert r.json()["label"] in ("positive", "negative")

    @pytest.mark.slow
    def test_predict_special_characters(self, train_job):
        r = requests.post(
            f"{FINETUNE_URL}/predict",
            json={"text": "10/10 — würde es wieder sehen! ★★★★★", "pipeline_id": PIPELINE_ID},
            timeout=30,
        )
        assert r.status_code == 200

    @pytest.mark.timeout(15)
    def test_predict_missing_text_field(self):
        r = requests.post(f"{FINETUNE_URL}/predict", json={}, timeout=10)
        assert r.status_code == 422   # FastAPI validation error

    @pytest.mark.timeout(35)
    def test_predict_before_training_returns_503_or_200(self):
        """
        If no model has been trained yet, /predict returns 503.
        If a model is already in S3 (from a prior run), it returns 200.
        Acceptable either way – we just verify it doesn't crash with 500.
        """
        r = requests.post(f"{FINETUNE_URL}/predict", json={"text": "test", "pipeline_id": PIPELINE_ID}, timeout=30)
        assert r.status_code in (200, 503)
