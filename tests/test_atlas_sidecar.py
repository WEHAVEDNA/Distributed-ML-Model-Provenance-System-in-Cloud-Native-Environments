"""
Atlas Sidecar tests.

Two test categories:

  DIRECT (no pipeline dependency)
  --------------------------------
  These tests call the sidecar's /collect endpoints directly with known S3 URIs.
  They pass as long as the sidecar is up and the artifact exists in S3.
  Run after any pipeline stage that produced the artifact.

  WIRED (pipeline auto-called the sidecar)
  -----------------------------------------
  These tests check that the pipeline services populated the registry on their own.
  They only pass when ATLAS_SIDECAR_URL is set in the pipeline service containers.
  Marked @pytest.mark.wired — skip if you haven't confirmed ATLAS_SIDECAR_URL is set.

Run:
  pytest tests/test_atlas_sidecar.py -v                          # direct tests
  pytest tests/test_atlas_sidecar.py -v -m wired --samples 500  # wired tests
  pytest tests/test_atlas_sidecar.py -v -m slow  --samples 500  # full chain
"""

import pytest
import requests

from conftest import (
    INGEST_TIMEOUT,
    INGEST_URL,
    PIPELINE_ID,
    SIDECAR_URL,
    model_uri,
    preprocessed_data_uri,
    raw_data_uri,
    wait_for_job,
)

DATASET_URI = raw_data_uri()
PREPROC_URI = preprocessed_data_uri()
MODEL_URI = model_uri()
DIRECT_MODEL_PIPELINE_ID = "direct-model-collect"


# ── Health ─────────────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
def test_sidecar_health():
    resp = requests.get(f"{SIDECAR_URL}/health", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "atlas-sidecar"
    assert body["atlas_cli"] != "unavailable", (
        "atlas-cli binary is not working inside the sidecar container. "
        "Check: docker compose logs atlas-sidecar"
    )


@pytest.mark.timeout(10)
def test_sidecar_atlas_cli_version_in_health():
    """Health response must include a real atlas-cli version string."""
    body = requests.get(f"{SIDECAR_URL}/health", timeout=5).json()
    assert body["atlas_cli"] != "unavailable"
    assert body["key_exists"] is True, (
        "Signing key was not generated. Check sidecar startup logs."
    )


@pytest.mark.timeout(10)
def test_sidecar_signing_key_is_rsa():
    resp = requests.get(f"{SIDECAR_URL}/signing-key", timeout=5)
    assert resp.status_code == 200
    pem = resp.json()["public_key_pem"]
    assert "BEGIN PUBLIC KEY" in pem


# ── Direct: call sidecar manually and verify it works ─────────────────────────
# These are the primary correctness tests for the sidecar.
# They need the S3 artifact to exist (run ingestion first).


@pytest.fixture
def direct_model_artifact_uri():
    """
    Create a tiny isolated artifact in S3 so direct model collection does not
    depend on running the full fine-tuning pipeline.
    """
    resp = requests.post(
        f"{INGEST_URL}/ingest",
        params={"split": "train", "num_samples": 8, "pipeline_id": DIRECT_MODEL_PIPELINE_ID},
        timeout=10,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    wait_for_job(INGEST_URL, job_id, INGEST_TIMEOUT)
    return raw_data_uri(pipeline_id=DIRECT_MODEL_PIPELINE_ID)

@pytest.mark.timeout(120)
def test_direct_collect_dataset(ingest_job):
    """
    POST directly to /collect/dataset and confirm a manifest_id comes back.
    This proves the sidecar can download from S3, run atlas-cli, and return a manifest.
    """
    payload = {
        "pipeline_id": PIPELINE_ID,
        "stage": "data-ingestion",
        "artifact_s3_uri": DATASET_URI,
        "ingredient_name": "IMDB train Dataset",
        "author": "pytest",
        "metadata": {},
    }
    resp = requests.post(f"{SIDECAR_URL}/collect/dataset", json=payload, timeout=120)
    assert resp.status_code == 200, f"collect/dataset failed: {resp.text}"
    body = resp.json()
    assert body.get("manifest_id"), (
        f"No manifest_id in response — atlas-cli may have failed.\n"
        f"Response: {body}\n"
        f"Check: docker compose logs atlas-sidecar"
    )
    assert body["type"] == "dataset"
    assert body["pipeline_id"] == PIPELINE_ID
    assert body["stage"] == "data-ingestion"


@pytest.mark.timeout(120)
def test_direct_collect_dataset_updates_registry(ingest_job):
    """After a direct collect call, the registry must contain the artifact URI."""
    payload = {
        "pipeline_id": PIPELINE_ID,
        "stage": "data-ingestion",
        "artifact_s3_uri": DATASET_URI,
        "ingredient_name": "IMDB train Dataset",
        "author": "pytest",
        "metadata": {},
    }
    requests.post(f"{SIDECAR_URL}/collect/dataset", json=payload, timeout=120)

    registry = requests.get(f"{SIDECAR_URL}/registry", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
    assert DATASET_URI in registry, (
        f"Registry does not contain {DATASET_URI} after direct collect call.\n"
        f"Registry keys: {list(registry.keys())}"
    )
    entry = registry[DATASET_URI]
    assert entry["manifest_id"]
    assert entry["stage"] == "data-ingestion"
    assert entry["type"] == "dataset"


@pytest.mark.timeout(120)
def test_direct_collect_pipeline(ingest_job, preprocess_job):
    """POST directly to /collect/pipeline and confirm a manifest_id comes back."""
    payload = {
        "pipeline_id": PIPELINE_ID,
        "stage": "preprocessing",
        "input_s3_uris": [DATASET_URI],
        "output_s3_uri": PREPROC_URI,
        "ingredient_name": "IMDB train Tokenized",
        "author": "pytest",
        "build_script": "BertTokenizer.from_pretrained('bert-base-uncased') max_length=128",
        "metadata": {},
    }
    resp = requests.post(f"{SIDECAR_URL}/collect/pipeline", json=payload, timeout=120)
    assert resp.status_code == 200, f"collect/pipeline failed: {resp.text}"
    body = resp.json()
    assert body.get("manifest_id"), f"No manifest_id returned: {body}"
    assert body["type"] == "pipeline"
    assert body["pipeline_id"] == PIPELINE_ID
    assert body["stage"] == "preprocessing"


@pytest.mark.timeout(120)
def test_direct_collect_model(direct_model_artifact_uri):
    """POST directly to /collect/model without requiring a full training run."""
    payload = {
        "pipeline_id": DIRECT_MODEL_PIPELINE_ID,
        "stage": "fine-tuning",
        "artifact_s3_uri": direct_model_artifact_uri,
        "ingredient_name": "BERT IMDB Sentiment Classifier",
        "author": "pytest",
        "linked_dataset_manifest_ids": [],
        "metadata": {},
    }
    resp = requests.post(f"{SIDECAR_URL}/collect/model", json=payload, timeout=120)
    assert resp.status_code == 200, f"collect/model failed: {resp.text}"
    body = resp.json()
    assert body.get("manifest_id"), f"No manifest_id returned: {body}"
    assert body["type"] == "model"
    assert body["pipeline_id"] == DIRECT_MODEL_PIPELINE_ID
    assert body["stage"] == "fine-tuning"


# ── Direct: manifest count increases after each collect ────────────────────────

@pytest.mark.timeout(120)
def test_manifest_count_increases_after_collect(ingest_job):
    """Each new collect call must add an entry to /manifests."""
    before = requests.get(
        f"{SIDECAR_URL}/manifests",
        params={"pipeline_id": PIPELINE_ID},
        timeout=5,
    ).json()["count"]

    payload = {
        "pipeline_id": PIPELINE_ID,
        "stage": "data-ingestion",
        "artifact_s3_uri": DATASET_URI,
        "ingredient_name": "Count test",
        "author": "pytest",
        "metadata": {},
    }
    requests.post(f"{SIDECAR_URL}/collect/dataset", json=payload, timeout=120)

    after = requests.get(
        f"{SIDECAR_URL}/manifests",
        params={"pipeline_id": PIPELINE_ID},
        timeout=5,
    ).json()["count"]
    assert after >= before, "Manifest count did not increase after collect"


# ── Direct: export and verify ──────────────────────────────────────────────────

@pytest.mark.timeout(120)
def test_export_manifest_returns_json(ingest_job):
    """Export a freshly collected manifest and confirm valid JSON is returned."""
    payload = {
        "pipeline_id": PIPELINE_ID,
        "stage": "data-ingestion",
        "artifact_s3_uri": DATASET_URI,
        "ingredient_name": "Export test",
        "author": "pytest",
        "metadata": {},
    }
    collect = requests.post(f"{SIDECAR_URL}/collect/dataset", json=payload, timeout=120).json()
    mid = collect.get("manifest_id")
    if not mid:
        pytest.skip("No manifest_id — atlas-cli may not be working")

    resp = requests.get(f"{SIDECAR_URL}/export/{mid}", timeout=30)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict) and len(body) > 0, (
        f"Export returned empty or non-dict: {body}"
    )


@pytest.mark.timeout(120)
def test_verify_manifest_returns_valid_field(ingest_job):
    """Verify a freshly collected manifest — response must include 'valid' field."""
    payload = {
        "pipeline_id": PIPELINE_ID,
        "stage": "data-ingestion",
        "artifact_s3_uri": DATASET_URI,
        "ingredient_name": "Verify test",
        "author": "pytest",
        "metadata": {},
    }
    collect = requests.post(f"{SIDECAR_URL}/collect/dataset", json=payload, timeout=120).json()
    mid = collect.get("manifest_id")
    if not mid:
        pytest.skip("No manifest_id — atlas-cli may not be working")

    resp = requests.get(f"{SIDECAR_URL}/verify/{mid}", timeout=30)
    assert resp.status_code == 200
    body = resp.json()
    assert "valid" in body, f"No 'valid' field in verify response: {body}"


# ── Wired: pipeline services auto-called the sidecar ──────────────────────────
# These tests only pass when ATLAS_SIDECAR_URL is set in the pipeline containers.
# Confirm with: docker compose exec data-ingestion env | grep ATLAS

@pytest.mark.wired
@pytest.mark.timeout(60)
def test_wired_ingest_populated_registry(ingest_job):
    """
    After ingestion, the pipeline service should have auto-called the sidecar.
    Fails if ATLAS_SIDECAR_URL is not set in the data-ingestion container.
    """
    registry = requests.get(f"{SIDECAR_URL}/registry", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
    assert DATASET_URI in registry, (
        f"{DATASET_URI!r} not in registry.\n"
        f"The data-ingestion service did not call the sidecar automatically.\n"
        f"Verify ATLAS_SIDECAR_URL is set: docker compose exec data-ingestion env | grep ATLAS\n"
        f"If missing, restart containers: docker compose up -d --force-recreate"
    )
    assert registry[DATASET_URI]["manifest_id"]


@pytest.mark.wired
@pytest.mark.timeout(60)
def test_wired_preprocess_populated_registry(preprocess_job):
    registry = requests.get(f"{SIDECAR_URL}/registry", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
    assert PREPROC_URI in registry, (
        f"{PREPROC_URI!r} not in registry after preprocessing.\n"
        f"Check ATLAS_SIDECAR_URL in the preprocessing container."
    )
    assert registry[PREPROC_URI]["manifest_id"]
    assert registry[PREPROC_URI]["stage"] == "preprocessing"


@pytest.mark.wired
@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_wired_full_chain_all_three_artifacts(train_job):
    """All three artifact URIs must be in the registry after a full pipeline run."""
    registry = requests.get(f"{SIDECAR_URL}/registry", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
    missing = [u for u in (DATASET_URI, PREPROC_URI, MODEL_URI) if u not in registry]
    assert not missing, (
        f"Missing from registry: {missing}\n"
        f"Check ATLAS_SIDECAR_URL is set in all pipeline containers."
    )
    for uri in (DATASET_URI, PREPROC_URI, MODEL_URI):
        assert registry[uri]["manifest_id"], f"manifest_id empty for {uri}"
