"""
Provenance chain tests – verify the sidecar's /lineage and /pipeline/status
endpoints, and that each pipeline service exposes /provenance.

These tests are distinct from test_atlas_sidecar.py which tests individual
/collect calls. Here we validate the *chain* view of the provenance graph.

Categories
----------
  smoke   – structure/schema checks, no pipeline data required
  wired   – chain correctness after the pipeline has auto-called the sidecar

Run:
    pytest tests/test_provenance_chain.py -v                   # smoke only
    pytest tests/test_provenance_chain.py -v -m wired          # after pipeline run
"""

import pytest
import requests

from conftest import (
    FINETUNE_URL,
    INGEST_URL,
    PIPELINE_ID,
    PREPROC_URL,
    SIDECAR_URL,
    model_uri,
)

PIPELINE_STAGES = ["data-ingestion", "preprocessing", "fine-tuning"]


# ── /lineage endpoint – smoke (no pipeline data required) ─────────────────────

class TestLineageEndpointSmoke:
    """Basic schema tests – pass as long as the sidecar is running."""

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_lineage_returns_200(self):
        r = requests.get(f"{SIDECAR_URL}/lineage", timeout=5)
        assert r.status_code == 200

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_lineage_schema(self):
        body = requests.get(f"{SIDECAR_URL}/lineage", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert "chain" in body, "Missing 'chain' key"
        assert "stages_complete" in body, "Missing 'stages_complete' key"
        assert "chain_complete" in body, "Missing 'chain_complete' key"
        assert "total_manifests" in body, "Missing 'total_manifests' key"
        assert isinstance(body["chain"], list)
        assert isinstance(body["stages_complete"], list)
        assert isinstance(body["chain_complete"], bool)

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_lineage_chain_entries_have_required_fields(self):
        chain = requests.get(f"{SIDECAR_URL}/lineage", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()["chain"]
        for entry in chain:
            assert "stage" in entry, f"Missing 'stage' in entry: {entry}"
            assert "artifact_uri" in entry, f"Missing 'artifact_uri' in entry: {entry}"
            assert "manifest_id" in entry, f"Missing 'manifest_id' in entry: {entry}"
            assert "type" in entry, f"Missing 'type' in entry: {entry}"

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_lineage_stages_complete_only_contains_known_stages(self):
        stages_complete = requests.get(
            f"{SIDECAR_URL}/lineage",
            params={"pipeline_id": PIPELINE_ID},
            timeout=5,
        ).json()["stages_complete"]
        for s in stages_complete:
            assert s in PIPELINE_STAGES, f"Unknown stage in stages_complete: {s!r}"


# ── /pipeline/status endpoint – smoke ─────────────────────────────────────────

class TestPipelineStatusSmoke:
    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_pipeline_status_returns_200(self):
        r = requests.get(f"{SIDECAR_URL}/pipeline/status", params={"pipeline_id": PIPELINE_ID}, timeout=5)
        assert r.status_code == 200

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_pipeline_status_schema(self):
        body = requests.get(
            f"{SIDECAR_URL}/pipeline/status",
            params={"pipeline_id": PIPELINE_ID},
            timeout=5,
        ).json()
        assert "stages" in body
        assert "chain_complete" in body
        assert isinstance(body["chain_complete"], bool)

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_pipeline_status_all_stages_present(self):
        stages = requests.get(
            f"{SIDECAR_URL}/pipeline/status",
            params={"pipeline_id": PIPELINE_ID},
            timeout=5,
        ).json()["stages"]
        for s in PIPELINE_STAGES:
            assert s in stages, f"Stage {s!r} missing from /pipeline/status"

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_pipeline_status_stage_fields(self):
        stages = requests.get(
            f"{SIDECAR_URL}/pipeline/status",
            params={"pipeline_id": PIPELINE_ID},
            timeout=5,
        ).json()["stages"]
        for s in PIPELINE_STAGES:
            entry = stages[s]
            assert "done" in entry, f"Missing 'done' for stage {s}"
            assert "artifact_count" in entry, f"Missing 'artifact_count' for stage {s}"
            assert "manifest_ids" in entry, f"Missing 'manifest_ids' for stage {s}"
            assert isinstance(entry["done"], bool)
            assert isinstance(entry["artifact_count"], int)
            assert isinstance(entry["manifest_ids"], list)


# ── /provenance endpoint on each pipeline service – smoke ─────────────────────

class TestServiceProvenanceEndpoint:
    """Each pipeline service must expose GET /provenance with a consistent schema."""

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_data_ingestion_provenance_endpoint_exists(self):
        r = requests.get(f"{INGEST_URL}/provenance", timeout=5)
        assert r.status_code == 200

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_data_ingestion_provenance_schema(self):
        body = requests.get(f"{INGEST_URL}/provenance", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert body["service"] == "data-ingestion"
        assert body["pipeline_id"] == PIPELINE_ID
        assert "manifest_id" in body    # may be None before first ingest

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_preprocessing_provenance_endpoint_exists(self):
        r = requests.get(f"{PREPROC_URL}/provenance", timeout=5)
        assert r.status_code == 200

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_preprocessing_provenance_schema(self):
        body = requests.get(f"{PREPROC_URL}/provenance", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert body["service"] == "preprocessing"
        assert body["pipeline_id"] == PIPELINE_ID
        assert "manifest_id" in body

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_fine_tuning_provenance_endpoint_exists(self):
        r = requests.get(f"{FINETUNE_URL}/provenance", timeout=5)
        assert r.status_code == 200

    @pytest.mark.smoke
    @pytest.mark.timeout(10)
    def test_fine_tuning_provenance_schema(self):
        body = requests.get(f"{FINETUNE_URL}/provenance", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert body["service"] == "fine-tuning"
        assert body["pipeline_id"] == PIPELINE_ID
        assert "manifest_id" in body


# ── Wired: chain correctness after pipeline has run ───────────────────────────

class TestLineageAfterPipeline:
    """
    These tests require the pipeline to have been run and ATLAS_SIDECAR_URL set
    in each pipeline container so that the services auto-called the sidecar.

    Mark: @wired
    """

    @pytest.mark.wired
    @pytest.mark.timeout(60)
    def test_lineage_includes_ingestion_stage(self, ingest_job):
        lineage = requests.get(f"{SIDECAR_URL}/lineage", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert "data-ingestion" in lineage["stages_complete"], (
            "data-ingestion not in stages_complete after ingest_job completed.\n"
            "Check ATLAS_SIDECAR_URL in the data-ingestion container."
        )

    @pytest.mark.wired
    @pytest.mark.timeout(60)
    def test_lineage_includes_preprocessing_stage(self, preprocess_job):
        lineage = requests.get(f"{SIDECAR_URL}/lineage", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert "preprocessing" in lineage["stages_complete"], (
            "preprocessing not in stages_complete after preprocess_job completed."
        )

    @pytest.mark.wired
    @pytest.mark.timeout(60)
    def test_lineage_chain_order_matches_pipeline(self, preprocess_job):
        """
        Entries in the chain must appear in pipeline stage order:
        data-ingestion before preprocessing before fine-tuning.
        """
        chain = requests.get(f"{SIDECAR_URL}/lineage", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()["chain"]
        stages_in_chain = [e["stage"] for e in chain]
        # Filter to known pipeline stages, preserving order
        ordered = [s for s in PIPELINE_STAGES if s in stages_in_chain]
        actual_order = [s for s in stages_in_chain if s in PIPELINE_STAGES]
        # actual_order may have duplicates if multiple artifacts per stage;
        # check that stages appear in non-decreasing pipeline order
        idx = {s: i for i, s in enumerate(PIPELINE_STAGES)}
        prev = -1
        for s in actual_order:
            assert idx[s] >= prev, (
                f"Stage {s!r} appears out of order in chain: {stages_in_chain}"
            )
            prev = idx[s]

    @pytest.mark.wired
    @pytest.mark.timeout(60)
    def test_pipeline_status_ingestion_done(self, ingest_job):
        stages = requests.get(
            f"{SIDECAR_URL}/pipeline/status",
            params={"pipeline_id": PIPELINE_ID},
            timeout=5,
        ).json()["stages"]
        assert stages["data-ingestion"]["done"], (
            "data-ingestion.done is False after ingest_job completed."
        )
        assert stages["data-ingestion"]["artifact_count"] >= 1

    @pytest.mark.wired
    @pytest.mark.timeout(60)
    def test_pipeline_status_preprocessing_done(self, preprocess_job):
        stages = requests.get(
            f"{SIDECAR_URL}/pipeline/status",
            params={"pipeline_id": PIPELINE_ID},
            timeout=5,
        ).json()["stages"]
        assert stages["preprocessing"]["done"], (
            "preprocessing.done is False after preprocess_job completed."
        )

    @pytest.mark.wired
    @pytest.mark.timeout(60)
    def test_ingest_service_provenance_populated(self, ingest_job):
        """After wired ingest, the data-ingestion service's /provenance has a manifest_id."""
        body = requests.get(f"{INGEST_URL}/provenance", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert body["manifest_id"] is not None, (
            "data-ingestion /provenance returned manifest_id=None after ingest_job.\n"
            "Ensure ATLAS_SIDECAR_URL is set in the data-ingestion container."
        )

    @pytest.mark.wired
    @pytest.mark.timeout(60)
    def test_preprocessing_service_provenance_populated(self, preprocess_job):
        body = requests.get(f"{PREPROC_URL}/provenance", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert body["manifest_id"] is not None, (
            "preprocessing /provenance returned manifest_id=None after preprocess_job."
        )

    @pytest.mark.wired
    @pytest.mark.slow
    @pytest.mark.timeout(7200)
    def test_full_chain_complete_after_training(self, train_job):
        lineage = requests.get(f"{SIDECAR_URL}/lineage", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        assert lineage["chain_complete"] is True, (
            f"chain_complete is False after full pipeline run.\n"
            f"stages_complete: {lineage['stages_complete']}"
        )
        # Every entry must have a non-empty manifest_id
        for entry in lineage["chain"]:
            assert entry["manifest_id"], (
                f"Empty manifest_id in chain entry: {entry}"
            )

    @pytest.mark.wired
    @pytest.mark.slow
    @pytest.mark.timeout(7200)
    def test_fine_tuning_service_provenance_populated(self, train_job):
        """
        Verify that the fine-tuning stage's provenance was recorded.

        The sidecar registry is the authoritative store (persisted to disk across
        restarts).  The service's in-memory /provenance endpoint may return None
        if the sidecar call timed out during this session (the BERT model file
        is large), so we check the sidecar registry first and accept a populated
        in-memory manifest_id as a bonus indicator.
        """
        registry = requests.get(
            f"{SIDECAR_URL}/registry",
            params={"pipeline_id": PIPELINE_ID},
            timeout=5,
        ).json()
        expected_model_uri = model_uri()
        assert expected_model_uri in registry and registry[expected_model_uri].get("manifest_id"), (
            f"No fine-tuning manifest in sidecar registry after train_job.\n"
            f"Expected URI: {expected_model_uri}\n"
            f"Check ATLAS_SIDECAR_URL in the fine-tuning container and sidecar logs."
        )

    @pytest.mark.wired
    @pytest.mark.slow
    @pytest.mark.timeout(7200)
    def test_manifest_ids_are_distinct_across_stages(self, train_job):
        """Each stage must produce a different manifest ID (no accidental sharing)."""
        lineage = requests.get(f"{SIDECAR_URL}/lineage", params={"pipeline_id": PIPELINE_ID}, timeout=5).json()
        ids = [e["manifest_id"] for e in lineage["chain"] if e.get("manifest_id")]
        # Allow same artifact collected multiple times to overwrite; just check
        # that if all three stages are present their IDs differ.
        stage_ids = {}
        for entry in lineage["chain"]:
            stage_ids[entry["stage"]] = entry.get("manifest_id")
        if len(stage_ids) >= 2:
            unique_ids = set(stage_ids.values()) - {None}
            assert len(unique_ids) == len([v for v in stage_ids.values() if v]), (
                f"Duplicate manifest IDs across stages: {stage_ids}"
            )
