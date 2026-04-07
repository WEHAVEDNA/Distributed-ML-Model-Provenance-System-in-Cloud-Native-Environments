import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo.py"
_DEMO_SPEC = importlib.util.spec_from_file_location("demo_under_test", _DEMO_PATH)
demo = importlib.util.module_from_spec(_DEMO_SPEC)
sys.modules[_DEMO_SPEC.name] = demo
_DEMO_SPEC.loader.exec_module(demo)


@pytest.fixture(autouse=True)
def ensure_services_ready():
    """Override the session-level health gate for demo unit tests."""
    return


def test_request_job_cancel_posts_cancel_endpoint(monkeypatch):
    calls = []
    warnings = []

    def fake_post_json(url, payload=None, timeout=10):
        calls.append((url, payload, timeout))
        return {"status": "cancel_requested"}

    monkeypatch.setattr(demo, "post_json", fake_post_json)
    monkeypatch.setattr(demo, "warn", warnings.append)

    demo.request_job_cancel(demo.FINETUNE_URL, "job123", "Fine-Tuning")

    assert calls == [(f"{demo.FINETUNE_URL}/jobs/job123/cancel", None, 10)]
    assert warnings == [
        "Fine-Tuning interruption requested. Remote job job123 status=cancel_requested"
    ]


def test_request_job_cancel_warns_when_cancel_fails(monkeypatch):
    warnings = []

    def fake_post_json(url, payload=None, timeout=10):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(demo, "post_json", fake_post_json)
    monkeypatch.setattr(demo, "warn", warnings.append)

    demo.request_job_cancel(demo.FINETUNE_URL, "job123", "Fine-Tuning")

    assert warnings == [
        "Could not request remote cancellation for Fine-Tuning: service unavailable"
    ]


def test_wait_for_job_exits_when_remote_job_is_cancelled(monkeypatch):
    monkeypatch.setattr(
        demo,
        "get_json",
        lambda url, timeout=10: {"status": "cancelled"},
    )
    monkeypatch.setattr(demo.time, "sleep", lambda _: None)

    with pytest.raises(SystemExit, match="Fine-Tuning cancelled"):
        demo.wait_for_job(demo.FINETUNE_URL, "job123", "Fine-Tuning", 30)


@pytest.mark.parametrize(
    ("base_url", "label"),
    (
        (demo.INGEST_URL, "Ingestion"),
        (demo.PREPROC_URL, "Preprocessing"),
        (demo.FINETUNE_URL, "Fine-Tuning"),
    ),
)
def test_wait_for_job_with_cancel_requests_remote_cancel(monkeypatch, base_url, label):
    calls = []

    def fake_post_json(url, payload=None, timeout=10):
        calls.append((url, payload, timeout))
        return {"status": "cancel_requested"}

    def fake_wait_for_job(base_url, job_id, label, timeout_s):
        raise KeyboardInterrupt

    monkeypatch.setattr(demo, "post_json", fake_post_json)
    monkeypatch.setattr(demo, "wait_for_job", fake_wait_for_job)
    monkeypatch.setattr(demo, "warn", lambda _: None)

    with pytest.raises(SystemExit, match="Remote cancellation requested"):
        demo.wait_for_job_with_cancel(base_url, "job123", label, 30)

    assert calls == [(f"{base_url}/jobs/job123/cancel", None, 10)]


def test_run_training_requests_remote_cancel_on_keyboard_interrupt(monkeypatch):
    calls = []
    wait_calls = []

    def fake_post_json(url, payload=None, timeout=10):
        calls.append((url, payload, timeout))
        return {"job_id": "job123", "device": "cpu", "epochs": 1}

    def fake_wait_for_job_with_cancel(base_url, job_id, label, timeout_s):
        wait_calls.append((base_url, job_id, label, timeout_s))
        return {"status": "completed"}

    monkeypatch.setattr(demo, "post_json", fake_post_json)
    monkeypatch.setattr(demo, "wait_for_job_with_cancel", fake_wait_for_job_with_cancel)
    monkeypatch.setattr(demo, "h2", lambda _: None)
    monkeypatch.setattr(demo, "warn", lambda _: None)
    monkeypatch.setattr(demo, "info", lambda _: None)
    monkeypatch.setattr(demo, "print_stage_result", lambda _: None)

    result = demo.run_training("train", "demo-500", 1)

    assert calls == [(f"{demo.FINETUNE_URL}/train?split=train&pipeline_id=demo-500&epochs=1", None, 10)]
    assert wait_calls == [(demo.FINETUNE_URL, "job123", "Fine-Tuning", demo.STAGE_TIMEOUTS["training"])]
    assert result == {"status": "completed"}


def test_parse_args_resolves_stage_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo.py"])
    args = demo.parse_args()
    assert args.stage == "pipeline"

    monkeypatch.setattr(sys, "argv", ["demo.py", "--train"])
    args = demo.parse_args()
    assert args.stage == "full"

    monkeypatch.setattr(sys, "argv", ["demo.py", "--stage", "ingest"])
    args = demo.parse_args()
    assert args.stage == "ingest"


def test_prove_provenance_reports_success(monkeypatch):
    ok_calls = []
    info_calls = []
    warn_calls = []

    def fake_get_json(url, timeout=10):
        if "/lineage?" in url:
            return {
                "chain": [
                    {
                        "stage": "data-ingestion",
                        "tracking_id": "track:data-ingestion",
                        "manifest_id": "urn:c2pa:test-manifest",
                        "has_manifest": True,
                        "artifact_uri": "s3://ml-provenance/pipelines/demo-500/raw/train_data.json",
                    }
                ],
                "chain_complete": True,
            }
        if "/pipeline/status?" in url:
            return {
                "stages": {
                    "data-ingestion": {"done": True, "artifact_count": 1},
                    "preprocessing": {"done": True, "artifact_count": 1},
                    "fine-tuning": {"done": True, "artifact_count": 1},
                }
            }
        if "/registry?" in url:
            return {
                "s3://ml-provenance/pipelines/demo-500/raw/train_data.json": {
                    "manifest_id": "urn:c2pa:test-manifest"
                }
            }
        if ":8001/provenance?" in url:
            return {"manifest_id": "urn:c2pa:ingest"}
        if ":8002/provenance?" in url:
            return {"manifest_id": "urn:c2pa:preprocess"}
        if ":8003/provenance?" in url:
            return {"manifest_id": "urn:c2pa:model"}
        if "/export/" in url:
            return {"root_id": "urn:c2pa:test-manifest", "nodes": {}, "edges": []}
        if "/verify/" in url:
            return {"valid": True}
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(demo, "get_json", fake_get_json)
    monkeypatch.setattr(demo, "h2", lambda _: None)
    monkeypatch.setattr(demo, "ok", ok_calls.append)
    monkeypatch.setattr(demo, "info", info_calls.append)
    monkeypatch.setattr(demo, "warn", warn_calls.append)

    demo.prove_provenance("demo-500")

    assert not warn_calls
    assert "Registry contains 1 artifact(s) for pipeline demo-500" in ok_calls
    assert "Every lineage entry has a sidecar tracking ID" in ok_calls
    assert "1/1 lineage entries have exportable manifest IDs" in ok_calls
    assert "Pipeline status marks these stages complete: data-ingestion, preprocessing, fine-tuning" in ok_calls
    assert "data-ingestion reports manifest_id=urn:c2pa:ingest" in ok_calls
    assert "preprocessing reports manifest_id=urn:c2pa:preprocess" in ok_calls
    assert "fine-tuning reports manifest_id=urn:c2pa:model" in ok_calls
    assert "Export succeeded for manifest urn:c2pa:test-manifest" in ok_calls
    assert "Verify returned valid=True (valid)" in ok_calls
    assert "Proof complete: full provenance chain was collected and linked" in ok_calls


def test_main_routes_requested_stage(monkeypatch):
    calls = []
    parsed_args = type(
        "Args",
        (),
        {
            "samples": 200,
            "split": "train",
            "pipeline_id": "demo-200",
            "train": False,
            "stage": "preprocess",
            "train_epochs": 1,
            "predict_text": None,
            "health_timeout": 10,
        },
    )()

    monkeypatch.setattr(demo, "parse_args", lambda: parsed_args)
    monkeypatch.setattr(demo, "h1", lambda _: None)
    monkeypatch.setattr(demo, "info", lambda _: None)
    monkeypatch.setattr(demo, "warn", lambda msg: calls.append(("warn", msg)))
    monkeypatch.setattr(demo, "wait_for_services", lambda timeout: calls.append(("health", timeout)))
    monkeypatch.setattr(
        demo,
        "ensure_raw_data_available",
        lambda split, pipeline_id, samples: calls.append(("ensure-raw", (split, pipeline_id, samples))),
    )
    monkeypatch.setattr(
        demo,
        "ensure_preprocessed_data_available",
        lambda split, pipeline_id, samples: calls.append(("ensure-preprocessed", (split, pipeline_id, samples))),
    )
    monkeypatch.setattr(demo, "run_ingestion", lambda *args: calls.append(("ingest", args)))
    monkeypatch.setattr(demo, "run_preprocessing", lambda *args: calls.append(("preprocess", args)))
    monkeypatch.setattr(demo, "run_training", lambda *args: calls.append(("train", args)))
    monkeypatch.setattr(demo, "run_inference_smoke", lambda *args: calls.append(("predict", args)))
    monkeypatch.setattr(demo, "show_lineage", lambda pipeline_id: calls.append(("lineage", pipeline_id)))
    monkeypatch.setattr(demo, "show_pipeline_status", lambda pipeline_id: calls.append(("status", pipeline_id)))
    monkeypatch.setattr(demo, "prove_provenance", lambda pipeline_id: calls.append(("proof", pipeline_id)))
    monkeypatch.setattr(demo, "print_summary", lambda pipeline_id: calls.append(("summary", pipeline_id)))

    demo.main()

    assert calls == [
        ("health", 10),
        ("warn", "Preprocess-only mode expects raw data for this pipeline and split to already exist."),
        ("ensure-raw", ("train", "demo-200", 200)),
        ("preprocess", ("train", "demo-200")),
        ("lineage", "demo-200"),
        ("status", "demo-200"),
        ("proof", "demo-200"),
        ("summary", "demo-200"),
    ]


def test_ensure_raw_data_available_exits_before_backend(monkeypatch):
    def fake_get_json(url, timeout=10):
        assert url == f"{demo.INGEST_URL}/status?split=train&pipeline_id=stage-demo"
        return {"available": False, "pipeline_id": "stage-demo"}

    monkeypatch.setattr(demo, "get_json", fake_get_json)

    with pytest.raises(SystemExit) as exc:
        demo.ensure_raw_data_available("train", "stage-demo", 200)

    message = str(exc.value)
    assert "preprocess-only mode requires existing raw data" in message
    assert "python demo.py --stage ingest --samples 200 --split train --pipeline-id stage-demo" in message
    assert "python demo.py --stage pipeline --samples 200 --split train --pipeline-id stage-demo" in message


def test_ensure_preprocessed_data_available_exits_before_backend(monkeypatch):
    def fake_get_json(url, timeout=10):
        assert url == f"{demo.PREPROC_URL}/status?split=train&pipeline_id=stage-demo"
        return {"available": False, "pipeline_id": "stage-demo"}

    monkeypatch.setattr(demo, "get_json", fake_get_json)

    with pytest.raises(SystemExit) as exc:
        demo.ensure_preprocessed_data_available("train", "stage-demo", 100)

    message = str(exc.value)
    assert "train-only mode requires existing preprocessed data" in message
    assert "python demo.py --stage pipeline --samples 100 --split train --pipeline-id stage-demo" in message
    assert "python demo.py --samples 100 --train --split train --pipeline-id stage-demo" in message


def test_main_train_stage_exits_before_backend_when_preprocessed_data_missing(monkeypatch):
    parsed_args = type(
        "Args",
        (),
        {
            "samples": 100,
            "split": "train",
            "pipeline_id": "stage-demo",
            "train": False,
            "stage": "train",
            "train_epochs": 1,
            "predict_text": "Loved it.",
            "health_timeout": 10,
        },
    )()
    calls = []

    monkeypatch.setattr(demo, "parse_args", lambda: parsed_args)
    monkeypatch.setattr(demo, "h1", lambda *_: None)
    monkeypatch.setattr(demo, "info", lambda *_: None)
    monkeypatch.setattr(demo, "warn", lambda msg: calls.append(("warn", msg)))
    monkeypatch.setattr(demo, "wait_for_services", lambda timeout: calls.append(("health", timeout)))
    monkeypatch.setattr(
        demo,
        "ensure_preprocessed_data_available",
        lambda split, pipeline_id, samples: (_ for _ in ()).throw(
            SystemExit("missing preprocessed data")
        ),
    )
    monkeypatch.setattr(demo, "run_training", lambda *args: calls.append(("train", args)))
    monkeypatch.setattr(demo, "run_inference_smoke", lambda *args: calls.append(("predict", args)))
    monkeypatch.setattr(demo, "show_lineage", lambda pipeline_id: calls.append(("lineage", pipeline_id)))
    monkeypatch.setattr(demo, "show_pipeline_status", lambda pipeline_id: calls.append(("status", pipeline_id)))
    monkeypatch.setattr(demo, "prove_provenance", lambda pipeline_id: calls.append(("proof", pipeline_id)))
    monkeypatch.setattr(demo, "print_summary", lambda pipeline_id: calls.append(("summary", pipeline_id)))

    with pytest.raises(SystemExit, match="missing preprocessed data"):
        demo.main()

    assert calls == [
        ("health", 10),
        ("warn", "Train-only mode expects preprocessed data for this pipeline and split to already exist."),
    ]
