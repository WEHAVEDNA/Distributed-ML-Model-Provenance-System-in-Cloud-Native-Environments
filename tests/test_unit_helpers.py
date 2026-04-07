"""
Unit tests for helper functions and runtime metadata paths.

These tests do not require live services or containers; they load modules with
lightweight dependency stubs and validate the pure helper behavior directly.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]

demo = None


class DummyFastAPI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get(self, *args, **kwargs):
        return lambda func: func

    def post(self, *args, **kwargs):
        return lambda func: func

    def on_event(self, *args, **kwargs):
        return lambda func: func


class DummyHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class DummyBackgroundTasks:
    def add_task(self, *args, **kwargs):
        return None


def _fastapi_stub() -> types.ModuleType:
    module = types.ModuleType("fastapi")
    module.FastAPI = DummyFastAPI
    module.HTTPException = DummyHTTPException
    module.BackgroundTasks = DummyBackgroundTasks
    return module


def _pydantic_stub() -> types.ModuleType:
    module = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    module.BaseModel = BaseModel
    module.Field = Field
    return module


def _boto_stub() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *args, **kwargs: object()

    botocore = types.ModuleType("botocore")
    botocore_exceptions = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        pass

    botocore_exceptions.ClientError = ClientError
    botocore.exceptions = botocore_exceptions
    return boto3, botocore, botocore_exceptions


def _datasets_stub() -> types.ModuleType:
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *args, **kwargs: []
    return module


def _transformers_stub() -> types.ModuleType:
    module = types.ModuleType("transformers")

    class BertTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class BertForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    def get_linear_schedule_with_warmup(*args, **kwargs):
        return object()

    module.BertTokenizer = BertTokenizer
    module.BertForSequenceClassification = BertForSequenceClassification
    module.get_linear_schedule_with_warmup = get_linear_schedule_with_warmup
    return module


def _numpy_stub() -> types.ModuleType:
    module = types.ModuleType("numpy")
    module.random = types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
    return module


def _torch_stub() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType, types.ModuleType]:
    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.manual_seed = lambda *_args, **_kwargs: None
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *_args, **_kwargs: None,
    )
    torch.tensor = lambda data, **_kwargs: data
    torch.long = "long"
    torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda *_args, **_kwargs: None))

    torch_optim = types.ModuleType("torch.optim")

    class AdamW:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    torch_optim.AdamW = AdamW

    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    class DataLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    torch_utils_data.Dataset = Dataset
    torch_utils_data.DataLoader = DataLoader
    torch_utils.data = torch_utils_data
    return torch, torch_optim, torch_utils, torch_utils_data


def load_module(monkeypatch, relative_path: str, module_name: str, env: dict[str, str] | None = None):
    env = env or {}
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setitem(sys.modules, "fastapi", _fastapi_stub())
    monkeypatch.setitem(sys.modules, "pydantic", _pydantic_stub())

    boto3, botocore, botocore_exceptions = _boto_stub()
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", botocore_exceptions)

    monkeypatch.setitem(sys.modules, "datasets", _datasets_stub())
    monkeypatch.setitem(sys.modules, "transformers", _transformers_stub())
    monkeypatch.setitem(sys.modules, "numpy", _numpy_stub())

    torch, torch_optim, torch_utils, torch_utils_data = _torch_stub()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.optim", torch_optim)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"{module_name}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def load_demo_module(monkeypatch):
    return load_module(monkeypatch, "demo.py", "demo")


def test_data_ingestion_helper_paths(monkeypatch):
    module = load_module(monkeypatch, "services/data-ingestion/app.py", "data_ingestion")

    assert module._normalize_pipeline_id(" Demo 200 ") == "demo-200"
    assert module._pipeline_prefix("demo-200") == "pipelines/demo-200"
    assert module._raw_data_key("demo-200", "train") == "pipelines/demo-200/raw/train_data.json"
    assert module._raw_meta_key("demo-200", "train") == "pipelines/demo-200/raw/train_meta.json"
    assert module._s3_uri("pipelines/demo-200/raw/train_data.json") == (
        "s3://ml-provenance/pipelines/demo-200/raw/train_data.json"
    )


def test_data_ingestion_health_includes_runtime_metadata(monkeypatch):
    module = load_module(
        monkeypatch,
        "services/data-ingestion/app.py",
        "data_ingestion",
        env={
            "DEPLOYMENT_MODE": "kubernetes",
            "POD_NAME": "ingest-pod",
            "POD_NAMESPACE": "ml-pipeline",
            "NODE_NAME": "desktop-worker",
        },
    )

    body = module.health()
    assert body["deployment_mode"] == "kubernetes"
    assert body["pod_name"] == "ingest-pod"
    assert body["pod_namespace"] == "ml-pipeline"
    assert body["node_name"] == "desktop-worker"


def test_data_ingestion_rejects_empty_pipeline_id(monkeypatch):
    module = load_module(monkeypatch, "services/data-ingestion/app.py", "data_ingestion")
    with pytest.raises(ValueError):
        module._normalize_pipeline_id("!!!")


def test_preprocessing_helper_paths(monkeypatch):
    module = load_module(monkeypatch, "services/preprocessing/app.py", "preprocessing")

    assert module._normalize_pipeline_id(" Demo 200 ") == "demo-200"
    assert module._preprocessed_data_key("demo-200", "train") == (
        "pipelines/demo-200/preprocessed/train_tokenized.json"
    )
    assert module._preprocessed_meta_key("demo-200", "train") == (
        "pipelines/demo-200/preprocessed/train_meta.json"
    )


def test_preprocessing_health_includes_runtime_metadata(monkeypatch):
    module = load_module(
        monkeypatch,
        "services/preprocessing/app.py",
        "preprocessing",
        env={
            "DEPLOYMENT_MODE": "kubernetes",
            "POD_NAME": "preprocess-pod",
            "POD_NAMESPACE": "ml-pipeline",
            "NODE_NAME": "desktop-control-plane",
        },
    )

    body = module.health()
    assert body["deployment_mode"] == "kubernetes"
    assert body["pod_name"] == "preprocess-pod"
    assert body["node_name"] == "desktop-control-plane"


def test_fine_tuning_path_helpers_and_lexical_bias(monkeypatch):
    module = load_module(monkeypatch, "services/fine-tuning/app.py", "fine_tuning")

    assert module._model_dir("demo-200") == "pipelines/demo-200/models/classifier"
    assert module._model_weights_key("demo-200") == "pipelines/demo-200/models/classifier/model.pt"
    assert module._model_meta_key("demo-200") == "pipelines/demo-200/models/classifier/meta.json"
    assert module._lexical_sentiment_bias("masterful wonderful best film ever") == "positive"
    assert module._lexical_sentiment_bias("awful boring terrible waste of time") == "negative"
    assert module._lexical_sentiment_bias("an ordinary movie night") is None


def test_fine_tuning_health_includes_runtime_metadata(monkeypatch):
    module = load_module(
        monkeypatch,
        "services/fine-tuning/app.py",
        "fine_tuning",
        env={
            "DEPLOYMENT_MODE": "kubernetes",
            "POD_NAME": "trainer-pod",
            "POD_NAMESPACE": "ml-pipeline",
            "NODE_NAME": "desktop-worker",
        },
    )

    body = module.health()
    assert body["deployment_mode"] == "kubernetes"
    assert body["device"] == "cpu"
    assert body["loaded_pipelines"] == []
    assert body["pod_name"] == "trainer-pod"


def test_atlas_sidecar_manifest_helpers(monkeypatch):
    module = load_module(monkeypatch, "services/atlas-sidecar/app.py", "atlas_sidecar")

    urn = "urn:c2pa:12345678-1234-1234-1234-1234567890ab"
    uuid_value = "12345678-1234-1234-1234-1234567890ab"
    assert module._is_resolvable_manifest_id(urn) is True
    assert module._is_resolvable_manifest_id(uuid_value) is True
    assert module._is_resolvable_manifest_id("0") is False
    assert module._parse_manifest_id(f"Manifest stored successfully with ID: {urn}") == urn


def test_atlas_sidecar_resolve_manifest_id_only_accepts_new_index_values(monkeypatch):
    module = load_module(monkeypatch, "services/atlas-sidecar/app.py", "atlas_sidecar")

    old_id = "urn:c2pa:11111111-1111-1111-1111-111111111111"
    new_id = "urn:c2pa:22222222-2222-2222-2222-222222222222"

    monkeypatch.setattr(module, "_manifest_ids_from_index", lambda: [old_id, new_id])
    assert module._resolve_manifest_id_from_command("", [old_id]) == new_id

    monkeypatch.setattr(module, "_manifest_ids_from_index", lambda: [old_id])
    assert module._resolve_manifest_id_from_command("", [old_id]) is None


def test_atlas_sidecar_registry_keys_and_stage_names(monkeypatch):
    module = load_module(monkeypatch, "services/atlas-sidecar/app.py", "atlas_sidecar")

    pipeline_key = module._registry_key_for_record(
        artifact_uri="s3://ml-provenance/pipelines/demo-200/preprocessed/train_tokenized.json",
        kind="pipeline",
        pipeline_id="demo-200",
        stage="preprocessing",
        stage_order=20,
    )
    dataset_key = module._registry_key_for_record(
        artifact_uri="s3://ml-provenance/pipelines/demo-200/preprocessed/train_tokenized.json",
        kind="dataset",
        pipeline_id="demo-200",
        stage="preprocessing",
        stage_order=20,
    )

    assert pipeline_key.startswith("pipeline::demo-200::00020::preprocessing::")
    assert dataset_key == "s3://ml-provenance/pipelines/demo-200/preprocessed/train_tokenized.json"

    stages = module._pipeline_stage_names(
        [
            ("a", {"stage": "data-ingestion"}),
            ("b", {"stage": "preprocessing"}),
            ("c", {"stage": "preprocessing"}),
            ("d", {"stage": "fine-tuning"}),
        ]
    )
    assert stages == ["data-ingestion", "preprocessing", "fine-tuning"]


def test_atlas_sidecar_health_includes_runtime_metadata(monkeypatch):
    module = load_module(
        monkeypatch,
        "services/atlas-sidecar/app.py",
        "atlas_sidecar",
        env={
            "DEPLOYMENT_MODE": "kubernetes",
            "POD_NAME": "atlas-pod",
            "POD_NAMESPACE": "ml-pipeline",
            "NODE_NAME": "desktop-worker",
        },
    )

    body = module.health()
    assert body["deployment_mode"] == "kubernetes"
    assert body["pod_name"] == "atlas-pod"
    assert body["node_name"] == "desktop-worker"


def test_atlas_sidecar_collect_pipeline_creates_pipeline_step_manifest(monkeypatch):
    module = load_module(monkeypatch, "services/atlas-sidecar/app.py", "atlas_sidecar")
    module.MANIFESTS_DIR = ROOT
    module.KEY_PATH = ROOT / "unit-test-signing-key.pem"

    pipeline_manifest_id = "urn:c2pa:33333333-3333-3333-3333-333333333333"
    atlas_calls = []
    registered = {}

    def fake_download(_uri: str):
        return ROOT / f"unit-test-{uuid.uuid4().hex}.json"

    def fake_run_atlas(*args: str, timeout: int = 120):
        atlas_calls.append(args)
        if args[:2] == ("pipeline", "generate-provenance"):
            return ("Manifest stored successfully with ID: 0", "", 0)
        if args[:2] == ("dataset", "create"):
            return (f"Manifest stored successfully with ID: {pipeline_manifest_id}", "", 0)
        if args[:2] == ("manifest", "link"):
            return ("linked", "", 0)
        raise AssertionError(f"Unexpected atlas call: {args}")

    def fake_register(
        s3_uri,
        manifest_id,
        pipeline_id,
        stage,
        stage_order,
        kind,
        ingredient_name,
        metadata,
        input_s3_uris=None,
        linked_manifest_ids=None,
    ):
        registered.update(
            {
                "s3_uri": s3_uri,
                "manifest_id": manifest_id,
                "pipeline_id": pipeline_id,
                "stage": stage,
                "stage_order": stage_order,
                "kind": kind,
                "ingredient_name": ingredient_name,
                "metadata": metadata,
                "input_s3_uris": input_s3_uris,
                "linked_manifest_ids": linked_manifest_ids,
            }
        )
        return {
            "tracking_id": "pipeline::demo-500::00020::preprocessing::s3://ml-provenance/pipelines/demo-500/preprocessed/train_tokenized.json",
            "manifest_id": manifest_id,
        }

    monkeypatch.setattr(module, "_download_s3_to_temp", fake_download)
    monkeypatch.setattr(module, "_run_atlas", fake_run_atlas)
    monkeypatch.setattr(module, "_register", fake_register)
    monkeypatch.setattr(module, "_manifest_ids_from_index", lambda: [])

    response = module.collect_pipeline(
        module.PipelineCollectRequest(
            pipeline_id="demo-500",
            stage="preprocessing",
            stage_order=20,
            input_s3_uris=["s3://ml-provenance/pipelines/demo-500/raw/train_data.json"],
            linked_manifest_ids=["urn:c2pa:11111111-1111-1111-1111-111111111111"],
            output_s3_uri="s3://ml-provenance/pipelines/demo-500/preprocessed/train_tokenized.json",
            ingredient_name="demo-500 train tokenized dataset",
            author="preprocessing-service",
            build_script="tokenizer",
            metadata={"stage": "preprocessing"},
        )
    )

    assert response["manifest_id"] == pipeline_manifest_id
    assert response["type"] == "pipeline"
    assert response["linked_manifest_ids"] == ["urn:c2pa:11111111-1111-1111-1111-111111111111"]
    assert registered["manifest_id"] == pipeline_manifest_id
    assert registered["kind"] == "pipeline"
    assert registered["linked_manifest_ids"] == ["urn:c2pa:11111111-1111-1111-1111-111111111111"]
    assert ("pipeline", "generate-provenance") == atlas_calls[0][:2]
    assert any(call[:2] == ("dataset", "create") for call in atlas_calls)
    assert any(call[:2] == ("manifest", "link") for call in atlas_calls)


def test_wait_for_services_reports_kubernetes_backend(monkeypatch):
    demo = load_demo_module(monkeypatch)
    health_by_url = {
        service.health_url: {
            "status": "ok",
            "service": service.name,
            "deployment_mode": "kubernetes",
            "pod_name": f"{service.name}-pod",
            "node_name": "desktop-worker",
            "atlas_cli": "atlas-cli 0.2.0",
            "key_exists": True,
        }
        for service in demo.SERVICES
    }

    ok_calls = []
    info_calls = []
    monkeypatch.setattr(demo, "get_json", lambda url, timeout=5: health_by_url[url])
    monkeypatch.setattr(demo, "ok", ok_calls.append)
    monkeypatch.setattr(demo, "info", info_calls.append)

    service_health = demo.wait_for_services(1)

    assert service_health["atlas-sidecar"]["deployment_mode"] == "kubernetes"
    assert any("Runtime backend: kubernetes" in msg for msg in ok_calls)
    assert any("traffic path: localhost -> kubectl port-forward -> ClusterIP services" in msg for msg in info_calls)


def test_print_stage_result_warns_without_manifest(monkeypatch):
    demo = load_demo_module(monkeypatch)
    info_calls = []
    warn_calls = []
    monkeypatch.setattr(demo, "info", info_calls.append)
    monkeypatch.setattr(demo, "ok", lambda *_args: None)
    monkeypatch.setattr(demo, "warn", warn_calls.append)

    demo.print_stage_result(
        {
            "pipeline_id": "demo-200",
            "num_samples": 200,
            "sha256": "a" * 64,
            "s3_uri": "s3://ml-provenance/pipelines/demo-200/raw/train_data.json",
        }
    )

    assert any("pipeline_id" in msg for msg in info_calls)
    assert "No Atlas manifest recorded for this stage" in warn_calls


def test_show_lineage_renders_tracked_only_pipeline_entry(monkeypatch):
    demo = load_demo_module(monkeypatch)
    info_calls = []
    warn_calls = []
    monkeypatch.setattr(
        demo,
        "get_json",
        lambda *_args, **_kwargs: {
            "chain": [
                {
                    "stage": "preprocessing",
                    "type": "pipeline",
                    "manifest_id": None,
                    "artifact_uri": "s3://ml-provenance/pipelines/demo-200/preprocessed/train_tokenized.json",
                    "input_artifact_uris": [
                        "s3://ml-provenance/pipelines/demo-200/raw/train_data.json"
                    ],
                }
            ],
            "stages_complete": ["preprocessing"],
            "chain_complete": False,
        },
    )
    monkeypatch.setattr(demo, "h2", lambda *_args: None)
    monkeypatch.setattr(demo, "info", info_calls.append)
    monkeypatch.setattr(demo, "warn", warn_calls.append)

    demo.show_lineage("demo-200")

    assert any("tracked-only" in msg for msg in info_calls + warn_calls) is False
    assert any("inputs:" in msg for msg in info_calls)
    assert "Chain incomplete for this pipeline" in warn_calls
