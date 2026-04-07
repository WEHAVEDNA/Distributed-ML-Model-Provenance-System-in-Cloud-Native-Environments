"""
Unit tests for the sample src/app.py lambda handler.
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


def load_src_app(monkeypatch):
    boto3 = types.ModuleType("boto3")

    class DummyS3Client:
        def __init__(self):
            self.downloads = []
            self.uploads = []

        def download_file(self, bucket, key, path):
            self.downloads.append((bucket, key, path))

        def upload_file(self, path, bucket, key):
            self.uploads.append((path, bucket, key))

    boto3.client = lambda *_args, **_kwargs: DummyS3Client()

    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")

    class DummyOpenedImage:
        def __init__(self):
            self.size = (800, 600)
            self.thumbnail_calls = []
            self.saved_to = None

        def thumbnail(self, size):
            self.thumbnail_calls.append(size)

        def save(self, path):
            self.saved_to = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    pil_image.open = lambda *_args, **_kwargs: DummyOpenedImage()
    pil.Image = pil_image

    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image)

    path = ROOT / "src/app.py"
    spec = importlib.util.spec_from_file_location(f"src_app_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_resize_image_halves_image_dimensions(monkeypatch):
    module = load_src_app(monkeypatch)
    opened = []

    class DummyOpenedImage:
        size = (800, 600)

        def __init__(self):
            self.thumbnail_calls = []
            self.saved_to = None

        def thumbnail(self, size):
            self.thumbnail_calls.append(size)

        def save(self, path):
            self.saved_to = path

        def __enter__(self):
            opened.append(self)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(module.Image, "open", lambda *_args, **_kwargs: DummyOpenedImage())

    module.resize_image("input.jpg", "resized.jpg")

    assert opened[0].thumbnail_calls == [(400.0, 300.0)]
    assert opened[0].saved_to == "resized.jpg"


def test_lambda_handler_downloads_resizes_and_uploads(monkeypatch):
    module = load_src_app(monkeypatch)
    resize_calls = []
    monkeypatch.setenv("DESTINATION_BUCKETNAME", "dest-bucket")
    monkeypatch.setattr(module, "resize_image", lambda src, dst: resize_calls.append((src, dst)))

    event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "source-bucket"},
                    "object": {"key": "images/cat photo.jpg"},
                }
            }
        ]
    }

    module.lambda_handler(event, context=None)

    assert len(module.s3_client.downloads) == 1
    bucket, key, download_path = module.s3_client.downloads[0]
    assert bucket == "source-bucket"
    assert key == "images/cat photo.jpg"
    assert download_path.endswith("imagescat photo.jpg")

    assert len(resize_calls) == 1
    assert resize_calls[0][0] == download_path
    assert resize_calls[0][1].endswith("resized-imagescat photo.jpg")

    assert module.s3_client.uploads == [
        (resize_calls[0][1], "dest-bucket", "resized-images/cat photo.jpg")
    ]
