import pytest
import requests

import conftest


@pytest.fixture(autouse=True)
def ensure_services_ready():
    """Override the session-level health gate for harness unit tests."""
    return


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self):
        return self._payload


def test_request_with_transient_retry_retries_safe_get(monkeypatch):
    attempts = {"count": 0}

    def fake_request(session, method, url, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.exceptions.ConnectionError("transient")
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(conftest, "_ORIGINAL_SESSION_REQUEST", fake_request)
    monkeypatch.setattr(conftest.time, "sleep", lambda _: None)

    resp = conftest._request_with_transient_retry(requests.Session(), "GET", "http://example.test")

    assert attempts["count"] == 3
    assert resp.json() == {"ok": True}


def test_request_with_transient_retry_does_not_retry_post(monkeypatch):
    attempts = {"count": 0}

    def fake_request(session, method, url, **kwargs):
        attempts["count"] += 1
        raise requests.exceptions.ConnectionError("transient")

    monkeypatch.setattr(conftest, "_ORIGINAL_SESSION_REQUEST", fake_request)
    monkeypatch.setattr(conftest.time, "sleep", lambda _: None)

    with pytest.raises(requests.exceptions.ConnectionError):
        conftest._request_with_transient_retry(requests.Session(), "POST", "http://example.test")

    assert attempts["count"] == 1


def test_request_with_transient_retry_honors_elapsed_timeout(monkeypatch):
    attempts = {"count": 0}
    now = {"value": 1000.0}

    def fake_request(session, method, url, **kwargs):
        attempts["count"] += 1
        now["value"] += 20
        raise requests.exceptions.ConnectionError("transient")

    monkeypatch.setattr(conftest, "_ORIGINAL_SESSION_REQUEST", fake_request)
    monkeypatch.setattr(conftest.time, "sleep", lambda _: None)
    monkeypatch.setattr(conftest.time, "time", lambda: now["value"])

    with pytest.raises(requests.exceptions.ConnectionError):
        conftest._request_with_transient_retry(requests.Session(), "GET", "http://example.test")

    assert attempts["count"] == 2


def test_wait_for_job_recovers_from_disconnect(monkeypatch):
    responses = iter(
        [
            requests.exceptions.ConnectionError("transient"),
            _FakeResponse({"status": "running"}),
            _FakeResponse({"status": "completed", "job_id": "abc123"}),
        ]
    )

    def fake_get(url, timeout):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(conftest.requests, "get", fake_get)
    monkeypatch.setattr(conftest.time, "sleep", lambda _: None)

    body = conftest.wait_for_job("http://service.test", "abc123", timeout=30)

    assert body["status"] == "completed"
    assert body["job_id"] == "abc123"


def test_ensure_services_ready_waits_until_all_healthy(monkeypatch):
    calls = {"count": 0}
    now = {"value": 0.0}

    def fake_get(url, timeout):
        calls["count"] += 1
        if calls["count"] <= len(conftest.HEALTH_ENDPOINTS):
            raise requests.exceptions.ConnectionError("starting")
        return _FakeResponse({"status": "ok"})

    monkeypatch.setattr(conftest.requests, "get", fake_get)
    monkeypatch.setattr(conftest.time, "sleep", lambda seconds: now.__setitem__("value", now["value"] + seconds))
    monkeypatch.setattr(conftest.time, "time", lambda: now["value"])

    conftest.ensure_services_ready.__wrapped__()

    assert calls["count"] > len(conftest.HEALTH_ENDPOINTS)


def test_ensure_services_ready_exits_after_timeout(monkeypatch):
    now = {"value": 0.0}

    def fake_get(url, timeout):
        raise requests.exceptions.ConnectionError("still down")

    monkeypatch.setattr(conftest.requests, "get", fake_get)
    monkeypatch.setattr(conftest.time, "sleep", lambda seconds: now.__setitem__("value", now["value"] + seconds))
    monkeypatch.setattr(conftest.time, "time", lambda: now["value"])
    monkeypatch.setattr(conftest, "HEALTHCHECK_TIMEOUT", 3)

    with pytest.raises(pytest.exit.Exception):
        conftest.ensure_services_ready.__wrapped__()
