"""Live translation is optional: no key → calm 503, results cached, no network in tests."""

import httpx
import pytest
from fastapi import HTTPException

from backend.app.routers import translate as translate_router
from backend.app.services import translation_service as svc


@pytest.fixture(autouse=True)
def _clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_no_key_is_503(monkeypatch):
    monkeypatch.setattr(svc, "get_secret", lambda name: None)
    with pytest.raises(HTTPException) as excinfo:
        translate_router.translate(translate_router.TranslateRequest(texts=["Approved"], target="hi"))
    assert excinfo.value.status_code == 503


def test_translates_and_caches(monkeypatch):
    monkeypatch.setattr(svc, "get_secret", lambda name: "test-key")
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return httpx.Response(200, json={"translated_text": "स्वीकृत"})

    monkeypatch.setattr(httpx, "post", fake_post)
    first = translate_router.translate(translate_router.TranslateRequest(texts=["Approved"], target="hi"))
    second = translate_router.translate(translate_router.TranslateRequest(texts=["Approved"], target="hi"))

    assert first == second == {"translations": ["स्वीकृत"]}
    assert len(calls) == 1 and calls[0]["target_language_code"] == "hi-IN"


def test_one_failed_text_does_not_sink_the_batch(monkeypatch):
    monkeypatch.setattr(svc, "get_secret", lambda name: "test-key")
    monkeypatch.setattr(svc.time, "sleep", lambda seconds: None)

    def fake_post(url, headers, json, timeout):
        if json["input"] == "bad":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"translated_text": "ठीक"})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = translate_router.translate(translate_router.TranslateRequest(texts=["good", "bad"], target="hi"))
    assert result == {"translations": ["ठीक", None]}


def test_unknown_target_is_rejected():
    with pytest.raises(HTTPException) as excinfo:
        translate_router.translate(translate_router.TranslateRequest(texts=["x"], target="fr"))
    assert excinfo.value.status_code == 422
