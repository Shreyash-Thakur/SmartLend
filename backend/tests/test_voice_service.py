"""Tests for the voice module.

Two invariants are under test:
  1. With no API key configured, nothing raises — not at import, not at any
     entry point, not at any endpoint. Endpoints answer 503, never 500.
  2. No test ever makes a real HTTP request. `voice_service._http_post` is the
     single network chokepoint and is monkeypatched (to an exploding stub, or a
     recorded payload) in every test that could reach it.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.config import ELEVENLABS_API_KEY, SARVAM_API_KEY
from backend.app.routers.voice import router as voice_router
from backend.app.services import voice_service


@pytest.fixture(autouse=True)
def no_keys_and_no_network(monkeypatch):
    """Default world for every test: no credentials, no network reachable."""
    monkeypatch.delenv(ELEVENLABS_API_KEY, raising=False)
    monkeypatch.delenv(SARVAM_API_KEY, raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    monkeypatch.delenv("SARVAM_STT_MODEL", raising=False)

    def _boom(*args, **kwargs):  # pragma: no cover - firing this IS the failure
        raise AssertionError("a test attempted a real network call")

    monkeypatch.setattr(voice_service, "_http_post", _boom)
    # Restore the module-level providers in case a test swapped them.
    tts, stt = voice_service.get_tts_provider(), voice_service.get_stt_provider()
    yield
    voice_service.set_providers(tts=tts, stt=stt)


@pytest.fixture()
def client():
    """Voice router mounted on a bare app: no DB, no model load, just HTTP."""
    app = FastAPI()
    app.include_router(voice_router, prefix="/api")
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Import safety
# --------------------------------------------------------------------------- #


def test_module_imports_without_any_key(monkeypatch):
    monkeypatch.delenv(ELEVENLABS_API_KEY, raising=False)
    monkeypatch.delenv(SARVAM_API_KEY, raising=False)
    module = importlib.reload(importlib.import_module("backend.app.services.voice_service"))
    assert module is not None


def test_main_app_imports_and_registers_voice_routes():
    from backend.app.main import app

    paths = {route.path for route in app.routes}
    assert {"/api/voice/status", "/api/voice/synthesize", "/api/voice/transcribe"} <= paths


# --------------------------------------------------------------------------- #
# voice_available()
# --------------------------------------------------------------------------- #


def test_voice_available_reports_unavailable_with_reasons():
    status = voice_service.voice_available()
    assert status["available"] is False
    assert status["tts"]["configured"] is False
    assert status["stt"]["configured"] is False
    assert ELEVENLABS_API_KEY in status["tts"]["reason"]
    assert SARVAM_API_KEY in status["stt"]["reason"]
    assert status["reasons"], "unavailability must come with at least one reason"


def test_voice_available_flips_when_a_key_appears(monkeypatch):
    monkeypatch.setenv(ELEVENLABS_API_KEY, "test-key")
    status = voice_service.voice_available()
    assert status["tts"]["configured"] is True
    assert status["tts"]["reason"] is None
    assert status["available"] is True  # one side configured is enough


def test_voice_available_survives_a_broken_provider():
    class Exploding(voice_service.TextToSpeech):
        name = "exploding"

        def is_configured(self):
            raise RuntimeError("provider is broken")

        def unavailable_reason(self):
            raise RuntimeError("provider is broken")

        def synthesize(self, text, language=None):
            raise RuntimeError("provider is broken")

    voice_service.set_providers(tts=Exploding())
    status = voice_service.voice_available()
    assert status["tts"]["configured"] is False
    assert "raised" in status["tts"]["reason"]


# --------------------------------------------------------------------------- #
# Entry-point degradation
# --------------------------------------------------------------------------- #


def test_synthesize_decision_returns_none_without_key():
    assert voice_service.synthesize_decision("approved", "Your income is stable.") is None


def test_synthesize_decision_returns_none_when_provider_fails(monkeypatch):
    monkeypatch.setenv(ELEVENLABS_API_KEY, "test-key")

    def _fail(*args, **kwargs):
        raise voice_service.VoiceProviderError("provider returned HTTP 401")

    monkeypatch.setattr(voice_service, "_http_post", _fail)
    assert voice_service.synthesize_decision("rejected", "Too much existing debt.") is None


def test_synthesize_decision_returns_audio_bytes_when_configured(monkeypatch):
    monkeypatch.setenv(ELEVENLABS_API_KEY, "test-key")
    captured = {}

    def _fake_post(url, *, headers, body):
        captured["url"] = url
        captured["headers"] = headers
        return b"ID3-fake-mp3"

    monkeypatch.setattr(voice_service, "_http_post", _fake_post)
    audio = voice_service.synthesize_decision("approved", "Stable income.")
    assert audio == b"ID3-fake-mp3"
    assert captured["headers"]["xi-api-key"] == "test-key"
    assert "text-to-speech" in captured["url"]


def test_decision_script_includes_decision_and_explanation():
    script = voice_service.build_decision_script("approved", "Your income is stable.")
    assert "approved" in script
    assert "Your income is stable." in script


def test_transcribe_returns_structured_unavailable_without_key():
    result = voice_service.transcribe(b"fake-audio")
    assert result["available"] is False
    assert result["text"] is None
    assert result["words"] == []
    assert SARVAM_API_KEY in result["reason"]


def test_transcribe_handles_empty_audio(monkeypatch):
    monkeypatch.setenv(SARVAM_API_KEY, "test-key")
    result = voice_service.transcribe(b"")
    assert result["available"] is False
    assert "no audio" in result["reason"]


def test_transcribe_returns_unavailable_when_provider_fails(monkeypatch):
    monkeypatch.setenv(SARVAM_API_KEY, "test-key")

    def _fail(*args, **kwargs):
        raise voice_service.VoiceProviderError("could not reach provider: timed out")

    monkeypatch.setattr(voice_service, "_http_post", _fail)
    result = voice_service.transcribe(b"fake-audio")
    assert result["available"] is False
    assert "could not reach provider" in result["reason"]


def test_transcribe_survives_unparseable_provider_response(monkeypatch):
    monkeypatch.setenv(SARVAM_API_KEY, "test-key")
    monkeypatch.setattr(voice_service, "_http_post", lambda *a, **k: b"<html>oops</html>")
    result = voice_service.transcribe(b"fake-audio")
    assert result["available"] is False
    assert "parse" in result["reason"]


def test_transcribe_preserves_per_word_confidence(monkeypatch):
    monkeypatch.setenv(SARVAM_API_KEY, "test-key")
    payload = (
        b'{"transcript": "mera naam", "language_code": "hi-IN",'
        b' "timestamps": {"words": ["mera", "naam"],'
        b' "start_time_seconds": [0.0, 0.5], "end_time_seconds": [0.4, 0.9],'
        b' "confidences": [0.9, 0.7]}}'
    )
    monkeypatch.setattr(voice_service, "_http_post", lambda *a, **k: payload)
    result = voice_service.transcribe(b"fake-audio", language="hi")
    assert result["available"] is True
    assert result["text"] == "mera naam"
    assert result["language"] == "hi-IN"
    assert [w["word"] for w in result["words"]] == ["mera", "naam"]
    assert [w["confidence"] for w in result["words"]] == [0.9, 0.7]
    assert result["confidence"] == pytest.approx(0.8)


def test_transcribe_emits_none_confidence_when_provider_gives_none(monkeypatch):
    """Providers without confidence must still yield the full word shape."""
    monkeypatch.setenv(SARVAM_API_KEY, "test-key")
    payload = b'{"transcript": "hello there", "language_code": "en-IN"}'
    monkeypatch.setattr(voice_service, "_http_post", lambda *a, **k: payload)
    result = voice_service.transcribe(b"fake-audio", language="en")
    assert [w["word"] for w in result["words"]] == ["hello", "there"]
    assert all(w["confidence"] is None for w in result["words"])
    # None means "unknown", never "certain".
    assert result["confidence"] is None


def test_transcribe_never_raises_on_a_misbehaving_provider():
    class Rogue(voice_service.SpeechToText):
        name = "rogue"

        def is_configured(self):
            return True

        def unavailable_reason(self):
            return None

        def transcribe(self, audio_bytes, language=None):
            raise ValueError("kaboom")

    voice_service.set_providers(stt=Rogue())
    result = voice_service.transcribe(b"fake-audio")
    assert result["available"] is False
    assert "kaboom" in result["reason"]


# --------------------------------------------------------------------------- #
# Endpoints — 503, never 500
# --------------------------------------------------------------------------- #


def test_status_endpoint_works_without_any_key(client):
    response = client.get("/api/voice/status")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["tts"]["reason"]
    assert body["stt"]["reason"]


def test_synthesize_endpoint_returns_503_without_key(client):
    response = client.post(
        "/api/voice/synthesize",
        json={"decision": "approved", "explanation": "Stable income.", "language": "en"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "VOICE_UNAVAILABLE"
    assert ELEVENLABS_API_KEY in detail["details"]


def test_transcribe_endpoint_returns_503_without_key(client):
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("clip.wav", b"fake-audio", "audio/wav")},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "VOICE_UNAVAILABLE"
    assert SARVAM_API_KEY in detail["details"]


def test_synthesize_endpoint_returns_503_when_provider_call_fails(client, monkeypatch):
    """Key present, provider down: still 503 with guidance, never 500."""
    monkeypatch.setenv(ELEVENLABS_API_KEY, "test-key")

    def _fail(*args, **kwargs):
        raise voice_service.VoiceProviderError("provider returned HTTP 429")

    monkeypatch.setattr(voice_service, "_http_post", _fail)
    response = client.post(
        "/api/voice/synthesize", json={"decision": "approved", "explanation": ""}
    )
    assert response.status_code == 503
    assert "quota" in response.json()["detail"]["details"]


def test_transcribe_endpoint_returns_503_when_provider_call_fails(client, monkeypatch):
    monkeypatch.setenv(SARVAM_API_KEY, "test-key")

    def _fail(*args, **kwargs):
        raise voice_service.VoiceProviderError("could not reach provider: timed out")

    monkeypatch.setattr(voice_service, "_http_post", _fail)
    response = client.post(
        "/api/voice/transcribe", files={"file": ("clip.wav", b"fake-audio", "audio/wav")}
    )
    assert response.status_code == 503


def test_synthesize_endpoint_returns_audio_when_configured(client, monkeypatch):
    monkeypatch.setenv(ELEVENLABS_API_KEY, "test-key")
    monkeypatch.setattr(voice_service, "_http_post", lambda *a, **k: b"ID3-fake-mp3")
    response = client.post(
        "/api/voice/synthesize", json={"decision": "approved", "explanation": "Stable income."}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"ID3-fake-mp3"


def test_transcribe_endpoint_rejects_empty_upload(client, monkeypatch):
    monkeypatch.setenv(SARVAM_API_KEY, "test-key")
    response = client.post(
        "/api/voice/transcribe", files={"file": ("clip.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400


def test_transcribe_endpoint_returns_words_when_configured(client, monkeypatch):
    monkeypatch.setenv(SARVAM_API_KEY, "test-key")
    payload = b'{"transcript": "namaste", "language_code": "hi-IN"}'
    monkeypatch.setattr(voice_service, "_http_post", lambda *a, **k: payload)
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("clip.wav", b"fake-audio", "audio/wav")},
        data={"language": "hi"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "namaste"
    assert body["words"][0]["word"] == "namaste"
    assert body["words"][0]["confidence"] is None
