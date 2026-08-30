"""Voice I/O for SmartLend: decision readout (TTS) and spoken intake (STT).

Why this module exists
----------------------
Two separate jobs, deliberately kept behind two separate abstractions:

* ``TextToSpeech``  — read a loan decision + its explanation aloud, so an
  applicant who cannot comfortably read the on-screen explanation still gets it.
* ``SpeechToText``  — accept spoken application intake in Indian languages, so
  form-filling is not a literacy test.

Provider-agnostic by construction
---------------------------------
Callers never import a provider. They call :func:`synthesize_decision`,
:func:`transcribe` and :func:`voice_available`, which resolve a provider through
:func:`get_tts_provider` / :func:`get_stt_provider`. **That resolver pair is the
swap seam.** Swapping ElevenLabs for a locally-hosted open-source model means
writing one new class that implements the ABC and returning it from the
resolver — no caller, router, or test changes. See ``docs/VOICE-MODULE.md``.

Currently shipped implementations:
  * :class:`ElevenLabsTextToSpeech` — hosted TTS, multilingual.
  * :class:`SarvamSpeechToText`     — hosted ASR tuned for Indian languages.

Documented drop-in open-source alternatives (NOT implemented here — API credits
are limited, and these are the intended migration targets once GPU time exists):
  * ``ai4bharat/indic-conformer-600m-multilingual`` — 22 Indian languages, MIT
    licence. Implements ``SpeechToText``; runs locally, no key, no per-call cost.
  * ``openai/whisper-large-v3`` — strong Hindi / Indian-English, word-level
    timestamps and per-token logprobs (a genuine per-word confidence source,
    which the hosted APIs mostly do not give us). Implements ``SpeechToText``.
  * ``ai4bharat/indic-parler-tts`` — TTS side. Implements ``TextToSpeech``.

Graceful degradation (the load-bearing property)
------------------------------------------------
A missing API key must never take down the API. Therefore:
  * importing this module never touches the network and never raises;
  * key lookups happen lazily, per call, via ``backend.app.config``;
  * every public entry point returns a structured "unavailable" result
    (``None`` for :func:`synthesize_decision`, an ``available: False`` dict for
    :func:`transcribe`) instead of raising;
  * provider/network/parse failures are caught and folded into the same shape.
Only the HTTP layer in ``backend/app/routers/voice.py`` turns "unavailable" into
a 503 with an actionable message — never a 500.

Per-word confidence
-------------------
:func:`transcribe` always returns a ``words`` list. Each entry carries
``{word, start, end, confidence}`` with ``confidence`` set to ``None`` when the
provider does not report one. This is intentional and load-bearing: a downstream
uncertainty layer is meant to consume it, so that a low-confidence transcription
of a critical field (income, dependants) can be flagged for confirmation or
routed to deferral rather than silently accepted. Provider reality as of now:

  * Sarvam ``saarika`` returns word/segment timestamps when asked but does not
    expose per-word acoustic confidence; we emit ``confidence: None``.
  * Whisper (local) exposes token logprobs, from which a real per-word
    confidence can be derived — one more reason the open-source path matters.

Never assume ``confidence`` is a float. Treat ``None`` as "unknown", not "1.0".

No new dependencies: HTTP uses ``urllib`` from the stdlib.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from abc import ABC, abstractmethod
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from backend.app.config import ELEVENLABS_API_KEY, SARVAM_API_KEY, get_secret

logger = logging.getLogger(__name__)

# Network timeout (seconds). Voice calls sit in a request path; a hung provider
# must degrade to an error result quickly rather than pinning a worker.
HTTP_TIMEOUT_SECONDS = 30

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
# "Rachel" — ElevenLabs' default public voice. Override with ELEVENLABS_VOICE_ID.
DEFAULT_ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_SARVAM_MODEL = "saarika:v2"

# Short ISO codes -> the BCP-47-ish codes Sarvam expects. Unknown codes are
# passed through unchanged so a new language does not require a code change.
LANGUAGE_CODES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "od": "od-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}


class VoiceProviderError(RuntimeError):
    """Internal-only: a provider call failed.

    Never escapes the module — the public entry points catch it and convert it
    into a structured result so callers do not have to handle exceptions.
    """


# --------------------------------------------------------------------------- #
# HTTP helper (stdlib only — see module docstring)
# --------------------------------------------------------------------------- #


def _http_post(
    url: str,
    *,
    headers: dict[str, str],
    body: bytes,
) -> bytes:
    """POST and return the raw response body.

    Isolated in one function on purpose: tests monkeypatch this to guarantee no
    real network call is ever made.
    """
    req = urllib_request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return response.read()
    except urllib_error.HTTPError as exc:  # provider said no
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:  # pragma: no cover - best-effort diagnostics only
            pass
        raise VoiceProviderError(f"provider returned HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:  # DNS/TLS/connection/timeout
        raise VoiceProviderError(f"could not reach provider: {exc.reason}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise VoiceProviderError(f"unexpected transport failure: {exc}") from exc


def _encode_multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes]],
) -> tuple[bytes, str]:
    """Build a multipart/form-data body without pulling in `requests`."""
    boundary = f"----smartlend{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    for name, (filename, content) in files.items():
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        )
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _normalise_language(language: str | None) -> str | None:
    if not language:
        return None
    return LANGUAGE_CODES.get(language.lower(), language)


# --------------------------------------------------------------------------- #
# Abstractions — the provider seam
# --------------------------------------------------------------------------- #


class TextToSpeech(ABC):
    """Render text to audio bytes.

    Implement this (and return it from :func:`get_tts_provider`) to swap in a
    different engine — e.g. a locally hosted ``ai4bharat/indic-parler-tts``.
    Implementations MUST NOT raise from :meth:`is_configured` or
    :meth:`unavailable_reason`, and MUST raise only :class:`VoiceProviderError`
    from :meth:`synthesize`.
    """

    #: Stable identifier surfaced in API responses and logs.
    name: str = "tts"

    @abstractmethod
    def is_configured(self) -> bool:
        """True when this provider has everything it needs to run."""

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """Human-readable, actionable reason, or None when configured."""

    @abstractmethod
    def synthesize(self, text: str, language: str | None = None) -> bytes:
        """Return encoded audio (MP3) for `text`."""

    #: MIME type of the bytes returned by :meth:`synthesize`.
    media_type: str = "audio/mpeg"


class SpeechToText(ABC):
    """Transcribe audio bytes to text plus per-word detail.

    Implementations MUST return the shape documented on
    :meth:`transcribe` so the uncertainty layer downstream can rely on it —
    in particular the ``words`` list must always be present, with
    ``confidence: None`` where the provider gives no confidence.
    """

    name: str = "stt"

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def unavailable_reason(self) -> str | None: ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> dict[str, Any]:
        """Return ``{"text", "language", "words", "confidence"}``.

        ``words`` is a list of ``{"word", "start", "end", "confidence"}``.
        """


# --------------------------------------------------------------------------- #
# Concrete providers
# --------------------------------------------------------------------------- #


class ElevenLabsTextToSpeech(TextToSpeech):
    """Hosted multilingual TTS. Needs ELEVENLABS_API_KEY."""

    name = "elevenlabs"

    def is_configured(self) -> bool:
        # Lazy, per-call lookup: a key added to .env after boot works without a
        # restart, and a missing key is a value, never an exception.
        return get_secret(ELEVENLABS_API_KEY) is not None

    def unavailable_reason(self) -> str | None:
        if self.is_configured():
            return None
        return (
            f"{ELEVENLABS_API_KEY} is not set. Add it to the gitignored .env at the "
            "project root (see .env.example) and restart the API."
        )

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        api_key = get_secret(ELEVENLABS_API_KEY)
        if api_key is None:  # belt-and-braces; callers gate on is_configured()
            raise VoiceProviderError(self.unavailable_reason() or "not configured")

        voice_id = get_secret("ELEVENLABS_VOICE_ID") or DEFAULT_ELEVENLABS_VOICE_ID
        model_id = get_secret("ELEVENLABS_MODEL_ID") or DEFAULT_ELEVENLABS_MODEL_ID
        # `language` is informational for ElevenLabs: the multilingual model
        # infers language from the text itself, so we do not send a code.
        payload = json.dumps(
            {
                "text": text,
                "model_id": model_id,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
        ).encode("utf-8")
        audio = _http_post(
            ELEVENLABS_TTS_URL.format(voice_id=voice_id),
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": self.media_type,
            },
            body=payload,
        )
        if not audio:
            raise VoiceProviderError("provider returned an empty audio body")
        return audio


class SarvamSpeechToText(SpeechToText):
    """Hosted ASR for Indian languages. Needs SARVAM_API_KEY."""

    name = "sarvam"

    def is_configured(self) -> bool:
        return get_secret(SARVAM_API_KEY) is not None

    def unavailable_reason(self) -> str | None:
        if self.is_configured():
            return None
        return (
            f"{SARVAM_API_KEY} is not set. Add it to the gitignored .env at the "
            "project root (see .env.example) and restart the API. Alternatively "
            "run a local open-source ASR — see docs/VOICE-MODULE.md."
        )

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> dict[str, Any]:
        api_key = get_secret(SARVAM_API_KEY)
        if api_key is None:
            raise VoiceProviderError(self.unavailable_reason() or "not configured")

        fields = {
            "model": get_secret("SARVAM_STT_MODEL") or DEFAULT_SARVAM_MODEL,
            # Ask for timestamps so the words list carries positions even though
            # the provider gives us no acoustic confidence to go with them.
            "with_timestamps": "true",
        }
        code = _normalise_language(language)
        if code:
            fields["language_code"] = code
        else:
            # Sarvam auto-detects when given the unknown-language sentinel.
            fields["language_code"] = "unknown"

        body, content_type = _encode_multipart(fields, {"file": ("audio.wav", audio_bytes)})
        raw = _http_post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": api_key, "Content-Type": content_type},
            body=body,
        )
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise VoiceProviderError(f"could not parse provider response: {exc}") from exc
        return self._to_result(data, requested_language=code)

    @staticmethod
    def _to_result(data: dict[str, Any], requested_language: str | None) -> dict[str, Any]:
        """Map the provider payload onto our stable shape.

        Kept static and separate so it can be unit-tested against a recorded
        payload without any network access.
        """
        text = data.get("transcript") or data.get("text") or ""
        timestamps = data.get("timestamps") or {}
        words: list[dict[str, Any]] = []

        # Sarvam returns parallel arrays: words[], start_time_seconds[],
        # end_time_seconds[]. There is no confidence array today — we still emit
        # the key as None so the uncertainty layer has one shape to consume.
        raw_words = timestamps.get("words") or []
        starts = timestamps.get("start_time_seconds") or []
        ends = timestamps.get("end_time_seconds") or []
        confidences = timestamps.get("confidences") or []  # future-proofing
        for index, word in enumerate(raw_words):
            words.append(
                {
                    "word": word,
                    "start": starts[index] if index < len(starts) else None,
                    "end": ends[index] if index < len(ends) else None,
                    "confidence": confidences[index] if index < len(confidences) else None,
                }
            )
        if not words and text:
            # No timestamps came back: still expose per-word entries with
            # unknown confidence rather than an empty list, so consumers do not
            # need two code paths.
            words = [
                {"word": token, "start": None, "end": None, "confidence": None}
                for token in text.split()
            ]

        known = [w["confidence"] for w in words if isinstance(w.get("confidence"), (int, float))]
        overall = sum(known) / len(known) if known else None
        return {
            "text": text,
            "language": data.get("language_code") or requested_language,
            "words": words,
            "confidence": overall,  # None == unknown, NOT high confidence
        }


# --------------------------------------------------------------------------- #
# Provider resolution — THE SWAP SEAM
# --------------------------------------------------------------------------- #
# Change these two functions (or the two module globals they read) to point at a
# different implementation. Nothing above the seam — routers, services, tests of
# callers — needs to change. To run open-source locally you would add e.g.
# `class IndicConformerSpeechToText(SpeechToText)` and return it here.

_tts_provider: TextToSpeech = ElevenLabsTextToSpeech()
_stt_provider: SpeechToText = SarvamSpeechToText()


def get_tts_provider() -> TextToSpeech:
    return _tts_provider


def get_stt_provider() -> SpeechToText:
    return _stt_provider


def set_providers(
    tts: TextToSpeech | None = None,
    stt: SpeechToText | None = None,
) -> None:
    """Install alternative providers at runtime (used by the swap and by tests)."""
    global _tts_provider, _stt_provider
    if tts is not None:
        _tts_provider = tts
    if stt is not None:
        _stt_provider = stt


# --------------------------------------------------------------------------- #
# Public entry points — none of these raise
# --------------------------------------------------------------------------- #


def voice_available() -> dict[str, Any]:
    """Report which providers are configured, and why the others are not.

    Safe to call with no keys, no network, and no restart. ``available`` is the
    OR of the two sides: it is False, with reasons, when nothing is configured.
    """
    tts = get_tts_provider()
    stt = get_stt_provider()

    def _describe(provider: TextToSpeech | SpeechToText, kind: str) -> dict[str, Any]:
        # Even the availability probe is defensive: a broken provider must not
        # be able to break the status endpoint.
        try:
            configured = bool(provider.is_configured())
            reason = None if configured else provider.unavailable_reason()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("voice provider %s failed its availability check", kind)
            configured, reason = False, f"provider raised during check: {exc}"
        return {
            "kind": kind,
            "provider": getattr(provider, "name", kind),
            "configured": configured,
            "reason": reason,
        }

    tts_status = _describe(tts, "tts")
    stt_status = _describe(stt, "stt")
    reasons = [s["reason"] for s in (tts_status, stt_status) if s["reason"]]
    return {
        "available": tts_status["configured"] or stt_status["configured"],
        "tts": tts_status,
        "stt": stt_status,
        "reasons": reasons,
        "docs": "docs/VOICE-MODULE.md",
    }


def _unavailable_transcript(reason: str, provider: str) -> dict[str, Any]:
    """The single canonical 'we could not transcribe' shape."""
    return {
        "available": False,
        "provider": provider,
        "text": None,
        "language": None,
        "words": [],
        "confidence": None,
        "reason": reason,
    }


def build_decision_script(decision: str, explanation: str, language: str = "en") -> str:
    """Compose the sentence that gets read aloud.

    Separate from synthesis so the wording can be reviewed, tested and
    translated without touching any provider code.
    """
    decision_text = (decision or "").strip() or "pending"
    explanation_text = (explanation or "").strip()
    script = f"Your loan application decision is: {decision_text}."
    if explanation_text:
        script = f"{script} {explanation_text}"
    return script


def synthesize_decision(
    decision: str,
    explanation: str,
    language: str = "en",
) -> bytes | None:
    """Read a decision aloud. Returns MP3 bytes, or None when unavailable.

    None means "voice is off or the provider failed" — it is a normal outcome,
    not an error condition. The caller decides whether that is a 503 (the
    router) or simply no audio player (a batch job).
    """
    provider = get_tts_provider()
    try:
        if not provider.is_configured():
            # Degradation path #1: no key. Log once at INFO (not ERROR — this is
            # an expected configuration state) and return None.
            logger.info("TTS unavailable: %s", provider.unavailable_reason())
            return None
        script = build_decision_script(decision, explanation, language)
        return provider.synthesize(script, language=language)
    except VoiceProviderError as exc:
        # Degradation path #2: key present, provider failed. Still no raise.
        logger.warning("TTS provider %s failed: %s", getattr(provider, "name", "?"), exc)
        return None
    except Exception:  # pragma: no cover - degradation path #3: anything else
        logger.exception("Unexpected TTS failure; degrading to no audio")
        return None


def transcribe(audio_bytes: bytes, language: str | None = None) -> dict[str, Any]:
    """Transcribe uploaded audio. Always returns a dict; never raises.

    Success::

        {"available": True, "provider": "sarvam", "text": "...",
         "language": "hi-IN", "confidence": None,
         "words": [{"word": "...", "start": 0.0, "end": 0.4,
                    "confidence": None}]}

    ``confidence`` is ``None`` when the provider does not report one — see the
    module docstring. Failure returns the same keys with ``available: False``
    and a ``reason``.
    """
    provider = get_stt_provider()
    name = getattr(provider, "name", "stt")
    try:
        if not provider.is_configured():
            return _unavailable_transcript(provider.unavailable_reason() or "not configured", name)
        if not audio_bytes:
            return _unavailable_transcript("no audio was supplied", name)
        result = provider.transcribe(audio_bytes, language=language)
        # Normalise defensively: a third-party implementation of the ABC might
        # omit a key, and callers must still see the full shape.
        return {
            "available": True,
            "provider": name,
            "text": result.get("text"),
            "language": result.get("language"),
            "words": result.get("words") or [],
            "confidence": result.get("confidence"),
            "reason": None,
        }
    except VoiceProviderError as exc:
        logger.warning("STT provider %s failed: %s", name, exc)
        return _unavailable_transcript(str(exc), name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected STT failure; degrading to unavailable")
        return _unavailable_transcript(f"unexpected transcription failure: {exc}", name)


__all__ = [
    "ElevenLabsTextToSpeech",
    "SarvamSpeechToText",
    "SpeechToText",
    "TextToSpeech",
    "VoiceProviderError",
    "build_decision_script",
    "get_stt_provider",
    "get_tts_provider",
    "set_providers",
    "synthesize_decision",
    "transcribe",
    "voice_available",
]
