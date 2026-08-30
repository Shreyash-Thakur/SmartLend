"""HTTP surface for the voice module.

Contract: a missing API key is a *configuration* problem, not a server fault, so
these endpoints answer 503 (Service Unavailable) with an actionable message —
never 500. `/api/voice/status` answers 200 in every configuration, including
none, so a client can decide whether to show the microphone button at all.

All provider logic lives in `backend.app.services.voice_service`; this module
only translates its structured results into HTTP.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.app.services import voice_service

router = APIRouter(prefix="/voice", tags=["voice"])

# Cap uploads so a large body cannot be used to exhaust memory before the
# provider ever sees it. ~10 MB is minutes of speech at typical bitrates.
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class SynthesizeRequest(BaseModel):
    """Body for POST /api/voice/synthesize.

    Defined here rather than in schemas.py to keep the voice module self-
    contained (and because schemas.py is owned elsewhere).
    """

    decision: str = Field(..., description="Decision to read aloud, e.g. 'approved'.")
    explanation: str = Field("", description="Plain-language reason, read after the decision.")
    language: str = Field("en", description="ISO code, e.g. 'en' or 'hi'.")


def _unavailable(reason: str) -> HTTPException:
    """503, never 500 — the deployment is fine, the credential is missing."""
    return HTTPException(
        status_code=503,
        detail={
            "error": "VOICE_UNAVAILABLE",
            "details": reason,
            "docs": "docs/VOICE-MODULE.md",
        },
    )


@router.get("/status")
def voice_status() -> dict[str, object]:
    """Which voice providers are configured. Always 200, key or no key."""
    return voice_service.voice_available()


@router.post("/synthesize")
def synthesize(payload: SynthesizeRequest) -> Response:
    """Decision text in, audio out."""
    status = voice_service.voice_available()
    if not status["tts"]["configured"]:
        # Check before doing any work so the 503 message names the exact fix.
        raise _unavailable(status["tts"]["reason"] or "Text-to-speech is not configured.")

    audio = voice_service.synthesize_decision(
        payload.decision, payload.explanation, language=payload.language
    )
    if audio is None:
        # Key present but the call failed — still 503 (upstream dependency),
        # still not a 500.
        raise _unavailable(
            "Text-to-speech is configured but the provider call failed. "
            "Check the API key's validity and quota, and the server logs."
        )
    return Response(content=audio, media_type=voice_service.get_tts_provider().media_type)


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/ogg)."),
    language: str | None = Form(None, description="ISO code, e.g. 'hi'. Omit to auto-detect."),
) -> dict[str, object]:
    """Audio upload in, transcript (with per-word detail) out."""
    status = voice_service.voice_available()
    if not status["stt"]["configured"]:
        raise _unavailable(status["stt"]["reason"] or "Speech-to-text is not configured.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail={"error": "EMPTY_AUDIO", "details": "The uploaded audio file was empty."},
        )
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "AUDIO_TOO_LARGE",
                "details": f"Audio exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)} MB limit.",
            },
        )

    result = voice_service.transcribe(audio_bytes, language=language)
    if not result.get("available"):
        raise _unavailable(result.get("reason") or "Transcription is unavailable.")
    return result
