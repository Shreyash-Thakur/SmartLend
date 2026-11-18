"""POST /api/translate — live Hindi/Marathi for sentences with numbers in them.

503 (never 500) when SARVAM_API_KEY is missing or Sarvam is unreachable; the
frontend keeps the English text in that case.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services import translation_service

router = APIRouter(tags=["translate"])

MAX_TEXTS = 40


class TranslateRequest(BaseModel):
    texts: list[str] = Field(..., description="English strings to translate.")
    target: str = Field(..., description='"hi" or "mr".')


@router.post("/translate")
def translate(payload: TranslateRequest) -> dict[str, list[str | None]]:
    """One entry per input; null where that text could not be translated, so
    one rate-limited call does not throw away the rest of the batch."""
    if payload.target not in translation_service.SUPPORTED_TARGETS:
        raise HTTPException(status_code=422, detail='target must be "hi" or "mr"')
    if len(payload.texts) > MAX_TEXTS:
        raise HTTPException(status_code=422, detail=f"at most {MAX_TEXTS} texts per request")
    if not translation_service.translation_configured():
        raise HTTPException(
            status_code=503,
            detail={"error": "TRANSLATION_UNAVAILABLE", "details": "Set SARVAM_API_KEY in .env to enable Hindi/Marathi."},
        )
    translations: list[str | None] = []
    for text in payload.texts:
        try:
            translations.append(translation_service.translate(text, payload.target))
        except translation_service.TranslationUnavailableError:
            translations.append(None)
    if payload.texts and all(item is None for item in translations):
        raise HTTPException(
            status_code=503,
            detail={"error": "TRANSLATION_UNAVAILABLE", "details": "Sarvam could not be reached."},
        )
    return {"translations": translations}
