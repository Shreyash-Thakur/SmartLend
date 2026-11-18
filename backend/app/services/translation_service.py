"""English → Hindi / Marathi translation through Sarvam, for the customer UI.

Static interface labels are translated once, offline, by
`scripts/translate_ui_strings.py` into `frontend/src/i18n/{hi,mr}.json`; the
page reads those files and never calls Sarvam for them. This service exists for
the few sentences that carry numbers (the decision explanation), which cannot
be pre-translated. Results are cached in-process so a page reload costs nothing.

Optional by construction: no SARVAM_API_KEY → TranslationUnavailableError and
the router answers a calm 503; the frontend then simply keeps the English text.
"""

from __future__ import annotations

import logging
import time

import httpx

from backend.app.config import get_secret

logger = logging.getLogger(__name__)

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_TRANSLATE_MODEL = "sarvam-translate:v1"
SUPPORTED_TARGETS = {"hi": "hi-IN", "mr": "mr-IN"}
MAX_TEXT_CHARS = 1000  # sarvam-translate:v1 input limit is 2000; stay well under
TIMEOUT_SECONDS = 20.0
RATE_LIMIT_RETRIES = 3

_cache: dict[tuple[str, str], str] = {}


class TranslationUnavailableError(RuntimeError):
    """No key configured, or the provider could not be reached."""


def translation_configured() -> bool:
    return get_secret("SARVAM_API_KEY") is not None


def translate(text: str, target: str) -> str:
    """Translate one English string; `target` is "hi" or "mr"."""
    code = SUPPORTED_TARGETS.get(target)
    if code is None:
        raise ValueError(f"target must be one of {sorted(SUPPORTED_TARGETS)}")
    text = text.strip()
    if not text:
        return text
    key = (text, code)
    if key in _cache:
        return _cache[key]

    api_key = get_secret("SARVAM_API_KEY")
    if api_key is None:
        raise TranslationUnavailableError("Set SARVAM_API_KEY in .env to enable Hindi/Marathi translation.")
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            response = httpx.post(
                SARVAM_TRANSLATE_URL,
                headers={"api-subscription-key": api_key},
                json={
                    "input": text[:MAX_TEXT_CHARS],
                    "source_language_code": "en-IN",
                    "target_language_code": code,
                    "model": SARVAM_TRANSLATE_MODEL,
                },
                timeout=TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise TranslationUnavailableError(f"could not reach Sarvam: {exc}") from exc
        # The free tier rate-limits bursts; a short backoff clears it.
        if response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
            break
        time.sleep(0.5 * 2**attempt)
    if response.status_code != 200:
        logger.warning("Sarvam translate failed: HTTP %s %s", response.status_code, response.text[:200])
        raise TranslationUnavailableError(f"Sarvam returned HTTP {response.status_code}")
    translated = str(response.json().get("translated_text", "")).strip()
    if not translated:
        raise TranslationUnavailableError("Sarvam returned an empty translation")
    _cache[key] = translated
    return translated
