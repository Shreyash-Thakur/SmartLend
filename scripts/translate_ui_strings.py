"""Translate the customer UI strings into Hindi and Marathi with Sarvam, once.

    node frontend/scripts/extract-ui-strings.mjs    # refresh the English list
    python scripts/translate_ui_strings.py           # fills only missing entries

Reads frontend/src/i18n/en.json and writes frontend/src/i18n/{hi,mr}.json as
{english: translation}. Existing translations are kept, so a hand-corrected
entry is never overwritten; delete an entry to have it re-translated.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.translation_service import TranslationUnavailableError, translate  # noqa: E402

I18N = ROOT / "frontend" / "src" / "i18n"


def _translate_with_retry(text: str, target: str) -> str:
    # The free tier rate-limits bursts (HTTP 429): go one at a time and back off.
    for attempt in range(6):
        try:
            return translate(text, target)
        except TranslationUnavailableError as exc:
            if "429" not in str(exc) or attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _write(path: Path, english: list[str], translations: dict[str, str]) -> None:
    ordered = {text: translations[text] for text in english if text in translations}
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    english = json.loads((I18N / "en.json").read_text(encoding="utf-8"))
    for target in ("hi", "mr"):
        path = I18N / f"{target}.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        missing = [text for text in english if text not in existing]
        for index, text in enumerate(missing, 1):
            existing[text] = _translate_with_retry(text, target)
            if index % 20 == 0:  # progress survives an interrupted run
                _write(path, english, existing)
        _write(path, english, existing)
        print(f"{target}: {len(missing)} translated -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
