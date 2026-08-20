"""Centralised configuration and secret loading.

Secrets live in the gitignored `.env` at the project root. Nothing in this module
logs or echoes a secret value; callers get the value or a clear error, never a
partially-populated config that fails deep inside a request handler.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env_file() -> None:
    """Populate os.environ from .env without clobbering real environment vars.

    Uses python-dotenv when available and falls back to a minimal parser so a
    missing dependency degrades to "secret not configured" rather than an
    import error at startup.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file_fallback()
        return

    load_dotenv(ENV_PATH, override=False)


def _load_env_file_fallback() -> None:
    if not ENV_PATH.exists():
        return

    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env_file()


class MissingSecretError(RuntimeError):
    """Raised when a feature is invoked without its required credential."""


def get_secret(name: str) -> str | None:
    """Return a secret, or None when it is unset or blank."""
    value = os.environ.get(name, "").strip()
    return value or None


def require_secret(name: str) -> str:
    """Return a secret, raising an actionable error when it is not configured."""
    value = get_secret(name)
    if value is None:
        raise MissingSecretError(
            f"{name} is not set. Add it to {ENV_PATH} (see .env.example) "
            "or export it in the environment."
        )
    return value


ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"


def elevenlabs_api_key() -> str:
    return require_secret(ELEVENLABS_API_KEY)


def elevenlabs_configured() -> bool:
    """Check availability without raising — for health checks and feature flags."""
    return get_secret(ELEVENLABS_API_KEY) is not None
