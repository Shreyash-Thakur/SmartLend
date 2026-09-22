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


# ---------------------------------------------------------------------------
# Relearning loop — exploration arm
# ---------------------------------------------------------------------------
# Spec section 3 ("Also capture a control arm"): route a small random 2-5% of
# would-be-auto-decided applications into human review anyway. 3% is the
# midpoint default; the band is enforced as a hard clamp so a fat-fingered env
# var cannot silently push a large slice of production traffic into manual
# review (or switch the control arm off without anyone noticing).
EXPLORATION_RATE_ENV = "SMARTLEND_EXPLORATION_RATE"
DEFAULT_EXPLORATION_RATE = 0.03
EXPLORATION_RATE_BOUNDS = (0.0, 0.05)


def exploration_rate() -> float:
    """Fraction of would-be-auto decisions routed to a human as the control arm.

    Defaults to 3%. A malformed value falls back to the default rather than
    raising: the exploration arm is a research instrument, and it must never be
    the reason a lending decision fails.
    """
    raw = get_secret(EXPLORATION_RATE_ENV)
    if raw is None:
        return DEFAULT_EXPLORATION_RATE
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_EXPLORATION_RATE
    low, high = EXPLORATION_RATE_BOUNDS
    return max(low, min(high, value))


ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"


def elevenlabs_api_key() -> str:
    return require_secret(ELEVENLABS_API_KEY)


def elevenlabs_configured() -> bool:
    """Check availability without raising — for health checks and feature flags."""
    return get_secret(ELEVENLABS_API_KEY) is not None


SARVAM_API_KEY = "SARVAM_API_KEY"


def sarvam_api_key() -> str:
    return require_secret(SARVAM_API_KEY)


def sarvam_configured() -> bool:
    """Check availability without raising — for health checks and feature flags."""
    return get_secret(SARVAM_API_KEY) is not None


# ---------------------------------------------------------------------------
# Underwriting agent (Claude API)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"


def anthropic_api_key() -> str:
    return require_secret(ANTHROPIC_API_KEY)


def anthropic_configured() -> bool:
    """Check availability without raising — for health checks and feature flags."""
    return get_secret(ANTHROPIC_API_KEY) is not None


# ---------------------------------------------------------------------------
# Remote TabPFN endpoint (Colab-hosted GPU inference)
# ---------------------------------------------------------------------------
# TabPFN-2.5 needs more VRAM than the local 8 GB GPU can give it, so it can be
# hosted on a Colab GPU via colab/tabpfn_colab_server.ipynb, which prints a
# public tunnel URL and a session token. The backend treats that endpoint as a
# strictly OPTIONAL research instrument: nothing on the lending decision path
# calls it, and every consumer must degrade gracefully when it is absent.
# Licence note: TabPFN-2.5 outputs are non-commercial (models/README.md), so
# the remote scores are research/demo material, never decision inputs.
TABPFN_URL_ENV = "SMARTLEND_TABPFN_URL"
TABPFN_TOKEN_ENV = "SMARTLEND_TABPFN_TOKEN"


def tabpfn_remote_url() -> str | None:
    """Base URL of the Colab-hosted TabPFN endpoint, or None when not configured."""
    url = get_secret(TABPFN_URL_ENV)
    return url.rstrip("/") if url else None


def tabpfn_remote_token() -> str | None:
    return get_secret(TABPFN_TOKEN_ENV)


def tabpfn_remote_configured() -> bool:
    """Check availability without raising — for health checks and feature flags."""
    return tabpfn_remote_url() is not None
