"""Client for the Colab-hosted TabPFN-2.5 inference endpoint.

The server side lives in ``colab/tabpfn_colab_server.ipynb``: a Colab GPU
session that loads the pinned TabPFN-2.5 checkpoint with an in-context
training set exported by ``backend/export_tabpfn_context.py``, and exposes
``/health`` and ``/score`` behind a session token through a Cloudflare quick
tunnel. This module is the only place the backend talks to it.

Two properties are load-bearing:

1. **Optional by construction.** The endpoint exists only while someone has a
   Colab tab open. Every function here either answers "not configured" or
   raises ``RemoteTabPFNError`` for the caller to translate into a 503 — a
   missing tunnel must never look like a scoring bug.
2. **Never on the decision path.** TabPFN-2.5's licence covers outputs and is
   non-commercial (models/README.md), and the endpoint is a research
   instrument. Remote scores are surfaced as a *second opinion* next to a
   decision that has already been made; no code path feeds them into
   ``hybrid_decision`` or a threshold.

Wire protocol (matched by the notebook):

    GET  {url}/health                     -> {status, model, device, context_rows,
                                              tabpfn_version, feature_names}
    POST {url}/score {"rows": [{...}]}    -> {"predictions": [{"p_default": f,
                                              "p_approve": f}, ...], "seconds": f}

Both carry the session token in the ``X-SmartLend-Token`` header.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.app.config import tabpfn_remote_token, tabpfn_remote_url

logger = logging.getLogger(__name__)

# TabPFN holds its whole training context in VRAM per forward pass, so remote
# inference is seconds-per-batch, not milliseconds. Health stays snappy.
HEALTH_TIMEOUT_SECONDS = 10.0
SCORE_TIMEOUT_SECONDS = 120.0
MAX_ROWS_PER_CALL = 500


class RemoteTabPFNError(RuntimeError):
    """The remote endpoint is configured but did not produce usable scores."""


def configured() -> bool:
    return tabpfn_remote_url() is not None


def _headers() -> dict[str, str]:
    token = tabpfn_remote_token()
    return {"X-SmartLend-Token": token} if token else {}


def remote_status() -> dict[str, Any]:
    """One honest snapshot: configured? reachable? what is actually loaded?

    Never raises — this feeds status endpoints and feature flags, and an
    unreachable tunnel is an expected state, not an exception.
    """
    url = tabpfn_remote_url()
    if url is None:
        return {
            "configured": False,
            "reachable": False,
            "detail": (
                "SMARTLEND_TABPFN_URL is not set. Start colab/tabpfn_colab_server.ipynb, "
                "copy the printed URL and token into .env, then restart the backend."
            ),
        }
    try:
        response = httpx.get(f"{url}/health", headers=_headers(), timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 — any transport/parse failure means "unreachable"
        logger.warning("Remote TabPFN health check failed for %s: %s", url, exc)
        return {"configured": True, "reachable": False, "url": url, "detail": str(exc)}

    return {"configured": True, "reachable": True, "url": url, "server": payload}


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Score feature rows on the remote GPU. Returns one dict per input row.

    Rows must use the serving feature vocabulary (the artifact's
    ``feature_names``); the notebook was fitted on exactly those columns via
    ``backend/export_tabpfn_context.py``, and it reindexes on its own
    ``feature_names`` so extra keys are ignored and missing ones are imputed
    server-side with the same CBES worst-case defaults.
    """
    url = tabpfn_remote_url()
    if url is None:
        raise RemoteTabPFNError(
            "Remote TabPFN is not configured (SMARTLEND_TABPFN_URL unset)."
        )
    if not rows:
        return []
    if len(rows) > MAX_ROWS_PER_CALL:
        raise RemoteTabPFNError(
            f"Refusing to send {len(rows)} rows in one call (max {MAX_ROWS_PER_CALL}); "
            "batch the request."
        )

    try:
        response = httpx.post(
            f"{url}/score",
            json={"rows": rows},
            headers=_headers(),
            timeout=SCORE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise RemoteTabPFNError(
            f"Remote TabPFN returned HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — transport failures collapse to one caller-facing type
        raise RemoteTabPFNError(f"Remote TabPFN call failed: {exc}") from exc

    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise RemoteTabPFNError(
            f"Remote TabPFN answered with {len(predictions) if isinstance(predictions, list) else 'no'} "
            f"predictions for {len(rows)} rows."
        )

    out: list[dict[str, float]] = []
    for item in predictions:
        try:
            p_default = float(item["p_default"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteTabPFNError(f"Malformed prediction from remote TabPFN: {item!r}") from exc
        if not 0.0 <= p_default <= 1.0:
            raise RemoteTabPFNError(f"Remote TabPFN produced an out-of-range probability: {p_default}")
        out.append({"p_default": p_default, "p_approve": 1.0 - p_default})
    return out
