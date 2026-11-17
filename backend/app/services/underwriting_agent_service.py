"""LLM reviewer-briefing agent for deferred/under-review applications.

What this is: a Claude-powered analyst assistant that reads one application's
full decision report (the same audit record `GET /api/applications/{id}/report`
serves) and produces a structured briefing for the human reviewer — a summary,
a plain reading of the SHAP factors, risk factors and mitigants, and checks
worth doing before ruling.

What this is NOT — three hard boundaries, all load-bearing:

1. **Advisory only, never a decision.** Nothing here writes to
   `final_decision`, feeds `hybrid_decision`, or moves a threshold. The RBI
   FREE-AI framework's human-in-the-loop expectation is the design constraint:
   the agent prepares the reviewer, the reviewer decides.
2. **Optional by construction.** No provider configured → a calm 503 with setup
   instructions, mirroring the voice module's 503-never-500 contract.
   Providers, first configured wins: ANTHROPIC_API_KEY, GEMINI_API_KEY (free
   tier), OLLAMA_MODEL (a local model, no key, no internet).
3. **Grounded in the report.** The prompt forbids inventing applicant facts;
   every flag must cite fields that exist in the payload it was given.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.app.config import anthropic_api_key, anthropic_configured, get_secret

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5"
# Google retires Gemini models for new keys (2.5-flash went 404); override
# with GEMINI_MODEL in .env instead of editing code.
GEMINI_MODEL = "gemini-3.8-flash"


def _gemini_model() -> str:
    return get_secret("GEMINI_MODEL") or GEMINI_MODEL
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Local models are slower than hosted ones; a 7-8B model on a laptop CPU/GPU
# can take a minute or two for a structured briefing.
OLLAMA_TIMEOUT_SECONDS = 240.0
MAX_TOKENS = 4096

# Structured output: the briefing arrives as validated JSON, not prose to parse.
BRIEFING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "3-5 sentence plain-language briefing of the case for the reviewer",
        },
        "riskFactors": {"type": "array", "items": {"type": "string"}},
        "mitigants": {"type": "array", "items": {"type": "string"}},
        "suggestedChecks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete verifications the reviewer should perform before ruling",
        },
        "factorExplanation": {
            "type": "string",
            "description": "2-4 plain-language sentences explaining the SHAP topFactors: "
            "which raised and which lowered the default risk, and what that means for this applicant",
        },
    },
    "required": [
        "summary",
        "riskFactors",
        "mitigants",
        "suggestedChecks",
        "factorExplanation",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an underwriting analyst assistant inside SmartLend, \
a loan decisioning research system. You brief the human reviewer on ONE loan \
application that has been routed to them. The engine's scores (p_ml, p_cbes, \
thresholds, SHAP factors, CBES pillar breakdown) and the applicant's submitted \
data are provided as JSON.

Rules:
- You are advisory only. The human reviewer decides; never recommend APPROVE or \
REJECT as a verdict, and never claim authority over the outcome.
- Ground every statement in the provided JSON. If a value is missing, say it \
is missing — never invent applicant facts.
- Look for internal inconsistencies (e.g. income vs EMIs vs requested amount, \
tenure vs age, CBES pillars contradicting the headline score) and for places \
where the ML score and CBES disagree, and say what would explain them.
- engine.topFactors are SHAP attributions re-signed for the decision taken: \
impact > 0 ("supports_decision") pushed toward that decision, impact < 0 \
("opposes_decision") pushed against it. For APPROVE and DEFER, > 0 means the \
feature LOWERED the default risk; for REJECT, > 0 means it RAISED it. In \
factorExplanation, explain them in plain words a bank officer understands, say \
when a factor's direction looks counter-intuitive (e.g. a larger loan lowering \
risk is learned from the training data, not a rule), and do not quote raw SHAP \
numbers. If the decision reason starts with "affordability_", say that the \
bank's affordability policy, not the model, set the outcome.
- Keep the briefing focused and brief; the reviewer reads this in under a minute."""


class AgentUnavailableError(RuntimeError):
    """The agent is not configured or the API could not be reached."""


class AgentBriefingError(RuntimeError):
    """The API answered but did not produce a usable briefing."""


def _provider() -> str | None:
    """Which LLM backs the briefing: 'anthropic' (native), 'gemini' or 'ollama'.

    First configured wins, in that order, so a GEMINI_API_KEY alone or an
    OLLAMA_MODEL alone is enough to light the panel up.
    """
    if anthropic_configured():
        return "anthropic"
    if get_secret("GEMINI_API_KEY"):
        return "gemini"
    if get_secret("OLLAMA_MODEL"):
        return "ollama"
    return None


def active_model() -> str | None:
    """Model the briefing will use, for /agent/status; None when unconfigured."""
    return {
        "anthropic": MODEL,
        "gemini": _gemini_model(),
        "ollama": get_secret("OLLAMA_MODEL"),
    }.get(_provider() or "")


def configured() -> bool:
    return _provider() is not None


def _client():
    import anthropic

    return anthropic.Anthropic(api_key=anthropic_api_key())


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini's responseSchema speaks an OpenAPI subset: same type/properties/
    required/items/description, but no additionalProperties — strip it."""
    cleaned = {k: v for k, v in schema.items() if k != "additionalProperties"}
    if "properties" in cleaned:
        cleaned["properties"] = {k: _gemini_schema(v) for k, v in cleaned["properties"].items()}
    if "items" in cleaned:
        cleaned["items"] = _gemini_schema(cleaned["items"])
    return cleaned


def _generate_gemini(user_text: str) -> tuple[str, str]:
    """(briefing JSON text, model name) via Gemini's native API; errors are
    mapped onto the same exception contract as the Anthropic path."""
    import httpx

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{_gemini_model()}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_TOKENS,
            "responseMimeType": "application/json",
            "responseSchema": _gemini_schema(BRIEFING_SCHEMA),
        },
    }
    try:
        response = httpx.post(
            url, json=body, timeout=60.0,
            headers={"x-goog-api-key": get_secret("GEMINI_API_KEY") or ""},
        )
    except httpx.HTTPError as exc:
        raise AgentUnavailableError(f"Could not reach the Gemini API: {exc}") from exc
    if response.status_code in (401, 403):
        raise AgentUnavailableError(f"Gemini API key rejected (HTTP {response.status_code}).")
    if response.status_code == 429:
        raise AgentUnavailableError("Gemini API rate limited; retry shortly.")
    if response.status_code != 200:
        raise AgentUnavailableError(
            f"Gemini API error {response.status_code}: {response.text[:200]}")

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise AgentBriefingError(f"Gemini produced no briefing ({reason}).")
    finish = candidates[0].get("finishReason")
    if finish not in (None, "STOP", "MAX_TOKENS"):
        raise AgentBriefingError(f"Gemini stopped abnormally ({finish}).")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = next((p.get("text") for p in parts if p.get("text")), None)
    if not text:
        raise AgentBriefingError("Gemini returned no text content.")
    return text, _gemini_model()


def _generate_ollama(user_text: str) -> tuple[str, str]:
    """(briefing JSON text, model name) via a local Ollama server.

    Ollama's `format` takes a JSON schema and constrains decoding to it, so the
    briefing arrives in the same shape as the hosted providers'.
    """
    import httpx

    model = get_secret("OLLAMA_MODEL") or ""
    base_url = (get_secret("OLLAMA_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
    body = {
        "model": model,
        "stream": False,
        "format": BRIEFING_SCHEMA,
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }
    try:
        response = httpx.post(f"{base_url}/api/chat", json=body, timeout=OLLAMA_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise AgentUnavailableError(
            f"Could not reach Ollama at {base_url} ({exc}). Start it with `ollama serve`."
        ) from exc
    if response.status_code == 404:
        raise AgentUnavailableError(
            f"Ollama has no model '{model}'. Download it with `ollama pull {model}`.")
    if response.status_code != 200:
        raise AgentUnavailableError(
            f"Ollama error {response.status_code}: {response.text[:200]}")

    text = ((response.json().get("message") or {}).get("content") or "").strip()
    if not text:
        raise AgentBriefingError("Ollama returned no text content.")
    return text, f"ollama/{model}"


def generate_briefing(report: dict[str, Any]) -> dict[str, Any]:
    """One decision report in, one structured reviewer briefing out.

    Raises AgentUnavailableError (missing key / transport / rate limit — the
    caller answers 503) or AgentBriefingError (malformed model output — 502).
    """
    provider = _provider()
    if provider is None:
        raise AgentUnavailableError(
            "No agent is configured. Add GEMINI_API_KEY (free at aistudio.google.com/apikey), "
            "ANTHROPIC_API_KEY, or OLLAMA_MODEL (a local model) to .env and restart the backend."
        )

    import anthropic

    # The report is already PII-minimal for its audience (the reviewer sees the
    # same payload); underscore-prefixed internals were stripped upstream.
    user_payload = {
        "decision_report": {
            "application": report.get("application"),
            "engine": report.get("engine"),
            "humanReview": report.get("humanReview"),
        },
    }

    user_text = "Brief the reviewer on this application:\n" + json.dumps(
        user_payload, indent=1, default=str
    )

    if provider == "gemini":
        text, model_name = _generate_gemini(user_text)
    elif provider == "ollama":
        text, model_name = _generate_ollama(user_text)
    else:
        try:
            response = _client().messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                output_config={"format": {"type": "json_schema", "schema": BRIEFING_SCHEMA}},
                messages=[{"role": "user", "content": user_text}],
            )
        except anthropic.AuthenticationError as exc:
            raise AgentUnavailableError(f"Anthropic API key rejected: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise AgentUnavailableError("Anthropic API rate limited; retry shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise AgentUnavailableError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise AgentUnavailableError(f"Could not reach the Anthropic API: {exc}") from exc

        if response.stop_reason == "refusal":
            raise AgentBriefingError("The model declined to produce a briefing for this case.")
        text = next((block.text for block in response.content if block.type == "text"), None)
        model_name = response.model
        if not text:
            raise AgentBriefingError("The model returned no text content.")

    try:
        briefing = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentBriefingError(f"The model returned non-JSON output: {text[:200]}") from exc

    briefing["model"] = model_name
    briefing["advisoryOnly"] = True
    return briefing
