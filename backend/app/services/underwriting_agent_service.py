"""LLM reviewer-briefing agent for deferred/under-review applications.

What this is: a Claude-powered analyst assistant that reads one application's
full decision report (the same audit record `GET /api/applications/{id}/report`
serves) and produces a structured briefing for the human reviewer — a summary,
data-consistency flags, risk factors and mitigants, checks worth doing before
ruling, and suggested reason codes drawn from the live taxonomy.

What this is NOT — three hard boundaries, all load-bearing:

1. **Advisory only, never a decision.** Nothing here writes to
   `final_decision`, feeds `hybrid_decision`, or moves a threshold. The RBI
   FREE-AI framework's human-in-the-loop expectation is the design constraint:
   the agent prepares the reviewer, the reviewer decides.
2. **Optional by construction.** No ANTHROPIC_API_KEY → a calm 503 with setup
   instructions, mirroring the voice module's 503-never-500 contract.
3. **Grounded in the report.** The prompt forbids inventing applicant facts;
   every flag must cite fields that exist in the payload it was given.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.app.config import anthropic_api_key, anthropic_configured, get_secret
from backend.app.services.review_reason_codes import catalog as reason_code_catalog

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5"
GEMINI_MODEL = "gemini-2.5-flash"
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
        "dataQualityFlags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Internal inconsistencies, implausible or missing values, citing the exact fields",
        },
        "suggestedChecks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete verifications the reviewer should perform before ruling",
        },
        "suggestedReasonCodes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Codes from the provided taxonomy that the evidence could support",
        },
    },
    "required": [
        "summary",
        "riskFactors",
        "mitigants",
        "dataQualityFlags",
        "suggestedChecks",
        "suggestedReasonCodes",
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
- Ground every statement in the provided JSON. Cite field names for every data \
quality flag. If a value is missing, say it is missing — never invent applicant facts.
- Look for internal inconsistencies (e.g. income vs EMIs vs requested amount, \
tenure vs age, CBES pillars contradicting the headline score) and for places \
where the ML score and CBES disagree, and say what would explain them.
- suggestedReasonCodes must come from the provided taxonomy codes verbatim; \
suggest only codes the evidence could genuinely support, for either verdict.
- Keep the briefing focused and brief; the reviewer reads this in under a minute."""


class AgentUnavailableError(RuntimeError):
    """The agent is not configured or the API could not be reached."""


class AgentBriefingError(RuntimeError):
    """The API answered but did not produce a usable briefing."""


def _provider() -> str | None:
    """Which LLM backs the briefing: 'anthropic' (native) or 'gemini' (adapter).

    Anthropic wins when both keys are present; Gemini is the fallback so a
    GEMINI_API_KEY alone is enough to light the panel up.
    """
    if anthropic_configured():
        return "anthropic"
    if get_secret("GEMINI_API_KEY"):
        return "gemini"
    return None


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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
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
    return text, GEMINI_MODEL


def generate_briefing(report: dict[str, Any]) -> dict[str, Any]:
    """One decision report in, one structured reviewer briefing out.

    Raises AgentUnavailableError (missing key / transport / rate limit — the
    caller answers 503) or AgentBriefingError (malformed model output — 502).
    """
    provider = _provider()
    if provider is None:
        raise AgentUnavailableError(
            "No agent API key is set. Add ANTHROPIC_API_KEY (or GEMINI_API_KEY) "
            "to .env to enable the reviewer briefing agent."
        )

    import anthropic

    taxonomy = [
        {"code": item["code"], "label": item["label"], "direction": item["direction"]}
        for item in reason_code_catalog()
    ]
    # The report is already PII-minimal for its audience (the reviewer sees the
    # same payload); underscore-prefixed internals were stripped upstream.
    user_payload = {
        "decision_report": {
            "application": report.get("application"),
            "engine": report.get("engine"),
            "humanReview": report.get("humanReview"),
        },
        "reason_code_taxonomy": taxonomy,
    }

    user_text = "Brief the reviewer on this application:\n" + json.dumps(
        user_payload, indent=1, default=str
    )

    if provider == "gemini":
        text, model_name = _generate_gemini(user_text)
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
