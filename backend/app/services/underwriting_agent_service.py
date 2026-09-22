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

from backend.app.config import anthropic_api_key, anthropic_configured
from backend.app.services.review_reason_codes import catalog as reason_code_catalog

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5"
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


def configured() -> bool:
    return anthropic_configured()


def _client():
    import anthropic

    return anthropic.Anthropic(api_key=anthropic_api_key())


def generate_briefing(report: dict[str, Any]) -> dict[str, Any]:
    """One decision report in, one structured reviewer briefing out.

    Raises AgentUnavailableError (missing key / transport / rate limit — the
    caller answers 503) or AgentBriefingError (malformed model output — 502).
    """
    if not configured():
        raise AgentUnavailableError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to enable the reviewer briefing agent."
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

    try:
        response = _client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": BRIEFING_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Brief the reviewer on this application:\n"
                        + json.dumps(user_payload, indent=1, default=str)
                    ),
                }
            ],
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
    if not text:
        raise AgentBriefingError("The model returned no text content.")
    try:
        briefing = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentBriefingError(f"The model returned non-JSON output: {text[:200]}") from exc

    briefing["model"] = response.model
    briefing["advisoryOnly"] = True
    return briefing
