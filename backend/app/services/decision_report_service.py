"""Per-application audit record: what the engine did, and what the human did.

`GET /api/applications/{id}/report` is the single place an auditor, a reviewer,
or a defence committee can see one application's whole decision in one payload:
the applicant data, the engine's probabilities and the thresholds they were
compared against, the CBES pillar breakdown, the SHAP top factors, and — if a
human has ruled on it — the reviewer's verdict, structured reason codes,
self-rated confidence, time spent, and whether they overrode the engine.

TWO INDEPENDENT HALVES. The engine half is always present; the human half is
`None` until a reviewer actually decides. "Not reviewed yet" is a normal state
for a deferred case, not an error, so the endpoint answers 200 with
`humanReview: null` rather than 404 — a report that fails whenever it is most
needed (right after the defer) would be useless.

READ-ONLY. This module builds a dict out of rows that already exist. It writes
nothing, decides nothing, and — like everything else touching `deferred_reviews`
— treats `human_decision` strictly as audit metadata, never as a label.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.app.models import DeferredReview, LoanApplication

# Imported, not re-declared: the blend weight must come from the engine so a
# derived p_blend in a report can never disagree with the engine's own.
from backend.app.services.decision_engine import _BLEND_ALPHA
from backend.app.services.decision_service import build_application_response
from backend.app.services.explainability_service import build_explainability_payload
from backend.app.services.review_reason_codes import describe_reason_codes


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    if isinstance(value, str) and value:
        return value
    return None


def _meta(app_item: LoanApplication) -> dict[str, Any]:
    data = app_item.input_data or {}
    meta = data.get("_decision_meta")
    return meta if isinstance(meta, dict) else {}


def build_engine_section(
    app_item: LoanApplication, review: DeferredReview | None
) -> dict[str, Any]:
    """The engine's side of the record.

    Where the same quantity exists on both the capture row and the stored
    decision metadata, the capture row wins: it was written from the live
    `DecisionResult` and carries fields (`p_blend`, `t_base`, `engine_version`,
    `threshold_artifact_hash`) that the application's `_decision_meta` never
    stored. For an application that was auto-decided — and so has no capture row
    — `p_blend` is reconstructed from the engine's own blend weight and flagged
    `pBlendSource: "derived"`, so a reader can tell a recorded number from a
    recomputed one.
    """
    meta = _meta(app_item)
    explain = build_explainability_payload(app_item)

    p_ml = _as_float(app_item.ml_prob)
    p_cbes = _as_float(app_item.cbes_prob)

    if review is not None:
        p_blend = _as_float(review.p_blend)
        p_blend_source = "captured"
    elif p_ml is not None and p_cbes is not None:
        p_blend = round((1.0 - _BLEND_ALPHA) * p_ml + _BLEND_ALPHA * p_cbes, 6)
        p_blend_source = "derived"
    else:
        p_blend = None
        p_blend_source = "unavailable"

    thresholds = {
        "approve": _as_float(review.t_approve if review is not None else meta.get("approval_threshold")),
        "reject": _as_float(review.t_reject if review is not None else meta.get("rejection_threshold")),
        "base": _as_float(review.t_base) if review is not None else None,
    }

    cbes_breakdown = (
        dict(review.cbes_breakdown_json or {})
        if review is not None and isinstance(review.cbes_breakdown_json, dict)
        else (meta.get("cbes_components") if isinstance(meta.get("cbes_components"), dict) else {})
    )

    return {
        "decision": app_item.final_decision,
        "decisionReason": str(
            review.decision_reason if review is not None else meta.get("decision_reason", "model_ensemble")
        ),
        "selectedModel": str(meta.get("selected_model", "unknown")),
        "engineVersion": str(review.engine_version) if review is not None else None,
        "thresholdArtifactHash": str(review.threshold_artifact_hash) if review is not None else None,
        "pMl": p_ml,
        "pCbes": p_cbes,
        "pBlend": p_blend,
        "pBlendSource": p_blend_source,
        "disagreement": _as_float(
            review.disagreement if review is not None else meta.get("disagreement")
        ),
        "confidence": _as_float(
            review.confidence if review is not None else app_item.confidence
        ),
        "confidenceLabel": str(meta.get("confidence_label", "")) or None,
        "riskScore": _as_float(meta.get("risk_score")),
        "thresholds": thresholds,
        "cbesBreakdown": {k: _as_float(v) for k, v in dict(cbes_breakdown or {}).items()},
        "cbesWeights": {
            k: _as_float(v)
            for k, v in dict(meta.get("cbes_weights") or {}).items()
        },
        "topFactors": list(explain.get("topFactors", [])),
        "explanation": str(explain.get("explanation", "")),
        "routedToHumanReview": bool(meta.get("routed_to_human_review", review is not None)),
        "explorationFlag": bool(
            review.exploration_flag if review is not None else meta.get("exploration_flag", False)
        ),
    }


def build_human_review_section(review: DeferredReview | None) -> dict[str, Any] | None:
    """The reviewer's side, or `None` when nobody has decided yet.

    A capture row with `human_decision IS NULL` is a case still sitting in the
    queue — it yields `None` here rather than a half-filled block, so the caller
    can render "awaiting review" without inspecting individual fields.
    """
    if review is None or review.human_decision is None:
        return None

    return {
        "reviewId": review.review_id,
        "reviewerId": review.reviewer_id,
        "decision": review.human_decision,
        "reviewedAt": _iso(review.reviewed_at),
        "reasonCodes": describe_reason_codes(review.human_reason_codes),
        "freeText": review.human_free_text,
        "reviewerConfidence": review.reviewer_confidence,
        "timeSpentSeconds": _as_float(review.time_spent_seconds),
        "agreedWithEngine": review.agreed_with_engine,
        "overrideDirection": review.override_direction,
        "explorationFlag": bool(review.exploration_flag),
        "outcomeCensored": bool(review.outcome_censored),
        "realizedOutcome": review.realized_outcome,
    }


def build_decision_report(
    app_item: LoanApplication, review: DeferredReview | None
) -> dict[str, Any]:
    """Assemble the full audit record for one application."""
    response = build_application_response(app_item)
    input_data = app_item.input_data or {}

    return {
        "applicationId": app_item.id,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "application": {
            "applicantId": app_item.applicant_id,
            "applicantName": response.get("applicantName"),
            "email": response.get("email"),
            "phone": response.get("phone"),
            "createdAt": _iso(app_item.created_at),
            "status": response.get("status"),
            "loanAmount": _as_float(response.get("loanAmount")),
            "loanPurpose": response.get("loanPurpose"),
            "loanTenureMonths": response.get("loanTenure"),
            "data": {k: v for k, v in input_data.items() if not str(k).startswith("_")},
        },
        "engine": build_engine_section(app_item, review),
        "humanReview": build_human_review_section(review),
        "analystNotes": response.get("analystNotes") or None,
        "manualDecisionApplied": bool(response.get("manualDecisionApplied")),
    }


def latest_review_for(session, application_id: str) -> DeferredReview | None:
    """The most recent capture row for an application, decided or not.

    Ordered newest-first so an application that was deferred, decided, and later
    re-deferred reports the review that matches its current state.
    """
    return (
        session.query(DeferredReview)
        .filter(DeferredReview.application_id == application_id)
        .order_by(DeferredReview.created_at.desc())
        .first()
    )
