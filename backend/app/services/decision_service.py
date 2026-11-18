from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from backend.app.models import LoanApplication
from backend.app.services.decision_engine import decision_economics
from backend.app.services.explainability_service import build_explainability_payload


def _decision_to_status(decision: str) -> str:
    mapping = {
        "APPROVE": "approved",
        "REJECT": "rejected",
        "DEFER": "deferred",
    }
    return mapping.get(decision.upper(), "submitted")


_APPROVE_REASONS = {"ml_approve", "cbes_fallback_approve"}
_REJECT_REASONS = {"ml_reject", "cbes_fallback_reject", "affordability_reject"}


def engine_decision(app_item: LoanApplication) -> str:
    """What the engine decided, even after a reviewer overwrote final_decision.

    apply_manual_decision stores it as _engine_decision. Rows adjudicated before
    that existed recover it from the stored decision_reason; everything that is
    not an approve/reject reason was routed to review.
    """
    input_data = app_item.input_data or {}
    stored = input_data.get("_engine_decision")
    if stored:
        return str(stored)
    if not input_data.get("_manual_decision_applied"):
        return app_item.final_decision
    meta = input_data.get("_decision_meta") or {}
    reason = str(meta.get("decision_reason", "")) if isinstance(meta, dict) else ""
    if reason in _APPROVE_REASONS:
        return "APPROVE"
    if reason in _REJECT_REASONS:
        return "REJECT"
    return "DEFER"


def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _economics_payload(ml_prob: float) -> dict[str, Any]:
    """camelCase view of decision_engine.decision_economics for the API.

    Computed at response time from the stored ml_prob (a pure function of it
    and the wired cost constants), so rows persisted before this field existed
    show it too.
    """
    econ = decision_economics(float(ml_prob))
    return {
        "lossIfApproved": econ["loss_if_approved"],
        "profitForgoneIfRejected": econ["profit_forgone_if_rejected"],
        "lossRatio": econ["loss_ratio"],
        "lowerCostAction": econ["lower_cost_action"],
        "deferCutoff": econ["defer_cutoff"],
        "costFalseApproval": econ["cost_false_approval"],
        "costFalseRejection": econ["cost_false_rejection"],
    }


def _full_name(input_data: dict[str, Any]) -> str:
    first_name = str(input_data.get("firstName", "Customer")).strip()
    last_name = str(input_data.get("lastName", "Applicant")).strip()
    return f"{first_name} {last_name}".strip()


def _created_at(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    return dt


def build_application_response(app_item: LoanApplication) -> dict[str, Any]:
    input_data = app_item.input_data or {}
    meta = input_data.get("_decision_meta", {}) if isinstance(input_data.get("_decision_meta", {}), dict) else {}
    created_at = _created_at(app_item.created_at)
    model_recommendation = _decision_to_status(engine_decision(app_item))
    manual_decision_applied = bool(input_data.get("_manual_decision_applied", False))
    # Customer-facing status: APPROVE/REJECT are final at submit time and a
    # DEFER stays "deferred" until a human decides. A manual decision overwrites
    # final_decision, so status follows the human while modelRecommendation
    # keeps the engine's own verdict.
    status = _decision_to_status(app_item.final_decision)
    manual_decided_at = input_data.get("_manual_decided_at")

    risk_score = clamp_float(float(meta.get("risk_score", 1 - app_item.ml_prob)))
    confidence = clamp_float(app_item.confidence)
    confidence_label = str(meta.get("confidence_label", _confidence_band(confidence))).lower()
    documents = list(app_item.documents or [])
    explain_payload = build_explainability_payload(app_item)
    analyst_notes = str(input_data.get("_manual_notes", ""))

    # "shap" | "heuristic" — carried onto every feature row so the chart can label
    # itself honestly without the page needing to thread an extra prop.
    explanation_source = str(explain_payload.get("explanationSource", "shap"))

    def _base_value(item: dict[str, Any]) -> float:
        # targetValue is None when no meaningful counterfactual exists (immutable
        # feature, or already at target). Fall back to the current value so the
        # chart still has a number; the *suggestion* is omitted upstream.
        target = item.get("targetValue")
        if target is None:
            target = item.get("value", 0.0)
        return float(target)

    feature_importance = [
        {
            "name": str(item.get("name", "Feature")),
            "impact": float(item.get("impact", 0.0)),
            "value": float(item.get("value", 0.0)),
            "baseValue": _base_value(item),
            "source": str(item.get("source", explanation_source)),
        }
        for item in explain_payload.get("topFactors", [])
    ]

    return {
        "id": app_item.id,
        "responseStatus": "success",
        "createdAt": created_at,
        "updatedAt": created_at,
        "status": status,
        "source": "customer",
        "applicantId": app_item.applicant_id,
        "applicantName": _full_name(input_data),
        "email": str(input_data.get("email", "customer@example.com")),
        "phone": str(input_data.get("phone", "+91 9000000000")),
        "loanAmount": float(input_data.get("loanAmount", input_data.get("loan_amount", 0))),
        "loanPurpose": str(input_data.get("loanPurpose", input_data.get("loan_purpose", "personal"))),
        "loanTenure": int(input_data.get("loanTenure", input_data.get("loan_tenure", 36))),
        "interestRate": float(input_data.get("interestRate", input_data.get("interest_rate", 12.0))),
        "ml_prob": round(app_item.ml_prob, 4),
        "cbes_prob": round(app_item.cbes_prob, 4),
        "cbes_score": round(app_item.cbes_prob, 4),
        "confidence": confidence,
        "decisionCode": app_item.final_decision,
        "finalDecision": app_item.final_decision,
        "modelRecommendation": model_recommendation,
        "manualDecisionApplied": manual_decision_applied,
        "manualDecidedAt": manual_decided_at,
        # Exploration-arm cases keep an auto status but still need a reviewer;
        # the review queue keys off these instead of a blanket "submitted".
        "explorationFlag": bool(meta.get("exploration_flag", False)),
        "routedToHumanReview": bool(meta.get("routed_to_human_review", False)),
        "analystNotes": analyst_notes,
        "applicationData": input_data,
        "decision": {
            "id": f"dec-{uuid.uuid4().hex[:12]}",
            "status": status,
            "decidedAt": created_at,
            "decidedBy": "human" if manual_decision_applied else "model",
            "riskScore": risk_score,
            "cbessScore": round(app_item.cbes_prob * 100, 2),
            "uncertainty": clamp_float(1 - confidence),
            "confidence": confidence_label,
            "reason": str(meta.get("decision_reason", "model_ensemble")),
            "topFeatures": [str(f.get("name", "Unknown")) for f in meta.get("shap_explanation", [])],
            "cbesBreakdown": meta.get("cbes_components", {}),
            "explanation": analyst_notes if manual_decision_applied else str(explain_payload.get("explanation", "Dynamic hybrid ML + CBES decision applied.")),
            "positiveFactors": list(explain_payload.get("positiveFactors", [])),
            "negativeFactors": list(explain_payload.get("negativeFactors", [])),
            "featureImportance": feature_importance,
            "explanationSource": explanation_source,
            "explanationNote": str(explain_payload.get("explanationNote", "")),
            # Real provenance stamped at decision time; "cbes-v2" only for rows
            # written before engine_version reached _decision_meta.
            "modelVersion": str(meta.get("engine_version") or "cbes-v2"),
            "modelArtifact": meta.get("model_artifact"),
            "analystNotes": analyst_notes,
            "allModelPredictions": meta.get("all_model_predictions", {}),
            "decisionEconomics": _economics_payload(app_item.ml_prob),
        },
        "documents": documents,
    }



def apply_manual_decision(app_item: LoanApplication, status: str, notes: str) -> dict[str, Any]:
    decision_map = {
        "approved": "APPROVE",
        "rejected": "REJECT",
        "deferred": "DEFER",
    }
    mapped = decision_map.get(status.lower(), "DEFER")

    # Keep notes in persisted JSON for traceability without schema changes, and
    # remember the engine's verdict before final_decision is overwritten.
    input_data = dict(app_item.input_data or {})
    input_data.setdefault("_engine_decision", engine_decision(app_item))
    input_data["_manual_notes"] = notes
    input_data["_manual_decision_applied"] = True
    input_data["_manual_decided_at"] = datetime.now(timezone.utc).isoformat()
    app_item.input_data = input_data
    app_item.final_decision = mapped

    payload = build_application_response(app_item)
    payload["status"] = status.lower()
    payload["decision"]["status"] = status.lower()
    payload["decision"]["decidedBy"] = "human"
    payload["decision"]["explanation"] = notes
    payload["decision"]["analystId"] = "analyst-placeholder"
    payload["decision"]["analystNotes"] = notes
    return payload


def clamp_float(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_dashboard_metrics(applications: list[LoanApplication]) -> dict[str, int]:
    if not applications:
        return {
            "totalApplications": 0,
            "approved": 0,
            "rejected": 0,
            "deferred": 0,
            "averageProcessingTime": 120,
            "approvalRate": 0,
            "avgLoanAmount": 0,
            "automationRate": 100,
        }

    approved = sum(1 for item in applications if item.final_decision == "APPROVE")
    rejected = sum(1 for item in applications if item.final_decision == "REJECT")
    deferred = sum(1 for item in applications if item.final_decision == "DEFER")
    avg_loan = round(
        sum(float((item.input_data or {}).get("loanAmount", (item.input_data or {}).get("loan_amount", 0))) for item in applications)
        / len(applications)
    )

    return {
        "totalApplications": len(applications),
        "approved": approved,
        "rejected": rejected,
        "deferred": deferred,
        "averageProcessingTime": 120,
        "approvalRate": round((approved / len(applications)) * 100),
        "avgLoanAmount": avg_loan,
        "automationRate": 100,
    }
