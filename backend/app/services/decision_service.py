from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from backend.app.models import LoanApplication
from backend.app.services.explainability_service import build_explainability_payload


def _decision_to_status(decision: str) -> str:
    mapping = {
        "APPROVE": "approved",
        "REJECT": "rejected",
        "DEFER": "deferred",
    }
    return mapping.get(decision.upper(), "submitted")


def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


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
    created_at = _created_at(app_item.created_at)
    model_recommendation = _decision_to_status(app_item.final_decision)
    manual_decision_applied = bool(input_data.get("_manual_decision_applied", False))
    status = model_recommendation if manual_decision_applied else "submitted"

    risk_score = clamp_float(1 - app_item.ml_prob)
    confidence = clamp_float(app_item.confidence)
    documents = list(app_item.documents or [])
    explain_payload = build_explainability_payload(app_item)

    feature_importance = [
        {
            "name": str(item.get("name", "Feature")),
            "impact": float(item.get("impact", 0.0)),
            "value": float(item.get("value", 0.0)),
            "baseValue": float(item.get("targetValue", item.get("value", 0.0))),
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
        "applicationData": input_data,
        "decision": {
            "id": f"dec-{uuid.uuid4().hex[:12]}",
            "status": model_recommendation,
            "decidedAt": created_at,
            "decidedBy": "human" if manual_decision_applied else "model",
            "riskScore": risk_score,
            "cbessScore": round(app_item.cbes_prob * 100, 2),
            "uncertainty": clamp_float(1 - confidence),
            "confidence": _confidence_band(confidence),
            "explanation": str(explain_payload.get("explanation", "Dynamic hybrid ML + CBES decision applied.")),
            "positiveFactors": list(explain_payload.get("positiveFactors", [])),
            "negativeFactors": list(explain_payload.get("negativeFactors", [])),
            "featureImportance": feature_importance,
            "modelVersion": "cbes-v2",
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
    app_item.final_decision = mapped

    payload = build_application_response(app_item)
    payload["status"] = status.lower()
    payload["decision"]["status"] = status.lower()
    payload["decision"]["decidedBy"] = "human"
    payload["decision"]["explanation"] = notes
    payload["decision"]["analystId"] = "analyst-placeholder"
    payload["decision"]["analystNotes"] = notes

    # Keep notes in persisted JSON for traceability without schema changes.
    input_data = dict(app_item.input_data or {})
    input_data["_manual_notes"] = notes
    input_data["_manual_decision_applied"] = True
    app_item.input_data = input_data

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
