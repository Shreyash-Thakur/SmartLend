from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
import csv

from backend.app.services.ml_service import DATASET_PATH
from backend.app.services.ml_service import dynamic_hybrid_decision


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_loan_purpose(value: str) -> str:
    normalized = str(value or "personal").strip().lower()
    allowed = {"home", "auto", "personal", "business", "education"}
    return normalized if normalized in allowed else "personal"


def _cbes_probability(row: dict[str, Any]) -> float:
    cibil_score = _to_float(row.get("cibil_score"), 650.0)
    missed_payments = _to_float(row.get("missed_payments"))
    total_loans = _to_float(row.get("total_loans"))
    credit_utilization = _to_float(row.get("credit_utilization_ratio"))
    debt_to_income = _to_float(row.get("debt_to_income_ratio"))
    emi_income = _to_float(row.get("emi_income_ratio"))
    loan_income = _to_float(row.get("loan_income_ratio"))
    asset_coverage = _to_float(row.get("total_assets")) / max(_to_float(row.get("loan_amount")), 1.0)
    liquidity_ratio = _to_float(row.get("bank_balance")) / max(_to_float(row.get("loan_amount")), 1.0)
    employment_stability = _to_float(row.get("years_employed")) / max(_to_float(row.get("age")), 1.0)

    missed_payment_ratio = missed_payments / (total_loans + 1.0)
    cibil_norm = max(0.0, min((cibil_score - 300.0) / 600.0, 1.0))
    payment_penalty = max(0.0, min(1.0 - missed_payment_ratio, 1.0))
    util_penalty = max(0.0, min(1.0 - credit_utilization, 1.0))
    credit_component = 0.5 * cibil_norm + 0.3 * payment_penalty + 0.2 * util_penalty

    dti_score = max(0.2, min(1.0 - debt_to_income, 1.0))
    emi_score = max(0.0, min(1.0 - emi_income, 1.0))
    loan_income_score = max(0.0, min(1.0 - loan_income, 1.0))
    capacity_component = 0.5 * dti_score + 0.3 * emi_score + 0.2 * loan_income_score

    asset_score = max(0.0, min(asset_coverage, 2.0)) / 2.0
    liquidity_score = max(0.0, min(liquidity_ratio, 1.0))
    asset_component = 0.7 * asset_score + 0.3 * liquidity_score
    stability_component = max(0.0, min(employment_stability, 1.0))

    cbes_score = (
        0.35 * max(0.0, min(credit_component, 1.0))
        + 0.3 * max(0.0, min(capacity_component, 1.0))
        + 0.25 * max(0.0, min(asset_component, 1.0))
        + 0.1 * stability_component
    )
    return max(0.0, min(cbes_score, 1.0))


def _ml_probability(row: dict[str, Any]) -> float:
    confidence = _to_float(row.get("confidence_score"), 0.5)
    approved = _to_int(row.get("loan_approved"))
    if approved == 1:
        return max(0.5, min(confidence, 1.0))
    return max(0.0, min(1.0 - confidence, 0.5))


def _final_decision(ml_prob: float, cbes_prob: float) -> str:
    final_decision, _, _, _ = dynamic_hybrid_decision(ml_prob, cbes_prob)
    return final_decision


def _status_for_decision(decision: str) -> str:
    if decision == "APPROVE":
        return "approved"
    if decision == "REJECT":
        return "rejected"
    return "deferred"


def _training_timestamp(index: int) -> datetime:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(minutes=index)


def _build_training_application(row: dict[str, Any], index: int) -> dict[str, Any]:
    applicant_id = str(row.get("applicant_id") or f"TRAIN{index + 1:06d}")
    ml_prob = _ml_probability(row)
    cbes_prob = _cbes_probability(row)
    final_decision = _final_decision(ml_prob, cbes_prob)
    confidence = max(0.0, min(abs(ml_prob - 0.5), 1.0))
    created_at = _training_timestamp(index)

    application_data = {
        "firstName": "Applicant",
        "lastName": applicant_id,
        "gender": str(row.get("gender", "other")).lower(),
        "maritalStatus": str(row.get("marital_status", "single")).lower(),
        "education": str(row.get("education", "")).strip().lower().replace("-", "_"),
        "monthlyIncome": _to_float(row.get("monthly_income")),
        "annualIncome": _to_float(row.get("annual_income")),
        "emi": _to_float(row.get("emi")),
        "existingEmis": _to_float(row.get("existing_emis")),
        "employmentType": str(row.get("employment_type", "salaried")).lower(),
        "yearsOfEmployment": _to_int(row.get("years_employed")),
        "assets": _to_float(row.get("total_assets")),
        "totalAssets": _to_float(row.get("total_assets")),
        "residentialAssetsValue": _to_float(row.get("residential_assets_value")),
        "commercialAssetsValue": _to_float(row.get("commercial_assets_value")),
        "bankBalance": _to_float(row.get("bank_balance")),
        "creditScore": _to_int(row.get("cibil_score")),
        "cibilScore": _to_int(row.get("cibil_score")),
        "totalLoans": _to_int(row.get("total_loans")),
        "activeLoans": _to_int(row.get("active_loans")),
        "closedLoans": _to_int(row.get("closed_loans")),
        "missedPayments": _to_int(row.get("missed_payments")),
        "creditUtilizationRatio": _to_float(row.get("credit_utilization_ratio")),
        "emiIncomeRatio": _to_float(row.get("emi_income_ratio")) * 100,
        "loanIncomeRatio": _to_float(row.get("loan_income_ratio")) * 100,
        "debtToIncomeRatio": _to_float(row.get("debt_to_income_ratio")) * 100,
        "age": _to_int(row.get("age"), 30),
        "dependents": _to_int(row.get("dependents")),
        "region": str(row.get("region", "unknown")).lower(),
        "city": str(row.get("city", "Unknown")).strip(),
    }

    return {
        "id": f"train-{applicant_id}",
        "createdAt": created_at,
        "updatedAt": created_at,
        "status": _status_for_decision(final_decision),
        "source": "seed",
        "applicantId": applicant_id,
        "applicantName": f"Applicant {applicant_id}",
        "email": f"{applicant_id.lower()}@training.local",
        "phone": "N/A",
        "loanAmount": _to_float(row.get("loan_amount")),
        "loanPurpose": _normalize_loan_purpose(str(row.get("loan_type", "personal"))),
        "loanTenure": _to_int(row.get("loan_term"), 36),
        "interestRate": _to_float(row.get("interest_rate"), 12.0),
        "ml_prob": round(ml_prob, 4),
        "cbes_prob": round(cbes_prob, 4),
        "cbes_score": round(cbes_prob, 4),
        "confidence": round(confidence, 4),
        "decisionCode": final_decision,
        "finalDecision": final_decision,
        "applicationData": application_data,
        "decision": {
            "id": f"dec-train-{applicant_id}",
            "status": _status_for_decision(final_decision),
            "decidedAt": created_at,
            "decidedBy": "model",
            "riskScore": round(max(0.0, min(1.0 - ml_prob, 1.0)), 4),
            "cbessScore": round(cbes_prob * 100, 2),
            "uncertainty": round(max(0.0, min(1.0 - confidence, 1.0)), 4),
            "confidence": "high" if confidence >= 0.75 else "medium" if confidence >= 0.55 else "low",
            "explanation": "Converted from training dataset row for unified dashboard view.",
            "positiveFactors": [],
            "negativeFactors": [],
            "featureImportance": [],
            "modelVersion": "cbes-v2",
        },
        "documents": [],
    }


@lru_cache(maxsize=1)
def get_training_applications() -> list[dict[str, Any]]:
    dataset = Path(DATASET_PATH)
    if not dataset.exists():
        return []

    applications: list[dict[str, Any]] = []
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            applications.append(_build_training_application(row, index))

    return applications


def get_training_application_by_id(application_id: str) -> dict[str, Any] | None:
    for item in get_training_applications():
        if item["id"] == application_id:
            return item
    return None
