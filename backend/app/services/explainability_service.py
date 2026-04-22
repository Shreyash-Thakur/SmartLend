from __future__ import annotations

from typing import Any

from backend.app.models import LoanApplication


def _impact_sign_for_decision(decision: str, raw_impact: float) -> float:
    # Negative impact means risk-increasing; flip for approve view so top factors align with final decision semantics.
    return raw_impact if decision == "REJECT" else -raw_impact


def _build_top_factors(app_item: LoanApplication) -> list[dict[str, Any]]:
    data = app_item.input_data or {}
    meta = data.get("_decision_meta", {}) if isinstance(data.get("_decision_meta", {}), dict) else {}
    engineered = meta.get("engineered_features", {}) if isinstance(meta.get("engineered_features", {}), dict) else {}
    components = meta.get("cbes_components", {}) if isinstance(meta.get("cbes_components", {}), dict) else {}

    dti = float(engineered.get("debt_to_income_ratio", 0))
    emi_ratio = float(engineered.get("emi_income_ratio", 0))
    loan_income = float(engineered.get("loan_income_ratio", 0))
    missed_ratio = float(engineered.get("missed_payment_ratio", 0))
    employment_stability = float(engineered.get("employment_stability", 0))
    asset_coverage = float(engineered.get("asset_coverage", 0))

    cibil = float(data.get("cibilScore", data.get("cibil_score", 650)) or 650)
    cibil_norm = max(0.0, min((cibil - 300) / 600, 1.0))

    raw_factors = [
        {
            "feature": "debt_to_income_ratio",
            "impact": dti - 0.45,
            "reason": "High debt burden relative to income" if dti > 0.45 else "Debt burden is within controllable range",
        },
        {
            "feature": "emi_income_ratio",
            "impact": emi_ratio - 0.4,
            "reason": "EMI consumes a large share of monthly income" if emi_ratio > 0.4 else "EMI load is manageable against monthly income",
        },
        {
            "feature": "cibil_score",
            "impact": 0.65 - cibil_norm,
            "reason": "Below recommended credit score" if cibil_norm < 0.65 else "Credit score supports repayment trust",
        },
        {
            "feature": "credit_component",
            "impact": 0.55 - float(components.get("credit_component", 0.5)),
            "reason": "Credit behavior weakens CBES credit component"
            if float(components.get("credit_component", 0.5)) < 0.55
            else "Credit behavior strengthens CBES credit component",
        },
        {
            "feature": "capacity_component",
            "impact": 0.55 - float(components.get("capacity_component", 0.5)),
            "reason": "Income capacity and leverage are below expected levels"
            if float(components.get("capacity_component", 0.5)) < 0.55
            else "Income capacity and leverage are favorable",
        },
        {
            "feature": "asset_component",
            "impact": 0.5 - float(components.get("asset_component", max(0.0, min(asset_coverage / 2, 1.0)))),
            "reason": "Asset and liquidity backing is limited"
            if float(components.get("asset_component", max(0.0, min(asset_coverage / 2, 1.0)))) < 0.5
            else "Asset and liquidity backing improves resilience",
        },
        {
            "feature": "stability_component",
            "impact": 0.5 - float(components.get("stability_component", employment_stability)),
            "reason": "Employment stability is limited"
            if float(components.get("stability_component", employment_stability)) < 0.5
            else "Employment stability supports continuity of repayments",
        },
        {
            "feature": "missed_payment_ratio",
            "impact": missed_ratio - 0.1,
            "reason": "Past missed payments indicate repayment volatility" if missed_ratio > 0.1 else "Past repayment behavior is stable",
        },
        {
            "feature": "loan_income_ratio",
            "impact": loan_income - 0.7,
            "reason": "Requested loan is high relative to annual income"
            if loan_income > 0.7
            else "Requested loan size is proportionate to income",
        },
    ]

    decision = app_item.final_decision
    scored = [
        {
            "feature": item["feature"],
            "impact": round(_impact_sign_for_decision(decision, float(item["impact"])), 4),
            "reason": item["reason"],
        }
        for item in raw_factors
    ]
    scored.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return scored[:5]


def _build_suggestions(app_item: LoanApplication, top_factors: list[dict[str, Any]]) -> list[str]:
    feature_set = {item["feature"] for item in top_factors}
    suggestions: list[str] = []

    if "debt_to_income_ratio" in feature_set or "emi_income_ratio" in feature_set:
        suggestions.append("Reduce EMI or loan amount")
    if "cibil_score" in feature_set or "credit_component" in feature_set:
        suggestions.append("Improve credit score")
    if "asset_component" in feature_set:
        suggestions.append("Provide additional collateral or improve liquid balance")
    if "stability_component" in feature_set:
        suggestions.append("Share stronger employment continuity proof")

    if not suggestions:
        if app_item.final_decision == "APPROVE":
            suggestions.append("Proceed with standard disbursement checks")
        elif app_item.final_decision == "DEFER":
            suggestions.append("Collect additional verification documents for analyst review")
        else:
            suggestions.append("Re-apply after improving repayment profile")

    return suggestions[:3]


def build_explainability_payload(app_item: LoanApplication) -> dict[str, Any]:
    data = app_item.input_data or {}
    meta = data.get("_decision_meta", {}) if isinstance(data.get("_decision_meta", {}), dict) else {}

    top_factors = _build_top_factors(app_item)
    suggestions = _build_suggestions(app_item, top_factors)
    reasons = [factor["reason"] for factor in top_factors[:3]]

    return {
        "id": app_item.id,
        "decision": app_item.final_decision,
        "topFactors": top_factors,
        "reasons": reasons,
        "suggestions": suggestions,
        "mlProb": round(app_item.ml_prob, 4),
        "cbesProb": round(app_item.cbes_prob, 4),
        "confidence": round(app_item.confidence, 4),
        "riskScore": round(1 - app_item.ml_prob, 4),
        "explanation": "Decision factors ranked from engineered ML + CBES components.",
        "modelVersion": "cbes-v2",
        "thresholds": {
            "approval": round(float(meta.get("approval_threshold", 0.5)), 4),
            "rejection": round(float(meta.get("rejection_threshold", 0.5)), 4),
        },
    }
