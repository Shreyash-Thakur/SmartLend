from __future__ import annotations

from typing import Any

from backend.app.models import LoanApplication


FEATURE_LABELS = {
    "debt_to_income_ratio": "Debt To Income Ratio",
    "emi_income_ratio": "EMI To Income Ratio",
    "cibil_score": "CIBIL Score",
    "credit_component": "Credit Strength",
    "capacity_component": "Repayment Capacity",
    "asset_component": "Collateral And Liquidity",
    "stability_component": "Employment Stability",
    "missed_payment_ratio": "Missed Payment Ratio",
    "loan_income_ratio": "Loan To Income Ratio",
}


def _impact_sign_for_decision(decision: str, raw_impact: float) -> float:
    # Negative impact means risk-increasing; flip for approve view so top factors align with final decision semantics.
    return raw_impact if decision == "REJECT" else -raw_impact


def _to_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _counterfactual_target(feature: str, value: float) -> float:
    targets = {
        "debt_to_income_ratio": 0.4,
        "emi_income_ratio": 0.35,
        "cibil_score": 720,
        "credit_component": 0.6,
        "capacity_component": 0.6,
        "asset_component": 0.55,
        "stability_component": 0.55,
        "missed_payment_ratio": 0.08,
        "loan_income_ratio": 0.65,
    }
    return float(targets.get(feature, value))


def _build_top_factors(app_item: LoanApplication) -> list[dict[str, Any]]:
    data = app_item.input_data or {}
    meta = data.get("_decision_meta", {}) if isinstance(data.get("_decision_meta", {}), dict) else {}
    engineered = meta.get("engineered_features", {}) if isinstance(meta.get("engineered_features", {}), dict) else {}
    components = meta.get("cbes_components", {}) if isinstance(meta.get("cbes_components", {}), dict) else {}
    shap_explanation = meta.get("shap_explanation", []) if isinstance(meta.get("shap_explanation", []), list) else []

    if shap_explanation:
        normalized: list[dict[str, Any]] = []
        for item in shap_explanation:
            feature = str(item.get("feature", "feature")) if isinstance(item, dict) else "feature"
            impact = float(item.get("impact", 0.0)) if isinstance(item, dict) else 0.0
            direction_impact = _impact_sign_for_decision(app_item.final_decision, impact)
            normalized.append(
                {
                    "feature": feature,
                    "name": _to_label(feature),
                    "impact": round(direction_impact, 4),
                    "direction": "supports_decision" if direction_impact >= 0 else "opposes_decision",
                    "severity": round(min(1.0, abs(direction_impact) * 2.0), 4),
                    "value": 0.0,
                    "targetValue": 0.0,
                    "reason": f"{_to_label(feature)} contributes {'positively' if direction_impact >= 0 else 'negatively'} to the decision.",
                }
            )

        normalized.sort(key=lambda entry: abs(float(entry.get("impact", 0.0))), reverse=True)
        return normalized[:5]

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
    value_lookup = {
        "debt_to_income_ratio": dti,
        "emi_income_ratio": emi_ratio,
        "cibil_score": cibil,
        "credit_component": float(components.get("credit_component", 0.5)),
        "capacity_component": float(components.get("capacity_component", 0.5)),
        "asset_component": float(components.get("asset_component", max(0.0, min(asset_coverage / 2, 1.0)))),
        "stability_component": float(components.get("stability_component", employment_stability)),
        "missed_payment_ratio": missed_ratio,
        "loan_income_ratio": loan_income,
    }

    scored = [
        {
            "feature": item["feature"],
            "name": _to_label(item["feature"]),
            "impact": round(_impact_sign_for_decision(decision, float(item["impact"])), 4),
            "direction": "supports_decision" if _impact_sign_for_decision(decision, float(item["impact"])) >= 0 else "opposes_decision",
            "severity": round(min(1.0, abs(float(item["impact"])) * 1.6), 4),
            "value": round(float(value_lookup.get(item["feature"], 0.0)), 4),
            "targetValue": round(_counterfactual_target(item["feature"], float(value_lookup.get(item["feature"], 0.0))), 4),
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


def _build_counterfactuals(top_factors: list[dict[str, Any]], decision: str) -> list[dict[str, Any]]:
    if decision == "APPROVE":
        return []

    recs: list[dict[str, Any]] = []
    for item in top_factors:
        if item["impact"] < 0:
            current = float(item.get("value", 0.0))
            target = float(item.get("targetValue", current))
            recs.append(
                {
                    "feature": item["feature"],
                    "name": item["name"],
                    "current": round(current, 4),
                    "target": round(target, 4),
                    "delta": round(target - current, 4),
                    "priority": "high" if abs(item["impact"]) >= 0.2 else "medium",
                }
            )
    return recs[:3]


def build_explainability_payload(app_item: LoanApplication) -> dict[str, Any]:
    data = app_item.input_data or {}
    meta = data.get("_decision_meta", {}) if isinstance(data.get("_decision_meta", {}), dict) else {}

    top_factors = _build_top_factors(app_item)
    suggestions = _build_suggestions(app_item, top_factors)
    reasons = [factor["reason"] for factor in top_factors[:3]]
    positive_factors = [factor["reason"] for factor in top_factors if factor["impact"] > 0][:3]
    negative_factors = [factor["reason"] for factor in top_factors if factor["impact"] < 0][:3]
    counterfactuals = _build_counterfactuals(top_factors, app_item.final_decision)

    components = meta.get("cbes_components", {}) if isinstance(meta.get("cbes_components", {}), dict) else {}
    weights = meta.get("cbes_weights", {}) if isinstance(meta.get("cbes_weights", {}), dict) else {}

    credit_component = float(components.get("credit_component", components.get("credit", 0.0)))
    capacity_component = float(components.get("capacity_component", components.get("capacity", 0.0)))
    behaviour_component = float(components.get("behaviour", 0.0))
    asset_component = float(components.get("asset_component", components.get("liquidity", 0.0)))
    stability_component = float(components.get("stability_component", components.get("stability", 0.0)))

    credit_weight = float(weights.get("credit", 0.35))
    capacity_weight = float(weights.get("capacity", 0.30))
    asset_weight = float(weights.get("asset", 0.25))
    stability_weight = float(weights.get("stability", 0.10))

    factor_buckets = {
        "credit": round(credit_component, 4),
        "capacity": round(capacity_component, 4),
        "behaviour": round(behaviour_component, 4),
        "collateral": round(asset_component, 4),
        "stability": round(stability_component, 4),
        "creditWeighted": round(credit_component * credit_weight, 4),
        "capacityWeighted": round(capacity_component * capacity_weight, 4),
        "behaviourWeighted": round(behaviour_component * 0.2, 4),
        "collateralWeighted": round(asset_component * asset_weight, 4),
        "stabilityWeighted": round(stability_component * stability_weight, 4),
    }

    explanation_text = (
        "Top factors are ranked by directional impact on the final hybrid decision, combining tuned ML probability with CBES components."
    )

    return {
        "id": app_item.id,
        "decision": app_item.final_decision,
        "topFactors": top_factors,
        "reasons": reasons,
        "positiveFactors": positive_factors,
        "negativeFactors": negative_factors,
        "suggestions": suggestions,
        "counterfactuals": counterfactuals,
        "factorBuckets": factor_buckets,
        "mlProb": round(app_item.ml_prob, 4),
        "cbesProb": round(app_item.cbes_prob, 4),
        "confidence": round(app_item.confidence, 4),
        "riskScore": round(1 - app_item.ml_prob, 4),
        "explanation": explanation_text,
        "modelVersion": "cbes-v2",
        "thresholds": {
            "approval": round(float(meta.get("approval_threshold", 0.5)), 4),
            "rejection": round(float(meta.get("rejection_threshold", 0.5)), 4),
        },
    }
