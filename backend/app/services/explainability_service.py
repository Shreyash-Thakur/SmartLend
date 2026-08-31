from __future__ import annotations

from typing import Any

from backend.app.models import LoanApplication

# NOTE (2026-08-31): this module is a PRESENTATION layer. It does not compute
# SHAP — `ml_service.MLPredictor.predict_application` does, and hands the top-3
# down through `_decision_meta["shap_explanation"]`.
#
# Vocabulary: FEATURE_LABELS and the counterfactual tables below now key on the
# 15 `feature_names` of the live serving artifact
# (backend/artifacts/pipeline_v3_real.joblib): age, dependents, years_employed,
# annual_income, monthly_income, existing_emis, cibil_score, total_loans,
# active_loans, closed_loans, missed_payments, credit_utilization_ratio,
# debt_to_income_ratio, loan_amount, loan_income_ratio. The old India-specific
# keys (emi_income_ratio, credit_component, asset_component, ...) survive only
# in the heuristic fallback below, which is built on CBES component names and
# engineered features rather than on the artifact's feature vocabulary.
#
# CAVEAT (inherent, not a bug): SHAP explains
# `pipeline.named_steps["model"]` — the plain LogisticRegression — while the
# served probability comes from the sibling `CalibratedClassifierCV(isotonic)`.
# Attributions therefore rank the drivers faithfully but do NOT decompose the
# served probability; isotonic calibration is a non-linear monotone map and no
# additive decomposition survives it. Documented in
# docs/DEFENCE-SHAP-CBES.md §1.6 (item 1) and §3.1.

# --- Live serving vocabulary (all 15 artifact features) ---------------------
FEATURE_LABELS = {
    "age": "Age",
    "dependents": "Dependents",
    "years_employed": "Years Employed",
    "annual_income": "Annual Income",
    "monthly_income": "Monthly Income",
    "existing_emis": "Existing EMIs",
    "cibil_score": "CIBIL Score",
    "total_loans": "Total Loans",
    "active_loans": "Active Loans",
    "closed_loans": "Closed Loans",
    "missed_payments": "Missed Payments",
    "credit_utilization_ratio": "Credit Utilization Ratio",
    "debt_to_income_ratio": "Debt To Income Ratio",
    "loan_amount": "Loan Amount",
    "loan_income_ratio": "Loan To Income Ratio",
    # --- legacy / CBES-component keys, still emitted by the heuristic fallback
    "emi_income_ratio": "EMI To Income Ratio",
    "credit_component": "Credit Strength",
    "capacity_component": "Repayment Capacity",
    "asset_component": "Collateral And Liquidity",
    "stability_component": "Employment Stability",
    "missed_payment_ratio": "Missed Payment Ratio",
}


def _impact_sign_for_decision(decision: str, raw_impact: float) -> float:
    # Negative impact means risk-increasing; flip for approve view so top factors align with final decision semantics.
    return raw_impact if decision == "REJECT" else -raw_impact


def _to_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


# Features for which a counterfactual is meaningless or the attribute is
# immutable/not actionable by the applicant. No suggestion is emitted for these
# — "improve Age from 35 to 35" is worse than saying nothing.
#   age, dependents      : immutable / not a lending lever we may ask them to pull
#   total_loans          : historical count (= active + closed); cannot be reduced
#   closed_loans         : already-settled history; cannot be un-closed
IMMUTABLE_FEATURES = frozenset({"age", "dependents", "total_loans", "closed_loans"})

# feature -> (absolute target, "higher" | "lower")
_ABSOLUTE_TARGETS: dict[str, tuple[float, str]] = {
    "cibil_score": (720.0, "higher"),
    "years_employed": (5.0, "higher"),
    "missed_payments": (0.0, "lower"),
    "active_loans": (1.0, "lower"),
    "existing_emis": (0.0, "lower"),
    "credit_utilization_ratio": (0.30, "lower"),
    "debt_to_income_ratio": (0.35, "lower"),
    # legacy / CBES-component keys used by the heuristic fallback
    "emi_income_ratio": (0.35, "lower"),
    "missed_payment_ratio": (0.08, "lower"),
    "credit_component": (0.60, "higher"),
    "capacity_component": (0.60, "higher"),
    "asset_component": (0.55, "higher"),
    "stability_component": (0.55, "higher"),
}

# feature -> (multiplicative factor, "higher" | "lower"). Used where no absolute
# target is meaningful because the quantity has no natural scale (rupee amounts,
# and a loan/income ratio whose Home Credit median is ~3.3, not ~0.65).
_RELATIVE_TARGETS: dict[str, tuple[float, str]] = {
    "annual_income": (1.20, "higher"),
    "monthly_income": (1.20, "higher"),
    "loan_amount": (0.80, "lower"),
    "loan_income_ratio": (0.80, "lower"),
}


def _counterfactual_target(feature: str, value: float) -> float | None:
    """Target value the applicant would need to reach for this factor.

    Returns ``None`` when no meaningful counterfactual exists — the feature is
    immutable, is unknown to the tables, or the applicant is already at/past the
    target. Callers MUST omit the suggestion in that case rather than emit a
    zero-delta one.

    This is a hand-written lookup, not a model-derived counterfactual: nothing is
    re-scored. See docs/DEFENCE-SHAP-CBES.md §1.6 (item 6).
    """
    if feature in IMMUTABLE_FEATURES:
        return None

    target: float | None = None
    direction: str | None = None

    if feature in _ABSOLUTE_TARGETS:
        target, direction = _ABSOLUTE_TARGETS[feature]
    elif feature in _RELATIVE_TARGETS:
        factor, direction = _RELATIVE_TARGETS[feature]
        if value == 0.0:
            return None  # 0 * factor == 0 -> zero delta
        target = float(value) * factor

    if target is None or direction is None:
        return None

    # Already at or beyond the target: nothing to suggest.
    if direction == "higher" and target <= value:
        return None
    if direction == "lower" and target >= value:
        return None

    return float(target)


def _build_top_factors(app_item: LoanApplication) -> tuple[list[dict[str, Any]], str]:
    """Returns (top factors, source) where source is "shap" or "heuristic".

    "shap" means the factors were derived from the SHAP attributions computed in
    ml_service. "heuristic" means SHAP was unavailable (explainer failed to build,
    or `shap_values` raised) and the hand-written rule table below produced them.
    The two have the same output shape; only this flag distinguishes them, so it
    must be carried all the way to the UI.
    """
    data = app_item.input_data or {}
    meta = data.get("_decision_meta", {}) if isinstance(data.get("_decision_meta", {}), dict) else {}
    engineered = meta.get("engineered_features", {}) if isinstance(meta.get("engineered_features", {}), dict) else {}
    components = meta.get("cbes_components", {}) if isinstance(meta.get("cbes_components", {}), dict) else {}
    shap_explanation = meta.get("shap_explanation", []) if isinstance(meta.get("shap_explanation", []), list) else []

    if shap_explanation:
        normalized: list[dict[str, Any]] = []
        for item in shap_explanation:
            feature = str(item.get("feature", item.get("name", "feature"))) if isinstance(item, dict) else "feature"
            impact = float(item.get("impact", 0.0)) if isinstance(item, dict) else 0.0
            value = float(item.get("value", 0.0)) if isinstance(item, dict) else 0.0
            direction_impact = _impact_sign_for_decision(app_item.final_decision, impact)
            
            label = _to_label(feature)
            if app_item.final_decision == "APPROVE":
                reason = f"Applicant's {label} ({value}) strongly supports the approval." if direction_impact >= 0 else f"Applicant's {label} ({value}) was a slight risk factor."
            elif app_item.final_decision == "REJECT":
                reason = f"Applicant's {label} ({value}) strongly contributed to the rejection." if direction_impact >= 0 else f"Applicant's {label} ({value}) was a positive factor, but insufficient."
            else:
                reason = f"Applicant's {label} ({value}) requires analyst review." if direction_impact >= 0 else f"Applicant's {label} ({value}) is generally favorable."

            target = _counterfactual_target(feature, value)
            normalized.append(
                {
                    "feature": feature,
                    "name": label,
                    "impact": round(direction_impact, 4),
                    "direction": "supports_decision" if direction_impact >= 0 else "opposes_decision",
                    "severity": round(min(1.0, abs(direction_impact) * 2.0), 4),
                    "value": round(value, 2),
                    # None => no meaningful counterfactual; downstream must omit
                    # the suggestion rather than render a zero-delta one.
                    "targetValue": None if target is None else round(target, 2),
                    "reason": reason,
                    "source": "shap",
                }
            )

        normalized.sort(key=lambda entry: abs(float(entry.get("impact", 0.0))), reverse=True)
        return normalized[:5], "shap"

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

    def _heuristic_entry(item: dict[str, Any]) -> dict[str, Any]:
        current = float(value_lookup.get(item["feature"], 0.0))
        signed = _impact_sign_for_decision(decision, float(item["impact"]))
        target = _counterfactual_target(item["feature"], current)
        return {
            "feature": item["feature"],
            "name": _to_label(item["feature"]),
            "impact": round(signed, 4),
            "direction": "supports_decision" if signed >= 0 else "opposes_decision",
            "severity": round(min(1.0, abs(float(item["impact"])) * 1.6), 4),
            "value": round(current, 4),
            "targetValue": None if target is None else round(target, 4),
            "reason": item["reason"],
            "source": "heuristic",
        }

    scored = [_heuristic_entry(item) for item in raw_factors]
    scored.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return scored[:5], "heuristic"


def _build_suggestions(app_item: LoanApplication, top_factors: list[dict[str, Any]]) -> list[str]:
    feature_set = {item["feature"] for item in top_factors}
    suggestions: list[str] = []

    if feature_set & {"debt_to_income_ratio", "emi_income_ratio", "existing_emis"}:
        suggestions.append("Reduce EMI or loan amount")
    if feature_set & {"loan_amount", "loan_income_ratio"}:
        suggestions.append("Request a smaller loan relative to income")
    if feature_set & {"annual_income", "monthly_income"}:
        suggestions.append("Evidence additional or co-applicant income")
    if feature_set & {"cibil_score", "credit_component", "credit_utilization_ratio"}:
        suggestions.append("Improve credit score")
    if feature_set & {"missed_payments", "missed_payment_ratio"}:
        suggestions.append("Maintain an unbroken on-time repayment record")
    if feature_set & {"active_loans"}:
        suggestions.append("Close one or more active credit lines before reapplying")
    if "asset_component" in feature_set:
        suggestions.append("Provide additional collateral or improve liquid balance")
    if feature_set & {"years_employed", "stability_component"}:
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
        if item["impact"] >= 0:
            continue
        raw_target = item.get("targetValue")
        if raw_target is None:
            # Immutable / unknown / already-at-target feature: omit entirely.
            # A "change X from 35 to 35" row is worse than no row at all.
            continue
        current = float(item.get("value", 0.0))
        target = float(raw_target)
        delta = round(target - current, 4)
        if delta == 0.0:
            continue
        recs.append(
            {
                "feature": item["feature"],
                "name": item["name"],
                "current": round(current, 4),
                "target": round(target, 4),
                "delta": delta,
                "priority": "high" if abs(item["impact"]) >= 0.2 else "medium",
            }
        )
    return recs[:3]


def build_explainability_payload(app_item: LoanApplication) -> dict[str, Any]:
    data = app_item.input_data or {}
    meta = data.get("_decision_meta", {}) if isinstance(data.get("_decision_meta", {}), dict) else {}

    top_factors, explanation_source = _build_top_factors(app_item)
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

    if explanation_source == "shap":
        explanation_text = (
            "Top factors are SHAP attributions from the serving logistic-regression model, "
            "re-signed to show which features supported the decision actually taken. "
            "They rank the drivers of the model's log-odds; they do not sum to the displayed risk score, "
            "because the served probability comes from the isotonic-calibrated sibling model."
        )
    else:
        explanation_text = (
            "SHAP attributions were unavailable for this application, so top factors come from a "
            "hand-written rule table over the CBES components and engineered features. "
            "These are NOT SHAP values."
        )

    return {
        "id": app_item.id,
        "decision": app_item.final_decision,
        "topFactors": top_factors,
        # "shap" | "heuristic" — which mechanism actually produced topFactors.
        "explanationSource": explanation_source,
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
