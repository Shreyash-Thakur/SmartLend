"""CBES: hand-designed heuristic risk score, redesigned for Home Credit fields.

Replaces the prior 15-key India-specific vocabulary (cibil_score,
residential_assets_value, ...), which has no Home Credit equivalent. This
version consumes exactly the 7 fields the canonical data layer
(research/data/canonical.py) can actually populate for Home Credit:
credit_score (EXT_SOURCE_2 proxy), delinquencies, active_loans (both from
bureau.csv via research/data/bureau_aggregates.py), dti, employment_tenure_years,
an income/loan-amount affordability ratio, and an optional region rule.

Thresholds are percentile breakpoints computed from the real training
distribution by cbes_calibration.py — see load_thresholds() — not hand-picked
bank conventions. EXT_SOURCE_2 has no established "prime/subprime" cutoff the
way a real bureau score does, so treating it as a real-world scale would be
scientifically dishonest; percentile bands make the rule set legible instead
("bottom 20% of applicants by this dataset's own score distribution").
"""

from __future__ import annotations

import math
from typing import Any

from backend.app.services.cbes_calibration import load_thresholds

# Conservative fallback defaults: as bad as observed data in this dataset gets,
# so a missing field never masks risk as neutral.
DEFAULTS: dict[str, float] = {
    "credit_score": 0.0,
    "delinquencies": 10.0,
    "active_loans": 10.0,
    "dti": 1.0,
    "employment_tenure_years": 0.0,
    "annual_income": 1.0,
    "loan_amount": 10_000_000.0,
    "region": 3.0,
}

# Loaded lazily, not at import time: Task 5 (cbes_calibration.py) may not have
# produced a real artifact yet when this module is first imported (e.g. during
# Task 4's own test collection), and this module must not crash on import
# because of that. Tests stub this via monkeypatch before compute_cbes runs;
# real callers get it filled in on first use.
_THRESHOLDS: dict[str, list[float]] | None = None


def _get_thresholds() -> dict[str, list[float]]:
    global _THRESHOLDS
    if _THRESHOLDS is None:
        _THRESHOLDS = load_thresholds()
    return _THRESHOLDS


def _safe_float(val: Any, default: float) -> float:
    if val is None:
        return default
    try:
        f = float(val)
    except (ValueError, TypeError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _percentile_score(value: float, breakpoints: list[float], higher_is_better: bool) -> float:
    """Map a raw value onto [0, 1] via 5 percentile breakpoints (p10/p30/p50/p70/p90).

    Values below breakpoints[0] or above breakpoints[-1] clip to 0.0/1.0.
    `higher_is_better=False` inverts the scale (e.g. more delinquencies = worse).
    """
    edges = breakpoints
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = float(_interp(value, edges, positions))
    return score if higher_is_better else 1.0 - score


def _interp(value: float, edges: list[float], positions: list[float]) -> float:
    if value <= edges[0]:
        return positions[0]
    if value >= edges[-1]:
        return positions[-1]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= value <= hi:
            if hi == lo:
                return positions[i]
            frac = (value - lo) / (hi - lo)
            return positions[i] + frac * (positions[i + 1] - positions[i])
    return positions[-1]


def component_sigmoid(x: float) -> float:
    """k=4: softer curve. For x in [0, 1] the output spans [0.1192, 0.8808].

    (Verified empirically: sigmoid(-4*0.5)=0.11920292, sigmoid(4*0.5)=0.88079708.
    An earlier version of this docstring claimed [0.27, 0.73]; that is the k=2
    span, not k=4. The code is and always was k=4 - only the comment was wrong.)
    """
    return 1.0 / (1.0 + math.exp(-4.0 * (x - 0.5)))


def compute_cbes(data: dict[str, Any]) -> tuple[float, dict[str, float]]:
    thresholds = _get_thresholds()
    credit_score = _safe_float(data.get("credit_score"), DEFAULTS["credit_score"])
    delinquencies = _safe_float(data.get("delinquencies"), DEFAULTS["delinquencies"])
    active_loans = _safe_float(data.get("active_loans"), DEFAULTS["active_loans"])
    dti = _safe_float(data.get("dti"), DEFAULTS["dti"])
    tenure = _safe_float(data.get("employment_tenure_years"), DEFAULTS["employment_tenure_years"])
    income = max(_safe_float(data.get("annual_income"), DEFAULTS["annual_income"]), 1.0)
    loan_amount = max(_safe_float(data.get("loan_amount"), DEFAULTS["loan_amount"]), 0.0)
    region = _safe_float(data.get("region"), DEFAULTS["region"])

    loan_to_income = loan_amount / income

    # 1. CREDIT (w=0.35): external score + delinquency history
    credit_raw = 0.70 * _percentile_score(
        credit_score, thresholds["credit_score"], higher_is_better=True
    ) + 0.30 * _percentile_score(
        delinquencies, thresholds["delinquencies"], higher_is_better=False
    )
    credit_final = component_sigmoid(credit_raw)

    # 2. CAPACITY (w=0.30): debt-to-income + loan-to-income affordability
    capacity_raw = 0.60 * _percentile_score(
        dti, thresholds["dti"], higher_is_better=False
    ) + 0.40 * _percentile_score(
        loan_to_income, thresholds["loan_to_income"], higher_is_better=False
    )
    capacity_final = component_sigmoid(capacity_raw)

    # 3. BEHAVIOUR (w=0.20): concurrent active credit lines
    behaviour_raw = _percentile_score(
        active_loans, thresholds["active_loans"], higher_is_better=False
    )
    behaviour_final = component_sigmoid(behaviour_raw)

    # 4. STABILITY (w=0.10): employment tenure
    stability_raw = _percentile_score(
        tenure, thresholds["employment_tenure_years"], higher_is_better=True
    )
    stability_final = component_sigmoid(stability_raw)

    # 5. REGION (w=0.05): urbanicity proxy, low weight, deliberately not a
    # geography claim (REGION_RATING_CLIENT is ordinal 1=best, 3=worst).
    region_raw = 1.0 - _percentile_score(region, [1, 1, 2, 3, 3], higher_is_better=True)
    region_final = component_sigmoid(region_raw)

    CBES_raw = (
        0.35 * credit_final
        + 0.30 * capacity_final
        + 0.20 * behaviour_final
        + 0.10 * stability_final
        + 0.05 * region_final
    )
    p_cbes = 1.0 / (1.0 + math.exp(-5.0 * (CBES_raw - 0.5)))

    breakdown = {
        "credit": float(credit_final),
        "capacity": float(capacity_final),
        "behaviour": float(behaviour_final),
        "stability": float(stability_final),
        "region": float(region_final),
    }
    return float(p_cbes), breakdown
