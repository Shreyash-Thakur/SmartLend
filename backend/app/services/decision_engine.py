"""
decision_engine.py
------------------
Research-grade hybrid decision engine combining ML probability (p_ml)
and CBES probability (p_cbes) through the exact 6-step specification.

Decision order, threshold formulas, and confidence weights must not be
altered. All five steps are implemented verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _confidence_label(confidence: float) -> str:
    """Step 5 label mapping: HIGH >= 0.75, MEDIUM >= 0.55, else LOW."""
    if confidence >= 0.75:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    """Full output payload as specified in Step 5."""
    decision: str
    confidence: float
    confidence_label: str
    risk_score: float                           # = 1 - p_ml
    p_ml: float
    p_cbes: float
    disagreement: float
    decision_reason: str
    t_approve: float
    t_reject: float
    shap_explanation: list[dict[str, Any]] = field(default_factory=list)
    cbes_breakdown: dict[str, float] = field(default_factory=dict)
    
    @property
    def ml_prob(self) -> float: return self.p_ml
    @property
    def cbes_prob(self) -> float: return self.p_cbes
    @property
    def final_decision(self) -> str: return self.decision
    @property
    def approval_threshold(self) -> float: return self.t_approve
    @property
    def rejection_threshold(self) -> float: return self.t_reject

# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def hybrid_decision(
    p_ml: float,
    p_cbes: float,
    tau_d: float,
    shap_explanation: list[dict[str, Any]] | None = None,
    cbes_breakdown: dict[str, float] | None = None,
) -> DecisionResult:
    """Execute the 6-step hybrid decision process exactly as specified.

    Parameters
    ----------
    p_ml : float
        ML model probability (approval likelihood).
    p_cbes : float
        CBES probability (approval likelihood).
    tau_d : float
        Disagreement threshold.  Clamped to [0.15, 0.45].
    shap_explanation : list[dict], optional
        Top-3 SHAP features forwarded from ml_service.
    cbes_breakdown : dict, optional
        Per-component CBES scores forwarded from cbes_engine.
    """
    # -- type safety --------------------------------------------------------
    p_ml  = _clip(float(p_ml),  0.0, 1.0)
    p_cbes = _clip(float(p_cbes), 0.0, 1.0)
    tau_d  = _clip(float(tau_d),  0.15, 0.45)

    # STEP 1 — DISAGREEMENT -------------------------------------------------
    D = abs(p_ml - p_cbes)

    # STEP 2 — CONFIDENCE ---------------------------------------------------
    confidence = (
        0.60 * abs(p_ml   - 0.5)
        + 0.20 * abs(p_cbes - 0.5)
        + 0.20 * (1.0 - D)
    )
    confidence = _clip(confidence, 0.0, 1.0)

    # STEP 3 — DYNAMIC THRESHOLDS -------------------------------------------
    # Exact spec: tightened boundaries restricting arbitrary approvals safely
    tilt      = p_cbes - 0.5
    t_approve = 0.60 - 0.10 * tilt
    t_reject  = 0.50 - 0.10 * tilt

    # STEP 4 — DECISION LOGIC (order is mandatory) --------------------------
    if D > tau_d:
        decision = "DEFER"
        reason   = "disagreement"
    elif confidence < 0.15:
        decision = "DEFER"
        reason   = "low_confidence"
    elif p_ml >= t_approve:
        decision = "APPROVE"
        reason   = "ml_approve"
    elif p_ml <= t_reject:
        decision = "REJECT"
        reason   = "ml_reject"
    elif p_cbes > 0.62:
        decision = "APPROVE"
        reason   = "cbes_fallback"
    elif p_cbes < 0.38:
        decision = "REJECT"
        reason   = "cbes_fallback"
    else:
        decision = "DEFER"
        reason   = "grey_zone"

    # STEP 5 — OUTPUT -------------------------------------------------------
    risk_score = 1.0 - p_ml

    return DecisionResult(
        decision=decision,
        confidence=round(confidence, 6),
        confidence_label=_confidence_label(confidence),
        risk_score=round(risk_score, 6),
        p_ml=round(p_ml, 6),
        p_cbes=round(p_cbes, 6),
        disagreement=round(D, 6),
        decision_reason=reason,
        t_approve=round(t_approve, 6),
        t_reject=round(t_reject, 6),
        shap_explanation=shap_explanation or [],
        cbes_breakdown=cbes_breakdown or {},
    )
