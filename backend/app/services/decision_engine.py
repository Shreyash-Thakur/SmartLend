"""
decision_engine.py
------------------
Research-grade hybrid decision engine combining ML probability (p_ml)
and CBES probability (p_cbes) through a two-stage blend + 5-gate specification.

ROOT CAUSE FIX (2026-04-26):
  The original implementation used a static t_approve=0.60, causing nearly every
  non-deferred case to be rejected (100% recall, ~23% accuracy).
  Fix: T_base comes from F1-optimal validation sweep; CBES tilt applies ±0.0075.

STRUCTURAL DIVERGENCE FIX (2026-04-27):
  Stage A — Blend: p_blend = 0.75*p_ml + 0.25*p_cbes.
  Stage B — Gate comparisons use p_blend (not raw p_ml) so CBES nudges the
            effective signal, not just the thresholds. Disagreement D still
            uses raw signals for gating, preserving abstention sensitivity.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _confidence_label(confidence: float) -> str:
    """HIGH >= 0.75, MEDIUM >= 0.55, else LOW."""
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
    """Full output payload."""
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
    all_model_predictions: dict[str, float] = field(default_factory=dict)
    p_blend: float = 0.0                        # blended approval probability (observability)

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

# Blend weight: CBES contributes 25%, ML contributes 75%.
_BLEND_ALPHA = 0.25

# Provenance stamp written onto every relearning-loop capture row. Bump this
# whenever the gate logic, blend weight, or confidence formula above changes:
# it is what lets a future analysis separate a fixed router's labels from a
# broken router's (spec section 3, `engine_version`). Without it, capture rows
# from two different routers are indistinguishable and all of them are useless.
ENGINE_VERSION = "hybrid-2stage-5gate/2026-04-27+cbes-v2"

# ---------------------------------------------------------------------------
# Deferral routing mode (2026-08-31 deferral fix, see docs/DEFERRAL-FIX.md)
# ---------------------------------------------------------------------------
# The historical rule defers on raw disagreement D = |p_ml - p_cbes|, which is
# dominated by a fixed ~0.31 calibration offset between the two scores and was
# measured (research/deferral/evaluate.py, reports/deferral_fix.json) to defer
# the EASIEST cases: gate condition 1 z = +24/+28 on the held-out split.
# The fixed rule defers on ML uncertainty |p_ml - t_approve| < TAU_U, which
# scored z = -92.5 / -99.7 (negative = harder-than-random, i.e. correct) at a
# 22.7% deferral rate.
#
# DEFAULT IS THE OLD BEHAVIOUR. Nothing changes under the running demo unless
# the environment explicitly opts in:
#
#   SMARTLEND_DEFERRAL_MODE=uncertainty   # switch the router
#   SMARTLEND_TAU_U=0.2458                # optional half-width override
#
# When flipping the flag in production, bump ENGINE_VERSION alongside it so
# relearning-loop capture rows from the two routers stay separable.
_DEFERRAL_MODE_ENV = "SMARTLEND_DEFERRAL_MODE"
_TAU_U_ENV = "SMARTLEND_TAU_U"
DEFERRAL_MODE_DISAGREEMENT = "disagreement"   # legacy default
DEFERRAL_MODE_UNCERTAINTY = "uncertainty"     # measured fix

# Uncertainty half-width: the tune-split quantile that defers 22.5% of cases
# (the mid-point of the 20-25% underwriter-capacity band). Source of truth:
# reports/deferral_fix.json -> winner_detail.threshold_from_tune_split.
DEFAULT_TAU_U = 0.2458


def _resolve_deferral_mode(explicit: str | None) -> str:
    """Explicit argument wins; else the env flag; else the legacy default."""
    mode = (explicit or os.environ.get(_DEFERRAL_MODE_ENV, "")).strip().lower()
    if mode == DEFERRAL_MODE_UNCERTAINTY:
        return DEFERRAL_MODE_UNCERTAINTY
    return DEFERRAL_MODE_DISAGREEMENT


def _resolve_tau_u(explicit: float | None) -> float:
    """Explicit argument wins; else the env override; else the measured default."""
    if explicit is not None:
        return _clip(float(explicit), 0.0, 0.5)
    raw = os.environ.get(_TAU_U_ENV, "").strip()
    if raw:
        try:
            return _clip(float(raw), 0.0, 0.5)
        except ValueError:
            pass  # malformed override degrades to the measured default
    return DEFAULT_TAU_U


def hybrid_decision(
    p_ml: float,
    p_cbes: float,
    tau_d: float,
    t_base: float = 0.50,
    shap_explanation: list[dict[str, Any]] | None = None,
    cbes_breakdown: dict[str, float] | None = None,
    all_model_predictions: dict[str, float] | None = None,
    deferral_mode: str | None = None,
    tau_u: float | None = None,
) -> DecisionResult:
    """Execute the two-stage blend + 5-gate hybrid decision process.

    Parameters
    ----------
    p_ml : float
        ML model probability (approval likelihood, i.e. P(no default)).
    p_cbes : float
        CBES probability (approval likelihood).
    tau_d : float
        Disagreement threshold.  Clamped to [0.10, 0.90].
    t_base : float
        F1-optimal base threshold from validation calibration.
        Defaults to 0.50; the real value is loaded from the artifact.
    shap_explanation : list[dict], optional
        Top-3 SHAP features forwarded from ml_service.
    cbes_breakdown : dict, optional
        Per-component CBES scores forwarded from cbes_engine.
    deferral_mode : str, optional
        "disagreement" (legacy default) or "uncertainty" (measured fix, see
        docs/DEFERRAL-FIX.md). When None, the SMARTLEND_DEFERRAL_MODE
        environment variable decides; unset means legacy behaviour.
    tau_u : float, optional
        Half-width of the uncertainty deferral band around t_approve. When
        None, SMARTLEND_TAU_U or the measured default (0.2458) applies.
        Ignored in "disagreement" mode.
    """
    # -- type / range safety ------------------------------------------------
    p_ml   = _clip(float(p_ml),   0.0, 1.0)
    p_cbes = _clip(float(p_cbes), 0.0, 1.0)
    tau_d  = _clip(float(tau_d),  0.10, 0.90)
    # Upper bound widened 0.75 -> 0.98. With an 8% default rate a calibrated
    # p_ml concentrates near 0.92, so ANY balanced threshold lands above 0.75
    # and the old clamp silently clipped it back - which is why the system
    # approved ~99.5% of applicants and caught under 3% of defaults no matter
    # what threshold selection produced. Youden's J picks 0.925 here.
    t_base = _clip(float(t_base), 0.30, 0.98)

    # ── STAGE A — BLEND ─────────────────────────────────────────────────────
    # p_blend softens structural divergence between ML and CBES signals.
    # CBES contributes 25%; this nudges the effective approval signal without
    # overriding ML discrimination.
    p_blend = (1.0 - _BLEND_ALPHA) * p_ml + _BLEND_ALPHA * p_cbes

    # ── STAGE B — GATE LOGIC ────────────────────────────────────────────────

    # GATE 0 — DISAGREEMENT (computed on raw signals, not blend) ────────────
    D = abs(p_ml - p_cbes)

    # GATE 0b — CONFIDENCE --------------------------------------------------
    # Scaled 0→1: abs(x - 0.5) * 2 maps [0, 0.5] → [0, 1]
    confidence = (
        0.60 * abs(p_ml   - 0.5) * 2.0     # ML certainty
        + 0.20 * abs(p_cbes - 0.5) * 2.0   # CBES certainty
        + 0.20 * (1.0 - D)                  # agreement bonus
    )
    confidence = _clip(confidence, 0.0, 1.0)

    # GATE 1 — CALIBRATED THRESHOLDS ----------------------------------------
    # CBES tilt nudges T_base by at most ±0.0075 in either direction.
    tilt         = p_cbes - 0.5                            # ∈ [−0.5, +0.5]
    tilt_clamped = _clip(tilt, -0.15, 0.15)
    t_approve    = t_base - 0.05 * tilt_clamped            # max shift ±0.0075
    t_reject     = t_base + 0.05 * tilt_clamped

    # GATE 2 — DECISION LOGIC (mandatory order) -----------------------------
    # Threshold comparisons use p_blend so CBES nudges the effective decision
    # signal (not just the thresholds).  Reason strings kept for API compat.

    mode = _resolve_deferral_mode(deferral_mode)

    if mode == DEFERRAL_MODE_UNCERTAINTY:
        # MEASURED FIX (docs/DEFERRAL-FIX.md): defer on model uncertainty —
        # distance of p_ml from the actual decision boundary. This is exactly
        # the rule validated in research/deferral/evaluate.py (gate condition 1
        # z = -92.5/-99.7 on the held-out split); the auto decision below is
        # the same hard p_ml >= t_approve rule the evaluation scored, so the
        # measured properties carry over to production unchanged.
        tau_u_eff = _resolve_tau_u(tau_u)
        if abs(p_ml - t_approve) < tau_u_eff:
            decision = "DEFER"
            reason   = "ml_uncertainty"
        elif p_ml >= t_approve:
            decision = "APPROVE"
            reason   = "ml_approve"
        else:
            decision = "REJECT"
            reason   = "ml_reject"

    # Gate 2a: Disagreement gate (legacy default path)
    elif D > tau_d:
        decision = "DEFER"
        reason   = "disagreement"

    # Gate 2b: Low-confidence gate
    elif confidence < 0.18:
        decision = "DEFER"
        reason   = "low_confidence"

    # Gate 2c: Approve uses p_blend (CBES nudges upward bar for safer approvals)
    elif p_blend >= t_approve:
        decision = "APPROVE"
        reason   = "ml_approve"

    # Gate 2d: Reject uses raw p_ml (conservative; preserves recall on defaults)
    elif p_ml <= t_reject:
        decision = "REJECT"
        reason   = "ml_reject"

    # Gate 2d: Grey zone — bounded CBES tiebreak
    elif p_cbes >= 0.60:
        decision = "APPROVE"
        reason   = "cbes_fallback_approve"

    elif p_cbes <= 0.40:
        decision = "REJECT"
        reason   = "cbes_fallback_reject"

    # Gate 2e: True grey zone — defer
    else:
        decision = "DEFER"
        reason   = "grey_zone"

    # GATE 3 — OUTPUT -------------------------------------------------------
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
        p_blend=round(p_blend, 6),
        shap_explanation=shap_explanation or [],
        cbes_breakdown=cbes_breakdown or {},
        all_model_predictions=all_model_predictions or {},
    )
