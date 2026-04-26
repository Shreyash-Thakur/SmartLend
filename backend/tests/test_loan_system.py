"""
test_loan_system.py
-------------------
Unit tests for the SmartLend hybrid loan scoring system.

Updated to align with the calibrated decision engine:
  - `hybrid_decision()` now accepts `t_base` parameter
  - Thresholds are derived from T_base, not a hard-coded 0.60 / 0.50
  - Confidence formula scales certainty 0→1 (×2 factor)
  - confidence gate is 0.18 (raised from 0.15)
"""
import pytest
import numpy as np
import pandas as pd
import math

from backend.app.services.cbes_engine import compute_cbes, DEFAULTS
from backend.app.services.decision_engine import hybrid_decision
from backend.app.services.calibrate import calibrate_tau_d, find_t_base

# ==============================================================================
# CBES Engine Boundary Tests
# ==============================================================================

def test_cbes_boundary_missing_values():
    """Ensure undefined payload falls back to absolute worst-case conservative constraints."""
    empty_payload = {}
    p_cbes, breakdown = compute_cbes(empty_payload)

    # Due to pessimistic boundaries (CIBIL 300, max utilization, massive debt defaults)
    # CBES should evaluate extremely poorly.
    assert 0.0 <= p_cbes <= 0.5, "Missing payload should evaluate strictly as high risk (low approval)."
    assert all(0.0 <= val <= 1.0 for val in breakdown.values())

def test_cbes_perfect_profile_bounds():
    """Evaluate absolute upper bounds tracking."""
    perfect = {
        "cibil_score": 900.0,
        "missed_payment_ratio": 0.0,
        "credit_utilization": 0.0,
        "gross_monthly_income": 100000.0,
        "net_monthly_income": 95000.0,
        "total_monthly_debt": 0.0,
        "monthly_emi": 0.0,
        "repayments_on_time_last_12": 12.0,
        "active_loans": 0.0,
        "total_loans": 5.0,
        "bank_balance": 5000000.0,
        "loan_amount": 10000.0,
        "total_assets": 10000000.0,
        "years_employed": 20.0,
        "age": 45.0
    }
    p_cbes, breakdown = compute_cbes(perfect)
    assert p_cbes > 0.8, "Perfect profile must yield highly optimal probability mappings."
    assert all(val > 0.5 for val in breakdown.values()), "All component transformations must recognize peak efficiency."

def test_cbes_zero_vectors():
    """Division-by-zero integrity tests using flat inputs."""
    flat = {k: 0.0 for k in DEFAULTS.keys()}
    # Safe float should absorb zeros securely without NaNs triggering exceptions
    p_cbes, breakdown = compute_cbes(flat)
    assert not math.isnan(p_cbes)

# ==============================================================================
# Decision Engine Tests (calibrated thresholds)
# ==============================================================================

def test_decision_branches_exhaustive():
    """Evaluates the strict precedence of the 5-gate decision engine paths.

    All tests use explicit t_base to make threshold arithmetic deterministic
    regardless of what is stored in the artifact.
    """
    T = 0.50   # t_base for most tests

    # 1. Disagreement > TAU_D → DEFER (disagreement)
    res1 = hybrid_decision(p_ml=0.9, p_cbes=0.1, tau_d=0.3, t_base=T)
    assert res1.decision == "DEFER"
    assert res1.decision_reason == "disagreement"

    # 2. ML approve: p_ml well above t_approve
    #    p_cbes=0.8 → tilt_clamped=0.15 → t_approve = 0.50 - 0.05*0.15 = 0.4925
    #    p_ml=0.9 >= 0.4925 → APPROVE
    res3 = hybrid_decision(p_ml=0.9, p_cbes=0.8, tau_d=0.3, t_base=T)
    assert res3.decision == "APPROVE"
    assert res3.decision_reason == "ml_approve"

    # 3. ML reject: p_ml well below t_reject
    #    p_cbes=0.2 → tilt=-0.3, clamped=-0.15 → t_reject = 0.50 + 0.05*(-0.15) = 0.4925
    #    p_ml=0.1 <= 0.4925 → REJECT
    res4 = hybrid_decision(p_ml=0.1, p_cbes=0.2, tau_d=0.3, t_base=T)
    assert res4.decision == "REJECT"
    assert res4.decision_reason == "ml_reject"

    # 4. CBES fallback reject:
    #    Need t_reject < p_ml < t_approve (gap exists when tilt_clamped < 0).
    #    p_cbes=0.30 → tilt=-0.20, clamped=-0.15
    #    With t_base=0.55: t_approve=0.5575, t_reject=0.5425
    #    p_ml=0.55: not >= 0.5575 (approve), not <= 0.5425 (reject) → CBES gate.
    #    p_cbes=0.30 <= 0.40 → cbes_fallback_reject.
    #    D=|0.55-0.30|=0.25 < tau_d=0.50 → passes gate 1.
    res5_rej = hybrid_decision(p_ml=0.55, p_cbes=0.30, tau_d=0.50, t_base=0.55)
    assert res5_rej.decision == "REJECT"
    assert res5_rej.decision_reason == "cbes_fallback_reject"

    # 5. Grey zone: ML between thresholds, CBES neutral (0.40-0.60)
    #    p_cbes=0.45 → tilt=-0.05, clamped=-0.05
    #    t_approve = 0.55 - 0.05*(-0.05) = 0.5525
    #    t_reject  = 0.55 + 0.05*(-0.05) = 0.5475
    #    p_ml=0.55: not >= 0.5525, not <= 0.5475 → CBES gate
    #    p_cbes=0.45: not >= 0.60, not <= 0.40 → grey_zone
    #    D=|0.55-0.45|=0.10 < tau_d=0.50 → passes disagreement gate
    res6 = hybrid_decision(p_ml=0.55, p_cbes=0.45, tau_d=0.50, t_base=0.55)
    assert res6.decision == "DEFER"
    assert res6.decision_reason == "grey_zone"

def test_confidence_formula_is_scaled():
    """Confidence components are scaled 0→1 (×2 factor), so max ML-only contribution is 0.60."""
    # p_ml=1.0, p_cbes=0.5, D=0.5 (agreement=0.5)
    # ml_certainty  = 0.60 * |1.0-0.5| * 2 = 0.60
    # cbes_certainty = 0.20 * |0.5-0.5| * 2 = 0.0
    # agreement = 0.20 * (1 - 0.5) = 0.10
    # total = 0.70
    res = hybrid_decision(p_ml=1.0, p_cbes=0.5, tau_d=0.3, t_base=0.5)
    assert abs(res.confidence - 0.70) < 0.01, f"Expected ~0.70, got {res.confidence}"

def test_threshold_shift_is_bounded():
    """CBES tilt shifts thresholds by at most ±0.0075 regardless of extreme CBES."""
    # Extreme favorable CBES (p_cbes=1.0 → tilt=0.5, clamped=0.15)
    res_high = hybrid_decision(p_ml=0.5, p_cbes=1.0, tau_d=0.9, t_base=0.50)
    # t_approve = 0.50 - 0.05*0.15 = 0.4925
    # t_reject  = 0.50 + 0.05*0.15 = 0.5075
    assert abs(res_high.t_approve - 0.4925) < 0.001
    assert abs(res_high.t_reject  - 0.5075) < 0.001

    # Extreme risky CBES (p_cbes=0.0 → tilt=-0.5, clamped=-0.15)
    res_low = hybrid_decision(p_ml=0.5, p_cbes=0.0, tau_d=0.9, t_base=0.50)
    # t_approve = 0.50 - 0.05*(-0.15) = 0.5075
    # t_reject  = 0.50 + 0.05*(-0.15) = 0.4925
    assert abs(res_low.t_approve - 0.5075) < 0.001
    assert abs(res_low.t_reject  - 0.4925) < 0.001

def test_no_collapsed_thresholds_on_realistic_data():
    """Guard against every non-deferred decision being the same outcome.

    With calibrated thresholds (t_base≈0.5, small tilt), a dataset with p_ml
    spread across [0.1, 0.9] should produce BOTH approve and reject decisions
    in the non-deferred set — not all-REJECT (the broken-threshold failure mode).
    """
    np.random.seed(99)
    N = 500
    # Bimodal: half strongly approve (p_ml 0.65-0.95), half strongly reject (p_ml 0.05-0.35)
    p_ml_approve = np.random.uniform(0.65, 0.95, N // 2)
    p_ml_reject  = np.random.uniform(0.05, 0.35, N // 2)
    p_ml   = np.concatenate([p_ml_approve, p_ml_reject])
    p_cbes = np.random.uniform(0.35, 0.65, N)   # neutral CBES — low tilt effect

    t_base = 0.50
    tau_d  = 0.30

    decisions = [hybrid_decision(float(pm), float(pc), tau_d, t_base).decision
                 for pm, pc in zip(p_ml, p_cbes)]
    labels    = np.array(decisions)

    non_mask = labels != "DEFER"
    assert np.any(non_mask), "All cases deferred — TAU_D too low or thresholds broken"

    non_decisions = labels[non_mask]
    # Both outcomes must appear — if only REJECT appears, thresholds have collapsed
    assert "APPROVE" in non_decisions, (
        f"No APPROVEs in non-deferred decisions ({len(non_decisions)} decided, "
        f"all={dict(zip(*np.unique(non_decisions, return_counts=True)))}). "
        "Threshold collapse: p_ml can never clear t_approve."
    )
    assert "REJECT" in non_decisions, "No REJECTs in non-deferred decisions"

    # Approve rate among non-deferred should be meaningfully non-zero
    approve_rate = np.mean(non_decisions == "APPROVE")
    assert approve_rate > 0.05, f"Approve rate too low ({approve_rate:.2%}) — likely threshold collapse"

# ==============================================================================
# Calibration Tests
# ==============================================================================

def test_find_t_base_returns_valid_range():
    """F1-optimal T_base must be in [0.30, 0.75] (clamped range)."""
    np.random.seed(7)
    probs = np.random.uniform(0.2, 0.8, 500)
    y     = (probs < 0.50).astype(int)   # high p_ml → approve → label 0
    t     = find_t_base(probs, y)
    assert 0.30 <= t <= 0.75, f"T_base={t} out of clamped range [0.30, 0.75]"

def test_calibrate_tau_target_bands():
    """Ensure tau calibration aligns strictly to evaluating bounds correctly."""
    np.random.seed(42)
    p_ml   = np.random.uniform(0, 1, 500)
    p_cbes = np.random.uniform(0, 1, 500)
    y_true = (p_ml < 0.5).astype(int)

    res = calibrate_tau_d(p_ml, p_cbes, y_true,
                          t_base=0.50,
                          lower_target=0.20,
                          upper_target=0.30)

    assert 0.10 <= res.tau_d <= 0.50, "Calibrated TAU_D violates system parameter limits."
    assert len(res.curve) > 0

def test_calibrate_t_base_kwarg_accepted():
    """calibrate_tau_d must accept t_base without raising."""
    np.random.seed(1)
    p_ml   = np.random.uniform(0.3, 0.7, 200)
    p_cbes = np.random.uniform(0.3, 0.7, 200)
    y_true = (p_ml < 0.5).astype(int)
    res = calibrate_tau_d(p_ml, p_cbes, y_true, t_base=0.48)
    assert res.t_base == 0.48

# ==============================================================================
# ML Prediction State Parity
# ==============================================================================

import os
from unittest.mock import patch
from backend.app.services.ml_service import MLPredictor

def test_ml_system_state_isolation_load(tmp_path):
    """
    Ensures that MLPredictor throws immediately if joblib is not present,
    enforcing exactly the 'no-retraining-at-startup' constraint.
    """
    with patch("backend.app.services.ml_service.PIPELINE_PATH", tmp_path / "non_existent.joblib"):
        with pytest.raises(FileNotFoundError):
            MLPredictor()

# ==============================================================================
# dynamic_hybrid_decision wrapper
# ==============================================================================

def test_dynamic_hybrid_decision_returns_four_tuple():
    """dynamic_hybrid_decision must return (decision, confidence, t_approve, t_reject)."""
    from backend.app.services.ml_service import dynamic_hybrid_decision
    result = dynamic_hybrid_decision(0.7, 0.6)
    assert len(result) == 4
    decision, confidence, t_approve, t_reject = result
    assert decision in {"APPROVE", "REJECT", "DEFER"}
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= t_approve <= 1.0
    assert 0.0 <= t_reject  <= 1.0
