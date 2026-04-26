import pytest
import numpy as np
import pandas as pd
import math

from backend.app.services.cbes_engine import compute_cbes, DEFAULTS
from backend.app.services.decision_engine import hybrid_decision
from backend.app.services.calibrate import calibrate_tau_d

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
# Decision Engine Tests
# ==============================================================================

def test_decision_branches_exhaustive():
    """Evaluates the strict precedence of the 6 decision engine paths."""
    
    # 1. Disagreement > TAU_D
    res1 = hybrid_decision(p_ml=0.9, p_cbes=0.1, tau_d=0.3)
    assert res1.decision == "DEFER"
    assert res1.decision_reason == "disagreement"
    
    # 2. Low Confidence (< 0.15) 
    # Mathematically unreachable since minimum possible confidence = 0.20 under the provided formula.
    # res2 = hybrid_decision(p_ml=0.5, p_cbes=0.5, tau_d=0.3)
    # assert res2.decision == "DEFER"
    # assert res2.decision_reason == "low_confidence"
    
    # 3. Approve via ML strictly (high p_ml) + High confidence + Agreement
    res3 = hybrid_decision(p_ml=0.9, p_cbes=0.8, tau_d=0.3)
    assert res3.decision == "APPROVE"
    assert res3.decision_reason == "ml_approve"
    
    # 4. Reject via ML strictly (low p_ml) + High confidence + Agreement
    res4 = hybrid_decision(p_ml=0.1, p_cbes=0.2, tau_d=0.3)
    assert res4.decision == "REJECT"
    assert res4.decision_reason == "ml_reject"
    
    # 5. CBES Fallback Paths (ML is indecisive between T_approve and T_reject, but CBES takes a firm stance)
    # Ensure Tilt triggers explicitly.
    # Let p_cbes = 0.8. tilt = 0.3. t_approve = 0.55 - 0.03 = 0.52.
    # We want ML inside [t_reject, t_approve], say p_ml = 0.49
    # But wait, T_approve = 0.55 - 0.1*(0.8-0.5) = 0.52.
    # T_reject = 0.45 - 0.03 = 0.42.
    # We need p_ml between 0.42 and 0.52 to miss ML rules.
    # Confidence must be > 0.15. D = abs(0.48 - 0.8) = 0.32. If tau=0.4, it survives.
    res5_app = hybrid_decision(p_ml=0.48, p_cbes=0.80, tau_d=0.4)
    assert res5_app.decision == "APPROVE"
    assert res5_app.decision_reason == "cbes_fallback"
    
    # CBES Reject Fallback
    # let p_cbes = 0.2. tilt = -0.3. t_approve = 0.58. t_reject = 0.48.
    # ML = 0.50
    res5_rej = hybrid_decision(p_ml=0.50, p_cbes=0.20, tau_d=0.4)
    assert res5_rej.decision == "REJECT"
    assert res5_rej.decision_reason == "cbes_fallback"
    
    # 6. Grey Zone (Everything indecisive but passing confidence baseline)
    # To hit grey zone, ML is inside thresholds, CBES is inside [0.38, 0.62], confidence > 0.15, disagreement < tau_d.
    res6 = hybrid_decision(p_ml=0.49, p_cbes=0.51, tau_d=0.3)
    # Wait, confidence = 0.6*(0.01) + 0.2*(0.01) + 0.2*(0.98) = 0.006 + 0.002 + 0.196 = 0.204.
    # This survives confidence filter.
    # t_approve = 0.55 - 0.1*(0.01) = 0.549
    # t_reject = 0.45 - 0.001 = 0.449
    # ML is 0.49 (inside bounds). CBES is 0.51 (inside 0.38-0.62).
    assert res6.decision == "DEFER"
    assert res6.decision_reason == "grey_zone"

# ==============================================================================
# Calibration Tests
# ==============================================================================

def test_calibrate_tau_target_bands():
    """Ensure the tau calibration aligns strictly to evaluating bounds correctly."""
    np.random.seed(42)
    # Generate mock probabilities spanning all decisions uniformly
    p_ml = np.random.uniform(0, 1, 500)
    p_cbes = np.random.uniform(0, 1, 500)
    # Fake y_true mapped slightly accurately to ML mapping
    y_true = (p_ml < 0.5).astype(int) 
    
    res = calibrate_tau_d(p_ml, p_cbes, y_true, lower_target=0.20, upper_target=0.30)
    
    assert 0.15 <= res.tau_d <= 0.45, "Calibrated TAU_D violates system parameter limits."
    # Since we use complete random noise, disagreement is high, deferral should map easily.
    # The selector grabs the best overlap value mapping
    assert len(res.curve) > 0

# ==============================================================================
# ML Prediction State Parity
# ==============================================================================
import os
from unittest.mock import patch
from backend.app.services.ml_service import MLPredictor

def test_ml_system_state_isolation_load(tmp_path):
    """
    Ensures that MLPredictor throws immediately if joblib is not present 
    enforcing exactly the 'no-retraining-at-startup' constraint requirement.
    """
    with patch("backend.app.services.ml_service.PIPELINE_PATH", tmp_path / "non_existent.joblib"):
        with pytest.raises(FileNotFoundError):
            MLPredictor()
