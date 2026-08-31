"""Unit tests for the metric helpers in research/analysis/complementarity.py.

Run:  python -m pytest research/tests/test_complementarity.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from complementarity import (  # noqa: E402
    cv_stack_auc,
    error_confusion,
    paired_bootstrap_auc_delta,
    rank_average,
    weight_sweep,
)


@pytest.fixture()
def synth():
    """Synthetic binary problem with two noisy views of a latent score."""
    rng = np.random.default_rng(0)
    n = 4000
    latent = rng.normal(size=n)
    y = (latent + rng.normal(scale=1.0, size=n) > 1.0).astype(int)
    # Two models: same latent signal, independent noise -> complementary.
    p_a = 1 / (1 + np.exp(-(latent + rng.normal(scale=0.8, size=n))))
    p_b = 1 / (1 + np.exp(-(latent + rng.normal(scale=0.8, size=n))))
    return y, p_a, p_b


# ---- rank_average ---------------------------------------------------------

def test_rank_average_is_order_only():
    """Rank average must be invariant to monotone rescaling of inputs."""
    p1 = np.array([0.1, 0.4, 0.2, 0.9])
    p2 = np.array([0.3, 0.8, 0.5, 0.7])
    base = rank_average(p1, p2)
    squashed = rank_average(p1 ** 3, np.sqrt(p2))  # same orderings
    np.testing.assert_allclose(base, squashed)


def test_rank_average_range_and_symmetry():
    rng = np.random.default_rng(1)
    p1, p2 = rng.random(100), rng.random(100)
    r = rank_average(p1, p2)
    assert r.min() > 0 and r.max() <= 1
    np.testing.assert_allclose(rank_average(p1, p2), rank_average(p2, p1))


def test_rank_average_single_input_is_scaled_rank():
    p = np.array([0.5, 0.1, 0.9])
    np.testing.assert_allclose(rank_average(p), np.array([2, 1, 3]) / 3)


# ---- weight_sweep ---------------------------------------------------------

def test_weight_sweep_endpoints_match_single_models(synth):
    y, p_a, p_b = synth
    _, _, curve = weight_sweep(y, p_a, p_b)
    assert curve[0]["w_a"] == 0.0
    assert curve[-1]["w_a"] == 1.0
    assert curve[0]["auc"] == pytest.approx(roc_auc_score(y, p_b), abs=1e-4)
    assert curve[-1]["auc"] == pytest.approx(roc_auc_score(y, p_a), abs=1e-4)


def test_weight_sweep_best_is_max_of_curve(synth):
    y, p_a, p_b = synth
    best_w, best_auc, curve = weight_sweep(y, p_a, p_b)
    assert best_auc == pytest.approx(max(c["auc"] for c in curve), abs=1e-5)
    assert len(curve) == 21  # 0.0 .. 1.0 step 0.05 inclusive


def test_weight_sweep_useless_partner():
    """When model B is pure noise, the best weight should sit at/near w_a=1."""
    rng = np.random.default_rng(2)
    n = 4000
    latent = rng.normal(size=n)
    y = (latent > 0.5).astype(int)
    p_a = 1 / (1 + np.exp(-latent))       # perfect ranking
    p_b = rng.random(n)                    # noise
    best_w, best_auc, _ = weight_sweep(y, p_a, p_b)
    assert best_w >= 0.9
    assert best_auc <= roc_auc_score(y, p_a) + 1e-9


# ---- cv_stack_auc ---------------------------------------------------------

def test_cv_stack_beats_or_matches_singles_on_complementary(synth):
    y, p_a, p_b = synth
    stack = cv_stack_auc(y, p_a, p_b)
    single = max(roc_auc_score(y, p_a), roc_auc_score(y, p_b))
    # Independent-noise views: the stack should genuinely help here.
    assert stack > single


def test_cv_stack_no_leakage_on_pure_noise():
    """Two noise 'models': an honestly-evaluated stack must stay ~0.5.
    A leaky stack (fit and scored on the same rows) would drift above 0.5."""
    rng = np.random.default_rng(3)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    p_a, p_b = rng.random(n), rng.random(n)
    stack = cv_stack_auc(y, p_a, p_b)
    assert abs(stack - 0.5) < 0.03


def test_cv_stack_deterministic(synth):
    y, p_a, p_b = synth
    assert cv_stack_auc(y, p_a, p_b) == cv_stack_auc(y, p_a, p_b)


# ---- paired_bootstrap_auc_delta -------------------------------------------

def test_bootstrap_delta_detects_real_gap():
    rng = np.random.default_rng(4)
    n = 3000
    latent = rng.normal(size=n)
    y = (latent + rng.normal(scale=0.5, size=n) > 1).astype(int)
    p_good = 1 / (1 + np.exp(-latent))
    p_bad = 1 / (1 + np.exp(-(latent + rng.normal(scale=2.0, size=n))))
    res = paired_bootstrap_auc_delta(y, p_good, p_bad, n_boot=300)
    assert res["delta_mean"] > 0
    assert res["delta_ci95"][0] > 0          # CI excludes zero
    assert res["frac_positive"] > 0.99


def test_bootstrap_delta_null_when_identical():
    rng = np.random.default_rng(5)
    n = 2000
    y = (rng.random(n) < 0.2).astype(int)
    p = rng.random(n)
    res = paired_bootstrap_auc_delta(y, p, p.copy(), n_boot=100)
    assert res["delta_mean"] == pytest.approx(0.0, abs=1e-12)


# ---- error_confusion ------------------------------------------------------

def test_error_confusion_counts_sum_to_n():
    rng = np.random.default_rng(6)
    n = 500
    y = (rng.random(n) < 0.5).astype(int)
    res = error_confusion(y, rng.random(n), rng.random(n))
    assert sum(res["counts"].values()) == n


def test_error_confusion_hand_checked():
    y = np.array([1, 1, 0, 0])
    p_a = np.array([0.9, 0.2, 0.1, 0.8])   # right, wrong, right, wrong
    p_b = np.array([0.9, 0.8, 0.9, 0.8])   # right, right, wrong, wrong
    res = error_confusion(y, p_a, p_b)
    assert res["counts"] == {"both_right": 1, "only_a_right": 1,
                             "only_b_right": 1, "both_wrong": 1}


def test_error_confusion_threshold_boundary():
    """p == threshold must classify as the positive class (>=)."""
    y = np.array([1, 0])
    p = np.array([0.5, 0.5])
    res = error_confusion(y, p, p)
    assert res["counts"]["both_right"] == 1
    assert res["counts"]["both_wrong"] == 1
