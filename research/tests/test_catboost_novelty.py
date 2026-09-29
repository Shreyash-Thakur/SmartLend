"""Unit tests for the pure-math helpers of the three CatBoost novelty studies.

Data-free by design: the experiments themselves need the Home Credit extract,
but the uncertainty decomposition, cost accounting and constraint plumbing
must hold unconditionally.
"""

import numpy as np
import pytest

from research.analysis.cost_sensitive_catboost import (
    best_cost_threshold,
    paired_bootstrap_cost_delta,
    per_row_cost,
)
from research.analysis.monotone_catboost import CONSTRAINTS, build_constraint_vector
from research.analysis.uncertainty_deferral_catboost import (
    binary_entropy,
    decompose_uncertainty,
    race_candidate,
)


# ---------------------------------------------------------------------------
# uncertainty decomposition
# ---------------------------------------------------------------------------

def test_binary_entropy_extremes_and_peak():
    assert binary_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2))
    assert binary_entropy(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-5)
    assert binary_entropy(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-5)


def test_agreeing_members_have_zero_knowledge_uncertainty():
    # All virtual-ensemble members say the same thing: total == data, knowledge == 0.
    members = np.full((4, 10), 0.3)
    parts = decompose_uncertainty(members)
    assert np.allclose(parts["knowledge"], 0.0, atol=1e-9)
    assert np.allclose(parts["total"], parts["data"])
    assert np.allclose(parts["pd"], 0.3)


def test_disagreeing_members_have_positive_knowledge_uncertainty():
    # Half the members are confident-good, half confident-bad: the mean sits at
    # 0.5 (max total entropy) while each member is near-certain (low data
    # entropy) — the textbook epistemic case.
    members = np.column_stack([np.full((3, 5), 0.05), np.full((3, 5), 0.95)])
    parts = decompose_uncertainty(members)
    assert (parts["knowledge"] > 0.3).all()
    assert np.allclose(parts["pd"], 0.5)


def test_knowledge_never_negative():
    rng = np.random.default_rng(0)
    parts = decompose_uncertainty(rng.uniform(0.01, 0.99, size=(200, 10)))
    assert (parts["knowledge"] >= 0).all()


# ---------------------------------------------------------------------------
# deferral race mechanics
# ---------------------------------------------------------------------------

def test_race_oracle_signal_reaches_position_one():
    rng = np.random.default_rng(1)
    errors = (rng.uniform(size=2000) < 0.25).astype(int)
    # A signal that IS the error indicator (with tie-breaking jitter) defers
    # errors first: position must be ~1 and selective risk ~0 at 30% deferral.
    signal = errors + rng.uniform(0, 0.01, size=2000)
    result = race_candidate(signal, signal, errors, target_rate=0.30)
    assert result["beats_random"]
    assert result["position_random0_oracle1"] == pytest.approx(1.0, abs=0.05)


def test_race_random_signal_sits_near_position_zero():
    rng = np.random.default_rng(2)
    errors = (rng.uniform(size=5000) < 0.25).astype(int)
    signal = rng.uniform(size=5000)
    result = race_candidate(signal, signal, errors, target_rate=0.30)
    assert abs(result["position_random0_oracle1"]) < 0.25


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------

def test_per_row_cost_charges_only_mistakes():
    y = np.array([1, 1, 0, 0])           # 1 = default
    pd_scores = np.array([0.9, 0.1, 0.9, 0.1])
    costs = per_row_cost(y, pd_scores, threshold=0.5, cost_fn=0.75, cost_fp=0.10)
    # caught defaulter, missed defaulter, rejected good, approved good
    assert costs.tolist() == [0.0, 0.75, 0.10, 0.0]


def test_best_cost_threshold_moves_with_the_cost_ratio():
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=20000) < 0.2).astype(int)
    pd_scores = np.clip(0.2 + 0.4 * (y - 0.5) + rng.normal(0, 0.18, size=20000), 0.001, 0.999)
    t_fn_heavy = best_cost_threshold(y, pd_scores, cost_fn=0.9, cost_fp=0.05)
    t_fp_heavy = best_cost_threshold(y, pd_scores, cost_fn=0.05, cost_fp=0.9)
    # When missing defaulters is the expensive mistake, reject more (lower t).
    assert t_fn_heavy < t_fp_heavy


def test_paired_bootstrap_detects_a_real_cost_gap():
    rng = np.random.default_rng(4)
    base = rng.uniform(0, 0.2, size=5000)
    better = base - 0.01  # uniformly 0.01 cheaper per row
    boot = paired_bootstrap_cost_delta(better, base, n_boot=200, seed=0)
    assert boot["delta_mean"] == pytest.approx(-0.01, abs=1e-9)
    assert boot["delta_ci95"][1] < 0


# ---------------------------------------------------------------------------
# monotone constraint plumbing
# ---------------------------------------------------------------------------

def test_constraint_vector_alignment_and_directions():
    cols = ["EXT_SOURCE_1", "SOMETHING_ELSE", "APP_ANNUITY_TO_INCOME"]
    assert build_constraint_vector(cols) == [-1, 0, 1]


def test_constraint_set_uses_only_valid_directions():
    assert set(CONSTRAINTS.values()) <= {-1, 1}
    # Safer-signal features must be negative; distress features positive.
    assert CONSTRAINTS["APP_EXT_MEAN"] == -1
    assert CONSTRAINTS["BURO_overdue_sum"] == 1
