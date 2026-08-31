"""Tests for the deferral-fix candidates and their evaluation machinery.

Two fixture routers anchor the risk-coverage tests the same way the gate tests
anchor label semantics: a deliberately GOOD router (defers exactly the cases
the model gets wrong) and a deliberately BAD router (defers only cases the
model gets right). The evaluation code must place the good one between random
and oracle, and the bad one below random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.services import decision_engine as de
from research.deferral.evaluate import (
    defer_mask_at_rate,
    rebuild_decisions,
    risk_coverage_curve,
    risk_coverage_point,
    split_tune_test,
)
from research.deferral.signals import (
    CurrentDisagreement,
    IsotonicDisagreement,
    MLUncertainty,
    PercentileMap,
    RankDisagreement,
    ZScaler,
    ZScoreDisagreement,
    all_candidates,
)

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# fixture frames
# ---------------------------------------------------------------------------


def _frame(p_ml, y, threshold=0.5):
    n = len(p_ml)
    return pd.DataFrame(
        {
            "y_true": np.asarray(y, dtype=int),
            "best_model_prob": np.asarray(p_ml, dtype=float),
            "cbes_prob": np.full(n, 0.5),
            "approval_threshold": np.full(n, float(threshold)),
            "final_decision": ["APPROVE"] * n,
        }
    )


@pytest.fixture()
def scored_population():
    """600 cases; the model is wrong on a known subset."""
    n = 600
    p_ml = RNG.uniform(0.05, 0.95, n)
    y = (p_ml + RNG.normal(0, 0.25, n) > 0.5).astype(int)
    frame = _frame(p_ml, y)
    errors = ((p_ml >= 0.5).astype(int) != y).astype(bool)
    return frame, errors


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_percentile_map_is_uniform_on_its_own_data(self):
        v = RNG.normal(3.0, 2.0, 5000)
        u = PercentileMap().fit(v).transform(v)
        assert 0.0 < u.min() and u.max() <= 1.0
        # ECDF of its own sample is uniform: mean ~ (n+1)/2n
        assert abs(u.mean() - 0.5) < 0.02

    def test_percentile_map_removes_monotone_scale_mismatch(self):
        base = RNG.uniform(0, 1, 4000)
        shifted = base - 0.31  # the production offset, exaggerated to exact
        m1 = PercentileMap().fit(base)
        m2 = PercentileMap().fit(shifted)
        np.testing.assert_allclose(m1.transform(base), m2.transform(shifted), atol=1e-9)

    def test_zscaler_standardises_with_tune_statistics(self):
        v = RNG.normal(0.61, 0.05, 3000)
        z = ZScaler().fit(v).transform(v)
        assert abs(z.mean()) < 1e-9 and abs(z.std() - 1.0) < 1e-9

    def test_zscaler_constant_input_degrades_to_zero_not_nan(self):
        z = ZScaler().fit(np.full(10, 0.4)).transform(np.full(3, 0.4))
        assert np.all(z == 0.0)


# ---------------------------------------------------------------------------
# candidate signals
# ---------------------------------------------------------------------------


class TestSignals:
    def test_current_rule_is_contaminated_by_a_pure_offset(self):
        """A constant offset with IDENTICAL ordering: no genuine disagreement."""
        p_ml = RNG.uniform(0.35, 0.95, 2000)
        p_cbes = p_ml - 0.31
        y = (p_ml > 0.5).astype(int)
        th = np.full(2000, 0.5)
        raw = CurrentDisagreement().score(p_ml, p_cbes, th)
        assert raw.mean() > 0.30  # incumbent sees huge "disagreement"

        for cls in (RankDisagreement, ZScoreDisagreement):
            sig = cls().fit(p_ml, p_cbes, y, th)
            assert sig.score(p_ml, p_cbes, th).mean() < 0.01, cls.__name__

    def test_rank_diff_detects_genuine_ordering_disagreement(self):
        p_ml = np.array([0.1, 0.2, 0.8, 0.9])
        p_cbes = np.array([0.9, 0.8, 0.2, 0.1])  # reversed ordering
        y = np.array([0, 0, 1, 1])
        th = np.full(4, 0.5)
        sig = RankDisagreement().fit(p_ml, p_cbes, y, th)
        assert sig.score(p_ml, p_cbes, th).mean() > 0.4

    def test_isotonic_diff_kills_offset_but_keeps_real_disagreement(self):
        n = 4000
        p_ml = RNG.uniform(0.05, 0.95, n)
        y = (RNG.uniform(0, 1, n) < p_ml).astype(int)
        th = np.full(n, 0.5)
        offset_only = np.clip(p_ml - 0.31, 0, 1)
        sig = IsotonicDisagreement().fit(p_ml, offset_only, y, th)
        aligned = sig.score(p_ml, offset_only, th)
        # a shuffled (informationless) cbes must disagree more after calibration
        shuffled = RNG.permutation(offset_only)
        sig2 = IsotonicDisagreement().fit(p_ml, shuffled, y, th)
        assert sig2.score(p_ml, shuffled, th).mean() > aligned.mean()

    def test_ml_uncertainty_is_highest_at_the_boundary(self):
        p_ml = np.array([0.05, 0.55, 0.6425, 0.99])
        th = np.full(4, 0.6425)
        s = MLUncertainty().score(p_ml, None, th)
        assert np.argmax(s) == 2  # the case sitting on the threshold
        assert s[2] > s[1] > s[0]

    def test_all_candidates_unique_names(self):
        names = [c.name for c in all_candidates()]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# rate matching + decision rebuild
# ---------------------------------------------------------------------------


class TestRateMatching:
    def test_threshold_comes_from_tune_and_hits_target_on_test(self):
        tune = RNG.uniform(0, 1, 20000)
        test = RNG.uniform(0, 1, 20000)
        mask, threshold = defer_mask_at_rate(tune, test, 0.225)
        assert abs(float(np.quantile(tune, 0.775)) - threshold) < 1e-12
        assert 0.20 < mask.mean() < 0.25

    def test_rebuild_decisions_defers_masked_and_hard_decides_rest(self):
        frame = _frame([0.9, 0.4, 0.6], [1, 0, 1], threshold=0.5)
        out = rebuild_decisions(frame, np.array([False, False, True]))
        assert list(out["final_decision"]) == ["APPROVE", "REJECT", "DEFER"]

    def test_split_is_disjoint_and_covers_everything(self):
        frame = _frame(RNG.uniform(0, 1, 101), RNG.integers(0, 2, 101))
        frame["applicant_id"] = np.arange(101)
        tune, test = split_tune_test(frame, seed=1)
        assert len(tune) + len(test) == 101
        assert set(tune["applicant_id"]).isdisjoint(set(test["applicant_id"]))


# ---------------------------------------------------------------------------
# risk-coverage: the good and bad router fixtures
# ---------------------------------------------------------------------------


class TestRiskCoverage:
    def test_good_router_sits_between_random_and_oracle(self, scored_population):
        frame, errors = scored_population
        # deliberately GOOD: defer every error plus a little noise
        defer = errors.copy()
        rc = risk_coverage_point(frame, defer)
        assert rc["beats_random"] is True
        assert rc["selective_risk"] == 0.0  # deferred all errors -> oracle-grade
        assert rc["position_random0_oracle1"] == pytest.approx(1.0)

    def test_bad_router_is_worse_than_random(self, scored_population):
        frame, errors = scored_population
        # deliberately BAD: defer ONLY correctly-decided cases
        correct_idx = np.flatnonzero(~errors)
        defer = np.zeros(len(frame), dtype=bool)
        defer[correct_idx[: int(errors.sum())]] = True
        rc = risk_coverage_point(frame, defer)
        assert rc["beats_random"] is False
        assert rc["selective_risk"] > rc["random_risk_at_matched_coverage"]
        assert rc["position_random0_oracle1"] < 0

    def test_random_reference_equals_overall_error_rate(self, scored_population):
        frame, errors = scored_population
        rc = risk_coverage_point(frame, np.zeros(len(frame), dtype=bool))
        assert rc["random_risk_at_matched_coverage"] == pytest.approx(errors.mean())

    def test_curve_oracle_signal_reaches_zero_risk_when_capacity_covers_errors(
        self, scored_population
    ):
        frame, errors = scored_population
        err = errors.astype(int)
        cov = np.array([1.0 - err.mean() - 0.01])  # keep just fewer than the correct cases
        pts = risk_coverage_curve(err, err.astype(float), cov)
        assert pts[0]["selective_risk"] == 0.0

    def test_curve_full_coverage_recovers_overall_risk(self, scored_population):
        frame, errors = scored_population
        err = errors.astype(int)
        pts = risk_coverage_curve(err, RNG.uniform(0, 1, len(err)), np.array([1.0]))
        assert pts[0]["selective_risk"] == pytest.approx(err.mean())


# ---------------------------------------------------------------------------
# engine wiring: flag defaults to legacy, switches cleanly
# ---------------------------------------------------------------------------


class TestEngineWiring:
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("SMARTLEND_DEFERRAL_MODE", raising=False)
        monkeypatch.delenv("SMARTLEND_TAU_U", raising=False)

    def test_default_behaviour_is_unchanged_legacy_disagreement(self, monkeypatch):
        self._clear_env(monkeypatch)
        # Big raw disagreement -> legacy rule must DEFER exactly as before.
        res = de.hybrid_decision(p_ml=0.95, p_cbes=0.30, tau_d=0.43, t_base=0.6425)
        assert res.decision == "DEFER" and res.decision_reason == "disagreement"

    def test_uncertainty_mode_defers_near_boundary_not_on_offset(self, monkeypatch):
        self._clear_env(monkeypatch)
        # same offset-dominated pair: confident model must now AUTO-decide
        confident = de.hybrid_decision(
            0.95, 0.30, tau_d=0.43, t_base=0.6425, deferral_mode="uncertainty"
        )
        assert confident.decision == "APPROVE"
        # boundary case must defer
        boundary = de.hybrid_decision(
            0.64, 0.64, tau_d=0.43, t_base=0.6425, deferral_mode="uncertainty"
        )
        assert boundary.decision == "DEFER"
        assert boundary.decision_reason == "ml_uncertainty"
        # far-below-band case must reject
        low = de.hybrid_decision(
            0.10, 0.50, tau_d=0.43, t_base=0.6425, deferral_mode="uncertainty"
        )
        assert low.decision == "REJECT"

    def test_env_flag_switches_mode_without_code_changes(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("SMARTLEND_DEFERRAL_MODE", "uncertainty")
        res = de.hybrid_decision(0.95, 0.30, tau_d=0.43, t_base=0.6425)
        assert res.decision == "APPROVE"  # offset case no longer deferred

    def test_explicit_argument_beats_env(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("SMARTLEND_DEFERRAL_MODE", "uncertainty")
        res = de.hybrid_decision(
            0.95, 0.30, tau_d=0.43, t_base=0.6425, deferral_mode="disagreement"
        )
        assert res.decision == "DEFER" and res.decision_reason == "disagreement"

    def test_tau_u_env_override_and_malformed_fallback(self, monkeypatch):
        self._clear_env(monkeypatch)
        # tiny band: boundary-adjacent but outside band -> auto decision
        monkeypatch.setenv("SMARTLEND_TAU_U", "0.01")
        res = de.hybrid_decision(
            0.70, 0.50, tau_d=0.43, t_base=0.6425, deferral_mode="uncertainty"
        )
        assert res.decision == "APPROVE"
        # malformed value degrades to the measured default (0.2458) -> defers
        monkeypatch.setenv("SMARTLEND_TAU_U", "not-a-number")
        res = de.hybrid_decision(
            0.70, 0.50, tau_d=0.43, t_base=0.6425, deferral_mode="uncertainty"
        )
        assert res.decision == "DEFER"

    def test_unknown_mode_string_falls_back_to_legacy(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("SMARTLEND_DEFERRAL_MODE", "yolo")
        res = de.hybrid_decision(0.95, 0.30, tau_d=0.43, t_base=0.6425)
        assert res.decision_reason == "disagreement"
