"""Tests for t_base threshold selection (backend/app/services/threshold_selection.py).

Two of these are regression guards for the 2026-08 defect:
  * test_legacy_f1_sweep_is_degenerate_on_imbalanced_data reproduces the
    flat, edge-pinned F1 curve of the old hardcoded [0.30, 0.70) sweep on
    synthetic calibrated 8%-default-rate data, so the original failure mode
    stays documented and nobody quietly reinstates the sweep as the default.
  * test_data_driven_methods_stay_inside_observed_range FAILS if any
    data-driven selection method ever returns a threshold outside the
    observed probability range.
"""

import numpy as np
import pytest

from backend.app.services.threshold_selection import (
    DEFAULT_LGD,
    DEFAULT_ROI,
    confusion_metrics,
    expected_cost_per_applicant,
    select_t_base,
)


def _synthetic_calibrated(n: int = 60_000, default_rate: float = 0.08, seed: int = 0):
    """Synthetic (y_default, p_ml) resembling the real system: a calibrated
    score on a heavily imbalanced base rate, so p_ml = P(approval) piles up
    near 1 - default_rate ~ 0.92."""
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < default_rate).astype(int)
    # A discriminative latent score, then map through a monotone squash that
    # concentrates approval probabilities high — like the real OOF columns.
    latent = rng.normal(loc=np.where(y == 1, -0.5, 0.5), scale=1.0, size=n)
    p_default = 1.0 / (1.0 + np.exp(latent + 3.0))
    return y, 1.0 - p_default


DATA_DRIVEN_METHODS = ("cost", "youden", "f1", "approval_rate")


def test_data_driven_methods_stay_inside_observed_range():
    """A selection method that returns a threshold outside the observed
    probability range is broken by construction: every applicant would land
    on one side of it. This test must FAIL if that ever happens."""
    y, p = _synthetic_calibrated()
    lo, hi = float(p.min()), float(p.max())
    for method in DATA_DRIVEN_METHODS:
        t = select_t_base(y, p, method=method)["t_base"]
        assert lo <= t <= hi, (
            f"method={method!r} selected t_base={t}, outside the observed "
            f"probability range [{lo:.4f}, {hi:.4f}]"
        )


def test_legacy_f1_sweep_is_degenerate_on_imbalanced_data():
    """Reproduce the original defect on synthetic imbalanced data.

    With a calibrated ~8%-default-rate score, p_ml concentrates near 0.92:
    essentially nothing lies below 0.70, so over the legacy hardcoded sweep
    [0.30, 0.70) the F1 for catching defaulters is ~0 everywhere and monotone
    increasing, and the 'optimum' is pinned to the top edge of the range.
    This is why the v3 artifact shipped t_base=0.65 with F1=0.0024.
    """
    y, p = _synthetic_calibrated()
    # Precondition: the synthetic data has the real system's shape.
    assert float((p < 0.70).mean()) < 0.05, "synthetic p_ml should concentrate above 0.70"

    sel = select_t_base(y, p, method="f1_legacy")
    # (1) The F1 at the 'optimum' is negligible — the criterion carries no signal.
    assert sel["criterion_value"] < 0.10, (
        f"expected near-zero F1 across the legacy sweep, got {sel['criterion_value']}"
    )
    # (2) The argmax is pinned at the top edge of the hardcoded range: the
    # answer is set by the sweep bounds (0.69 is the last grid point of
    # np.arange(0.30, 0.70, 0.01)), not by the data.
    assert sel["t_base"] >= 0.68, (
        f"expected the legacy argmax pinned at the sweep's top edge, got {sel['t_base']}"
    )
    # (3) By contrast, the percentile-grid F1 search finds a real optimum in
    # the score mass, several times larger than anything the legacy window
    # could see.
    fixed = select_t_base(y, p, method="f1")
    assert fixed["t_base"] > 0.75
    assert fixed["criterion_value"] > 3 * sel["criterion_value"]


def test_cost_method_tracks_bayes_threshold_on_calibrated_scores():
    """On perfectly calibrated probabilities the expected-cost minimum should
    sit near the analytic Bayes threshold t* = 1 - ROI/(ROI + LGD)."""
    rng = np.random.RandomState(1)
    n = 200_000
    # Draw true approval probabilities with mass spread over (0.5, 1), then
    # sample outcomes from them -> perfectly calibrated by construction.
    p = 1.0 - rng.beta(1.3, 12.0, size=n)  # approval probs, mean ~0.90
    y = (rng.rand(n) > p).astype(int)  # 1 = default
    t_star = 1.0 - DEFAULT_ROI / (DEFAULT_ROI + DEFAULT_LGD)
    t = select_t_base(y, p, method="cost")["t_base"]
    assert abs(t - t_star) < 0.05, f"cost-optimal {t} should be near Bayes {t_star:.4f}"


def test_approval_rate_method_hits_target():
    y, p = _synthetic_calibrated(seed=3)
    res = select_t_base(y, p, method="approval_rate", target_approval_rate=0.85)
    approved = (p >= res["t_base"]).mean()
    assert abs(approved - 0.85) < 0.01


def test_engine_clip_flag():
    y, p = _synthetic_calibrated(seed=4)
    # Youden on this data lands far above 0.75 -> must be flagged.
    res = select_t_base(y, p, method="youden")
    if res["t_base"] > 0.75:
        assert res["engine_will_clip"] is True
    inside = select_t_base(y, p, method="approval_rate", target_approval_rate=0.999)
    # a ~99.9% approval-rate threshold sits deep in the left tail; the flag
    # must reflect the [0.30, 0.75] engine clamp truthfully either way
    assert inside["engine_will_clip"] == (not 0.30 <= inside["t_base"] <= 0.75)


def test_expected_cost_and_metrics_consistency():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.40, 0.80, 0.55, 0.90, 0.95, 0.99])
    t = 0.60
    # approved: p >= 0.6 -> [0.80(bad), 0.90, 0.95, 0.99]; rejected good: 0.55
    expect = DEFAULT_LGD * (1 / 6) + DEFAULT_ROI * (1 / 6)
    assert expected_cost_per_applicant(t, p, y, DEFAULT_LGD, DEFAULT_ROI) == pytest.approx(expect)
    m = confusion_metrics(t, p, y)
    assert m["approval_rate"] == pytest.approx(4 / 6)
    assert m["precision_default"] == pytest.approx(1 / 2)  # rejected: 0.40(bad), 0.55(good)
    assert m["recall_default"] == pytest.approx(1 / 2)


def test_unknown_method_and_shape_mismatch_raise():
    y, p = _synthetic_calibrated(n=1000, seed=5)
    with pytest.raises(ValueError):
        select_t_base(y, p, method="magic")
    with pytest.raises(ValueError):
        select_t_base(y[:-1], p, method="cost")
