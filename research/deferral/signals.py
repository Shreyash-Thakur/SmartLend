"""Candidate deferral signals for the SmartLend router.

THE DEFECT BEING FIXED
----------------------
The production router defers when ``D = |p_ml - p_cbes| > TAU_D``. Measured on
real out-of-fold predictions, ``p_ml - p_cbes`` has mean +0.307 with sd 0.137
and is positive on 98.4% of rows: the two scores live on different scales, so
D is dominated by a fixed calibration OFFSET, not by case-by-case
disagreement. Because the offset widens with model confidence
(corr(D, |p_ml - 0.5|) = +0.38), the rule defers the cases the model is MOST
sure about — the exact inverse of a working router.

Every candidate here is a *fit/score* pair so that any data-dependent pieces
(percentile maps, means/sds, isotonic calibrators) are learned on a TUNE split
and applied unchanged to a held-out TEST split. Scores follow one convention:

    HIGHER SIGNAL VALUE == MORE DESERVING OF DEFERRAL.

That lets the evaluator pick a single quantile threshold per candidate to hit
a matched deferral rate, which is the only fair way to compare signals.

No new dependencies: numpy + scikit-learn only.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# small fitted transforms
# ---------------------------------------------------------------------------


class PercentileMap:
    """Map a score to its percentile in [0, 1] under the TUNE distribution.

    Rank-normalising both scores before differencing removes any monotone
    scale mismatch (including the 0.307 offset) while preserving each score's
    ordering — the only part of a badly-calibrated score that carries
    information.
    """

    def fit(self, values: np.ndarray) -> "PercentileMap":
        self.sorted_ = np.sort(np.asarray(values, dtype=float))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        # searchsorted gives the count of tune values <= v; divide by n -> ECDF.
        n = len(self.sorted_)
        ranks = np.searchsorted(self.sorted_, np.asarray(values, dtype=float), side="right")
        return ranks / n


class ZScaler:
    """Standardise to zero mean / unit sd using TUNE statistics.

    Removes the offset and the scale difference but, unlike the percentile
    map, keeps each distribution's shape — a fatter-tailed score still
    produces larger |z| excursions.
    """

    def fit(self, values: np.ndarray) -> "ZScaler":
        v = np.asarray(values, dtype=float)
        self.mean_ = float(v.mean())
        self.std_ = float(v.std())
        if self.std_ <= 0:
            self.std_ = 1.0  # degenerate constant score: map everything to 0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean_) / self.std_


# ---------------------------------------------------------------------------
# candidate signals
# ---------------------------------------------------------------------------


class BaseSignal:
    """Interface: fit on the tune split, score anywhere."""

    name: str = "base"
    description: str = ""

    def fit(self, p_ml: np.ndarray, p_cbes: np.ndarray, y: np.ndarray, threshold: np.ndarray) -> "BaseSignal":
        return self

    def score(self, p_ml: np.ndarray, p_cbes: np.ndarray, threshold: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CurrentDisagreement(BaseSignal):
    """The production rule, |p_ml - p_cbes|, kept as the incumbent baseline.

    Evaluated at the matched rate so we can separate "the signal is wrong"
    from "the threshold was wrong".
    """

    name = "current_abs_diff"
    description = "|p_ml - p_cbes| (production rule; scale-offset contaminated)"

    def score(self, p_ml, p_cbes, threshold):
        return np.abs(p_ml - p_cbes)


class RankDisagreement(BaseSignal):
    """Candidate 1: percentile-rank both scores, then |difference|."""

    name = "rank_diff"
    description = "|percentile(p_ml) - percentile(p_cbes)| with percentiles fit on tune"

    def fit(self, p_ml, p_cbes, y, threshold):
        self.ml_map_ = PercentileMap().fit(p_ml)
        self.cbes_map_ = PercentileMap().fit(p_cbes)
        return self

    def score(self, p_ml, p_cbes, threshold):
        return np.abs(self.ml_map_.transform(p_ml) - self.cbes_map_.transform(p_cbes))


class ZScoreDisagreement(BaseSignal):
    """Candidate 2: z-standardise both scores, then |difference|."""

    name = "zscore_diff"
    description = "|z(p_ml) - z(p_cbes)| with mean/sd fit on tune"

    def fit(self, p_ml, p_cbes, y, threshold):
        self.ml_scaler_ = ZScaler().fit(p_ml)
        self.cbes_scaler_ = ZScaler().fit(p_cbes)
        return self

    def score(self, p_ml, p_cbes, threshold):
        return np.abs(self.ml_scaler_.transform(p_ml) - self.cbes_scaler_.transform(p_cbes))


class IsotonicDisagreement(BaseSignal):
    """Candidate 3: calibrate BOTH scores onto the outcome, then |difference|.

    Isotonic regression of each score against y (fit on tune) puts both on a
    common, outcome-anchored probability scale; any residual difference is
    genuine case-level disagreement about P(good), not calibration error.
    """

    name = "isotonic_diff"
    description = "|iso(p_ml) - iso(p_cbes)| after isotonic calibration onto y (fit on tune)"

    def fit(self, p_ml, p_cbes, y, threshold):
        self.ml_iso_ = IsotonicRegression(out_of_bounds="clip").fit(p_ml, y)
        self.cbes_iso_ = IsotonicRegression(out_of_bounds="clip").fit(p_cbes, y)
        return self

    def score(self, p_ml, p_cbes, threshold):
        return np.abs(self.ml_iso_.predict(p_ml) - self.cbes_iso_.predict(p_cbes))


class MLUncertainty(BaseSignal):
    """Candidate 4: abandon disagreement — defer on model uncertainty.

    The standard selective-prediction baseline (Chow's rule / softmax-response):
    the hardest cases are those closest to the decision threshold, so the
    signal is -|p_ml - t|. ``t`` is the engine's own per-row approval
    threshold, i.e. the boundary the auto-decision is actually made at.
    Nothing to fit.
    """

    name = "ml_uncertainty"
    description = "-|p_ml - approval_threshold| (distance from the actual decision boundary)"

    def score(self, p_ml, p_cbes, threshold):
        return -np.abs(p_ml - threshold)


class MLUncertaintyHalf(BaseSignal):
    """Candidate 4b: -|p_ml - 0.5|, the textbook variant with a fixed 0.5 pivot.

    Included so the report can show whether pivoting at the engine's real
    threshold (0.6425) matters versus the naive 0.5.
    """

    name = "ml_uncertainty_0.5"
    description = "-|p_ml - 0.5| (fixed-pivot selective-prediction baseline)"

    def score(self, p_ml, p_cbes, threshold):
        return -np.abs(p_ml - 0.5)


def all_candidates() -> list[BaseSignal]:
    """Every signal to be raced, incumbent first."""
    return [
        CurrentDisagreement(),
        RankDisagreement(),
        ZScoreDisagreement(),
        IsotonicDisagreement(),
        MLUncertainty(),
        MLUncertaintyHalf(),
    ]
