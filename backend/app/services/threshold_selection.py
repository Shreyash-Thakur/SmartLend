"""T_base (approve/reject threshold) selection — shared by training and research.

Why this module exists
----------------------
Until 2026-08, `train_pipeline()` and `retrain_serving_model_v3.py` picked
t_base as the F1-argmax over a HARDCODED sweep of [0.30, 0.70). That is
degenerate for this system, and the failure is structural, not a tuning issue:

  * The engine's score is p_ml = P(approval) = 1 - P(default). On a properly
    calibrated model with an ~8% default rate, p_ml concentrates near 0.92
    (observed 5th percentile on the out-of-fold predictions: ~0.77).
  * Fewer than ~2% of applicants ever score below 0.70, so every threshold in
    [0.30, 0.70) rejects almost nobody. F1 for catching defaulters is near
    zero across the ENTIRE swept range (the freshly retrained v3 artifact
    reported F1 = 0.0024 at its "optimum").
  * Worse, F1 is monotonically INCREASING over that range, so the argmax is
    pinned to the top edge of the sweep. The selected t_base is an artifact of
    where the range was cut off, not a property of the model or the data.

DO NOT reinstate the fixed-range F1 sweep as the default. It is kept here only
as method="f1_legacy" so the defect stays reproducible and documented.

Methods provided
----------------
cost           (RECOMMENDED, default) Minimise expected misclassification
               cost per applicant over a percentile grid of the observed
               p_ml. Loss matrix, per unit EAD:
                 - approving a defaulter costs LGD  (loss given default)
                 - rejecting a good customer costs ROI (foregone return)
               Defaults: LGD = 0.45, the Basel II foundation-IRB supervisory
               LGD for senior unsecured exposures (BCBS 128, June 2006,
               para 287). ROI = 0.2644, the return-on-investment parameter of
               the EMP credit-scoring measure (Verbraken, Bravo, Weber &
               Baesens, "Development and application of consumer credit
               scoring models using profit-based classification measures",
               EJOR 238(2), 2014), itself estimated from real consumer-loan
               portfolios. For a calibrated score the Bayes-optimal threshold
               is t* = 1 - ROI / (ROI + LGD) ≈ 0.63 with these defaults; the
               empirical grid search agrees to within the flatness of the
               cost curve. Both parameters are configurable and the caller
               should report a sensitivity table (see research/thresholds).
youden         Max TPR - FPR for detecting defaulters. Cost-agnostic
               reference point; typically lands far above the engine's clip
               range (decision_engine clips t_base to [0.30, 0.75]), so it is
               a diagnostic, not an operating recommendation.
f1             F1-argmax like the legacy method, but over a percentile grid
               of the OBSERVED p_ml so the search covers where the mass is.
approval_rate  The threshold that yields a target approval rate — what a
               lender actually operates. Ignores y.
f1_legacy      The old degenerate fixed-range sweep. Kept ONLY to document
               the defect; never use for a real artifact.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

# Basel II foundation-IRB supervisory LGD, senior unsecured (BCBS 128 para 287).
DEFAULT_LGD = 0.45
# ROI of a performing consumer loan, Verbraken et al. 2014 (EMP), EJOR 238(2).
DEFAULT_ROI = 0.2644
# decision_engine.hybrid_decision clips t_base to this range. We do NOT clip
# here (the raw optimum is information), but we flag when the engine would.
ENGINE_CLIP_LO, ENGINE_CLIP_HI = 0.30, 0.75

METHODS = ("cost", "youden", "f1", "approval_rate", "f1_legacy")


def _percentile_grid(p_ml: np.ndarray, n: int = 199) -> np.ndarray:
    """Candidate thresholds = percentiles of the OBSERVED score distribution.

    This is the fix for the core defect: the legacy sweep searched a fixed
    [0.30, 0.70) window containing <2% of the probability mass. A percentile
    grid puts every candidate where applicants actually score, whatever the
    base rate or calibration does to the distribution.
    """
    qs = np.linspace(0.5, 99.5, n)
    return np.unique(np.round(np.percentile(p_ml, qs), 6))


def expected_cost_per_applicant(
    t: float, p_ml: np.ndarray, y_default: np.ndarray, lgd: float, roi: float
) -> float:
    """Mean cost per applicant, in units of EAD, under threshold t.

    Approve iff p_ml >= t. Approved defaulter costs `lgd`; rejected good
    customer costs `roi` (foregone return). EAD is normalised to 1 per
    applicant because the out-of-fold prediction file carries no loan amounts;
    with per-row EAD available, both terms would be weighted by it.
    """
    approved = p_ml >= t
    fa = np.mean(approved & (y_default == 1))   # false approval rate
    fr = np.mean(~approved & (y_default == 0))  # false rejection rate
    return float(lgd * fa + roi * fr)


def confusion_metrics(
    t: float,
    p_ml: np.ndarray,
    y_default: np.ndarray,
    lgd: float = DEFAULT_LGD,
    roi: float = DEFAULT_ROI,
) -> Dict[str, float]:
    """Operating metrics at threshold t. Positive class = default (reject)."""
    reject = p_ml < t
    tp = int(np.sum(reject & (y_default == 1)))
    fp = int(np.sum(reject & (y_default == 0)))
    fn = int(np.sum(~reject & (y_default == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    approved = ~reject
    n_app = int(approved.sum())
    return {
        "threshold": round(float(t), 6),
        "approval_rate": round(float(approved.mean()), 6),
        "rejection_rate": round(float(reject.mean()), 6),
        "precision_default": round(precision, 6),
        "recall_default": round(recall, 6),
        "f1_default": round(f1, 6),
        "expected_cost_per_applicant": round(
            expected_cost_per_applicant(t, p_ml, y_default, lgd, roi), 6
        ),
        "default_rate_among_approved": round(
            float(np.mean(y_default[approved] == 1)) if n_app else 0.0, 6
        ),
    }


def select_t_base(
    y_default: Sequence[int] | np.ndarray,
    p_ml: Sequence[float] | np.ndarray,
    method: str = "cost",
    *,
    lgd: float = DEFAULT_LGD,
    roi: float = DEFAULT_ROI,
    target_approval_rate: float = 0.85,
    grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Select the approve/reject threshold t_base on (y_default, p_ml).

    Parameters
    ----------
    y_default : 1 = the applicant defaulted, 0 = repaid.
    p_ml      : model approval probability, p_ml = 1 - P(default).
    method    : one of METHODS. "cost" is the recommended default; see the
                module docstring for citations. "f1_legacy" reproduces the
                pre-2026-08 degenerate behaviour and MUST NOT be used for a
                real artifact — its F1 curve is flat-near-zero and edge-pinned
                on this data (see docstring).
    lgd, roi  : cost parameters for method="cost".
    target_approval_rate : for method="approval_rate".
    grid      : override the candidate thresholds (rarely needed).

    Returns a dict with `t_base`, the selection `criterion_value` at the
    optimum, the searched grid bounds, and `engine_will_clip` (True when
    decision_engine's [0.30, 0.75] clamp would alter the value).

    IMPORTANT: select on one data split and report performance on another —
    a threshold argmax is itself a fitted parameter.
    """
    y = np.asarray(y_default).astype(int)
    p = np.asarray(p_ml, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_default and p_ml must have the same shape")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    if method == "f1_legacy":
        # ── THE OLD DEFECT, PRESERVED ON PURPOSE ──────────────────────────
        # Fixed range regardless of where p_ml actually lives. On calibrated
        # ~8%-default-rate data essentially no mass is below 0.70, F1 is near
        # zero and monotone over the range, and the argmax sits at the top
        # edge of the sweep. Kept only so tests can reproduce the failure.
        cand = np.arange(0.30, 0.70, 0.01)
    elif grid is not None:
        cand = np.asarray(grid, dtype=float)
    else:
        cand = _percentile_grid(p)

    if method == "approval_rate":
        if not 0.0 < target_approval_rate < 1.0:
            raise ValueError("target_approval_rate must be in (0, 1)")
        t = float(np.quantile(p, 1.0 - target_approval_rate))
        crit = target_approval_rate
    elif method == "cost":
        costs = np.array(
            [expected_cost_per_applicant(t, p, y, lgd, roi) for t in cand]
        )
        i = int(np.argmin(costs))
        t, crit = float(cand[i]), float(costs[i])
    elif method == "youden":
        n_bad = max(int((y == 1).sum()), 1)
        n_good = max(int((y == 0).sum()), 1)
        js = np.array(
            [
                np.sum((p < t) & (y == 1)) / n_bad - np.sum((p < t) & (y == 0)) / n_good
                for t in cand
            ]
        )
        i = int(np.argmax(js))
        t, crit = float(cand[i]), float(js[i])
    else:  # "f1" and "f1_legacy" share the F1 objective; only the grid differs
        f1s = []
        for tt in cand:
            reject = p < tt
            tp = np.sum(reject & (y == 1))
            fp = np.sum(reject & (y == 0))
            fn = np.sum(~reject & (y == 1))
            denom = 2 * tp + fp + fn
            f1s.append(2 * tp / denom if denom else 0.0)
        f1s = np.array(f1s)
        i = int(np.argmax(f1s))
        t, crit = float(cand[i]), float(f1s[i])

    return {
        "t_base": round(t, 6),
        "method": method,
        "criterion_value": round(float(crit), 6),
        "grid_lo": round(float(np.min(cand)), 6) if method != "approval_rate" else None,
        "grid_hi": round(float(np.max(cand)), 6) if method != "approval_rate" else None,
        "params": {
            "lgd": lgd,
            "roi": roi,
            "target_approval_rate": target_approval_rate,
        },
        # decision_engine clamps t_base into [0.30, 0.75]; a value outside
        # that range would be silently altered at decision time.
        "engine_will_clip": not (ENGINE_CLIP_LO <= t <= ENGINE_CLIP_HI),
    }
