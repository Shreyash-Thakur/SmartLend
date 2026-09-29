"""Cost-sensitive CatBoost — does training on lending economics beat thresholding?

THE NOVELTY CLAIM UNDER TEST. AUC treats a missed defaulter and a rejected
good customer as equally bad; a lender does not. With a stated cost matrix the
decision-theoretic question splits in two: (a) pick the cost-optimal THRESHOLD
on an unweighted model (Elkan 2001: with calibrated probabilities this is all
you should need), or (b) push the costs into TRAINING via class weights and
re-threshold. If (b) beats (a) on held-out expected cost, training-time
cost-sensitivity is a real contribution; if not, the finding is that the
economics belong in the decision layer — which is where SmartLend already
keeps them. Either answer is reportable.

STATED COST MODEL (per unit of principal, assumptions a reviewer can attack):
  * approve a defaulter  -> lose LGD ~= 0.75 (unsecured retail, Basel-style
    foundation-IRB LGD for uncollateralised exposures)
  * reject a good payer  -> forgo net margin ~= 0.10 (lifetime interest margin
    on the loan, order-of-magnitude for cash-loan books)
  * correct decisions cost 0. Ratio FN:FP = 7.5 : 1 with TARGET=1 = DEFAULT,
    "positive" = predict default = reject.

PRE-REGISTERED DECISION RULES (fixed before results exist):

1. Both arms share the same 5 folds (seed 42) and params as the recorded
   CatBoost FE-v2 baseline; the unweighted arm REUSES the committed OOF
   predictions in reports/v2_oof_predictions.csv (no retrain, no drift).
2. Thresholds are chosen on a 50/50 TUNE split (seed 20260831) by minimising
   expected cost there; all reported costs come from the TEST half.
3. The weighted arm may be CLAIMED only if its TEST expected cost per
   applicant is below the thresholded baseline's with a paired-bootstrap 95%
   CI excluding zero (1000 resamples, seed 20260831). AUC changes are
   reported for context, never as the criterion.
4. Robustness: the comparison is repeated across cost ratios {2, 5, 7.5, 10,
   20} at decision time (weights stay at the stated 7.5 — retraining per
   ratio would be tuning on the question).

Run:  python -m research.analysis.cost_sensitive_catboost [--quick]
Writes reports/cost_sensitive_catboost.json (+ oof cache in data/processed/).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "reports" / "cost_sensitive_catboost.json"
OOF_CACHE = REPO / "data" / "processed" / "cost_weighted_oof.pkl"
V2_OOF_CSV = REPO / "reports" / "v2_oof_predictions.csv"

SEED = 42
N_FOLDS = 5
GBM_PARAMS = {"n_estimators": 2000, "learning_rate": 0.05, "early_stopping": 100}
SPLIT_SEED = 20260831
BOOT_SEED = 20260831
N_BOOT = 1000

COST_FN = 0.75   # approve a defaulter: lose LGD of principal
COST_FP = 0.10   # reject a good payer: forgo margin
COST_RATIOS = [2.0, 5.0, 7.5, 10.0, 20.0]


# --------------------------------------------------------------------------
# cost math (unit-tested in research/tests/test_catboost_novelty.py)
# --------------------------------------------------------------------------

def per_row_cost(y: np.ndarray, pd_scores: np.ndarray, threshold: float,
                 cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> np.ndarray:
    """Cost of the threshold decision per applicant (y: 1 = default)."""
    reject = pd_scores >= threshold
    y = np.asarray(y, dtype=bool)
    return np.where(~reject & y, cost_fn, np.where(reject & ~y, cost_fp, 0.0))


def best_cost_threshold(y: np.ndarray, pd_scores: np.ndarray,
                        cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """Threshold minimising mean cost, searched on the score's own quantile
    grid (401 points) — resolution ~0.25% of the population per step."""
    grid = np.unique(np.quantile(pd_scores, np.linspace(0.0, 1.0, 401)))
    costs = [per_row_cost(y, pd_scores, t, cost_fn, cost_fp).mean() for t in grid]
    return float(grid[int(np.argmin(costs))])


def paired_bootstrap_cost_delta(costs_a: np.ndarray, costs_b: np.ndarray,
                                n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Bootstrap CI for mean(costs_a - costs_b); negative favours arm A."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(costs_a) - np.asarray(costs_b)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "delta_mean": round(float(diff.mean()), 6),
        "delta_ci95": [round(float(np.quantile(means, q)), 6) for q in (0.025, 0.975)],
        "frac_negative": round(float((means < 0).mean()), 4),
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def train_weighted_oof(X, y, quick: bool) -> tuple[np.ndarray, list[float]]:
    from catboost import CatBoostClassifier

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof, fold_aucs = np.zeros(len(y)), []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_fit, X_es, y_fit, y_es = train_test_split(
            X.iloc[tr], y[tr], test_size=0.10, random_state=SEED + k, stratify=y[tr])
        model = CatBoostClassifier(
            iterations=GBM_PARAMS["n_estimators"], learning_rate=GBM_PARAMS["learning_rate"],
            eval_metric="AUC", od_type="Iter", od_wait=GBM_PARAMS["early_stopping"],
            class_weights={0: 1.0, 1: COST_FN / COST_FP},  # 7.5 : 1, the stated ratio
            random_seed=SEED + k, verbose=0, allow_writing_files=False)
        model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[va], oof[va])))
        print(f"    fold {k + 1}/{N_FOLDS}: AUC {fold_aucs[-1]:.4f} ({time.time() - t0:.0f}s)")
    return oof, fold_aucs


def main(quick: bool) -> None:
    started = time.time()
    X, y, ids = build_feature_matrix()
    if quick:
        keep = np.random.default_rng(SEED).choice(len(y), 30_000, replace=False)
        X, y, ids = X.iloc[keep].reset_index(drop=True), y[keep], ids[keep]
        print(f"[quick] subsampled to {len(y):,} rows")

    # Arm A (baseline): the committed unweighted CatBoost FE-v2 OOF predictions.
    v2 = pd.read_csv(V2_OOF_CSV).set_index("SK_ID_CURR")
    pd_base = v2["pd_CatBoost"].reindex(ids).to_numpy()
    assert not np.isnan(pd_base).any(), "v2 OOF csv does not cover these ids"

    # Arm B (weighted): retrain with the stated class weights, same folds.
    if not quick and OOF_CACHE.exists():
        pd_wt, fold_aucs_wt = pd.read_pickle(OOF_CACHE)
        print(f"[weighted] cache hit ({OOF_CACHE.name})")
    else:
        print(f"[weighted] training {N_FOLDS}-fold CatBoost, class_weights 1:{COST_FN / COST_FP:g}")
        pd_wt, fold_aucs_wt = train_weighted_oof(X, y, quick)
        if not quick:
            OOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle((pd_wt, fold_aucs_wt), OOF_CACHE)

    # ---- tune/test; thresholds fit on tune only -----------------------------
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(y))
    half = len(y) // 2
    tune_ix, test_ix = perm[:half], perm[half:]
    y_tu, y_te = y[tune_ix], y[test_ix]

    arms = {"baseline_thresholded": pd_base, "weighted_training": pd_wt}
    evaluation, cost_rows = {}, {}
    for name, scores in arms.items():
        t = best_cost_threshold(y_tu, scores[tune_ix])
        costs = per_row_cost(y_te, scores[test_ix], t)
        cost_rows[name] = costs
        evaluation[name] = {
            "auc_oof": round(float(roc_auc_score(y, scores)), 4),
            "threshold_from_tune": round(t, 4),
            "test_expected_cost_per_applicant": round(float(costs.mean()), 6),
            "test_reject_rate": round(float((scores[test_ix] >= t).mean()), 4),
            "test_missed_defaulter_rate": round(float(
                ((scores[test_ix] < t) & (y_te == 1)).mean()), 5),
        }

    boot = paired_bootstrap_cost_delta(
        cost_rows["weighted_training"], cost_rows["baseline_thresholded"])
    claim = bool(boot["delta_mean"] < 0 and boot["delta_ci95"][1] < 0)

    # ---- robustness: decision-time cost-ratio sweep (weights fixed) ---------
    sweep = []
    for ratio in COST_RATIOS:
        fn, fp = ratio / (ratio + 1.0), 1.0 / (ratio + 1.0)  # normalised pair
        row = {"cost_ratio_fn_to_fp": ratio}
        for name, scores in arms.items():
            t = best_cost_threshold(y_tu, scores[tune_ix], fn, fp)
            row[name] = round(float(per_row_cost(y_te, scores[test_ix], t, fn, fp).mean()), 6)
        row["weighted_wins"] = bool(row["weighted_training"] < row["baseline_thresholded"])
        sweep.append(row)

    report = {
        "protocol": {
            "rows": int(len(y)), "quick_mode": quick,
            "folds": f"{N_FOLDS}-fold StratifiedKFold seed {SEED}; baseline arm reuses "
                     "reports/v2_oof_predictions.csv (committed, no retrain)",
            "cost_model": {"cost_fn_approve_defaulter": COST_FN,
                           "cost_fp_reject_good": COST_FP,
                           "ratio": COST_FN / COST_FP,
                           "assumptions": "LGD ~0.75 unsecured retail; ~0.10 forgone margin"},
            "split": f"50/50 tune/test seed {SPLIT_SEED}; thresholds minimise tune cost",
            "claim_rule": "weighted arm claimable only if paired-bootstrap 95% CI of "
                          "its test-cost delta vs the thresholded baseline excludes 0",
        },
        "arms": evaluation,
        "weighted_fold_aucs": [round(a, 4) for a in fold_aucs_wt],
        "paired_bootstrap_weighted_minus_baseline_cost": boot,
        "claim_granted": claim,
        "decision_time_cost_ratio_sweep": sweep,
        "reading": ("Elkan (2001): with calibrated probabilities, class-weighted "
                    "training should be equivalent to threshold shifting; a null "
                    "result here means SmartLend's economics rightly live in the "
                    "decision layer, not the loss function."),
        "generated_seconds": round(time.time() - started, 1),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name, e in evaluation.items():
        print(f"{name:<24} cost/applicant {e['test_expected_cost_per_applicant']:.6f}  "
              f"auc {e['auc_oof']}  reject {e['test_reject_rate']:.2%}")
    print(f"delta (weighted - baseline): {boot['delta_mean']:+.6f} "
          f"CI95 {boot['delta_ci95']}  claim_granted={claim}")
    print(f"(report -> {OUT_JSON})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="30k-row smoke run")
    main(quick=parser.parse_args().quick)
