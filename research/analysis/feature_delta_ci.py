"""Paired-bootstrap confidence intervals for the FE-v2 feature deltas.

Review item: the fold-paired deltas in ``reports/features_vs_fusion.json``
exceed the pre-registered practical floor (+-0.0036 AUC), but "exceeds the
practical threshold" and "statistically distinguishable from zero" are
different statements. This script adds the second one: for each model, a
paired bootstrap (resampling ROWS, so v1 and v2 predictions stay paired) of
AUC(v2) - AUC(v1) on the full joined OOF population.

Inputs (both already committed artifacts):
- backend/artifacts/prediction_outputs.csv  -- v1 OOF predictions.
  y_true == 1 means GOOD; prob_* are APPROVAL probabilities.
- reports/v2_oof_predictions.csv            -- FE-v2 OOF predictions.
  y_target == 1 means DEFAULT; pd_* are DEFAULT probabilities.
Both AUCs are computed in the default framing: AUC(y_default, pd) with
pd_v1 = 1 - prob_v1. AUC is invariant to this flip; the flip only aligns
the two files' conventions. Row identity is verified via the label check.

Run: ``python -m research.analysis.feature_delta_ci``
Writes ``reports/feature_delta_ci.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "reports" / "feature_delta_ci.json"

BOOT_ITERS = 1000
BOOT_SEED = 20260831  # same bootstrap convention as reports/complementarity.json

MODELS = {
    "LightGBM": ("prob_LightGBM", "pd_LightGBM"),
    "XGBoost": ("prob_XGBoost", "pd_XGBoost"),
    "CatBoost": ("prob_CatBoost", "pd_CatBoost"),
    "LogisticRegression": ("prob_Logistic Regression", "pd_LogisticRegression"),
}


def main() -> int:
    v1 = pd.read_csv(REPO_ROOT / "backend" / "artifacts" / "prediction_outputs.csv")
    v2 = pd.read_csv(REPO_ROOT / "reports" / "v2_oof_predictions.csv")

    v1["SK_ID_CURR"] = v1["applicant_id"].astype(str).str.removeprefix("HC").astype(int)
    joined = v1.merge(v2, on="SK_ID_CURR", how="inner", validate="one_to_one")
    assert len(joined) == len(v2) == len(v1), "row sets must match exactly"
    # label consistency across the two artifacts (y_true: 1=good; y_target: 1=default)
    mismatches = int((joined["y_true"] != (1 - joined["y_target"])).sum())
    assert mismatches == 0, f"label mismatch on {mismatches} rows - misaligned join"

    y_def = joined["y_target"].to_numpy(dtype=int)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(joined)
    idx_draws = rng.integers(0, n, size=(BOOT_ITERS, n))

    results = {}
    for name, (c1, c2) in MODELS.items():
        pd_v1 = 1.0 - joined[c1].to_numpy(dtype=float)
        pd_v2 = joined[c2].to_numpy(dtype=float)
        full_delta = float(roc_auc_score(y_def, pd_v2) - roc_auc_score(y_def, pd_v1))
        deltas = np.empty(BOOT_ITERS)
        for i in range(BOOT_ITERS):
            idx = idx_draws[i]
            yb = y_def[idx]
            deltas[i] = roc_auc_score(yb, pd_v2[idx]) - roc_auc_score(yb, pd_v1[idx])
        results[name] = {
            "auc_v1": float(roc_auc_score(y_def, pd_v1)),
            "auc_v2": float(roc_auc_score(y_def, pd_v2)),
            "delta_full_sample": full_delta,
            "bootstrap": {
                "iterations": BOOT_ITERS,
                "seed": BOOT_SEED,
                "delta_mean": float(deltas.mean()),
                "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                "prob_delta_nonpositive": float((deltas <= 0).mean()),
            },
            "exceeds_practical_floor_0.0036": bool(full_delta > 0.0036),
            "ci_excludes_zero": bool(np.percentile(deltas, 2.5) > 0),
        }
        print(f"{name:<20} v1 {results[name]['auc_v1']:.4f}  v2 {results[name]['auc_v2']:.4f}  "
              f"delta {full_delta:+.4f}  CI95 [{results[name]['bootstrap']['ci95'][0]:+.4f}, "
              f"{results[name]['bootstrap']['ci95'][1]:+.4f}]")

    report = {
        "protocol": {
            "n_rows": int(n),
            "join": "SK_ID_CURR one-to-one; label consistency asserted (0 mismatches)",
            "note": (
                "Pooled-OOF paired bootstrap; complements (does not replace) the "
                "fold-paired deltas in reports/features_vs_fusion.json. The "
                "pre-registered +-0.0036 floor is a PRACTICAL effect-size "
                "criterion; the CI answers statistical distinguishability."
            ),
        },
        "models": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
