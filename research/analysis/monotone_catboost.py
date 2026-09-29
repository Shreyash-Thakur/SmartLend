"""Monotone-constrained CatBoost — the measured price of guaranteed directions.

THE NOVELTY CLAIM UNDER TEST. RBI's FREE-AI framework and the Digital Lending
Directions ask lenders to explain adverse decisions in terms a customer can
act on. An unconstrained GBM cannot GUARANTEE that "your bureau score improved,
so your risk went down" — locally, SHAP can point the other way. Enforcing
monotonicity on the domain-signed features makes every directional explanation
true BY CONSTRUCTION, turning a compliance narrative into a model property.
The open question is what that guarantee costs in AUC; this experiment prices
it under the project's standard fold-paired protocol.

PRE-REGISTERED DECISION RULES (fixed before results exist):

1. Baseline = the CatBoost FE-v2 fold AUCs already recorded in
   reports/features_vs_fusion.json (same 5 folds seed 42, same params, same
   early-stopping discipline). This script re-runs ONLY the constrained arm.
2. The constraint set is fixed below, from domain knowledge only — no peeking
   at the data to pick which features to constrain.
3. VERDICT "guarantee is free": |fold-paired mean AUC delta| <= 0.0036 (the
   fold-noise floor). VERDICT "guarantee has a price": mean delta < -0.0036,
   and the price is reported, not hidden. A POSITIVE delta above +0.0036
   would mean the constraints act as regularisation — claimable only under
   the same rule that governed every other claim.
4. Monotonicity is verified empirically after training: sweep three
   constrained features over their 5th-95th percentile range on 200 random
   applicants and count direction violations (must be 0).

Direction convention: the model predicts P(default), training label 1 =
DEFAULT, so a feature that makes an applicant SAFER gets -1.

Run:  python -m research.analysis.monotone_catboost [--quick]
Writes reports/monotone_catboost.json.
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
OUT_JSON = REPO / "reports" / "monotone_catboost.json"
BASELINE_JSON = REPO / "reports" / "features_vs_fusion.json"

SEED = 42
N_FOLDS = 5
GBM_PARAMS = {"n_estimators": 2000, "learning_rate": 0.05, "early_stopping": 100}
NOISE_FLOOR = 0.0036

# The pre-registered constraint set: financial-behaviour features whose
# direction w.r.t. default risk is domain knowledge, not a data artifact.
# (Deliberately excludes age/family/region features — RBI fairness posture.)
CONSTRAINTS: dict[str, int] = {
    # External credit scores: higher score, safer. The core of every adverse-
    # action letter ("improve your credit score").
    "EXT_SOURCE_1": -1, "EXT_SOURCE_2": -1, "EXT_SOURCE_3": -1,
    "APP_EXT_MEAN": -1, "APP_EXT_MIN": -1, "APP_EXT_MAX": -1,
    "APP_EXT_PROD": -1, "APP_EXT_PROD3": -1, "APP_EXT2_X_EXT3": -1,
    # Payment burden: more of your income committed, riskier.
    "APP_ANNUITY_TO_INCOME": 1, "APP_CREDIT_TO_INCOME": 1,
    # Bureau distress: overdue amounts and days, riskier.
    "BURO_overdue_sum": 1, "BURO_overdue_max": 1, "BURO_max_overdue_max": 1,
    "BURO_day_overdue_max": 1, "BURO_day_overdue_mean": 1,
    # Leverage on the bureau file: more of your granted credit still owed, riskier.
    "BURO_debt_to_credit_mean": 1, "BURO_debt_to_credit_max": 1,
    "BURO_total_debt_to_credit": 1, "BURO_overdue_to_debt": 1,
    # Loan prolongations: needing extensions, riskier.
    "BURO_prolong_sum": 1,
}

VERIFY_FEATURES = ["APP_EXT_MEAN", "APP_ANNUITY_TO_INCOME", "BURO_total_debt_to_credit"]


def build_constraint_vector(columns) -> list[int]:
    """Per-column monotone directions in X's column order (0 = unconstrained)."""
    return [CONSTRAINTS.get(c, 0) for c in columns]


def verify_monotonicity(model, X: pd.DataFrame, n_rows: int = 200, n_grid: int = 25,
                        seed: int = SEED) -> dict:
    """Empirical check: sweeping a constrained feature must move P(default)
    only in its declared direction. Returns violation counts per feature."""
    rng = np.random.default_rng(seed)
    rows = X.iloc[rng.choice(len(X), size=min(n_rows, len(X)), replace=False)]
    checks = {}
    for feat in VERIFY_FEATURES:
        direction = CONSTRAINTS[feat]
        lo, hi = np.nanpercentile(X[feat], [5, 95])
        grid = np.linspace(lo, hi, n_grid)
        preds = np.empty((len(rows), n_grid))
        for j, v in enumerate(grid):
            sweep = rows.copy()
            sweep[feat] = np.float32(v)
            preds[:, j] = model.predict_proba(sweep)[:, 1]
        diffs = np.diff(preds, axis=1) * direction  # declared direction => >= 0
        checks[feat] = {
            "direction": direction,
            "violations": int((diffs < -1e-9).sum()),
            "checked_transitions": int(diffs.size),
        }
    return checks


def main(quick: bool) -> None:
    started = time.time()
    X, y, ids = build_feature_matrix()
    if quick:
        keep = np.random.default_rng(SEED).choice(len(y), 30_000, replace=False)
        X, y = X.iloc[keep].reset_index(drop=True), y[keep]
        print(f"[quick] subsampled to {len(y):,} rows")

    missing = [c for c in CONSTRAINTS if c not in X.columns]
    assert not missing, f"constraint refers to absent columns: {missing}"
    vector = build_constraint_vector(X.columns)
    n_constrained = sum(1 for v in vector if v != 0)
    print(f"[monotone] {n_constrained} constrained of {X.shape[1]} features")

    from catboost import CatBoostClassifier

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs, last_model = [], None
    for k, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_fit, X_es, y_fit, y_es = train_test_split(
            X.iloc[tr], y[tr], test_size=0.10, random_state=SEED + k, stratify=y[tr])
        model = CatBoostClassifier(
            iterations=GBM_PARAMS["n_estimators"], learning_rate=GBM_PARAMS["learning_rate"],
            eval_metric="AUC", od_type="Iter", od_wait=GBM_PARAMS["early_stopping"],
            monotone_constraints=vector,
            random_seed=SEED + k, verbose=0, allow_writing_files=False)
        model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
        fold_aucs.append(float(roc_auc_score(y[va], model.predict_proba(X.iloc[va])[:, 1])))
        last_model = model
        print(f"    fold {k + 1}/{N_FOLDS}: AUC {fold_aucs[-1]:.4f} ({time.time() - t0:.0f}s)")

    baseline = json.loads(BASELINE_JSON.read_text())["models"]["CatBoost"]["v2"]
    if quick:
        deltas, mean_delta = None, None
        print("[quick] baseline comparison skipped (baseline folds are full-run)")
    else:
        deltas = [round(m - b, 4) for b, m in zip(baseline["fold_aucs"], fold_aucs)]
        mean_delta = round(float(np.mean(deltas)), 4)

    checks = verify_monotonicity(last_model, X)
    total_violations = sum(c["violations"] for c in checks.values())

    verdict = None
    if mean_delta is not None:
        if abs(mean_delta) <= NOISE_FLOOR:
            verdict = "guarantee is free (|delta| within fold noise)"
        elif mean_delta < 0:
            verdict = f"guarantee costs {-mean_delta:.4f} AUC"
        else:
            verdict = f"constraints act as regularisation (+{mean_delta:.4f} AUC)"

    report = {
        "protocol": {
            "rows": int(len(y)), "quick_mode": quick,
            "folds": f"{N_FOLDS}-fold StratifiedKFold seed {SEED}, params and early "
                     "stopping identical to features_vs_fusion's CatBoost arm",
            "baseline": "CatBoost v2 fold AUCs from reports/features_vs_fusion.json",
            "noise_floor": NOISE_FLOOR,
        },
        "constraints": {
            "n_constrained": n_constrained,
            "of_features": int(X.shape[1]),
            "directions": CONSTRAINTS,
            "rationale": "financial-behaviour features only; age/family/region "
                         "deliberately unconstrained (fairness posture)",
        },
        "constrained_model": {
            "oof_fold_aucs": [round(a, 4) for a in fold_aucs],
            "mean_fold_auc": round(float(np.mean(fold_aucs)), 4),
        },
        "baseline_model": {"fold_aucs": baseline["fold_aucs"], "oof_auc": baseline["oof_auc"]},
        "fold_paired_delta_monotone_minus_baseline": deltas,
        "mean_delta": mean_delta,
        "verdict": verdict,
        "monotonicity_verification": {
            "method": "5th-95th percentile sweep, 200 applicants, 3 features",
            "per_feature": checks,
            "total_violations": total_violations,
            "holds": bool(total_violations == 0),
        },
        "generated_seconds": round(time.time() - started, 1),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[monotone] mean fold AUC {np.mean(fold_aucs):.4f} "
          f"(baseline {baseline['oof_auc']}) delta={mean_delta} "
          f"violations={total_violations}\nVERDICT: {verdict}  (report -> {OUT_JSON})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="30k-row smoke run")
    main(quick=parser.parse_args().quick)
