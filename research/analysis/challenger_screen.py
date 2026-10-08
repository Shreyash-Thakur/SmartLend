"""2026 tabular-SOTA challengers vs the FE-v2 CatBoost — the screening round.

WHY A SCREEN. The project's standing verdict is that headroom lives in
information, not models (reports/features_vs_fusion.json), and four model
hypotheses have already been killed under the pre-registered ±0.0036 rule.
Before spending a full 307k x 5-fold run on any new challenger, it must earn
it on the 30k quick protocol — the same funnel that correctly predicted
TabPFN's full-run failure from its quick-run numbers.

CHALLENGERS (chosen from a 2026 survey of TabArena and recent releases; both
have PERMISSIVE licenses, unlike the TabPFN family, so a win would also be
deployable):

* TabICLv2 (Inria, BSD-3-Clause, ~110MB checkpoint) — in-context tabular
  foundation model claiming million-scale support with memory offloading.
  arXiv:2602.11139.
* RealMLP-TD (pytabkit, Apache-2.0) — trainable deep tabular MLP with
  benchmark-tuned defaults; TabArena-competitive, no row ceiling.
  arXiv:2407.04491.

PRE-REGISTERED RULES (fixed before results exist):

1. Same 30k subsample and same 5 folds as features_vs_fusion --quick:
   rng(42) choice of 30,000 rows from the v2 matrix, StratifiedKFold(5,
   shuffle, seed 42). The CatBoost baseline is re-fit here with the exact
   features_vs_fusion protocol so deltas are fold-paired.
2. ADVANCE rule: a challenger earns a full-scale run only if its fold-paired
   mean OOF-AUC delta vs CatBoost is POSITIVE on the screen.
3. CLAIM rule (unchanged, decided only at full scale): fold-paired mean
   delta > +0.0036 with every fold positive.
4. No tuning on the screen: library defaults / benchmark-tuned defaults
   only. A challenger that needs dataset-specific tuning to pass a screen
   has already lost to an untuned CatBoost.

Run:  python -m research.analysis.challenger_screen
Writes reports/challenger_screen.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "reports" / "challenger_screen.json"

SEED = 42
N_FOLDS = 5
QUICK_N = 30_000
GBM_PARAMS = {"n_estimators": 2000, "learning_rate": 0.05, "early_stopping": 100}
NOISE_FLOOR = 0.0036


def _fold_catboost(X_tr, y_tr, X_va):
    from catboost import CatBoostClassifier

    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr, y_tr, test_size=0.10, random_state=SEED, stratify=y_tr)
    model = CatBoostClassifier(
        iterations=GBM_PARAMS["n_estimators"], learning_rate=GBM_PARAMS["learning_rate"],
        eval_metric="AUC", od_type="Iter", od_wait=GBM_PARAMS["early_stopping"],
        random_seed=SEED, verbose=0, allow_writing_files=False)
    model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
    return model.predict_proba(X_va)[:, 1]


def _fold_tabicl(X_tr, y_tr, X_va):
    from tabicl import TabICLClassifier

    # Defaults per rule 4, plus the documented memory mechanics an 8GB GPU
    # requires (disk offload + smaller batches are capacity plumbing, not
    # model tuning): first attempt estimated an 18GB inference output.
    offload = REPO / "data" / "processed" / "tabicl_offload"
    offload.mkdir(parents=True, exist_ok=True)
    model = TabICLClassifier(
        random_state=SEED, batch_size=2, disk_offload_dir=str(offload))
    model.fit(X_tr, y_tr)  # in-context "fit": stores the context
    return model.predict_proba(X_va)[:, 1]


def _fold_realmlp(X_tr, y_tr, X_va):
    from sklearn.impute import SimpleImputer
    from pytabkit import RealMLP_TD_Classifier

    # Neural nets cannot ingest NaN: median imputation fit on the training
    # fold is mandatory preprocessing (identical to the LogisticRegression
    # arm of features_vs_fusion), not dataset-specific tuning.
    imp = SimpleImputer(strategy="median").fit(X_tr)
    model = RealMLP_TD_Classifier(random_state=SEED, verbosity=0)
    model.fit(imp.transform(X_tr), y_tr)
    return model.predict_proba(imp.transform(X_va))[:, 1]


CHALLENGERS = {
    "CatBoost_baseline": _fold_catboost,
    "TabICLv2": _fold_tabicl,
    "RealMLP_TD": _fold_realmlp,
}


def main() -> None:
    started = time.time()
    X, y, _ids = build_feature_matrix()
    keep = np.random.default_rng(SEED).choice(len(y), QUICK_N, replace=False)
    X, y = X.iloc[keep].reset_index(drop=True), y[keep]
    print(f"[screen] {len(y):,} rows x {X.shape[1]} features")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    results: dict = {"protocol": {
        "rows": int(len(y)), "folds": N_FOLDS, "seed": SEED,
        "advance_rule": "fold-paired mean delta vs CatBoost > 0 on this screen",
        "claim_rule_at_full_scale": f"mean delta > +{NOISE_FLOOR} AND all folds positive",
        "tuning": "none - defaults only (rule 4)",
    }, "models": {}}

    fold_aucs: dict[str, list[float]] = {}
    for name, fit_predict in CHALLENGERS.items():
        aucs = []
        print(f"[{name}]")
        for k, (tr, va) in enumerate(folds):
            t0 = time.time()
            try:
                p = fit_predict(X.iloc[tr], y[tr], X.iloc[va])
                auc = float(roc_auc_score(y[va], p))
            except Exception as exc:  # a challenger that cannot run has failed the screen
                results["models"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"    FAILED: {type(exc).__name__}: {exc}")
                aucs = []
                break
            aucs.append(auc)
            print(f"    fold {k + 1}/{N_FOLDS}: AUC {auc:.4f} ({time.time() - t0:.0f}s)")
        if aucs:
            fold_aucs[name] = aucs
            results["models"][name] = {
                "fold_aucs": [round(a, 4) for a in aucs],
                "mean_auc": round(float(np.mean(aucs)), 4),
            }

    base = fold_aucs.get("CatBoost_baseline")
    for name, aucs in fold_aucs.items():
        if name == "CatBoost_baseline" or base is None:
            continue
        deltas = [round(c - b, 4) for b, c in zip(base, aucs)]
        mean_delta = round(float(np.mean(deltas)), 4)
        results["models"][name]["fold_paired_delta_vs_catboost"] = deltas
        results["models"][name]["mean_delta"] = mean_delta
        results["models"][name]["advances_to_full_run"] = bool(mean_delta > 0)
        print(f"[{name}] mean delta vs CatBoost {mean_delta:+.4f} "
              f"-> {'ADVANCES' if mean_delta > 0 else 'ELIMINATED'}")

    results["generated_seconds"] = round(time.time() - started, 1)
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[report] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
