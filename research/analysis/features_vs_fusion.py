"""Features vs fusion — where does accuracy headroom actually come from?

THE RESEARCH QUESTION. The project has a chain of pre-registered negative
results showing that FUSING models adds nothing on the current feature set
(CBES, the tree family, TabPFN-2.5, TabFM — all KILL;
reports/complementarity.json), while tuning is exhausted and the learning
curve says the remaining lever is information, not optimisation. This
experiment tests the complementary hypothesis directly: hold the models, the
folds and the evaluation fixed, change ONLY the features (v1 = the current
~130-column merged frame; v2 = engineered application ratios + deep bureau
aggregates, research/features/engineer.py), and then re-run the IDENTICAL
fusion suite on the v2 predictions. If v2 clears the noise floor where every
hybrid failed, the conclusion is that this dataset pays for feature
engineering and not for ensembling — with both halves measured under the same
rule.

PRE-REGISTERED DECISION RULES (fixed here, before the results exist; the same
±0.0036 fold-noise criterion that killed the CatBoost/TabPFN/TabFM gains):

1. FEATURE CLAIM — "v2 beats v1" may be claimed for a model only if the
   fold-paired mean AUC delta (v2 − v1, same 5 folds, seed 42) exceeds
   +0.0036 AND every fold's delta is positive.
2. FUSION CLAIM ON V2 — "ensembling helps on v2" may be claimed only if the
   best HONEST combination (simple average, rank average, or out-of-fold
   logistic stack — never the self-selected weight sweep) beats the best
   single v2 model by more than +0.0036.
3. No post-hoc metric shopping: ROC-AUC on the out-of-fold predictions is the
   criterion, PR-AUC is reported alongside for context.

HONESTY NOTES. Early stopping uses an inner 10% split carved from the
TRAINING folds only (never the evaluation fold). The v1/v2 frames are aligned
row-for-row by SK_ID_CURR so the folds are literally identical. This is the
research reference track; the serving artifact's 15-feature contract is
untouched.

Run:  python -m research.analysis.features_vs_fusion [--quick]
Outputs: reports/features_vs_fusion.json, reports/v2_oof_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.analysis.complementarity import (
    FOLD_STD,
    cv_stack_auc,
    paired_bootstrap_auc_delta,
    rank_average,
    weight_sweep,
)
from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "reports" / "features_vs_fusion.json"
OUT_OOF = REPO / "reports" / "v2_oof_predictions.csv"

SEED = 42
N_FOLDS = 5
GBM_PARAMS = {"n_estimators": 2000, "learning_rate": 0.05, "early_stopping": 100}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_v1_aligned(ids: np.ndarray) -> pd.DataFrame:
    """The CURRENT feature frame (merged CSV, numeric columns), reindexed to
    the v2 row order so both feature sets share identical folds."""
    from backend.app.services.customer_profile_service import _resolve_source_path

    csv_path = _resolve_source_path()
    if csv_path is None:
        raise SystemExit("Home Credit extract not found; set SMARTLEND_CUSTOMER_DATA.")
    raw = pd.read_csv(csv_path, low_memory=False)
    raw = raw.set_index("SK_ID_CURR")
    X1 = (raw.drop(columns=["TARGET"], errors="ignore")
             .select_dtypes(include=["number"])
             .reindex(ids)
             .reset_index(drop=True)
             .astype(np.float32))
    assert len(X1) == len(ids), "v1 frame does not cover the v2 ids"
    return X1


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def _fit_predict_fold(name: str, X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    """Fit one model on the training folds, return P(default) on the eval fold.

    GBMs early-stop on an inner 10% split of the TRAINING data only.
    """
    if name in ("LightGBM", "XGBoost", "CatBoost"):
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr, y_tr, test_size=0.10, random_state=seed, stratify=y_tr)
    if name == "LightGBM":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=GBM_PARAMS["n_estimators"], learning_rate=GBM_PARAMS["learning_rate"],
            num_leaves=63, random_state=seed, n_jobs=-1, verbose=-1)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(GBM_PARAMS["early_stopping"], verbose=False)])
        return model.predict_proba(X_va)[:, 1]
    if name == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=GBM_PARAMS["n_estimators"], learning_rate=GBM_PARAMS["learning_rate"],
            max_depth=6, tree_method="hist", eval_metric="auc",
            early_stopping_rounds=GBM_PARAMS["early_stopping"], random_state=seed, n_jobs=-1)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        return model.predict_proba(X_va)[:, 1]
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=GBM_PARAMS["n_estimators"], learning_rate=GBM_PARAMS["learning_rate"],
            eval_metric="AUC", od_type="Iter", od_wait=GBM_PARAMS["early_stopping"],
            random_seed=seed, verbose=0, allow_writing_files=False)
        model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
        return model.predict_proba(X_va)[:, 1]
    if name == "LogisticRegression":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_va)[:, 1]
    raise ValueError(name)


def oof_run(name: str, X: pd.DataFrame, y: np.ndarray, folds) -> tuple[np.ndarray, list[float]]:
    oof = np.zeros(len(y))
    fold_aucs = []
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        oof[va] = _fit_predict_fold(name, X.iloc[tr], y[tr], X.iloc[va], seed=SEED + k)
        fold_aucs.append(float(roc_auc_score(y[va], oof[va])))
        print(f"    fold {k + 1}/{len(folds)}: AUC {fold_aucs[-1]:.4f} ({time.time() - t0:.0f}s)")
    return oof, fold_aucs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(quick: bool) -> None:
    started = time.time()
    X2, y, ids = build_feature_matrix()
    X1 = load_v1_aligned(ids)

    if quick:
        keep = np.random.default_rng(SEED).choice(len(y), 30_000, replace=False)
        X1, X2, y, ids = X1.iloc[keep].reset_index(drop=True), X2.iloc[keep].reset_index(drop=True), y[keep], ids[keep]
        print(f"[quick] subsampled to {len(y):,} rows")

    print(f"[data] v1: {X1.shape[1]} features | v2: {X2.shape[1]} features | rows {len(y):,}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X2, y))

    models = ["LightGBM", "XGBoost", "CatBoost", "LogisticRegression"]
    results: dict = {
        "protocol": {
            "folds": N_FOLDS, "seed": SEED, "rows": int(len(y)),
            "v1_features": int(X1.shape[1]), "v2_features": int(X2.shape[1]),
            "early_stopping": "inner 10% split of the training folds only",
            "noise_floor": FOLD_STD, "quick_mode": quick,
        },
        "models": {},
    }
    oof_store: dict[str, np.ndarray] = {}

    for name in models:
        entry: dict = {}
        for label, X in (("v1", X1), ("v2", X2)):
            print(f"[{name}] {label} ({X.shape[1]} features)")
            oof, fold_aucs = oof_run(name, X, y, folds)
            entry[label] = {
                "oof_auc": round(float(roc_auc_score(y, oof)), 4),
                "oof_pr_auc": round(float(average_precision_score(y, oof)), 4),
                "fold_aucs": [round(a, 4) for a in fold_aucs],
            }
            if label == "v2":
                oof_store[name] = oof
        deltas = [round(b - a, 4) for a, b in zip(entry["v1"]["fold_aucs"], entry["v2"]["fold_aucs"])]
        mean_delta = float(np.mean(deltas))
        entry["fold_paired_delta_v2_minus_v1"] = deltas
        entry["mean_delta"] = round(mean_delta, 4)
        entry["feature_claim"] = bool(mean_delta > FOLD_STD and all(d > 0 for d in deltas))
        print(f"[{name}] v1 {entry['v1']['oof_auc']} -> v2 {entry['v2']['oof_auc']} "
              f"(mean fold delta {mean_delta:+.4f}, claim={entry['feature_claim']})")
        results["models"][name] = entry

    # ---- Fusion suite on the v2 OOF predictions (pre-registered rule 2) ----
    best_single = max(oof_store, key=lambda m: roc_auc_score(y, oof_store[m]))
    p_best = oof_store[best_single]
    others = [m for m in models if m != best_single]
    p_all = np.column_stack([oof_store[m] for m in models])
    combos = {
        "simple_average_all": p_all.mean(axis=1),
        "rank_average_all": rank_average(*[oof_store[m] for m in models]),
        "simple_average_gbms": np.column_stack(
            [oof_store[m] for m in ("LightGBM", "XGBoost", "CatBoost")]).mean(axis=1),
    }
    fusion: dict = {
        "best_single_model": best_single,
        "best_single_auc": round(float(roc_auc_score(y, p_best)), 4),
        "pairwise_oof_corr": {
            f"{a}|{b}": round(float(np.corrcoef(oof_store[a], oof_store[b])[0, 1]), 4)
            for i, a in enumerate(models) for b in models[i + 1:]
        },
    }
    honest_best_name, honest_best_auc = None, -1.0
    for cname, probs in combos.items():
        auc = float(roc_auc_score(y, probs))
        fusion[f"auc_{cname}"] = round(auc, 4)
        if auc > honest_best_auc:
            honest_best_name, honest_best_auc, honest_best_probs = cname, auc, probs
    stack_auc = cv_stack_auc(y, p_best, oof_store[others[0]])  # best + strongest partner
    fusion["auc_cv_stack_best_plus_" + others[0].lower()] = round(stack_auc, 4)
    if stack_auc > honest_best_auc:
        honest_best_name, honest_best_auc, honest_best_probs = "cv_stack", stack_auc, None
    w, w_auc, _ = weight_sweep(y, p_best, combos["simple_average_gbms"])
    fusion["weight_sweep_note"] = {
        "w_best_single": w, "auc": round(w_auc, 4),
        "note": "self-selected weight - optimistic upper bound, never claimable",
    }
    gain = honest_best_auc - fusion["best_single_auc"]
    fusion["best_honest_fusion"] = {
        "method": honest_best_name,
        "auc": round(honest_best_auc, 4),
        "gain_vs_best_single": round(gain, 4),
        "gain_exceeds_noise_floor": bool(gain > FOLD_STD),
    }
    if honest_best_probs is not None:
        fusion["paired_bootstrap_vs_best_single"] = paired_bootstrap_auc_delta(
            y, honest_best_probs, p_best)
    results["fusion_on_v2"] = fusion

    # ---- The headline comparison the experiment exists for ----------------
    best_v1 = max(results["models"][m]["v1"]["oof_auc"] for m in models)
    best_v2 = max(results["models"][m]["v2"]["oof_auc"] for m in models)
    results["headline"] = {
        "best_single_v1_auc": best_v1,
        "best_single_v2_auc": best_v2,
        "feature_gain": round(best_v2 - best_v1, 4),
        "best_fusion_gain_on_v2": fusion["best_honest_fusion"]["gain_vs_best_single"],
        "reading": (
            "features vs fusion: the feature gain and the fusion gain above are "
            "judged by the SAME +0.0036 rule; whichever clears it is where the "
            "headroom lives."
        ),
    }

    # ---- Persist -----------------------------------------------------------
    if not quick:
        oof_df = pd.DataFrame({"SK_ID_CURR": ids, "y_target": y})
        for m in models:
            oof_df[f"pd_{m}"] = np.round(oof_store[m], 6)
        oof_df.to_csv(OUT_OOF, index=False)
        print(f"[oof] wrote {OUT_OOF}")

    results["generated_seconds"] = round(time.time() - started, 1)
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[report] wrote {OUT_JSON}")
    print(json.dumps(results["headline"], indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="30k-row smoke run")
    main(quick=parser.parse_args().quick)
