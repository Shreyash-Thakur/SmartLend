"""Controlled distribution-shift stress tests (reviewer item).

Question: does epistemic (knowledge) uncertainty from SGLB virtual ensembles
detect when the model operates outside its training distribution, and does
uncertainty-routed deferral still protect decision quality under shift?

These are CONTROLLED STRESS TESTS on synthetic corruptions of the Home
Credit test split, labelled as such; they are not evidence of real-world
generalisation. One SGLB CatBoost (1000 iterations, posterior sampling,
K=10) is trained on a 60% train slice; a 20% validation slice fits the
Youden threshold and the clean 22.5% knowledge-deferral quantile; the 20%
test slice is then evaluated under five scenarios:

  clean            : untouched test slice
  missing_info_50  : for a random 50% of rows, all EXT_SOURCE and
                     bureau-derived columns set to missing (simulates
                     degraded data feeds / forced thin-file)
  thin_file_only   : the test slice restricted to applicants with no
                     bureau record (a real, not synthetic, subpopulation)
  prior_shift_2x   : goods subsampled until the default rate doubles
  noise_20pct      : zero-mean Gaussian noise (0.5 column SD) added to a
                     random 20% of numeric feature columns

Per scenario: AUC, 10-bin ECE, mean knowledge uncertainty relative to
clean, the deferral rate AT THE CLEAN THRESHOLD (a free monitoring signal:
a stable quantile rule should fire more often under shift), and selective
risk at a matched 22.5% knowledge deferral versus the scenario's own
overall error rate (does deferral still beat random under shift?).

Run: ``python -m research.analysis.shift_stress_tests``
Writes ``reports/shift_stress_tests.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from research.analysis.uncertainty_deferral_catboost import (
    decompose_uncertainty,
    youden_threshold,
)
from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "home_credit"
OUT_JSON = REPO / "reports" / "shift_stress_tests.json"

SEED = 42
ITERATIONS = 1000
K_MEMBERS = 10
TARGET_RATE = 0.225
NOISE_COL_SHARE = 0.20
NOISE_SD_MULT = 0.5
MISSING_ROW_SHARE = 0.50


def member_probs(model, X) -> np.ndarray:
    raw = np.asarray(model.virtual_ensembles_predict(
        X, prediction_type="VirtEnsembles", virtual_ensembles_count=K_MEMBERS))
    return 1.0 / (1.0 + np.exp(-raw[:, :, 0]))


def ece_10bin(pd_scores: np.ndarray, y_def: np.ndarray) -> float:
    bins = np.clip((pd_scores * 10).astype(int), 0, 9)
    ece, n = 0.0, len(pd_scores)
    for b in range(10):
        m = bins == b
        if m.any():
            ece += (m.sum() / n) * abs(pd_scores[m].mean() - y_def[m].mean())
    return float(ece)


def evaluate(model, X: pd.DataFrame, y: np.ndarray, t_star: float,
             know_thr_clean: float, mean_know_clean: float | None) -> dict:
    parts = decompose_uncertainty(member_probs(model, X))
    pd_s, know = parts["pd"], parts["knowledge"]
    errors = ((pd_s >= t_star).astype(int) != y).astype(int)
    # matched 22.5% deferral by knowledge, threshold from THIS scenario's scores
    thr = float(np.quantile(know, 1.0 - TARGET_RATE))
    defer = know > thr
    keep = ~defer
    out = {
        "n": int(len(y)),
        "default_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, pd_s)) if len(np.unique(y)) == 2 else None,
        "ece_10bin": ece_10bin(pd_s, y),
        "overall_error_rate": float(errors.mean()),
        "mean_knowledge": float(know.mean()),
        "deferral_rate_at_clean_threshold": float((know > know_thr_clean).mean()),
        "selective_risk_at_matched_rate": float(errors[keep].mean()),
        "deferral_beats_random": bool(errors[keep].mean() < errors.mean()),
    }
    if mean_know_clean:
        out["mean_knowledge_ratio_vs_clean"] = float(know.mean() / mean_know_clean)
    return out


def main() -> int:
    from catboost import CatBoostClassifier

    X, y, ids = build_feature_matrix()
    bureau_ids = set(pd.read_csv(RAW / "bureau.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"].unique())
    thin = ~pd.Series(ids).isin(bureau_ids).to_numpy()

    idx = np.arange(len(y))
    tr_ix, rest_ix = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=y)
    val_ix, te_ix = train_test_split(rest_ix, test_size=0.5, random_state=SEED, stratify=y[rest_ix])

    t0 = time.time()
    model = CatBoostClassifier(
        iterations=ITERATIONS, learning_rate=0.05, posterior_sampling=True,
        random_seed=SEED, verbose=0, allow_writing_files=False)
    model.fit(X.iloc[tr_ix], y[tr_ix])
    print(f"[shift] trained on {len(tr_ix):,} rows ({time.time() - t0:.0f}s)", flush=True)

    # thresholds from the validation slice only
    val_parts = decompose_uncertainty(member_probs(model, X.iloc[val_ix]))
    t_star = youden_threshold(y[val_ix], val_parts["pd"])
    know_thr_clean = float(np.quantile(val_parts["knowledge"], 1.0 - TARGET_RATE))

    X_te, y_te = X.iloc[te_ix].reset_index(drop=True), y[te_ix]
    thin_te = thin[te_ix]
    rng = np.random.default_rng(SEED)

    info_cols = [c for c in X.columns
                 if "EXT_SOURCE" in c or "APP_EXT" in c or "BUREAU" in c.upper()]
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "fc"]

    scenarios: dict[str, tuple[pd.DataFrame, np.ndarray]] = {"clean": (X_te, y_te)}

    x_miss = X_te.copy()
    rows = rng.random(len(x_miss)) < MISSING_ROW_SHARE
    x_miss.loc[rows, info_cols] = np.nan
    scenarios["missing_info_50"] = (x_miss, y_te)

    scenarios["thin_file_only"] = (X_te[thin_te].reset_index(drop=True), y_te[thin_te])

    bad = np.flatnonzero(y_te == 1)
    good = np.flatnonzero(y_te == 0)
    target_rate = 2 * float(y_te.mean())
    n_good_keep = int(len(bad) * (1 - target_rate) / target_rate)
    keep_ix = np.sort(np.concatenate([bad, rng.choice(good, n_good_keep, replace=False)]))
    scenarios["prior_shift_2x"] = (X_te.iloc[keep_ix].reset_index(drop=True), y_te[keep_ix])

    x_noise = X_te.copy()
    noisy_cols = list(rng.choice(numeric_cols, int(len(numeric_cols) * NOISE_COL_SHARE), replace=False))
    for c in noisy_cols:
        sd = float(X_te[c].std(skipna=True)) or 0.0
        x_noise[c] = x_noise[c] + rng.normal(0.0, NOISE_SD_MULT * sd, len(x_noise))
    scenarios["noise_20pct"] = (x_noise, y_te)

    results = {}
    mean_know_clean = None
    for name, (Xs, ys) in scenarios.items():
        results[name] = evaluate(model, Xs, ys, t_star, know_thr_clean, mean_know_clean)
        if name == "clean":
            mean_know_clean = results[name]["mean_knowledge"]
        r = results[name]
        print(f"[shift] {name:<16} auc {r['auc']:.4f} ece {r['ece_10bin']:.4f} "
              f"defer@clean_thr {r['deferral_rate_at_clean_threshold']:.3f} "
              f"sel_risk {r['selective_risk_at_matched_rate']:.3f} vs overall {r['overall_error_rate']:.3f}",
              flush=True)

    report = {
        "protocol": {
            "design": "60/20/20 train/val/test stratified (seed 42); one SGLB CatBoost; thresholds from val only",
            "model": f"SGLB CatBoost, {ITERATIONS} iterations, K={K_MEMBERS}",
            "scenarios_note": (
                "CONTROLLED synthetic stress tests (except thin_file_only, a real "
                "subpopulation); not evidence of real-world generalisation."
            ),
            "masked_columns_missing_scenario": len(info_cols),
            "noisy_columns_noise_scenario": len(noisy_cols),
            "target_deferral_rate": TARGET_RATE,
        },
        "scenarios": results,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
