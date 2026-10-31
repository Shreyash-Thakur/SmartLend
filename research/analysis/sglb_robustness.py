"""SGLB sensitivity study: is the epistemic-routing result an artifact of
K=10 or of one training run? (reviewer item)

For each training seed offset, retrain the 5-fold SGLB CatBoost exactly as
``uncertainty_deferral_catboost`` does (1000 iterations, lr 0.05,
posterior_sampling, fold seed SEED+k+offset) and, from the SAME trained
models, compute knowledge uncertainty at K in {5, 10, 20, 50} virtual
ensembles. For every (seed, K) report:

- OOF AUC of the mean prediction,
- routing quality at the 22.5% matched rate on the test half (selective
  risk, random/oracle references, random->oracle position; Youden threshold
  and deferral quantile fitted on the tune half only, split seed 20260831),
- Spearman rank correlation of the knowledge scores against the committed
  reference run (seed offset 0, K=10) -- rank stability is what matters for
  a quantile-based router.

Run: ``python -m research.analysis.sglb_robustness``
Writes ``reports/sglb_robustness.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from research.analysis.uncertainty_deferral_catboost import (
    SEED,
    N_FOLDS,
    SGLB_ITERATIONS,
    SPLIT_SEED,
    TARGET_RATE,
    decompose_uncertainty,
    youden_threshold,
)
from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "reports" / "sglb_robustness.json"

K_GRID = [5, 10, 20, 50]
SEED_OFFSETS = [0, 100, 200]  # 0 replicates the committed study's fold seeds


def member_probs(model, X, k: int) -> np.ndarray:
    raw = np.asarray(
        model.virtual_ensembles_predict(X, prediction_type="VirtEnsembles",
                                        virtual_ensembles_count=k)
    )
    return 1.0 / (1.0 + np.exp(-raw[:, :, 0]))


def routing_at_rate(scores: np.ndarray, errors: np.ndarray,
                    tune_ix: np.ndarray, test_ix: np.ndarray) -> dict:
    """Defer the top TARGET_RATE by score (threshold from tune); risk on test."""
    thr = float(np.quantile(scores[tune_ix], 1.0 - TARGET_RATE))
    defer = scores[test_ix] > thr
    err = errors[test_ix]
    n, n_defer = len(err), int(defer.sum())
    keep = ~defer
    overall = float(err.mean())
    selective = float(err[keep].mean())
    oracle = float(max(0, int(err.sum()) - n_defer) / max(1, n - n_defer))
    position = (overall - selective) / (overall - oracle) if overall > oracle else None
    return {
        "deferral_rate": float(n_defer / n),
        "selective_risk": selective,
        "random_risk": overall,
        "oracle_risk": oracle,
        "position_random0_oracle1": position,
    }


def main() -> int:
    X, y, ids = build_feature_matrix()
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(y))
    half = len(y) // 2
    tune_ix, test_ix = perm[:half], perm[half:]

    from catboost import CatBoostClassifier

    results: dict[str, dict] = {}
    reference_knowledge = None  # offset 0, K=10

    for offset in SEED_OFFSETS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = {k: {m: np.zeros(len(y)) for m in ("pd", "knowledge")} for k in K_GRID}
        t_seed = time.time()
        for f, (tr, va) in enumerate(skf.split(X, y)):
            t0 = time.time()
            model = CatBoostClassifier(
                iterations=SGLB_ITERATIONS, learning_rate=0.05,
                posterior_sampling=True, random_seed=SEED + f + offset,
                verbose=0, allow_writing_files=False)
            model.fit(X.iloc[tr], y[tr])
            for k in K_GRID:
                parts = decompose_uncertainty(member_probs(model, X.iloc[va], k))
                oof[k]["pd"][va] = parts["pd"]
                oof[k]["knowledge"][va] = parts["knowledge"]
            print(f"[robustness] offset {offset} fold {f + 1}/{N_FOLDS} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        for k in K_GRID:
            pd_scores = oof[k]["pd"]
            know = oof[k]["knowledge"]
            t_star = youden_threshold(y[tune_ix], pd_scores[tune_ix])
            errors = ((pd_scores >= t_star).astype(int) != y).astype(int)
            entry = {
                "oof_auc": float(roc_auc_score(y, pd_scores)),
                "routing": routing_at_rate(know, errors, tune_ix, test_ix),
            }
            if offset == 0 and k == 10:
                reference_knowledge = know.copy()
            if reference_knowledge is not None:
                rho = spearmanr(know, reference_knowledge).statistic
                entry["spearman_vs_reference"] = float(rho)
            results[f"offset{offset}_K{k}"] = entry
            r = entry["routing"]
            print(f"[robustness] offset {offset} K={k}: AUC {entry['oof_auc']:.4f} "
                  f"position {r['position_random0_oracle1']:.3f}", flush=True)
        print(f"[robustness] offset {offset} done ({time.time() - t_seed:.0f}s)", flush=True)

    report = {
        "protocol": {
            "folds": f"{N_FOLDS}-fold StratifiedKFold seed {SEED} (+offset on model seeds)",
            "iterations": SGLB_ITERATIONS,
            "K_grid": K_GRID,
            "seed_offsets": SEED_OFFSETS,
            "split": f"tune/test seed {SPLIT_SEED}; Youden threshold and deferral quantile from tune only",
            "target_deferral_rate": TARGET_RATE,
            "reference": "offset 0, K=10 (the committed uncertainty_deferral study configuration)",
        },
        "configurations": results,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
