"""Uncertainty-calibration study: does higher epistemic uncertainty actually
mean higher error? (reviewer item)

Reads the committed SGLB OOF cache (data/processed/sglb_uncertainty_oof.pkl,
produced by uncertainty_deferral_catboost.py: K=10 virtual ensembles, 5-fold
OOF, 307,511 rows) and reports, on the held-out half of the standard policy
split (seed 20260831, Youden threshold from the tune half):

- decision-error rate by knowledge-uncertainty decile (deciles from tune),
- the same by total-uncertainty and Chow-margin deciles for contrast,
- error-detection AUROC and AUPRC per signal,
- mean knowledge uncertainty for thin-file vs documented applicants.

Run: ``python -m research.analysis.sglb_decile_study``
Writes ``reports/sglb_decile_study.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from research.analysis.uncertainty_deferral_catboost import (
    OOF_CACHE,
    SPLIT_SEED,
    youden_threshold,
)
from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "home_credit"
OUT_JSON = REPO / "reports" / "sglb_decile_study.json"


def decile_table(signal_tune: np.ndarray, signal_test: np.ndarray,
                 errors_test: np.ndarray) -> list[dict]:
    edges = np.quantile(signal_tune, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for d in range(10):
        mask = (signal_test > edges[d]) & (signal_test <= edges[d + 1])
        out.append({
            "decile": d + 1,
            "n": int(mask.sum()),
            "error_rate": float(errors_test[mask].mean()) if mask.any() else None,
        })
    return out


def main() -> int:
    out, fold_aucs = pd.read_pickle(OOF_CACHE)
    X, y, ids = build_feature_matrix()
    assert len(y) == len(out["pd"]), "cache and feature matrix must align"

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(y))
    half = len(y) // 2
    tune_ix, test_ix = perm[:half], perm[half:]

    pd_scores = out["pd"]
    t_star = youden_threshold(y[tune_ix], pd_scores[tune_ix])
    errors = ((pd_scores >= t_star).astype(int) != y).astype(int)
    err_test = errors[test_ix]

    signals = {
        "knowledge_uncertainty": out["knowledge"],
        "total_uncertainty": out["total"],
        "chow_margin": -np.abs(pd_scores - t_star),
    }

    bureau_ids = set(pd.read_csv(RAW / "bureau.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"].unique())
    thin = ~pd.Series(ids).isin(bureau_ids).to_numpy()

    report = {
        "protocol": {
            "source": "data/processed/sglb_uncertainty_oof.pkl (committed study's K=10 OOF run)",
            "rows": int(len(y)),
            "split": f"tune/test seed {SPLIT_SEED}; deciles and Youden threshold from tune only",
            "error_definition": "predicted default iff pd >= Youden t*; error iff prediction != label",
            "fold_aucs": [round(float(a), 4) for a in fold_aucs],
        },
        "overall_error_rate_test": float(err_test.mean()),
        "deciles": {
            name: decile_table(s[tune_ix], s[test_ix], err_test)
            for name, s in signals.items()
        },
        "error_detection": {
            name: {
                "auroc": float(roc_auc_score(err_test, s[test_ix])),
                "auprc": float(average_precision_score(err_test, s[test_ix])),
            }
            for name, s in signals.items()
        },
        "thin_file": {
            "n_thin": int(thin.sum()),
            "mean_knowledge_thin": float(out["knowledge"][thin].mean()),
            "mean_knowledge_documented": float(out["knowledge"][~thin].mean()),
            "error_rate_thin_test": float(errors[test_ix][thin[test_ix]].mean()),
            "error_rate_documented_test": float(errors[test_ix][~thin[test_ix]].mean()),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name in signals:
        rates = [d["error_rate"] for d in report["deciles"][name]]
        print(f"{name:<24} d1 {rates[0]:.3f} -> d10 {rates[-1]:.3f}  "
              f"auroc {report['error_detection'][name]['auroc']:.3f}")
    print(f"report -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
