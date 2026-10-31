"""Independent-dataset replication of the deferral audit and epistemic
routing (reviewer item: external validity).

Dataset: UCI 'default of credit card clients' (Taiwan; Yeh & Lien 2009),
30,000 real credit-card customers, 23 features, target = default on the
next payment (22.1% positive rate). Fetched via OpenML (data id 42477,
a verbatim upload of the UCI file); the script validates shape and class
rate before using it. Provenance-clean and fully public.

Protocol mirrors the Home Credit study end to end:
- 5-fold StratifiedKFold seed 42; SGLB CatBoost (1000 iterations, lr 0.05,
  posterior_sampling, seed 42+fold); K=10 virtual ensembles; OOF
  pd / total / data / knowledge uncertainty.
- 50/50 tune/test split seed 20260831; Youden threshold and all deferral
  quantiles from TUNE only; matched 22.5% deferral rate.
- Candidates: reject-option margin (Chow), knowledge uncertainty, total
  uncertainty. Metrics: selective risk, random/oracle references,
  random->oracle position, error-detection AUROC, knowledge deciles.
- The UNMODIFIED Home Credit gate statistic audits each router at
  m=10,000 permutations: the Taiwan frame is expressed in the gate's own
  schema (y_true==1 good, approval probabilities), so the audit tooling
  runs unchanged across datasets.

Run: ``python -m research.analysis.taiwan_replication``
Writes ``reports/taiwan_replication.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from research.analysis.uncertainty_deferral_catboost import (
    decompose_uncertainty,
    youden_threshold,
)
from research.relearning.gate import evaluate_condition_1

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "raw" / "uci_taiwan"
OUT_JSON = REPO / "reports" / "taiwan_replication.json"

OPENML_ID = 42477
SEED = 42
N_FOLDS = 5
ITERATIONS = 1000
K_MEMBERS = 10
SPLIT_SEED = 20260831
TARGET_RATE = 0.225
GATE_TRIALS = 10_000
GATE_SEED = 20260830


def load_taiwan() -> tuple[pd.DataFrame, np.ndarray]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home=str(DATA_DIR))
    X = ds.data.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(ds.target, errors="coerce").to_numpy(dtype=int)
    assert len(y) == 30_000, f"expected 30,000 rows, got {len(y)}"
    rate = float(y.mean())
    assert 0.20 < rate < 0.24, f"unexpected default rate {rate:.3f}"
    return X, y


def member_probs(model, X) -> np.ndarray:
    raw = np.asarray(model.virtual_ensembles_predict(
        X, prediction_type="VirtEnsembles", virtual_ensembles_count=K_MEMBERS))
    return 1.0 / (1.0 + np.exp(-raw[:, :, 0]))


def gate_frame(y_def: np.ndarray, pd_scores: np.ndarray, t_star: float,
               defer: np.ndarray) -> pd.DataFrame:
    """Express the Taiwan split in the gate's own schema (y_true==1 GOOD,
    approval probabilities), so the Home Credit audit code runs unchanged."""
    auto_good = pd_scores < t_star
    decision = np.where(defer, "DEFER", np.where(auto_good, "APPROVE", "REJECT"))
    return pd.DataFrame({
        "y_true": (1 - y_def).astype(int),
        "best_model_prob": 1.0 - pd_scores,
        "approval_threshold": np.full(len(y_def), 1.0 - t_star),
        "final_decision": decision,
    })


def routing_point(errors: np.ndarray, defer: np.ndarray) -> dict:
    n, n_defer = len(errors), int(defer.sum())
    keep = ~defer
    overall = float(errors.mean())
    selective = float(errors[keep].mean())
    oracle = float(max(0, int(errors.sum()) - n_defer) / max(1, n - n_defer))
    return {
        "deferral_rate": float(n_defer / n),
        "selective_risk": selective,
        "random_risk": overall,
        "oracle_risk": oracle,
        "position_random0_oracle1": (overall - selective) / (overall - oracle)
        if overall > oracle else None,
    }


def decile_rates(sig_tune: np.ndarray, sig_test: np.ndarray, err_test: np.ndarray) -> list:
    edges = np.quantile(sig_tune, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    return [
        float(err_test[(sig_test > edges[d]) & (sig_test <= edges[d + 1])].mean())
        for d in range(10)
    ]


def main() -> int:
    from catboost import CatBoostClassifier

    X, y = load_taiwan()
    print(f"[taiwan] {len(y):,} rows, default rate {y.mean():.3f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(y)) for k in ("pd", "total", "data", "knowledge")}
    fold_aucs = []
    for f, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.time()
        model = CatBoostClassifier(
            iterations=ITERATIONS, learning_rate=0.05, posterior_sampling=True,
            random_seed=SEED + f, verbose=0, allow_writing_files=False)
        model.fit(X.iloc[tr], y[tr])
        parts = decompose_uncertainty(member_probs(model, X.iloc[va]))
        for key, values in parts.items():
            oof[key if key != "pd" else "pd"][va] = values
        fold_aucs.append(float(roc_auc_score(y[va], parts["pd"])))
        print(f"[taiwan] fold {f + 1}/{N_FOLDS}: AUC {fold_aucs[-1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(y))
    half = len(y) // 2
    tune_ix, test_ix = perm[:half], perm[half:]

    pd_s = oof["pd"]
    t_star = youden_threshold(y[tune_ix], pd_s[tune_ix])
    errors = ((pd_s >= t_star).astype(int) != y).astype(int)
    err_test = errors[test_ix]

    candidates = {
        "chow_margin": -np.abs(pd_s - t_star),
        "knowledge_uncertainty": oof["knowledge"],
        "total_uncertainty": oof["total"],
    }
    results = {}
    for name, sig in candidates.items():
        thr = float(np.quantile(sig[tune_ix], 1.0 - TARGET_RATE))
        defer = sig[test_ix] > thr
        frame = gate_frame(y[test_ix], pd_s[test_ix], t_star, defer)
        cond1 = evaluate_condition_1(frame, trials=GATE_TRIALS, seed=GATE_SEED)
        results[name] = {
            "routing": routing_point(err_test, defer),
            "gate_accuracy_z": cond1.metrics["accuracy_z"],
            "gate_balance_z": cond1.metrics["balance_distance_z"],
            "gate_status": cond1.status,
            "error_detection_auroc": float(roc_auc_score(err_test, sig[test_ix])),
        }
        r = results[name]["routing"]
        print(f"[taiwan] {name:<24} pos {r['position_random0_oracle1']:.3f} "
              f"sel_risk {r['selective_risk']:.3f} acc_z {results[name]['gate_accuracy_z']:+.1f}",
              flush=True)

    report = {
        "dataset": {
            "name": "UCI default of credit card clients (Taiwan, Yeh & Lien 2009)",
            "source": f"OpenML data id {OPENML_ID} (verbatim UCI upload), fetched to data/raw/uci_taiwan/",
            "rows": int(len(y)),
            "default_rate": float(y.mean()),
            "n_features": int(X.shape[1]),
        },
        "protocol": {
            "folds": f"{N_FOLDS}-fold StratifiedKFold seed {SEED}",
            "model": f"SGLB CatBoost, {ITERATIONS} iterations, K={K_MEMBERS} virtual ensembles",
            "split": f"50/50 tune/test seed {SPLIT_SEED}; Youden threshold and quantiles from tune only",
            "target_deferral_rate": TARGET_RATE,
            "gate": f"research.relearning.gate.evaluate_condition_1, unmodified, trials={GATE_TRIALS}",
            "note": "same seeds, budgets and audit code as the Home Credit study",
        },
        "oof_auc": float(roc_auc_score(y, pd_s)),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "youden_threshold_tune": float(t_star),
        "overall_error_rate_test": float(err_test.mean()),
        "candidates_at_matched_rate": results,
        "knowledge_deciles_error_rate": decile_rates(
            oof["knowledge"][tune_ix], oof["knowledge"][test_ix], err_test),
        "corr_knowledge_vs_chow": float(np.corrcoef(
            oof["knowledge"][test_ix], candidates["chow_margin"][test_ix])[0, 1]),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[taiwan] OOF AUC {report['oof_auc']:.4f}")
    print(f"report -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
