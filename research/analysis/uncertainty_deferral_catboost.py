"""Epistemic-uncertainty deferral from a single CatBoost — no second model.

THE NOVELTY CLAIM UNDER TEST. The deferral study (reports/deferral_fix.json)
showed the production disagreement router was inverted and that Chow's rule
(distance from the decision threshold) fixes it. But Chow's rule only sees
ALEATORIC hardness — cases near the boundary. A model can also be wrong far
from the boundary when it has never seen cases like this one (EPISTEMIC
uncertainty). CatBoost trained with Stochastic Gradient Langevin Boosting
(``posterior_sampling=True``) yields virtual ensembles: K approximate
posterior samples from ONE model, whose spread separates knowledge
uncertainty from data uncertainty (Malinin, Prokhorenkova, Ustimenko, ICLR
2021). If that signal routes review cases better than Chow's rule, SmartLend
gets a deferral trigger that needs no second model, no CBES score and no
calibration step — a system contribution, not a leaderboard tweak.

PRE-REGISTERED DECISION RULES (fixed before results exist):

1. Candidates are raced EXACTLY as in research/deferral/evaluate.py: 50/50
   tune/test split (seed 20260831), threshold = the TUNE quantile that defers
   22.5% (the underwriter-capacity mid-band), all numbers reported on TEST.
2. The error definition is fixed up front: predicted default iff
   p_default >= t*, where t* is the Youden threshold fit on TUNE only.
3. The uncertainty signal may be CLAIMED as the working router only if, at
   the matched rate on TEST, it (a) beats random abstention (selective risk
   below overall error rate) AND (b) reaches a position on the random(0) ->
   oracle(1) axis at least as high as Chow's rule, the zero-cost incumbent.
   Matching Chow without beating it is reported as a negative result for the
   novelty and a positive replication of Chow.
4. No metric shopping: selective risk / position at 22.5% is the criterion;
   error-detection AUROC and full risk-coverage curves are context only.

HONESTY NOTES. SGLB needs a fixed training budget (no early stopping), so
the uncertainty model is NOT the leaderboard model; its own OOF AUC is
reported so any accuracy gap is visible. Folds are the same 5-fold seed-42
split as features_vs_fusion, so the XGBoost disagreement baseline joins
row-for-row by SK_ID_CURR from reports/v2_oof_predictions.csv. Research track
only — the serving router is untouched (flipping it is a human decision that
bumps ENGINE_VERSION).

Run:  python -m research.analysis.uncertainty_deferral_catboost [--quick]
Writes reports/uncertainty_deferral.json (+ oof cache in data/processed/).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "reports" / "uncertainty_deferral.json"
OOF_CACHE = REPO / "data" / "processed" / "sglb_uncertainty_oof.pkl"
V2_OOF_CSV = REPO / "reports" / "v2_oof_predictions.csv"

SEED = 42                # fold seed — must match features_vs_fusion
N_FOLDS = 5
SGLB_ITERATIONS = 1000   # fixed budget: SGLB's posterior view needs no early stop
VIRTUAL_ENSEMBLES = 10
SPLIT_SEED = 20260831    # tune/test seed — must match research/deferral/evaluate.py
TARGET_RATE = 0.225


# --------------------------------------------------------------------------
# uncertainty math (unit-tested in research/tests/test_catboost_novelty.py)
# --------------------------------------------------------------------------

def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy of a Bernoulli(p) in nats, safe at 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def decompose_uncertainty(member_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Total / data / knowledge uncertainty from an (n, K) matrix of member
    probabilities (Malinin et al. decomposition):

        total     = H(mean_k p_k)          entropy of the averaged prediction
        data      = mean_k H(p_k)          expected entropy (aleatoric)
        knowledge = total - data           mutual information (epistemic, >= 0)
    """
    member_probs = np.asarray(member_probs, dtype=float)
    mean_p = member_probs.mean(axis=1)
    total = binary_entropy(mean_p)
    data = binary_entropy(member_probs).mean(axis=1)
    knowledge = np.maximum(total - data, 0.0)  # clip float noise below 0
    return {"pd": mean_p, "total": total, "data": data, "knowledge": knowledge}


def _member_probs(model, X) -> np.ndarray:
    """(n, K) member probabilities from a SGLB model's virtual ensembles.

    ``virtual_ensembles_predict(..., 'VirtEnsembles')`` returns raw log-odds
    per member for binary classification; shape can carry a trailing
    singleton dimension depending on version, so normalise defensively.
    """
    raw = np.asarray(model.virtual_ensembles_predict(
        X, prediction_type="VirtEnsembles",
        virtual_ensembles_count=VIRTUAL_ENSEMBLES))
    if raw.ndim == 3:
        raw = raw[..., -1]  # (n, K, dims) -> approve/last raw value per member
    assert raw.shape == (len(X), VIRTUAL_ENSEMBLES), f"unexpected shape {raw.shape}"
    return 1.0 / (1.0 + np.exp(-raw))


# --------------------------------------------------------------------------
# deferral-race machinery (protocol of research/deferral/evaluate.py)
# --------------------------------------------------------------------------

def youden_threshold(y: np.ndarray, pd_scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, pd_scores)
    return float(thr[np.argmax(tpr - fpr)])


def race_candidate(scores_tune, scores_test, errors_test, target_rate=TARGET_RATE):
    """Threshold on TUNE quantile, report selective risk on TEST (higher
    score == defer first). Mirrors evaluate.py's defer_mask_at_rate +
    risk_coverage_point."""
    threshold = float(np.quantile(scores_tune, 1.0 - target_rate))
    defer = scores_test > threshold
    n, n_defer = len(errors_test), int(defer.sum())
    n_keep = n - n_defer
    overall = float(errors_test.mean())
    selective = float(errors_test[~defer].mean()) if n_keep else None
    oracle = float(max(0, int(errors_test.sum()) - n_defer) / n_keep) if n_keep else None
    position = None
    if selective is not None and oracle is not None and overall > oracle:
        position = float((overall - selective) / (overall - oracle))
    return {
        "deferral_rate": float(n_defer / n),
        "selective_risk": selective,
        "random_risk": overall,
        "oracle_risk": oracle,
        "position_random0_oracle1": position,
        "beats_random": None if selective is None else bool(selective < overall),
        "error_detection_auroc": float(roc_auc_score(errors_test, scores_test))
        if 0 < errors_test.sum() < n else None,
    }


def risk_coverage_curve(errors: np.ndarray, scores: np.ndarray) -> list[dict]:
    order = np.argsort(scores)  # keep lowest-signal first
    cum_err = np.cumsum(errors[order])
    n = len(errors)
    return [
        {"coverage": round(k / n, 3), "selective_risk": round(float(cum_err[k - 1] / k), 5)}
        for k in (max(1, int(round(c * n))) for c in np.arange(0.05, 1.0, 0.05))
    ]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def train_sglb_oof(X, y, quick: bool):
    """5-fold OOF pd + uncertainty from SGLB CatBoost (fixed budget)."""
    from catboost import CatBoostClassifier

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = {k: np.zeros(len(y)) for k in ("pd", "total", "data", "knowledge")}
    fold_aucs = []
    iterations = 300 if quick else SGLB_ITERATIONS
    for k, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.time()
        model = CatBoostClassifier(
            iterations=iterations, learning_rate=0.05,
            posterior_sampling=True,  # SGLB: makes virtual ensembles posterior samples
            random_seed=SEED + k, verbose=0, allow_writing_files=False)
        model.fit(X.iloc[tr], y[tr])
        parts = decompose_uncertainty(_member_probs(model, X.iloc[va]))
        for key, values in parts.items():
            out[key][va] = values
        fold_aucs.append(float(roc_auc_score(y[va], parts["pd"])))
        print(f"    fold {k + 1}/{N_FOLDS}: AUC {fold_aucs[-1]:.4f} ({time.time() - t0:.0f}s)")
    return out, fold_aucs


def main(quick: bool) -> None:
    started = time.time()
    X, y, ids = build_feature_matrix()
    if quick:
        keep = np.random.default_rng(SEED).choice(len(y), 30_000, replace=False)
        X, y, ids = X.iloc[keep].reset_index(drop=True), y[keep], ids[keep]
        print(f"[quick] subsampled to {len(y):,} rows")

    if not quick and OOF_CACHE.exists():
        out, fold_aucs = pd.read_pickle(OOF_CACHE)
        print(f"[sglb] cache hit ({OOF_CACHE.name})")
    else:
        print(f"[sglb] training {N_FOLDS}-fold SGLB CatBoost "
              f"({300 if quick else SGLB_ITERATIONS} iterations, no early stop)")
        out, fold_aucs = train_sglb_oof(X, y, quick)
        if not quick:
            OOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle((out, fold_aucs), OOF_CACHE)

    # XGBoost pd for the disagreement baseline, joined by row id.
    v2 = pd.read_csv(V2_OOF_CSV).set_index("SK_ID_CURR")
    pd_xgb = v2["pd_XGBoost"].reindex(ids).to_numpy()
    assert not np.isnan(pd_xgb).any(), "v2 OOF csv does not cover these ids"

    # ---- tune/test split; everything data-dependent fit on TUNE only -------
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(y))
    half = len(y) // 2
    tune_ix, test_ix = perm[:half], perm[half:]

    pd_scores = out["pd"]
    t_star = youden_threshold(y[tune_ix], pd_scores[tune_ix])
    predicted_default = (pd_scores >= t_star).astype(int)
    errors = (predicted_default != y).astype(int)

    candidates = {
        "chow_distance": -np.abs(pd_scores - t_star),
        "knowledge_uncertainty": out["knowledge"],
        "data_uncertainty": out["data"],
        "total_uncertainty": out["total"],
        "xgb_disagreement": np.abs(pd_scores - pd_xgb),
    }
    results = {
        name: race_candidate(s[tune_ix], s[test_ix], errors[test_ix])
        for name, s in candidates.items()
    }
    curves = {name: risk_coverage_curve(errors[test_ix], s[test_ix])
              for name, s in candidates.items()}

    # ---- pre-registered verdict (rule 3) -----------------------------------
    chow_pos = results["chow_distance"]["position_random0_oracle1"] or 0.0
    ku = results["knowledge_uncertainty"]
    verdict = {
        "novelty_signal": "knowledge_uncertainty (SGLB virtual ensembles, single model)",
        "beats_random": ku["beats_random"],
        "position_vs_chow": [ku["position_random0_oracle1"], chow_pos],
        "claim_granted": bool(ku["beats_random"]
                              and (ku["position_random0_oracle1"] or 0.0) >= chow_pos),
        "rule": ("claim only if knowledge uncertainty beats random AND reaches "
                 "Chow's position at the matched 22.5% rate on TEST"),
    }

    report = {
        "protocol": {
            "rows": int(len(y)), "quick_mode": quick,
            "folds": f"{N_FOLDS}-fold StratifiedKFold seed {SEED} (same as features_vs_fusion)",
            "sglb": {"iterations": 300 if quick else SGLB_ITERATIONS,
                     "virtual_ensembles": VIRTUAL_ENSEMBLES,
                     "note": "posterior_sampling=True; fixed budget, no early stopping"},
            "split": f"50/50 tune/test seed {SPLIT_SEED}; thresholds fit on tune only",
            "target_deferral_rate": TARGET_RATE,
            "error_definition": "predicted default iff pd >= Youden t* (t* from tune)",
        },
        "sglb_model": {
            "oof_auc": round(float(roc_auc_score(y, pd_scores)), 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "leaderboard_catboost_fe_v2_auc_for_reference": 0.7757,
            "youden_threshold_tune": round(t_star, 4),
            "overall_error_rate_test": round(float(errors[test_ix].mean()), 5),
        },
        "uncertainty_separation": {
            "corr_knowledge_vs_chow": round(float(np.corrcoef(
                candidates["knowledge_uncertainty"], candidates["chow_distance"])[0, 1]), 4),
            "corr_knowledge_vs_total": round(float(np.corrcoef(
                candidates["knowledge_uncertainty"], candidates["total_uncertainty"])[0, 1]), 4),
            "corr_total_vs_pd": round(float(np.corrcoef(
                candidates["total_uncertainty"], pd_scores)[0, 1]), 4),
            "caveat": ("with nearly all pd below 0.5, binary entropy is close to "
                       "monotone in pd itself, so total/data uncertainty largely "
                       "rank-order like 'defer the highest-pd band'; knowledge "
                       "uncertainty (mutual information) is the signal that is NOT "
                       "reducible to pd and carries the novelty claim"),
            "mean_knowledge_thin_file": round(float(
                out["knowledge"][X["BURO_has_bureau"].to_numpy() == 0].mean()), 6),
            "mean_knowledge_has_bureau": round(float(
                out["knowledge"][X["BURO_has_bureau"].to_numpy() == 1].mean()), 6),
        },
        "candidates_at_matched_rate": results,
        "verdict": verdict,
        "risk_coverage_curves": curves,
        "generated_seconds": round(time.time() - started, 1),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'signal':<24}{'rate':>8}{'sel_risk':>10}{'random':>9}{'oracle':>9}{'pos':>7}{'err_auc':>9}")
    for name, r in results.items():
        print(f"{name:<24}{r['deferral_rate']:>8.2%}{r['selective_risk']:>10.4f}"
              f"{r['random_risk']:>9.4f}{r['oracle_risk']:>9.4f}"
              f"{(r['position_random0_oracle1'] or float('nan')):>7.2f}"
              f"{(r['error_detection_auroc'] or float('nan')):>9.4f}")
    print(f"VERDICT: claim_granted={verdict['claim_granted']}  (report -> {OUT_JSON})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="30k-row smoke run")
    main(quick=parser.parse_args().quick)
