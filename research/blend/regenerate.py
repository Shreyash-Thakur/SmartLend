"""Regenerate artifact decisions on p_blend (ML+CBES) and measure what it costs.

WHY
---
The Decision Landscape scatter (p_ml on x, p_cbes on y) showed VERTICAL decision
bands: the stored decisions in backend/artifacts/prediction_outputs.csv were a
pure function of best_model_prob (threshold 0.925484 +/- 0.022269), so p_cbes
never influenced a single decision, even though decision_engine.py documents a
two-stage blend p_blend = (1 - alpha) * p_ml + alpha * p_cbes with alpha = 0.25.
A blend threshold  (1-a)*x + a*y >= t  is the half-plane above the slanted line
y = (t - (1-a)*x) / a  in that scatter; a p_ml-only threshold is a vertical line.

WHAT THIS DOES
--------------
1. Reads alpha from the live engine constant (decision_engine._BLEND_ALPHA).
2. Splits the OOF artifact 50/50 (research.deferral.evaluate.split_tune_test,
   seed 20260831). Threshold selection happens on TUNE only; every reported
   metric comes from TEST. Nothing is tuned on the rows it is scored on.
3. On TUNE: Youden's J threshold t* on p_blend; deferral half-width tau_u as the
   22.5% quantile of |p_blend - t*| (mid-point of the 20-25% capacity band).
4. Rewrites final_decision / approval_threshold / rejection_threshold /
   confidence for ALL rows of the artifact:
       DEFER    iff |p_blend - t*| <  tau_u
       APPROVE  iff  p_blend      >= t* + tau_u   (= approval_threshold)
       REJECT   iff  p_blend      <= t* - tau_u   (= rejection_threshold)
       confidence = clip(2 * |p_blend - t*|, 0, 1)
   All other columns (prob_*, cbes_prob, y_true, ...) are preserved byte-for-byte.
5. MEASURES THE COST on TEST: AUC of p_ml vs p_blend with a paired-bootstrap CI
   on the difference, default capture / approve rate / full confusion matrix
   under the old (p_ml-only) and new (blend) decision policies, and the
   UNMODIFIED gate condition 1 (research.relearning.gate) under each.
6. Sweeps alpha 0.00..0.50 in 0.05 steps (same tune/test discipline per alpha).

LABEL SEMANTICS: y_true == 1 means GOOD / approve-worthy (did not default);
all prob_* / cbes_prob columns are APPROVAL probabilities.

Run:  python -m research.blend.regenerate
Writes reports/blend_decision.json and updates backend/artifacts/prediction_outputs.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from research.deferral.evaluate import (
    GATE_SEED,
    GATE_TRIALS,
    RATE_BAND,
    SPLIT_SEED,
    TARGET_RATE,
    split_tune_test,
)
from research.relearning.gate import evaluate_condition_1

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = REPO_ROOT / "backend" / "artifacts" / "prediction_outputs.csv"
REPORT_PATH = REPO_ROOT / "reports" / "blend_decision.json"

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 20260831


def blend_alpha_from_engine() -> float:
    """Read the live blend weight from decision_engine — never hardcode it."""
    backend = str(REPO_ROOT / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from app.services.decision_engine import _BLEND_ALPHA  # noqa: PLC0415

    return float(_BLEND_ALPHA)


def youden_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising TPR - FPR for the approval score."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def fit_policy(y_tune: np.ndarray, blend_tune: np.ndarray) -> tuple[float, float]:
    """(t_star, tau_u) fitted on the TUNE half only."""
    t_star = youden_threshold(y_tune, blend_tune)
    tau_u = float(np.quantile(np.abs(blend_tune - t_star), TARGET_RATE))
    return t_star, tau_u


def decide(blend: np.ndarray, t_star: float, tau_u: float) -> np.ndarray:
    dist = np.abs(blend - t_star)
    return np.where(dist < tau_u, "DEFER", np.where(blend >= t_star, "APPROVE", "REJECT"))


def policy_metrics(y: np.ndarray, decisions: np.ndarray) -> dict[str, Any]:
    """Decision-level metrics. y == 1 GOOD; default capture is about y == 0."""
    n = len(y)
    good, bad = y == 1, y == 0
    app, rej, dfr = decisions == "APPROVE", decisions == "REJECT", decisions == "DEFER"
    auto = ~dfr
    n_auto = int(auto.sum())
    confusion = {
        "approve_good": int((app & good).sum()),
        "approve_bad": int((app & bad).sum()),
        "reject_good": int((rej & good).sum()),
        "reject_bad": int((rej & bad).sum()),
        "defer_good": int((dfr & good).sum()),
        "defer_bad": int((dfr & bad).sum()),
    }
    return {
        "n": n,
        "deferral_rate": float(dfr.mean()),
        "approve_rate_all_rows": float(app.mean()),
        "approve_rate_auto_decided": float(app.sum() / n_auto) if n_auto else None,
        # Share of true defaulters the policy REJECTS outright...
        "default_capture_rejected": float((rej & bad).sum() / bad.sum()),
        # ...and the share it at least keeps away from auto-approval.
        "default_capture_rejected_or_deferred": float(((rej | dfr) & bad).sum() / bad.sum()),
        "auto_accuracy": float(((app & good) | (rej & bad)).sum() / n_auto) if n_auto else None,
        "confusion_matrix": confusion,
    }


def paired_bootstrap_auc_delta(
    y: np.ndarray, p_ml: np.ndarray, p_blend: np.ndarray,
    iters: int = BOOTSTRAP_ITERS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap CI for AUC(p_blend) - AUC(p_ml) on the same resamples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():  # degenerate resample; resample again
            idx = rng.integers(0, n, n)
            yb = y[idx]
        deltas[i] = roc_auc_score(yb, p_blend[idx]) - roc_auc_score(yb, p_ml[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "iterations": iters,
        "seed": seed,
        "delta_mean": float(deltas.mean()),
        "ci95": [float(lo), float(hi)],
        "prob_delta_negative": float((deltas < 0).mean()),
    }


def gate_condition_1(frame: pd.DataFrame) -> dict[str, Any]:
    cond = evaluate_condition_1(frame, trials=GATE_TRIALS, seed=GATE_SEED)
    return {
        "status": cond.status,
        "balance_distance_z": cond.metrics["balance_distance_z"],
        "accuracy_z": cond.metrics["accuracy_z"],
        "deferred_good_share": cond.metrics["deferred"]["good_share"],
        "auto_good_share": cond.metrics["auto_decided"]["good_share"],
        "deferred_accuracy": cond.metrics["deferred"]["accuracy"],
        "auto_accuracy": cond.metrics["auto_decided"]["accuracy"],
        "reason": cond.reason,
    }


def sweep_alpha(
    y_tune: np.ndarray, ml_tune: np.ndarray, cb_tune: np.ndarray,
    y_test: np.ndarray, ml_test: np.ndarray, cb_test: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for step in range(11):  # 0.00 .. 0.50
        a = round(step * 0.05, 2)
        bl_tune = (1 - a) * ml_tune + a * cb_tune
        bl_test = (1 - a) * ml_test + a * cb_test
        t_star, tau_u = fit_policy(y_tune, bl_tune)
        decisions = decide(bl_test, t_star, tau_u)
        m = policy_metrics(y_test, decisions)
        rows.append({
            "alpha": a,
            "auc_blend": float(roc_auc_score(y_test, bl_test)),
            "youden_threshold": round(t_star, 6),
            "tau_u": round(tau_u, 6),
            "deferral_rate": m["deferral_rate"],
            "approve_rate_all_rows": m["approve_rate_all_rows"],
            "default_capture_rejected": m["default_capture_rejected"],
            "default_capture_rejected_or_deferred": m["default_capture_rejected_or_deferred"],
            "auto_accuracy": m["auto_accuracy"],
        })
    return rows


def rewrite_artifact(alpha: float, t_star: float, tau_u: float) -> pd.DataFrame:
    """Update the 4 decision columns in place; every other column untouched.

    The file is re-read with dtype=str so unmodified columns round-trip
    byte-for-byte instead of being reformatted by pandas float printing.
    """
    raw = pd.read_csv(PREDICTIONS, dtype=str)
    p_ml = raw["best_model_prob"].astype(float).to_numpy()
    p_cbes = raw["cbes_prob"].astype(float).to_numpy()
    blend = (1 - alpha) * p_ml + alpha * p_cbes

    decisions = decide(blend, t_star, tau_u)
    confidence = np.clip(2.0 * np.abs(blend - t_star), 0.0, 1.0)

    raw["final_decision"] = decisions
    raw["approval_threshold"] = f"{t_star + tau_u:.6f}"
    raw["rejection_threshold"] = f"{t_star - tau_u:.6f}"
    raw["confidence"] = [f"{c:.6f}" for c in confidence]
    raw.to_csv(PREDICTIONS, index=False)
    return raw


def main() -> int:
    alpha = blend_alpha_from_engine()
    frame = pd.read_csv(PREDICTIONS)
    tune, test = split_tune_test(frame)

    def cols(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            f["y_true"].to_numpy(dtype=int),
            f["best_model_prob"].to_numpy(dtype=float),
            f["cbes_prob"].to_numpy(dtype=float),
        )

    y_tu, ml_tu, cb_tu = cols(tune)
    y_te, ml_te, cb_te = cols(test)
    bl_tu = (1 - alpha) * ml_tu + alpha * cb_tu
    bl_te = (1 - alpha) * ml_te + alpha * cb_te

    # -- fit the blend policy on TUNE ---------------------------------------
    t_star, tau_u = fit_policy(y_tu, bl_tu)
    new_decisions_test = decide(bl_te, t_star, tau_u)

    # -- the outgoing policy, exactly as stored in the artifact --------------
    old_decisions_test = test["final_decision"].astype(str).to_numpy()

    # -- discrimination cost --------------------------------------------------
    auc_ml = float(roc_auc_score(y_te, ml_te))
    auc_blend = float(roc_auc_score(y_te, bl_te))
    auc_cbes = float(roc_auc_score(y_te, cb_te))
    bootstrap = paired_bootstrap_auc_delta(y_te, ml_te, bl_te)

    # -- decision-policy cost -------------------------------------------------
    old_metrics = policy_metrics(y_te, old_decisions_test)
    new_metrics = policy_metrics(y_te, new_decisions_test)

    # -- gate condition 1 under each router (test half, as the gate will see it)
    gate_old = gate_condition_1(test)
    test_new = test.copy()
    test_new["final_decision"] = new_decisions_test
    test_new["approval_threshold"] = t_star + tau_u
    test_new["rejection_threshold"] = t_star - tau_u
    gate_new = gate_condition_1(test_new)

    # -- alpha sweep ------------------------------------------------------------
    sweep = sweep_alpha(y_tu, ml_tu, cb_tu, y_te, ml_te, cb_te)

    # -- rewrite the artifact for ALL rows with the tune-fitted policy -----------
    updated = rewrite_artifact(alpha, t_star, tau_u)
    full_decisions = updated["final_decision"].to_numpy()
    full_defer_rate = float((full_decisions == "DEFER").mean())

    report = {
        "spec": "decisions must be taken on p_blend = (1-alpha)*p_ml + alpha*p_cbes, "
                "not on p_ml alone (decision_engine.py Stage A/B)",
        "alpha_source": "backend.app.services.decision_engine._BLEND_ALPHA",
        "alpha": alpha,
        "label_semantics": {
            "y_true==1": "GOOD customer (approve-worthy, did not default)",
            "probabilities": "all prob_* / cbes_prob are APPROVAL probabilities",
        },
        "split": f"50/50 tune/test, seed {SPLIT_SEED}; thresholds fitted on tune, "
                 "all reported metrics on test (research.deferral.evaluate.split_tune_test)",
        "policy": {
            "rule": "DEFER iff |p_blend - t*| < tau_u; else APPROVE iff p_blend >= t*, else REJECT",
            "youden_threshold_t_star": round(t_star, 6),
            "tau_u": round(tau_u, 6),
            "approval_threshold_stored": round(t_star + tau_u, 6),
            "rejection_threshold_stored": round(t_star - tau_u, 6),
            "confidence_stored": "clip(2*|p_blend - t*|, 0, 1)",
            "target_deferral_rate": TARGET_RATE,
            "deferral_rate_band": list(RATE_BAND),
            "test_deferral_rate": new_metrics["deferral_rate"],
            "test_rate_within_band": bool(
                RATE_BAND[0] <= new_metrics["deferral_rate"] <= RATE_BAND[1]
            ),
            "full_artifact_deferral_rate": full_defer_rate,
        },
        "discrimination_cost": {
            "auc_p_ml": auc_ml,
            "auc_p_blend": auc_blend,
            "auc_p_cbes": auc_cbes,
            "auc_delta_blend_minus_ml": auc_blend - auc_ml,
            "paired_bootstrap": bootstrap,
        },
        "decision_policy_comparison": {
            "old_p_ml_only": old_metrics,
            "new_blend": new_metrics,
        },
        "gate_condition_1": {
            "note": "research.relearning.gate.evaluate_condition_1, unmodified, "
                    f"trials={GATE_TRIALS}, seed={GATE_SEED}, on the test half",
            "old_p_ml_only": gate_old,
            "new_blend": gate_new,
        },
        "alpha_sweep": sweep,
        "artifact": {
            "path": str(PREDICTIONS.relative_to(REPO_ROOT)),
            "rows": int(len(updated)),
            "columns_updated": [
                "final_decision", "approval_threshold", "rejection_threshold", "confidence",
            ],
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: report[k] for k in (
        "alpha", "policy", "discrimination_cost", "gate_condition_1")}, indent=2))
    print(f"\nwrote {REPORT_PATH}")
    print(f"updated {PREDICTIONS} ({len(updated)} rows, defer rate {full_defer_rate:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
