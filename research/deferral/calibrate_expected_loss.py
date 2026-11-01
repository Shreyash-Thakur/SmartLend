"""Calibrate the serving-frame EXPECTED-LOSS deferral threshold (theta_EL).

The cost-aware routing study (reports/cost_aware_routing.json) showed, in the
research frame, that ranking review cases by expected decision loss minimises
auto-pile cost (lambda* = 0). This script carries that result into the
SERVING frame so the decision engine can expose it as a third flag-gated
deferral mode, with the same discipline as the wired uncertainty rule:

- data: backend/artifacts/prediction_outputs.csv (y_true == 1 means GOOD;
  all probabilities are approval probabilities),
- 50/50 tune/test split, seed 20260831 (research.deferral.evaluate),
- the deferral threshold is the TUNE-split quantile that defers 22.5%
  (mid-point of the 20-25% capacity band); all reported numbers are TEST,
- the router is scored by the UNMODIFIED gate condition 1 plus the
  risk-coverage point, exactly like every other candidate.

Signal: the expected loss of the engine's own hard action at its stored
per-row approval threshold t,

    E[L] = C_FA * (1 - p_ml)   if p_ml >= t   (approve a possible defaulter)
           C_FR * p_ml         otherwise      (reject a possible good)

with the decision-layer cost pair C_FA = 0.45 (Basel II F-IRB supervisory
LGD) and C_FR = 0.2644 (EMP return parameter, Verbraken et al. 2014) -- the
same pair as reports/t_base_selection.json. The wired margin rule
|p_ml - t| is re-scored alongside for a like-for-like cost comparison.

Run: ``python -m research.deferral.calibrate_expected_loss``
Writes ``reports/expected_loss_calibration.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.deferral.evaluate import (
    GATE_SEED,
    GATE_TRIALS,
    PREDICTIONS,
    RATE_BAND,
    TARGET_RATE,
    defer_mask_at_rate,
    rebuild_decisions,
    risk_coverage_point,
    split_tune_test,
)
from research.relearning.gate import _predicted_good, evaluate_condition_1

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "reports" / "expected_loss_calibration.json"

COST_FALSE_APPROVAL = 0.45    # Basel II F-IRB supervisory LGD
COST_FALSE_REJECTION = 0.2644  # EMP return parameter (Verbraken et al. 2014)


def expected_loss(frame: pd.DataFrame) -> np.ndarray:
    """Expected loss of the engine's hard action at its stored threshold."""
    p = frame["best_model_prob"].to_numpy(dtype=float)
    t = frame["approval_threshold"].to_numpy(dtype=float)
    return np.where(p >= t,
                    COST_FALSE_APPROVAL * (1.0 - p),
                    COST_FALSE_REJECTION * p)


def margin_uncertainty(frame: pd.DataFrame) -> np.ndarray:
    """The wired rule's signal, inverted so that higher = defer first."""
    p = frame["best_model_prob"].to_numpy(dtype=float)
    t = frame["approval_threshold"].to_numpy(dtype=float)
    return -np.abs(p - t)


def realized_cost(frame: pd.DataFrame, defer: np.ndarray) -> dict[str, Any]:
    """Realized decision cost of the auto pile (deferred rows excluded)."""
    y = frame["y_true"].to_numpy(dtype=int)          # 1 == GOOD
    approve = np.asarray(_predicted_good(frame)) == 1
    fa = (~defer) & approve & (y == 0)               # approved defaulter
    fr = (~defer) & (~approve) & (y == 1)            # rejected good
    total = COST_FALSE_APPROVAL * fa.sum() + COST_FALSE_REJECTION * fr.sum()
    n_auto = int((~defer).sum())
    defaulters = int((y == 0).sum())
    return {
        "false_approvals": int(fa.sum()),
        "false_rejections": int(fr.sum()),
        "cost_per_auto_decided_applicant": round(float(total / n_auto), 6),
        "cost_per_applicant_overall": round(float(total / len(frame)), 6),
        "default_recall_among_deferred": round(
            float((defer & (y == 0)).sum() / defaulters), 4),
    }


def score_router(name: str, scores_tune: np.ndarray, scores_test: np.ndarray,
                 test: pd.DataFrame) -> dict[str, Any]:
    defer, threshold = defer_mask_at_rate(scores_tune, scores_test, TARGET_RATE)
    rebuilt = rebuild_decisions(test, defer)
    cond1 = evaluate_condition_1(rebuilt, trials=GATE_TRIALS, seed=GATE_SEED)
    rate = float(defer.mean())
    return {
        "router": name,
        "threshold_from_tune_split": float(threshold),
        "test_deferral_rate": rate,
        "rate_within_20_25_band": bool(RATE_BAND[0] <= rate <= RATE_BAND[1]),
        "gate_condition_1": {
            "status": cond1.status,
            "balance_distance_z": cond1.metrics["balance_distance_z"],
            "accuracy_z": cond1.metrics["accuracy_z"],
        },
        "risk_coverage": risk_coverage_point(test, defer),
        "cost": realized_cost(test, defer),
    }


def main() -> int:
    frame = pd.read_csv(PREDICTIONS)
    tune, test = split_tune_test(frame)

    no_defer = realized_cost(test, np.zeros(len(test), dtype=bool))

    results = [
        score_router("expected_loss", expected_loss(tune), expected_loss(test), test),
        score_router("ml_uncertainty_wired", margin_uncertainty(tune),
                     margin_uncertainty(test), test),
    ]

    el = results[0]
    report = {
        "protocol": {
            "data": str(PREDICTIONS.relative_to(REPO_ROOT)),
            "n_total": int(len(frame)),
            "split": "50/50 tune/test, seed 20260831 (research.deferral.evaluate.split_tune_test); threshold from tune, all numbers from test",
            "target_deferral_rate": TARGET_RATE,
            "gate": f"research.relearning.gate.evaluate_condition_1, unmodified, {GATE_TRIALS} trials seed {GATE_SEED}",
            "costs": {"false_approval": COST_FALSE_APPROVAL,
                      "false_rejection": COST_FALSE_REJECTION,
                      "provenance": "decision-layer pair, reports/t_base_selection.json"},
            "signal": "E[L] of the engine's hard action at its stored per-row approval threshold",
        },
        "no_deferral_baseline": no_defer,
        "routers_at_matched_rate": results,
        "wired_constant": {
            "name": "DEFAULT_THETA_EL",
            "value": round(el["threshold_from_tune_split"], 6),
            "meaning": "defer when E[L] of the auto action exceeds this; tune-split quantile deferring 22.5%",
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"no-deferral cost/applicant: {no_defer['cost_per_auto_decided_applicant']:.4f}")
    for r in results:
        g, c = r["gate_condition_1"], r["cost"]
        print(f"{r['router']:<22} rate={r['test_deferral_rate']:.2%} "
              f"acc_z={g['accuracy_z']:+.1f} cost/auto={c['cost_per_auto_decided_applicant']:.4f} "
              f"defaulter-recall-in-review={c['default_recall_among_deferred']:.1%} "
              f"theta={r['threshold_from_tune_split']:.6f}")
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
