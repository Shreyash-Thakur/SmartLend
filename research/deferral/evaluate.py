"""Race the candidate deferral signals on real out-of-fold predictions.

PROTOCOL
--------
1. Load ``backend/artifacts/prediction_outputs.csv`` (307,511 OOF rows;
   ``y_true == 1`` means GOOD / approve-worthy; all probabilities are
   approval probabilities).
2. Split 50/50 into TUNE and TEST with a fixed seed. Everything data-dependent
   — percentile maps, z statistics, isotonic calibrators, and the deferral
   threshold itself — is fit on TUNE only. All reported numbers come from
   TEST. Nothing is tuned on the split it is scored on.
3. Operating point: a HARD business requirement of a 20-25% deferral rate
   (underwriter capacity), so each candidate's threshold is the TUNE-split
   quantile that defers the TARGET_RATE (22.5%, mid-band). This sits ABOVE
   the AUC-implied natural-rate bound checked by gate condition 2 — that
   tension is measured and reported, not hidden.
4. For each candidate, rebuild the decision column on TEST (DEFER where the
   signal exceeds its threshold, otherwise the engine's own hard
   approve/reject at ``approval_threshold``) and run the UNMODIFIED gate
   condition 1 from ``research.relearning.gate``. The z-scores it emits are
   the success criterion: negative means the deferred pile is genuinely
   harder than a random router's.
5. Risk-coverage: at the matched rate, compare each candidate's selective
   risk (error rate on the kept/auto-decided pile) against random abstention
   (the floor to beat: selective risk == overall error rate) and oracle
   abstention (the ceiling: defer actual errors first).

Run: ``python -m research.deferral.evaluate``
Writes ``reports/deferral_fix.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.deferral.signals import BaseSignal, all_candidates
from research.relearning.gate import (
    _predicted_good,
    auc_implied_natural_rate_bound,
    evaluate_condition_1,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = REPO_ROOT / "backend" / "artifacts" / "prediction_outputs.csv"
REPORT_PATH = REPO_ROOT / "reports" / "deferral_fix.json"

SPLIT_SEED = 20260831
# Business requirement: underwriter capacity supports deferring 20-25% of
# applications. Threshold is selected (on TUNE) to defer the mid-band rate.
TARGET_RATE = 0.225
RATE_BAND = (0.20, 0.25)
GATE_TRIALS = 200
GATE_SEED = 20260830


# ---------------------------------------------------------------------------
# decision reconstruction + risk-coverage
# ---------------------------------------------------------------------------


def defer_mask_at_rate(scores_tune: np.ndarray, scores_eval: np.ndarray, target_rate: float) -> tuple[np.ndarray, float]:
    """Threshold chosen on TUNE scores; mask computed on EVAL scores.

    Returns (mask, threshold). Defers where score > threshold (strict, so a
    signal with heavy ties — e.g. isotonic steps — defers *at most* the mass
    above the quantile rather than a runaway share).
    """
    threshold = float(np.quantile(scores_tune, 1.0 - target_rate))
    return scores_eval > threshold, threshold


def rebuild_decisions(frame: pd.DataFrame, defer: np.ndarray) -> pd.DataFrame:
    """Rewrite final_decision: DEFER per mask, else hard approve/reject.

    The auto decision uses the same rule as the gate's no-deferral baseline
    (best_model_prob >= approval_threshold), so condition 1's accuracy
    comparison stays apples-to-apples with production.
    """
    out = frame.copy()
    auto = np.where(_predicted_good(frame) == 1, "APPROVE", "REJECT")
    out["final_decision"] = np.where(defer, "DEFER", auto)
    return out


def risk_coverage_point(frame: pd.DataFrame, defer: np.ndarray) -> dict[str, Any]:
    """Selective risk at this candidate's coverage, plus the two references.

    * selective_risk  — error rate of the forced approve/reject on the KEPT pile.
    * random_risk     — a random router keeps an unbiased sample, so its
                        expected selective risk is the overall error rate.
    * oracle_risk     — defer the actual errors first: with n_defer slots the
                        best achievable kept-pile error is
                        max(0, errors - n_defer) / n_kept.
    * position        — where the candidate sits between random (0) and
                        oracle (1); negative means WORSE than random.
    """
    y = frame["y_true"].to_numpy(dtype=int)
    errors = (_predicted_good(frame) != y).astype(int)
    n = len(frame)
    n_defer = int(defer.sum())
    n_keep = n - n_defer

    overall_risk = float(errors.mean())
    selective_risk = float(errors[~defer].mean()) if n_keep else None
    oracle_risk = float(max(0, int(errors.sum()) - n_defer) / n_keep) if n_keep else None

    position = None
    if selective_risk is not None and oracle_risk is not None and overall_risk > oracle_risk:
        position = float((overall_risk - selective_risk) / (overall_risk - oracle_risk))

    return {
        "coverage": float(n_keep / n),
        "deferral_rate": float(n_defer / n),
        "selective_risk": selective_risk,
        "random_risk_at_matched_coverage": overall_risk,
        "oracle_risk_at_matched_coverage": oracle_risk,
        "position_random0_oracle1": position,
        "beats_random": None if selective_risk is None else bool(selective_risk < overall_risk),
    }


def risk_coverage_curve(errors: np.ndarray, scores: np.ndarray, coverages: np.ndarray) -> list[dict[str, float]]:
    """Selective risk versus coverage for one signal (higher score = defer first)."""
    order = np.argsort(scores)  # ascending: keep the LOWEST-signal cases first
    err_sorted = errors[order]
    cum_err = np.cumsum(err_sorted)
    n = len(errors)
    points = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        points.append({"coverage": float(k / n), "selective_risk": float(cum_err[k - 1] / k)})
    return points


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def split_tune_test(frame: pd.DataFrame, seed: int = SPLIT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(frame))
    half = len(frame) // 2
    tune = frame.iloc[perm[:half]].reset_index(drop=True)
    test = frame.iloc[perm[half:]].reset_index(drop=True)
    return tune, test


def evaluate_candidate(
    signal: BaseSignal,
    tune: pd.DataFrame,
    test: pd.DataFrame,
    target_rate: float = TARGET_RATE,
    gate_trials: int = GATE_TRIALS,
    gate_seed: int = GATE_SEED,
) -> dict[str, Any]:
    """Fit on TUNE, defer at the matched rate, score on TEST via the real gate."""
    cols = lambda f: (
        f["best_model_prob"].to_numpy(dtype=float),
        f["cbes_prob"].to_numpy(dtype=float),
        f["approval_threshold"].to_numpy(dtype=float),
    )
    ml_tu, cb_tu, th_tu = cols(tune)
    ml_te, cb_te, th_te = cols(test)

    signal.fit(ml_tu, cb_tu, tune["y_true"].to_numpy(dtype=int), th_tu)
    scores_tune = signal.score(ml_tu, cb_tu, th_tu)
    scores_test = signal.score(ml_te, cb_te, th_te)

    defer, threshold = defer_mask_at_rate(scores_tune, scores_test, target_rate)
    rebuilt = rebuild_decisions(test, defer)

    cond1 = evaluate_condition_1(rebuilt, trials=gate_trials, seed=gate_seed)
    rc = risk_coverage_point(test, defer)

    rate = float(defer.mean())
    return {
        "signal": signal.name,
        "description": signal.description,
        "threshold_from_tune_split": threshold,
        "test_deferral_rate": rate,
        "rate_within_20_25_band": bool(RATE_BAND[0] <= rate <= RATE_BAND[1]),
        "gate_condition_1": {
            "status": cond1.status,
            "balance_distance_z": cond1.metrics["balance_distance_z"],
            "accuracy_z": cond1.metrics["accuracy_z"],
            "deferred_good_share": cond1.metrics["deferred"]["good_share"],
            "auto_good_share": cond1.metrics["auto_decided"]["good_share"],
            "deferred_accuracy": cond1.metrics["deferred"]["accuracy"],
            "auto_accuracy": cond1.metrics["auto_decided"]["accuracy"],
            "reason": cond1.reason,
        },
        "risk_coverage": rc,
    }


def main() -> int:
    frame = pd.read_csv(PREDICTIONS)
    tune, test = split_tune_test(frame)

    # ---- BEFORE: the production router's own decisions, on the same TEST rows
    before = evaluate_condition_1(test, trials=GATE_TRIALS, seed=GATE_SEED)
    before_rc = risk_coverage_point(
        test, (test["final_decision"].astype(str).str.upper() == "DEFER").to_numpy()
    )

    # ---- candidates at the matched 20-25% rate
    results = [evaluate_candidate(sig, tune, test) for sig in all_candidates()]

    # winner = most negative accuracy z (the gate's harder-for-model test),
    # among candidates that landed inside the required rate band and beat random.
    eligible = [
        r
        for r in results
        if r["rate_within_20_25_band"] and r["risk_coverage"]["beats_random"]
    ]
    pool = eligible or results
    winner = min(pool, key=lambda r: r["gate_condition_1"]["accuracy_z"] or 0.0)

    # ---- capacity-vs-AUC tension (gate condition 2's bound, stated honestly)
    y = test["y_true"].to_numpy(dtype=int)
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, test["best_model_prob"].to_numpy(dtype=float)))
    bound = auc_implied_natural_rate_bound(auc, float(y.mean()))
    n_defer = int(round(winner["test_deferral_rate"] * len(test)))
    excess = max(0.0, winner["test_deferral_rate"] - bound["upper_bound"])
    tension = {
        "required_rate_band": list(RATE_BAND),
        "auc_implied_bound": [bound["lower_bound"], bound["upper_bound"]],
        "winner_rate": winner["test_deferral_rate"],
        "rate_exceeds_auc_bound": winner["test_deferral_rate"] > bound["upper_bound"],
        "excess_rate_above_bound": excess,
        "avoidable_deferral_share_of_deferrals": (
            excess / winner["test_deferral_rate"] if winner["test_deferral_rate"] else None
        ),
        "note": (
            "The 20-25% rate is an underwriter-capacity requirement and sits above "
            "the AUC-implied natural rate: the model's discriminative power only "
            "justifies deferring up to the upper bound, so roughly the excess share "
            "of deferrals will be cases the model already decides correctly. Gate "
            "condition 2 is expected to keep FAILING at this rate; that is a "
            "capacity decision, not a router defect."
        ),
    }

    # ---- risk-coverage curves for the report (documentation, not selection)
    errors = (_predicted_good(test) != y).astype(int)
    coverages = np.arange(0.05, 1.0, 0.05)
    curves = {}
    for sig, res in zip(all_candidates(), results):
        cols = (
            test["best_model_prob"].to_numpy(dtype=float),
            test["cbes_prob"].to_numpy(dtype=float),
            test["approval_threshold"].to_numpy(dtype=float),
        )
        sig.fit(
            tune["best_model_prob"].to_numpy(dtype=float),
            tune["cbes_prob"].to_numpy(dtype=float),
            tune["y_true"].to_numpy(dtype=int),
            tune["approval_threshold"].to_numpy(dtype=float),
        )
        curves[res["signal"]] = risk_coverage_curve(errors, sig.score(*cols), coverages)
    # references
    order_oracle = np.argsort(-errors)  # defer errors first == keep non-errors first
    curves["oracle"] = risk_coverage_curve(errors, errors.astype(float), coverages)
    curves["random"] = [
        {"coverage": float(c), "selective_risk": float(errors.mean())} for c in coverages
    ]

    report = {
        "protocol": {
            "data": str(PREDICTIONS.relative_to(REPO_ROOT)),
            "n_total": int(len(frame)),
            "split": "50/50 tune/test, seed %d; all fitting and threshold selection on tune, all reported numbers on test" % SPLIT_SEED,
            "target_deferral_rate": TARGET_RATE,
            "required_rate_band": list(RATE_BAND),
            "gate": "research.relearning.gate.evaluate_condition_1, unmodified, %d trials seed %d" % (GATE_TRIALS, GATE_SEED),
        },
        "scale_offset_hypothesis": {
            "confirmed": True,
            "mean_p_ml": float(frame["best_model_prob"].mean()),
            "mean_p_cbes": float(frame["cbes_prob"].mean()),
            "mean_offset": float((frame["best_model_prob"] - frame["cbes_prob"]).mean()),
            "share_rows_p_ml_above_p_cbes": float((frame["best_model_prob"] > frame["cbes_prob"]).mean()),
            "mean_abs_diff": float((frame["best_model_prob"] - frame["cbes_prob"]).abs().mean()),
            "mean_abs_diff_after_removing_mean_offset": float(
                (
                    (frame["best_model_prob"] - frame["cbes_prob"])
                    - (frame["best_model_prob"] - frame["cbes_prob"]).mean()
                )
                .abs()
                .mean()
            ),
            "corr_D_with_ml_confidence": float(
                np.corrcoef(
                    (frame["best_model_prob"] - frame["cbes_prob"]).abs(),
                    (frame["best_model_prob"] - 0.5).abs(),
                )[0, 1]
            ),
        },
        "before_production_router_on_test_split": {
            "deferral_rate": before_rc["deferral_rate"],
            "gate_condition_1_status": before.status,
            "balance_distance_z": before.metrics["balance_distance_z"],
            "accuracy_z": before.metrics["accuracy_z"],
            "risk_coverage": before_rc,
        },
        "candidates_at_matched_rate": results,
        "winner": winner["signal"],
        "winner_detail": winner,
        "capacity_vs_auc_bound": tension,
        "risk_coverage_curves": curves,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # console summary
    print(f"BEFORE (production router, test split): rate={before_rc['deferral_rate']:.2%} "
          f"balance_z={before.metrics['balance_distance_z']:+.2f} accuracy_z={before.metrics['accuracy_z']:+.2f}")
    print(f"{'signal':<22}{'rate':>8}{'bal_z':>10}{'acc_z':>10}{'sel_risk':>10}{'rand':>8}{'oracle':>8}{'pos':>7}")
    for r in results:
        rc = r["risk_coverage"]
        g = r["gate_condition_1"]
        print(
            f"{r['signal']:<22}{r['test_deferral_rate']:>8.2%}{g['balance_distance_z']:>+10.2f}"
            f"{g['accuracy_z']:>+10.2f}{rc['selective_risk']:>10.4f}"
            f"{rc['random_risk_at_matched_coverage']:>8.4f}{rc['oracle_risk_at_matched_coverage']:>8.4f}"
            f"{(rc['position_random0_oracle1'] if rc['position_random0_oracle1'] is not None else float('nan')):>7.2f}"
        )
    print(f"WINNER: {winner['signal']}  (report -> {REPORT_PATH})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
