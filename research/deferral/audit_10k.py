"""Deferral audit with a 10,000-trial permutation null (reviewer item).

Addresses three review findings on the deferral audit:

1. 200 random-router trials cannot resolve small p-values; a z-score alone
   may reflect a tiny estimated null SD. -> Rerun gate condition 1 with
   trials=10,000 (the gate itself is UNMODIFIED; ``trials`` is an existing
   parameter) and report EMPIRICAL permutation p-values with the standard
   add-one correction: p = (1 + #{null at least as extreme}) / (trials + 1).
2. The audit statistic should not rest on accuracy alone for an 8.07%
   default-rate problem. -> Report, per router at the matched rate:
   default recall among deferred cases and the expected financial cost of
   the auto-decided pile (false-approval cost 0.45, false-rejection cost
   0.2644 -- the same Basel-II LGD / EMP-ROI constants as
   ``reports/t_base_selection.json``).
3. Stability: repeat the headline comparison across extra tune/test split
   seeds (2,000 trials each) to show the sign and magnitude are not an
   artifact of one split.

Raw null samples (thinned to 2,000 values) are saved for the permutation
figure; they reproduce the gate's own null exactly (same rng seed, same
draw order, rng consumed only by ``choice``).

LABEL DIRECTIONS (do not change): in prediction_outputs.csv, y_true == 1
means GOOD customer and all probabilities are APPROVAL probabilities.

Run: ``python -m research.deferral.audit_10k``
Writes ``reports/deferral_audit_10k.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.deferral.evaluate import (
    GATE_SEED,
    RATE_BAND,
    SPLIT_SEED,
    TARGET_RATE,
    defer_mask_at_rate,
    rebuild_decisions,
    risk_coverage_point,
    split_tune_test,
)
from research.deferral.signals import all_candidates
from research.relearning.gate import _predicted_good, evaluate_condition_1

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = REPO_ROOT / "backend" / "artifacts" / "prediction_outputs.csv"
REPORT_PATH = REPO_ROOT / "reports" / "deferral_audit_10k.json"

TRIALS_MAIN = 10_000
TRIALS_STABILITY = 2_000
EXTRA_SPLIT_SEEDS = [20260901, 20260902]
NULL_SAMPLE_KEEP = 2_000  # thinned samples stored for the figure

# Decision-time cost constants -- same provenance as reports/t_base_selection.json
COST_FALSE_APPROVAL = 0.45    # LGD x EAD, Basel II F-IRB senior unsecured (BCBS 128 para 287)
COST_FALSE_REJECTION = 0.2644  # foregone ROI on a rejected good customer (Verbraken et al. 2014)


def null_samples(frame: pd.DataFrame, n_defer: int, trials: int, seed: int) -> dict[str, np.ndarray]:
    """Replicate the gate's random-router null draws exactly.

    Same rng construction and the identical per-trial ``choice`` call as
    ``gate._random_router_baseline``; nothing else in that loop consumes the
    rng, so these samples ARE the gate's null distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(frame)
    y = frame["y_true"].to_numpy(dtype=int)
    predicted = _predicted_good(frame)
    balance = np.empty(trials)
    accuracy = np.empty(trials)
    for i in range(trials):
        idx = rng.choice(n, size=n_defer, replace=False)
        ys = y[idx]
        balance[i] = abs(float(ys.mean()) - 0.5)
        accuracy[i] = float((predicted[idx] == ys).mean())
    return {"balance_distance": balance, "accuracy": accuracy}


def empirical_p(null: np.ndarray, observed: float, direction: str) -> float:
    """Add-one permutation p-value. direction: 'le' means extreme = null <= observed."""
    hits = int((null <= observed).sum()) if direction == "le" else int((null >= observed).sum())
    return float((1 + hits) / (len(null) + 1))


def observed_statistics(frame: pd.DataFrame, defer: np.ndarray) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=int)
    predicted = _predicted_good(frame)
    ys = y[defer]
    return {
        "balance_distance": float(abs(ys.mean() - 0.5)),
        "accuracy": float((predicted[defer] == ys).mean()),
    }


def cost_metrics(frame: pd.DataFrame, defer: np.ndarray) -> dict[str, Any]:
    """Decision-cost view of a router at its operating point.

    default_recall_deferred: share of ALL actual defaulters routed to review.
    expected_cost_auto_per_applicant: cost of the forced approve/reject on the
    kept pile, averaged over the WHOLE test population (so routers with equal
    rates are directly comparable).
    """
    y = frame["y_true"].to_numpy(dtype=int)          # 1 = good
    predicted = _predicted_good(frame)                # 1 = approve
    bad = y == 0
    keep = ~defer
    false_approvals = int((keep & (predicted == 1) & bad).sum())
    false_rejections = int((keep & (predicted == 0) & ~bad).sum())
    n = len(frame)
    return {
        "default_recall_deferred": float((defer & bad).sum() / max(1, bad.sum())),
        "false_approvals_auto": false_approvals,
        "false_rejections_auto": false_rejections,
        "expected_cost_auto_per_applicant": float(
            (COST_FALSE_APPROVAL * false_approvals + COST_FALSE_REJECTION * false_rejections) / n
        ),
    }


def audit_router(
    name: str,
    description: str,
    frame_test: pd.DataFrame,
    defer: np.ndarray,
    trials: int,
    gate_seed: int,
    keep_null: bool,
) -> dict[str, Any]:
    """Full audit of one router's deferral mask on the test split."""
    rebuilt = rebuild_decisions(frame_test, defer)
    cond1 = evaluate_condition_1(rebuilt, trials=trials, seed=gate_seed)
    null = null_samples(frame_test, int(defer.sum()), trials, gate_seed)
    obs = observed_statistics(frame_test, defer)
    out: dict[str, Any] = {
        "router": name,
        "description": description,
        "deferral_rate": float(defer.mean()),
        "gate_status": cond1.status,
        "balance_distance_z": cond1.metrics["balance_distance_z"],
        "accuracy_z": cond1.metrics["accuracy_z"],
        "observed": obs,
        "null_mean": {k: float(v.mean()) for k, v in null.items()},
        "null_sd": {k: float(v.std(ddof=1)) for k, v in null.items()},
        # A working router defers HARDER cases: lower balance distance, lower
        # model accuracy on the deferred pile -> extreme = null <= observed
        # would mean the null beat us; we test observed <= null.
        "p_empirical": {
            "balance_distance": empirical_p(null["balance_distance"], obs["balance_distance"], "le"),
            "accuracy": empirical_p(null["accuracy"], obs["accuracy"], "le"),
        },
        "p_resolution_floor": float(1.0 / (trials + 1)),
        "risk_coverage": risk_coverage_point(frame_test, defer),
        "cost": cost_metrics(frame_test, defer),
    }
    if keep_null:
        step = max(1, trials // NULL_SAMPLE_KEEP)
        out["null_samples_thinned"] = {
            "accuracy": [round(float(v), 6) for v in null["accuracy"][::step]],
            "balance_distance": [round(float(v), 6) for v in null["balance_distance"][::step]],
        }
    return out


def candidate_defer_mask(signal, tune: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    cols = lambda f: (
        f["best_model_prob"].to_numpy(dtype=float),
        f["cbes_prob"].to_numpy(dtype=float),
        f["approval_threshold"].to_numpy(dtype=float),
    )
    ml_tu, cb_tu, th_tu = cols(tune)
    signal.fit(ml_tu, cb_tu, tune["y_true"].to_numpy(dtype=int), th_tu)
    defer, _ = defer_mask_at_rate(
        signal.score(ml_tu, cb_tu, th_tu), signal.score(*cols(test)), TARGET_RATE
    )
    return defer


def main() -> int:
    frame = pd.read_csv(PREDICTIONS)
    tune, test = split_tune_test(frame)

    routers: list[dict[str, Any]] = []

    # The regenerated artifact's own recorded decisions: the currently
    # deployed blend + uncertainty-band policy at the 22.5% capacity rate
    # (NOT the historical 52% disagreement router, which predates this
    # artifact and is reconstructed below as current_abs_diff).
    deployed_mask = (test["final_decision"].astype(str).str.upper() == "DEFER").to_numpy()
    print("[audit-10k] deployed band policy ...")
    routers.append(
        audit_router(
            "deployed_band_policy",
            "regenerated artifact decisions: blend + uncertainty band at capacity",
            test, deployed_mask, TRIALS_MAIN, GATE_SEED, keep_null=False,
        )
    )

    # Candidates, all at the matched 22.5% rate, thresholds from TUNE only.
    for sig in all_candidates():
        print(f"[audit-10k] candidate {sig.name} ...")
        defer = candidate_defer_mask(sig, tune, test)
        routers.append(
            audit_router(
                sig.name, sig.description, test, defer, TRIALS_MAIN, GATE_SEED,
                keep_null=(sig.name in ("ml_uncertainty", "current_abs_diff")),
            )
        )

    # Stability: repeat deployed policy / winner across extra split seeds.
    stability = []
    for split_seed in EXTRA_SPLIT_SEEDS:
        tune_s, test_s = split_tune_test(frame, seed=split_seed)
        before_s = (test_s["final_decision"].astype(str).str.upper() == "DEFER").to_numpy()
        row: dict[str, Any] = {"split_seed": split_seed, "trials": TRIALS_STABILITY}
        b = audit_router("deployed_band_policy", "", test_s, before_s, TRIALS_STABILITY, GATE_SEED, False)
        row["deployed_band_policy"] = {
            "accuracy_z": b["accuracy_z"],
            "balance_distance_z": b["balance_distance_z"],
            "p_accuracy": b["p_empirical"]["accuracy"],
        }
        for sig in all_candidates():
            if sig.name != "ml_uncertainty":
                continue
            defer = candidate_defer_mask(sig, tune_s, test_s)
            w = audit_router(sig.name, "", test_s, defer, TRIALS_STABILITY, GATE_SEED, False)
            row["ml_uncertainty"] = {
                "accuracy_z": w["accuracy_z"],
                "balance_distance_z": w["balance_distance_z"],
                "p_accuracy": w["p_empirical"]["accuracy"],
            }
        print(f"[audit-10k] stability split {split_seed} done")
        stability.append(row)

    # Random-reference cost: forced decisions on everything (no deferral).
    no_defer = np.zeros(len(test), dtype=bool)
    baseline_cost = cost_metrics(test, no_defer)

    report = {
        "protocol": {
            "data": str(PREDICTIONS.relative_to(REPO_ROOT)),
            "n_test": int(len(test)),
            "split": f"50/50 tune/test, seed {SPLIT_SEED}; thresholds fitted on tune only",
            "target_deferral_rate": TARGET_RATE,
            "required_rate_band": list(RATE_BAND),
            "gate": f"research.relearning.gate.evaluate_condition_1, unmodified, trials={TRIALS_MAIN}, seed={GATE_SEED}",
            "p_value_definition": "add-one empirical permutation p = (1 + #extreme) / (trials + 1); floor 1/(trials+1)",
            "cost_constants": {
                "false_approval": COST_FALSE_APPROVAL,
                "false_rejection": COST_FALSE_REJECTION,
                "provenance": "reports/t_base_selection.json (Basel II F-IRB LGD; Verbraken et al. 2014 EMP ROI)",
            },
        },
        "no_deferral_baseline_cost": baseline_cost,
        "routers": routers,
        "stability_across_split_seeds": stability,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'router':<22}{'rate':>8}{'acc_z':>10}{'p_acc':>12}{'dflt_recall':>12}{'cost/app':>10}")
    for r in routers:
        print(
            f"{r['router']:<22}{r['deferral_rate']:>8.2%}{r['accuracy_z']:>+10.2f}"
            f"{r['p_empirical']['accuracy']:>12.2e}{r['cost']['default_recall_deferred']:>12.3f}"
            f"{r['cost']['expected_cost_auto_per_applicant']:>10.4f}"
        )
    print(f"no-deferral baseline cost/applicant: {baseline_cost['expected_cost_auto_per_applicant']:.4f}")
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
