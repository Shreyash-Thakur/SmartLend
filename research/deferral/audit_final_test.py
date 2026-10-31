"""Selection-bias-free deferral audit: three-way split, winner audited once.

Reviewer item: racing six candidate routers on one held-out half and
reporting the winner's statistics on that same half uses the test set for
model selection. This script removes that bias:

  Stage 1 (TUNE, 50%):      fit every signal's data-dependent pieces
                            (percentile maps, z stats, isotonic maps) and
                            each signal's 22.5% deferral threshold.
  Stage 2 (SELECTION, 25%): race the candidates; pick the winner by a rule
                            fixed here before results: the candidate whose
                            deferred pile has the HIGHEST 0/1 decision loss
                            (hardest pile) among candidates inside the
                            20-25% rate band.
  Stage 3 (FINAL, 25%):     evaluate ONLY the pre-selected winner (plus the
                            deployed policy's recorded decisions as the
                            operational reference) with the unmodified gate
                            statistic at m=10,000 permutations, empirical
                            p-values, and the cost metrics.

The split derives from one permutation (seed 20260831, the project's
standard policy-split seed): first half TUNE, third quarter SELECTION,
fourth quarter FINAL. The FINAL slice is touched exactly once.

LABEL DIRECTIONS: y_true == 1 means GOOD; probabilities are APPROVAL
probabilities (backend/artifacts/prediction_outputs.csv).

Run: ``python -m research.deferral.audit_final_test``
Writes ``reports/deferral_audit_final_test.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.deferral.audit_10k import audit_router, cost_metrics
from research.deferral.evaluate import GATE_SEED, RATE_BAND, TARGET_RATE, defer_mask_at_rate
from research.deferral.signals import all_candidates
from research.relearning.gate import _predicted_good

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = REPO_ROOT / "backend" / "artifacts" / "prediction_outputs.csv"
REPORT_PATH = REPO_ROOT / "reports" / "deferral_audit_final_test.json"

SPLIT_SEED = 20260831
TRIALS = 10_000


def three_way_split(frame: pd.DataFrame, seed: int = SPLIT_SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(frame))
    half, three_q = len(frame) // 2, (len(frame) * 3) // 4
    tune = frame.iloc[perm[:half]].reset_index(drop=True)
    select = frame.iloc[perm[half:three_q]].reset_index(drop=True)
    final = frame.iloc[perm[three_q:]].reset_index(drop=True)
    return tune, select, final


def cols(f: pd.DataFrame):
    return (
        f["best_model_prob"].to_numpy(dtype=float),
        f["cbes_prob"].to_numpy(dtype=float),
        f["approval_threshold"].to_numpy(dtype=float),
    )


def deferred_pile_loss(frame: pd.DataFrame, defer: np.ndarray) -> float:
    """0/1 decision loss of the no-deferral engine on the deferred pile."""
    y = frame["y_true"].to_numpy(dtype=int)
    return float((_predicted_good(frame)[defer] != y[defer]).mean())


def main() -> int:
    frame = pd.read_csv(PREDICTIONS)
    tune, select, final = three_way_split(frame)

    # Stage 1 + 2: fit on tune, race on selection, pick by the fixed rule.
    selection_table: list[dict[str, Any]] = []
    fitted = []
    for sig in all_candidates():
        ml_tu, cb_tu, th_tu = cols(tune)
        sig.fit(ml_tu, cb_tu, tune["y_true"].to_numpy(dtype=int), th_tu)
        scores_tune = sig.score(ml_tu, cb_tu, th_tu)
        defer_sel, threshold = defer_mask_at_rate(scores_tune, sig.score(*cols(select)), TARGET_RATE)
        rate = float(defer_sel.mean())
        entry = {
            "signal": sig.name,
            "selection_deferral_rate": rate,
            "rate_within_band": bool(RATE_BAND[0] <= rate <= RATE_BAND[1]),
            "selection_deferred_pile_loss": deferred_pile_loss(select, defer_sel),
            "threshold_from_tune": threshold,
        }
        selection_table.append(entry)
        fitted.append((sig, threshold, entry))

    eligible = [e for e in selection_table if e["rate_within_band"]]
    winner_name = max(eligible or selection_table,
                      key=lambda e: e["selection_deferred_pile_loss"])["signal"]

    # Stage 3: the winner, evaluated once on the untouched FINAL slice.
    winner_sig, winner_thr, _ = next(t for t in fitted if t[0].name == winner_name)
    defer_final = winner_sig.score(*cols(final)) > winner_thr
    print(f"[final-test] winner on selection slice: {winner_name}; auditing on final ...")
    winner_audit = audit_router(
        winner_name, f"pre-selected on the selection slice; threshold from tune ({winner_thr:.4f})",
        final, defer_final, TRIALS, GATE_SEED, keep_null=True,
    )

    # Second pre-specified router: the production-WIRED rule. Its selection is
    # not data-dependent on this split: ml_uncertainty was fixed as the engine's
    # measured fix by commit 0a253bd (2026-08-31), before the current decision
    # artifact existed, so auditing it on the final slice involves no selection
    # on this data.
    wired_audit = None
    if winner_name != "ml_uncertainty":
        wired_sig, wired_thr, _ = next(t for t in fitted if t[0].name == "ml_uncertainty")
        defer_wired = wired_sig.score(*cols(final)) > wired_thr
        print("[final-test] historically wired rule (ml_uncertainty) ...")
        wired_audit = audit_router(
            "ml_uncertainty", "pre-specified by commit 0a253bd (2026-08-31), predating this artifact",
            final, defer_wired, TRIALS, GATE_SEED, keep_null=False,
        )

    # Operational reference: the deployed policy's own recorded decisions.
    deployed_final = (final["final_decision"].astype(str).str.upper() == "DEFER").to_numpy()
    print("[final-test] deployed policy reference ...")
    deployed_audit = audit_router(
        "deployed_band_policy", "regenerated artifact decisions on the final slice",
        final, deployed_final, TRIALS, GATE_SEED, keep_null=False,
    )

    report = {
        "protocol": {
            "data": str(PREDICTIONS.relative_to(REPO_ROOT)),
            "split": f"50/25/25 tune/selection/final from one permutation, seed {SPLIT_SEED}",
            "n_tune": int(len(tune)), "n_selection": int(len(select)), "n_final": int(len(final)),
            "selection_rule": ("fixed before results: highest deferred-pile 0/1 decision loss on the "
                               "selection slice among candidates inside the 20-25% rate band"),
            "final_slice_contract": "touched exactly once, by the pre-selected winner and the deployed reference",
            "gate": f"research.relearning.gate.evaluate_condition_1, unmodified, trials={TRIALS}, seed={GATE_SEED}",
        },
        "stage2_selection_race": selection_table,
        "winner": winner_name,
        "stage3_final_test": {
            "winner": winner_audit,
            "historically_wired_ml_uncertainty": wired_audit,
            "deployed_band_policy": deployed_audit,
            "no_deferral_baseline_cost": cost_metrics(final, np.zeros(len(final), dtype=bool)),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    w = winner_audit
    print(f"[final-test] {winner_name}: rate {w['deferral_rate']:.2%}  acc_z {w['accuracy_z']:+.1f}  "
          f"p {w['p_empirical']['accuracy']:.1e}  cost {w['cost']['expected_cost_auto_per_applicant']:.4f}")
    d = deployed_audit
    print(f"[final-test] deployed: acc_z {d['accuracy_z']:+.1f}  p {d['p_empirical']['accuracy']:.2f}")
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
