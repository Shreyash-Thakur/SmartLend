"""Capacity-constrained cost-aware deferral: does epistemic uncertainty add
value beyond expected-loss ranking when review capacity is scarce?
(reviewer item: the strongest recommended upgrade)

Routing strategies compared across a review-capacity grid, in the research
frame (SGLB OOF cache; y==1 DEFAULT; auto decision = reject iff pd >= Youden
t* fitted on tune):

- random            : seeded uniform deferral
- chow_margin       : defer smallest |pd - t*| (reject-option rule)
- knowledge         : defer highest SGLB knowledge uncertainty
- expected_cost     : defer highest expected loss of the auto decision,
                      E[L] = pd * c_FA if auto-approve else (1-pd) * c_FR
- combined(lambda)  : lambda * rank(knowledge) + (1-lambda) * rank(E[L]),
                      ranks fitted on the tune slice

Three-way split (seed 20260831 permutation): 50% TUNE fits t*, rank maps
and quantiles; 25% SELECTION picks lambda (lowest realized total cost at
the 22.5% operating rate); 25% FINAL is touched once per strategy to report
the capacity sweep. Realized cost on FINAL per applicant:
c_FA per approved defaulter + c_FR per rejected good among auto decisions,
plus c_review per deferral (sensitivity over c_review). Primary costs are
the decision-layer pair (0.45 / 0.2644); 200-row-bootstrap CIs per point.

Run: ``python -m research.deferral.cost_aware_routing``
Writes ``reports/cost_aware_routing.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis.uncertainty_deferral_catboost import OOF_CACHE, youden_threshold
from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "reports" / "cost_aware_routing.json"

SPLIT_SEED = 20260831
C_FA, C_FR = 0.45, 0.2644          # decision-layer costs (t_base study provenance)
C_REVIEW_GRID = [0.0, 0.01, 0.02, 0.05]
CAPACITIES = [0.05, 0.10, 0.15, 0.20, 0.225, 0.25, 0.30, 0.35, 0.40]
LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
OPERATING_RATE = 0.225
BOOT = 200
RANDOM_ROUTER_SEED = 20260830


def rank_map(train_values: np.ndarray):
    """ECDF rank transform fitted on the tune slice."""
    sorted_v = np.sort(train_values)
    return lambda v: np.searchsorted(sorted_v, v, side="right") / len(sorted_v)


def realized_auto_loss(y_def: np.ndarray, auto_reject: np.ndarray) -> np.ndarray:
    """Per-applicant realized loss of the forced auto decision."""
    approve = ~auto_reject
    return C_FA * (approve & (y_def == 1)) + C_FR * (auto_reject & (y_def == 0))


def sweep(score: np.ndarray, tune_score: np.ndarray, loss_final: np.ndarray,
          rng_boot: np.random.Generator) -> list[dict]:
    """Defer the top-capacity quantile by score (quantile from tune); report
    realized auto cost per applicant on FINAL with bootstrap CI."""
    points = []
    n = len(score)
    for cap in CAPACITIES:
        thr = float(np.quantile(tune_score, 1.0 - cap))
        defer = score > thr
        kept_loss = np.where(defer, 0.0, loss_final)
        cost = float(kept_loss.mean())
        idx = rng_boot.integers(0, n, size=(BOOT, n))
        boots = kept_loss[idx].mean(axis=1)
        points.append({
            "capacity": cap,
            "deferral_rate": float(defer.mean()),
            "auto_cost_per_applicant": cost,
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        })
    return points


def main() -> int:
    out, _ = pd.read_pickle(OOF_CACHE)
    X, y_def, ids = build_feature_matrix()

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(y_def))
    half, three_q = len(y_def) // 2, (len(y_def) * 3) // 4
    tune_ix, sel_ix, fin_ix = perm[:half], perm[half:three_q], perm[three_q:]

    pd_s, know = out["pd"], out["knowledge"]
    t_star = youden_threshold(y_def[tune_ix], pd_s[tune_ix])
    auto_reject = pd_s >= t_star
    exp_loss = np.where(auto_reject, (1.0 - pd_s) * C_FR, pd_s * C_FA)
    loss = realized_auto_loss(y_def, auto_reject)

    r_know = rank_map(know[tune_ix])
    r_loss = rank_map(exp_loss[tune_ix])

    # ---- lambda selection on the SELECTION slice at the operating rate -----
    lam_table = []
    for lam in LAMBDA_GRID:
        s_tu = lam * r_know(know[tune_ix]) + (1 - lam) * r_loss(exp_loss[tune_ix])
        s_sel = lam * r_know(know[sel_ix]) + (1 - lam) * r_loss(exp_loss[sel_ix])
        thr = float(np.quantile(s_tu, 1.0 - OPERATING_RATE))
        defer = s_sel > thr
        cost = float(np.where(defer, 0.0, loss[sel_ix]).mean())
        lam_table.append({"lambda": lam, "selection_auto_cost": cost,
                          "selection_defer_rate": float(defer.mean())})
    lam_star = min(lam_table, key=lambda e: e["selection_auto_cost"])["lambda"]
    print(f"[cost-aware] lambda* = {lam_star} (chosen on selection slice)")

    # ---- FINAL sweep, one pass per strategy ---------------------------------
    rng_rand = np.random.default_rng(RANDOM_ROUTER_SEED)
    random_score_fin = rng_rand.permutation(len(fin_ix)) / len(fin_ix)
    random_score_tun = np.linspace(0, 1, len(tune_ix))
    comb = lambda ix: lam_star * r_know(know[ix]) + (1 - lam_star) * r_loss(exp_loss[ix])
    strategies = {
        "random": (random_score_fin, random_score_tun),
        "chow_margin": (-np.abs(pd_s[fin_ix] - t_star), -np.abs(pd_s[tune_ix] - t_star)),
        "knowledge": (know[fin_ix], know[tune_ix]),
        "expected_cost": (exp_loss[fin_ix], exp_loss[tune_ix]),
        f"combined_lambda_{lam_star}": (comb(fin_ix), comb(tune_ix)),
    }
    rng_boot = np.random.default_rng(SPLIT_SEED + 1)
    curves = {}
    at_operating = {}
    for name, (s_fin, s_tun) in strategies.items():
        curves[name] = sweep(s_fin, s_tun, loss[fin_ix], rng_boot)
        op = min(curves[name], key=lambda p: abs(p["capacity"] - OPERATING_RATE))
        at_operating[name] = op
        print(f"[cost-aware] {name:<26} cost@22.5% {op['auto_cost_per_applicant']:.4f} "
              f"CI [{op['ci95'][0]:.4f}, {op['ci95'][1]:.4f}]")

    no_defer_cost = float(loss[fin_ix].mean())
    report = {
        "protocol": {
            "frame": "research (SGLB OOF cache, pd at Youden t* from tune); y==1 DEFAULT",
            "split": f"50/25/25 tune/selection/final, permutation seed {SPLIT_SEED}; final touched once per strategy",
            "costs": {"false_approval": C_FA, "false_rejection": C_FR,
                      "provenance": "decision-layer pair (reports/t_base_selection.json)"},
            "lambda_grid": LAMBDA_GRID,
            "lambda_selected_on": "selection slice, lowest realized auto cost at 22.5%",
            "bootstrap": BOOT,
            "review_cost_note": (
                "auto_cost excludes review cost; add c_review * deferral_rate for total cost. "
                "Grid given so the frontier can be read at any assumed per-review cost."
            ),
            "c_review_grid": C_REVIEW_GRID,
        },
        "lambda_selection_table": lam_table,
        "lambda_star": lam_star,
        "no_deferral_cost_final": no_defer_cost,
        "curves": curves,
        "at_operating_rate": at_operating,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[cost-aware] no-deferral cost: {no_defer_cost:.4f}")
    print(f"report -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
