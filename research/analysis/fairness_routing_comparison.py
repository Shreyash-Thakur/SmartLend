"""Subgroup routing comparison: does each deferral policy actually serve
thin-file applicants, or merely defer them more? (reviewer item)

For each routing policy at the matched 22.5% rate, measured separately for
thin-file (no credit-bureau record) and documented applicants on the
held-out half (thresholds from the tune half, split seed 20260831):

- deferral rate,
- selective risk (0/1 decision loss of the forced engine decision on the
  group's KEPT pile),
- default recall in the review set (share of the group's eventual
  defaulters that the policy routes to humans),
- false-approval and false-rejection rates on the group's auto pile.

Serving-frame policies (backend/artifacts/prediction_outputs.csv,
y_true==1 GOOD, approval probabilities): the deployed band policy's
recorded decisions, the margin-uncertainty rule, the historical
disagreement rule, and a seeded random router. Research-frame policy
(data/processed/sglb_uncertainty_oof.pkl, y==1 DEFAULT, pd scores):
SGLB knowledge uncertainty, reported separately because its frame and
threshold differ.

Run: ``python -m research.analysis.fairness_routing_comparison``
Writes ``reports/fairness_routing.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.analysis.uncertainty_deferral_catboost import OOF_CACHE, youden_threshold
from research.deferral.evaluate import TARGET_RATE, defer_mask_at_rate, split_tune_test
from research.deferral.signals import all_candidates
from research.features.engineer import build_feature_matrix
from research.relearning.gate import _predicted_good

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "home_credit"
PREDICTIONS = REPO / "backend" / "artifacts" / "prediction_outputs.csv"
OUT_JSON = REPO / "reports" / "fairness_routing.json"

SPLIT_SEED = 20260831
RANDOM_ROUTER_SEED = 20260830


def group_routing_metrics(y_good: np.ndarray, predicted_good: np.ndarray,
                          defer: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    """Routing metrics for one subgroup. y_good: 1 = good customer."""
    g = group
    bad = (y_good == 0) & g
    keep = ~defer & g
    err_keep = predicted_good[keep] != y_good[keep]
    return {
        "n": int(g.sum()),
        "defer_rate": float(defer[g].mean()),
        "selective_risk": float(err_keep.mean()) if keep.any() else None,
        "default_recall_in_review": float((defer & bad).sum() / max(1, bad.sum())),
        "false_approval_rate_auto": float((predicted_good[keep] & (y_good[keep] == 0)).mean()) if keep.any() else None,
        "false_rejection_rate_auto": float((~predicted_good[keep].astype(bool) & (y_good[keep] == 1)).mean()) if keep.any() else None,
    }


def main() -> int:
    frame = pd.read_csv(PREDICTIONS)
    frame["SK_ID_CURR"] = frame["applicant_id"].astype(str).str.removeprefix("HC").astype(int)
    bureau_ids = set(pd.read_csv(RAW / "bureau.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"].unique())

    tune, test = split_tune_test(frame, seed=SPLIT_SEED)
    thin_test = (~test["SK_ID_CURR"].isin(bureau_ids)).to_numpy()
    y_test = test["y_true"].to_numpy(dtype=int)
    pred_good = _predicted_good(test).astype(bool)

    cols = lambda f: (
        f["best_model_prob"].to_numpy(dtype=float),
        f["cbes_prob"].to_numpy(dtype=float),
        f["approval_threshold"].to_numpy(dtype=float),
    )

    policies: dict[str, np.ndarray] = {
        "deployed_band_policy": (test["final_decision"].astype(str).str.upper() == "DEFER").to_numpy(),
    }
    rng = np.random.default_rng(RANDOM_ROUTER_SEED)
    n_defer = int(round(TARGET_RATE * len(test)))
    rand = np.zeros(len(test), dtype=bool)
    rand[rng.choice(len(test), size=n_defer, replace=False)] = True
    policies["random_router"] = rand
    for sig in all_candidates():
        if sig.name not in ("ml_uncertainty", "current_abs_diff"):
            continue
        ml_tu, cb_tu, th_tu = cols(tune)
        sig.fit(ml_tu, cb_tu, tune["y_true"].to_numpy(dtype=int), th_tu)
        defer, _ = defer_mask_at_rate(sig.score(ml_tu, cb_tu, th_tu), sig.score(*cols(test)), TARGET_RATE)
        policies[sig.name] = defer

    serving = {}
    for name, defer in policies.items():
        serving[name] = {
            "overall_defer_rate": float(defer.mean()),
            "thin_file": group_routing_metrics(y_test, pred_good, defer, thin_test),
            "documented": group_routing_metrics(y_test, pred_good, defer, ~thin_test),
        }
        t, d = serving[name]["thin_file"], serving[name]["documented"]
        print(f"[serving] {name:<22} defer thin {t['defer_rate']:.3f} vs doc {d['defer_rate']:.3f}  "
              f"recall thin {t['default_recall_in_review']:.3f} vs doc {d['default_recall_in_review']:.3f}")

    # Research frame: SGLB knowledge uncertainty (labels are DEFAULT=1).
    out, _ = pd.read_pickle(OOF_CACHE)
    X, y_def, ids = build_feature_matrix()
    rng2 = np.random.default_rng(SPLIT_SEED)
    perm = rng2.permutation(len(y_def))
    half = len(y_def) // 2
    tune_ix, test_ix = perm[:half], perm[half:]
    t_star = youden_threshold(y_def[tune_ix], out["pd"][tune_ix])
    pred_default = (out["pd"] >= t_star).astype(bool)
    thr = float(np.quantile(out["knowledge"][tune_ix], 1.0 - TARGET_RATE))
    defer_k = out["knowledge"][test_ix] > thr
    thin_r = ~pd.Series(ids[test_ix]).isin(bureau_ids).to_numpy()
    y_good_r = (1 - y_def[test_ix]).astype(int)
    pred_good_r = ~pred_default[test_ix]
    research = {
        "sglb_knowledge_uncertainty": {
            "frame_note": "research frame: SGLB pd at Youden t*, not the serving engine",
            "overall_defer_rate": float(defer_k.mean()),
            "thin_file": group_routing_metrics(y_good_r, pred_good_r, defer_k, thin_r),
            "documented": group_routing_metrics(y_good_r, pred_good_r, defer_k, ~thin_r),
        }
    }
    t = research["sglb_knowledge_uncertainty"]["thin_file"]
    d = research["sglb_knowledge_uncertainty"]["documented"]
    print(f"[research] sglb_knowledge       defer thin {t['defer_rate']:.3f} vs doc {d['defer_rate']:.3f}  "
          f"recall thin {t['default_recall_in_review']:.3f} vs doc {d['default_recall_in_review']:.3f}")

    report = {
        "protocol": {
            "serving_frame": "prediction_outputs.csv held-out half; thresholds from tune (seed 20260831)",
            "research_frame": "SGLB OOF cache, same split seed; Youden threshold from tune",
            "target_rate": TARGET_RATE,
            "thin_file_definition": "no record in bureau.csv",
            "metrics_note": "selective risk = 0/1 loss of the forced engine decision on the group's kept pile",
        },
        "serving_frame_policies": serving,
        "research_frame_policies": research,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
