"""Decision-rate decomposition of the CBES ablation (reviewer request).

The ablation table reported AUC, deferral rate and default capture for
ML-only vs ML+CBES. A higher capture could come from rejecting more
applicants rather than allocating review better, so this script derives the
full decision mix from the confusion matrices already recorded in
``reports/blend_decision.json`` (committed; test half of population C,
153,756 rows, thresholds fitted on the tune half, seed 20260831):

- approval / rejection / deferral rates,
- false-approval and false-rejection rates (y_true==1 is GOOD in that
  artifact, so approve_bad = false approval, reject_good = false rejection),
- default recall among deferred applicants,
- expected decision cost of the auto-decided pile under BOTH project cost
  pairs (decision-layer 0.45/0.2644 and training-study 0.75/0.10), per
  auto-decided applicant and per applicant overall.

Pure derivation -- no model is trained and no threshold is fitted here.

Run: ``python -m research.analysis.ablation_decision_rates``
Writes ``reports/ablation_decision_rates.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "reports" / "blend_decision.json"
OUT = REPO / "reports" / "ablation_decision_rates.json"

COST_PAIRS = {
    "decision_layer_0.45_0.2644": {"false_approval": 0.45, "false_rejection": 0.2644},
    "training_study_0.75_0.10": {"false_approval": 0.75, "false_rejection": 0.10},
}


def decompose(cm: dict, n: int) -> dict:
    approve = cm["approve_good"] + cm["approve_bad"]
    reject = cm["reject_good"] + cm["reject_bad"]
    defer = cm["defer_good"] + cm["defer_bad"]
    assert approve + reject + defer == n
    defaulters = cm["approve_bad"] + cm["reject_bad"] + cm["defer_bad"]
    n_auto = approve + reject
    out = {
        "n": n,
        "n_defaulters": defaulters,
        "approval_rate": round(approve / n, 4),
        "rejection_rate": round(reject / n, 4),
        "deferral_rate": round(defer / n, 4),
        "false_approvals": cm["approve_bad"],
        "false_rejections": cm["reject_good"],
        "false_approval_rate": round(cm["approve_bad"] / n, 4),
        "false_rejection_rate": round(cm["reject_good"] / n, 4),
        "default_recall_among_deferred": round(cm["defer_bad"] / defaulters, 4),
        "default_capture_rejected_or_deferred": round(
            (cm["reject_bad"] + cm["defer_bad"]) / defaulters, 4),
        "expected_cost": {},
    }
    for name, pair in COST_PAIRS.items():
        total = (pair["false_approval"] * cm["approve_bad"]
                 + pair["false_rejection"] * cm["reject_good"])
        out["expected_cost"][name] = {
            "per_auto_decided_applicant": round(total / n_auto, 4),
            "per_applicant_overall": round(total / n, 4),
        }
    return out


def main() -> int:
    src = json.loads(SRC.read_text())
    comp = src["decision_policy_comparison"]
    report = {
        "derived_from": "reports/blend_decision.json decision_policy_comparison confusion matrices",
        "population": "test half of population C (153,756 rows), thresholds from tune half, split seed 20260831",
        "label_semantics": "y_true==1 is GOOD; approve_bad = false approval, reject_good = false rejection",
        "cost_convention": "deferred cases excluded from decision cost (they carry review cost, identical at the matched ~22.5% rate)",
        "configurations": {
            "ml_only_alpha_0": decompose(
                comp["old_p_ml_only"]["confusion_matrix"], comp["old_p_ml_only"]["n"]),
            "ml_plus_cbes_blend": decompose(
                comp["new_blend"]["confusion_matrix"], comp["new_blend"]["n"]),
        },
    }
    a = report["configurations"]["ml_only_alpha_0"]
    b = report["configurations"]["ml_plus_cbes_blend"]
    report["reading"] = {
        "capture_gain_mechanism": (
            f"blend cuts false approvals {a['false_approvals']} -> {b['false_approvals']} "
            f"but raises false rejections {a['false_rejections']} -> {b['false_rejections']} "
            f"(rejection rate {a['rejection_rate']:.1%} -> {b['rejection_rate']:.1%}); "
            "part of the capture gain is therefore rejecting more, not reviewing better"),
        "cost_direction_depends_on_ratio": {
            name: {
                "ml_only": a["expected_cost"][name]["per_auto_decided_applicant"],
                "blend": b["expected_cost"][name]["per_auto_decided_applicant"],
                "blend_cheaper": b["expected_cost"][name]["per_auto_decided_applicant"]
                < a["expected_cost"][name]["per_auto_decided_applicant"],
            }
            for name in COST_PAIRS
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["configurations"], indent=1))
    print(json.dumps(report["reading"], indent=1))
    print(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
