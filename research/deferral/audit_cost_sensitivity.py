"""Cost-model sensitivity for the deferral audit (reviewer item).

The project carries two literature-sourced cost models that must not be
conflated: the decision-layer costs used by the t_base study and the audit
(false approval 0.45 = Basel II F-IRB LGD, false rejection 0.2644 =
Verbraken et al. EMP ROI; ratio ~1.70:1), and the cost-sensitive TRAINING
study's stated pair (0.75 LGD / 0.10 margin; ratio 7.5:1,
reports/cost_sensitive_catboost.json). This script derives the audit's
expected auto-pile cost per applicant under BOTH pairs and a ratio sweep
(2, 5, 7.5, 10, 20 with the false-rejection cost fixed at 0.10), directly
from the false-approval / false-rejection counts recorded in
reports/deferral_audit_10k.json, and reports whether the ROUTER RANKING by
cost is stable across cost assumptions.

Pure derivation from a committed report; no model is run.

Run: ``python -m research.deferral.audit_cost_sensitivity``
Writes ``reports/deferral_cost_sensitivity.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "reports" / "deferral_audit_10k.json"
REPORT_PATH = REPO_ROOT / "reports" / "deferral_cost_sensitivity.json"

COST_PAIRS = {
    "decision_layer_basel_verbraken": (0.45, 0.2644),
    "training_study_lgd_margin": (0.75, 0.10),
    "ratio_2": (0.20, 0.10),
    "ratio_5": (0.50, 0.10),
    "ratio_7.5": (0.75, 0.10),
    "ratio_10": (1.00, 0.10),
    "ratio_20": (2.00, 0.10),
}


def main() -> int:
    audit = json.loads(AUDIT.read_text())
    n_test = audit["protocol"]["n_test"]

    rows = {}
    for r in audit["routers"]:
        fa, fr = r["cost"]["false_approvals_auto"], r["cost"]["false_rejections_auto"]
        rows[r["router"]] = {"false_approvals_auto": fa, "false_rejections_auto": fr}

    per_pair = {}
    rankings = {}
    for name, (c_fa, c_fr) in COST_PAIRS.items():
        costs = {
            router: (c_fa * v["false_approvals_auto"] + c_fr * v["false_rejections_auto"]) / n_test
            for router, v in rows.items()
        }
        per_pair[name] = {
            "cost_false_approval": c_fa,
            "cost_false_rejection": c_fr,
            "cost_per_applicant_by_router": {k: round(v, 6) for k, v in costs.items()},
        }
        rankings[name] = sorted(costs, key=costs.get)

    reference_ranking = rankings["decision_layer_basel_verbraken"]
    stable = all(rank == reference_ranking for rank in rankings.values())

    report = {
        "source": "reports/deferral_audit_10k.json (false-approval / false-rejection counts per router)",
        "n_test": n_test,
        "note": (
            "Two cost models coexist in the project by design: the decision-layer "
            "pair (0.45 Basel F-IRB LGD / 0.2644 Verbraken ROI, ratio ~1.70) used "
            "by the t_base study and the audit, and the cost-sensitive training "
            "study's pair (0.75 / 0.10, ratio 7.5). This table shows the audit's "
            "router ranking under both and under a ratio sweep."
        ),
        "cost_pairs": per_pair,
        "ranking_by_pair": rankings,
        "ranking_stable_across_all_pairs": stable,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ranking stable across all cost pairs:", stable)
    print("reference ranking:", " > ".join(reference_ranking))
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
