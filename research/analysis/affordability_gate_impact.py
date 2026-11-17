"""Decision mix before/after the FOIR affordability gate (policy_rules.py).

Scores every seeded customer (customer_profiles table, 510 rows) at a grid of
loan requests through the live serving path, once with the gate's verdict
ignored (the model + router alone) and once with it applied. This is the
measurement behind the thresholds in backend/app/services/policy_rules.py.

Caveat stated up front: seeded customers are Home Credit rows the serving
model was trained on, so this measures decision MIX, not accuracy.

Run: ``python -m research.analysis.affordability_gate_impact``
Writes ``reports/affordability_gate_impact.json``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path

from backend.app.database import DB_PATH, init_db
from backend.app.routers.applications import _validate_payload
from backend.app.services.decision_engine import DEFAULT_TAU_U
from backend.app.services.ml_service import get_predictor
from backend.app.services.policy_rules import FOIR_REJECT, FOIR_REVIEW

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "reports" / "affordability_gate_impact.json"
GRID = [(100_000, 36), (200_000, 36), (500_000, 60), (1_000_000, 60), (2_500_000, 24)]


def _shares(counter: Counter, n: int) -> dict[str, float]:
    return {d: round(counter[d] / n, 4) for d in ("APPROVE", "DEFER", "REJECT")}


def main() -> int:
    logging.disable(logging.WARNING)
    init_db()
    ids = [row[0] for row in sqlite3.connect(DB_PATH).execute("SELECT customer_id FROM customer_profiles")]
    predictor = get_predictor()

    rows = []
    for loan, months in GRID:
        before, after = Counter(), Counter()
        for cid in ids:
            payload = _validate_payload({"customer_id": cid, "loan_amount": loan,
                                         "loan_tenure_months": months, "end_use_declaration": True})
            result = predictor.predict_application(payload)
            after[result.decision] += 1
            # Model + router alone: undo the gate where it acted.
            before[result.decision if not result.decision_reason.startswith("affordability_")
                   else ("APPROVE" if result.decision == "DEFER" else _model_only(result))] += 1
        rows.append({"loan_amount": loan, "tenure_months": months,
                     "before": _shares(before, len(ids)), "after": _shares(after, len(ids))})
        print(f"{loan:>9} x {months}m  before {rows[-1]['before']}  after {rows[-1]['after']}")

    OUT_JSON.write_text(json.dumps({
        "population": f"{len(ids)} seeded customer profiles (in the serving model's training data)",
        "gate": {"reject_if_foir_above": FOIR_REJECT, "approve_to_defer_if_foir_above": FOIR_REVIEW},
        "assumptions": "interest 12% p.a.; no other-bank EMIs; no additional income",
        "grid": rows,
    }, indent=2), encoding="utf-8")
    print(f"report -> {OUT_JSON}")
    return 0


def _model_only(result) -> str:
    """Decision the router would have taken for a gate REJECT (thresholds unchanged)."""
    if abs(result.p_ml - result.t_approve) < DEFAULT_TAU_U:
        return "DEFER"
    return "APPROVE" if result.p_ml >= result.t_approve else "REJECT"


if __name__ == "__main__":
    raise SystemExit(main())
