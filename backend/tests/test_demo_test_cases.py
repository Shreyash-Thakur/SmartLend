"""Every case on the Demo Test Cases page must still produce its stated decision.

The page (frontend/src/pages/DemoTestCases.tsx) and docs/DEMO-TEST-CASES.md
both read frontend/src/data/demoTestCases.json. If the engine, the policy gate
or the seeded customers change, this test fails instead of the demo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.database import init_db
from backend.app.routers.applications import _validate_payload
from backend.app.services.decision_engine import ENGINE_VERSION
from backend.app.services.ml_service import get_predictor

CASES_PATH = Path(__file__).resolve().parents[2] / "frontend" / "src" / "data" / "demoTestCases.json"
DATA = json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def _seeded_db() -> None:
    init_db()


def test_cases_were_written_for_this_engine() -> None:
    assert DATA["engineVersion"] == ENGINE_VERSION


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda c: f"case{c['id']}-{c['expected']}")
def test_demo_case_decision(case: dict) -> None:
    payload = _validate_payload({
        "customer_id": case["customerId"],
        "loan_amount": case["loanAmount"],
        "loan_tenure_months": case["tenureMonths"],
        "loan_purpose": case["purpose"],
        "end_use_declaration": True,
    })
    result = get_predictor().predict_application(payload)
    assert result.decision == case["expected"]
    by_policy = result.decision_reason.startswith("affordability_")
    assert by_policy == (case["decidedBy"] == "policy")
