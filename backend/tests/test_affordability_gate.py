"""FOIR affordability gate (backend/app/services/policy_rules.py).

Pins the bug it fixes: customer 105969 (Rs 2.7L/yr income) asking for Rs 25L
over 24 months used to be DEFERRED with an 89% model approval score, because
the model rewards larger loans and nothing checked affordability.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.database import init_db
from backend.app.main import app
from backend.app.routers.applications import _validate_payload
from backend.app.services.ml_service import get_predictor
from backend.app.services.policy_rules import (
    REASON_REJECT,
    REASON_REVIEW,
    apply_affordability_gate,
    check_affordability,
    monthly_emi,
)


@pytest.fixture(scope="module", autouse=True)
def _seeded_db() -> None:
    init_db()


def _payload(loan: float, months: int = 36, income: float = 300_000, dti: float = 0.0) -> dict:
    return {"loan_amount": loan, "loan_tenure_months": months, "annual_income": income,
            "debt_to_income_ratio": dti, "interestRate": 12.0}


def test_emi_matches_standard_formula() -> None:
    assert monthly_emi(100_000, 12, 12.0) == pytest.approx(8884.88, abs=0.01)


def test_missing_inputs_give_no_verdict() -> None:
    assert check_affordability({"loan_amount": 100_000}) is None
    assert check_affordability(_payload(100_000, income=0)) is None


@pytest.mark.parametrize(
    ("loan", "expected_action"),
    [(100_000, None),          # FOIR ~0.40: affordable
     (1_000_000, "DEFER"),     # FOIR ~1.33: review band
     (2_500_000, "REJECT")],   # FOIR ~3.3: impossible
)
def test_bands(loan: float, expected_action: str | None) -> None:
    check = check_affordability(_payload(loan))
    assert check is not None
    assert check.action == expected_action


def test_gate_only_makes_decisions_stricter() -> None:
    review = check_affordability(_payload(1_000_000))
    assert apply_affordability_gate("APPROVE", "ml_approve", review) == ("DEFER", REASON_REVIEW)
    assert apply_affordability_gate("DEFER", "ml_uncertainty", review) == ("DEFER", "ml_uncertainty")
    assert apply_affordability_gate("REJECT", "ml_reject", review) == ("REJECT", "ml_reject")
    reject = check_affordability(_payload(2_500_000))
    for decision in ("APPROVE", "DEFER", "REJECT"):
        assert apply_affordability_gate(decision, "x", reject) == ("REJECT", REASON_REJECT)


def test_unaffordable_loan_for_weak_customer_is_rejected() -> None:
    payload = _validate_payload({"customer_id": "105969", "loan_amount": 2_500_000,
                                 "loan_tenure_months": 24, "end_use_declaration": True})
    result = get_predictor().predict_application(payload)
    assert result.decision == "REJECT"
    assert result.decision_reason == REASON_REJECT
    assert result.affordability["foir"] > 2.0


def test_main_form_requires_customer_id() -> None:
    response = TestClient(app).post("/api/applications", json={"loan_amount": 25_000})
    assert response.status_code == 422
    assert "Customer ID required" in response.text
