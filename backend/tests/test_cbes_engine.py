from __future__ import annotations

import math

import pytest

from backend.app.services import cbes_engine


FAKE_THRESHOLDS = {
    # percentile breakpoints: 5 edges -> 4 bands, mapped to [0, 0.25, 0.5, 0.75, 1.0]
    "credit_score": [0.10, 0.30, 0.50, 0.70, 0.90],
    "delinquencies": [0, 0, 1, 2, 5],
    "active_loans": [0, 1, 2, 4, 8],
    "dti": [0.05, 0.15, 0.25, 0.40, 0.80],
    "employment_tenure_years": [0.0, 1.0, 3.0, 7.0, 15.0],
    "loan_to_income": [0.5, 1.5, 3.0, 5.0, 10.0],
}


@pytest.fixture(autouse=True)
def stub_thresholds(monkeypatch):
    monkeypatch.setattr(cbes_engine, "_THRESHOLDS", FAKE_THRESHOLDS)


def test_returns_probability_and_five_component_breakdown():
    p_cbes, breakdown = cbes_engine.compute_cbes(
        {
            "credit_score": 0.75,
            "delinquencies": 0,
            "active_loans": 1,
            "dti": 0.10,
            "employment_tenure_years": 5.0,
            "annual_income": 500_000.0,
            "loan_amount": 800_000.0,
            "region": 1,
        }
    )
    assert 0.0 <= p_cbes <= 1.0
    assert set(breakdown) == {"credit", "capacity", "behaviour", "stability", "region"}
    assert all(0.0 <= v <= 1.0 for v in breakdown.values())


def test_strong_profile_scores_higher_than_weak_profile():
    strong = cbes_engine.compute_cbes(
        {
            "credit_score": 0.85,
            "delinquencies": 0,
            "active_loans": 1,
            "dti": 0.08,
            "employment_tenure_years": 10.0,
            "annual_income": 900_000.0,
            "loan_amount": 500_000.0,
            "region": 1,
        }
    )[0]
    weak = cbes_engine.compute_cbes(
        {
            "credit_score": 0.15,
            "delinquencies": 4,
            "active_loans": 6,
            "dti": 0.60,
            "employment_tenure_years": 0.2,
            "annual_income": 150_000.0,
            "loan_amount": 900_000.0,
            "region": 3,
        }
    )[0]
    assert strong > weak


def test_missing_fields_use_conservative_defaults_not_crash():
    p_cbes, breakdown = cbes_engine.compute_cbes({})
    assert 0.0 <= p_cbes <= 1.0
    assert not any(math.isnan(v) for v in breakdown.values())


def test_nan_employment_tenure_does_not_crash():
    # DAYS_EMPLOYED sentinel -> NaN, per research/data/specs/home_credit.py
    p_cbes, _ = cbes_engine.compute_cbes(
        {
            "credit_score": 0.5,
            "delinquencies": 0,
            "active_loans": 0,
            "dti": 0.2,
            "employment_tenure_years": float("nan"),
            "annual_income": 300_000.0,
            "loan_amount": 300_000.0,
        }
    )
    assert 0.0 <= p_cbes <= 1.0


def test_zero_income_does_not_produce_infinite_ratio():
    p_cbes, _ = cbes_engine.compute_cbes(
        {
            "credit_score": 0.5,
            "delinquencies": 0,
            "active_loans": 0,
            "dti": 0.2,
            "employment_tenure_years": 2.0,
            "annual_income": 0.0,
            "loan_amount": 300_000.0,
        }
    )
    assert 0.0 <= p_cbes <= 1.0
