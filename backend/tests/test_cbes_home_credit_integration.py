"""Proves the new CBES engine runs end-to-end on real Home Credit rows,
through to a hybrid decision, with the ML side stubbed (no model this pass)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.app.services.cbes_engine import compute_cbes
from backend.app.services.decision_engine import hybrid_decision
from research.data.adapters import build_bundle
from research.data.bureau_aggregates import attach_bureau_aggregates
from research.data.specs import home_credit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPLICATION_CSV = PROJECT_ROOT / home_credit.SOURCE_FILE
BUREAU_CSV = PROJECT_ROOT / "data/raw/home_credit/bureau.csv"


@pytest.fixture(scope="module")
def sample_core() -> pd.DataFrame:
    if not APPLICATION_CSV.exists() or not BUREAU_CSV.exists():
        pytest.skip("Home Credit raw data not present in data/raw/home_credit/")
    application_df = pd.read_csv(APPLICATION_CSV, nrows=200, low_memory=False)
    bureau_df = pd.read_csv(BUREAU_CSV, low_memory=False)
    merged = attach_bureau_aggregates(application_df, bureau_df)
    bundle = build_bundle(home_credit.SPEC, merged)
    core = bundle.core.copy()
    core["loan_to_income"] = core["loan_amount"] / core["annual_income"].replace(0, float("nan"))
    return core


def test_every_sample_row_produces_a_decision_without_key_errors(sample_core):
    for _, row in sample_core.iterrows():
        p_cbes, breakdown = compute_cbes(row.to_dict())
        assert 0.0 <= p_cbes <= 1.0
        result = hybrid_decision(p_ml=0.5, p_cbes=p_cbes, tau_d=0.43, cbes_breakdown=breakdown)
        assert result.decision in {"APPROVE", "REJECT", "DEFER"}
