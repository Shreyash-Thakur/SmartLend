from __future__ import annotations

import pandas as pd
import pytest

from research.data.bureau_aggregates import (
    attach_bureau_aggregates,
    compute_bureau_aggregates,
)


@pytest.fixture
def bureau() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2, 2, 3],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed", "Closed", "Active"],
            "CREDIT_DAY_OVERDUE": [0, 5, 0, 0, 12, 0],
        }
    )


def test_active_loan_count_per_applicant(bureau):
    agg = compute_bureau_aggregates(bureau)
    assert agg.loc[1, "BUREAU_ACTIVE_LOAN_COUNT"] == 2
    assert agg.loc[2, "BUREAU_ACTIVE_LOAN_COUNT"] == 0
    assert agg.loc[3, "BUREAU_ACTIVE_LOAN_COUNT"] == 1


def test_delinquency_count_counts_overdue_rows_not_days(bureau):
    agg = compute_bureau_aggregates(bureau)
    # applicant 1 has one row with CREDIT_DAY_OVERDUE=5 -> one delinquent credit line
    assert agg.loc[1, "BUREAU_DELINQUENCY_COUNT"] == 1
    assert agg.loc[2, "BUREAU_DELINQUENCY_COUNT"] == 1


def test_attach_merges_onto_application_frame(bureau):
    application_df = pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4]})
    merged = attach_bureau_aggregates(application_df, bureau)
    assert list(merged["BUREAU_ACTIVE_LOAN_COUNT"]) == [2, 0, 1, 0]
    assert list(merged["BUREAU_DELINQUENCY_COUNT"]) == [1, 1, 0, 0]


def test_attach_does_not_mutate_input(bureau):
    application_df = pd.DataFrame({"SK_ID_CURR": [1, 2]})
    original_columns = list(application_df.columns)
    attach_bureau_aggregates(application_df, bureau)
    assert list(application_df.columns) == original_columns
