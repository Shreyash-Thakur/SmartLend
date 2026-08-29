"""Aggregate bureau.csv into per-applicant counts for the canonical core.

bureau.csv is many-rows-per-applicant (one row per prior credit line at
another institution). The canonical schema needs one row per SK_ID_CURR, so
this module reduces it before `research/data/specs/home_credit.py` can treat
`delinquencies`/`active_loans` as NATIVE fields instead of ABSENT.

An applicant with no rows in bureau.csv gets 0 for both counts. This is a
judgment call, not a neutral default: it reads "no bureau record" as "no
active loans / no delinquencies found", which is the standard convention for
this field in the Home Credit competition, not a claim that we know their
true bureau history. Documented here so it is not mistaken for imputation of
an unknown value.
"""

from __future__ import annotations

import pandas as pd

ACTIVE_LOAN_COL = "BUREAU_ACTIVE_LOAN_COUNT"
DELINQUENCY_COL = "BUREAU_DELINQUENCY_COUNT"


def compute_bureau_aggregates(bureau: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR: active loan count and delinquency count."""
    grouped = bureau.groupby("SK_ID_CURR")
    active = grouped["CREDIT_ACTIVE"].apply(lambda s: int((s == "Active").sum()))
    delinquent = grouped["CREDIT_DAY_OVERDUE"].apply(lambda s: int((s > 0).sum()))
    return pd.DataFrame({ACTIVE_LOAN_COL: active, DELINQUENCY_COL: delinquent})


def attach_bureau_aggregates(
    application_df: pd.DataFrame, bureau_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-merge bureau aggregates onto an application frame by SK_ID_CURR."""
    aggregates = compute_bureau_aggregates(bureau_df)
    merged = application_df.merge(
        aggregates, how="left", left_on="SK_ID_CURR", right_index=True
    )
    merged[ACTIVE_LOAN_COL] = merged[ACTIVE_LOAN_COL].fillna(0).astype(int)
    merged[DELINQUENCY_COL] = merged[DELINQUENCY_COL].fillna(0).astype(int)
    return merged
