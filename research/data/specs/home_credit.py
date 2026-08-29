"""Canonical mapping for Home Credit Default Risk (Kaggle).

Primary dataset. 307,511 applications, 122 columns in `application_train.csv`,
plus six relational tables.

Two facts that shape the whole project:

1. `application_train.csv` contains **only accepted applications**, so it cannot
   on its own support selection modelling. The reject data lives in
   `previous_application.csv`, where `NAME_CONTRACT_STATUS` takes values
   Approved / Refused / Canceled / Unused offer. That table is therefore the
   entry point for the propensity model, not `application_train`.

2. One canonical field (credit_utilization) is NOT in application_train and
   requires aggregating credit_card_balance.csv, which has not been fetched.
   It is marked ABSENT deliberately so `missing_required()` reports the gap
   rather than hiding it. delinquencies/active_loans ARE available, via
   bureau.csv aggregation (research/data/bureau_aggregates.py) — the caller
   must merge those columns onto application_train before calling
   build_bundle(); the spec only declares where the values live once merged.

Column names are from the competition data dictionary; verify against the actual
download before trusting the mapping (see `adapters.validate_spec`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.data.canonical import Availability as A
from research.data.canonical import DatasetSpec, FieldSpec, Unit

SOURCE_FILE = "data/raw/home_credit/application_train.csv"

# The notorious sentinel: DAYS_EMPLOYED == 365243 (~55,374 rows) encodes
# "pensioner / not employed", not 1000 years of service. It is also informative
# — those rows default at ~5.4% vs ~8.7% for the rest — so it must become
# NaN + indicator, never an imputed number and never a literal value.
DAYS_EMPLOYED_SENTINEL = 365243


def _age_years(df: pd.DataFrame) -> pd.Series:
    # DAYS_BIRTH is negative days relative to application date.
    return -df["DAYS_BIRTH"] / 365.25


def _employment_tenure_years(df: pd.DataFrame) -> pd.Series:
    days = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    return -days / 365.25


def _dti(df: pd.DataFrame) -> pd.Series:
    """Annuity-to-income ratio.

    Home Credit has no native DTI. AMT_ANNUITY / AMT_INCOME_TOTAL is the
    standard stand-in; it is a *derived approximation* and must be described as
    one, since it omits obligations to other lenders.
    """
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    return df["AMT_ANNUITY"] / income


def _gender(df: pd.DataFrame) -> pd.Series:
    # CODE_GENDER contains a handful of 'XNA' rows; surface them as missing
    # rather than silently folding them into M or F.
    return df["CODE_GENDER"].replace("XNA", pd.NA).astype("string")


SPEC = DatasetSpec(
    name="home_credit",
    notes=(
        "Primary dataset. application_train.csv holds accepted applications "
        "only; selection/propensity modelling must use previous_application.csv "
        "(NAME_CONTRACT_STATUS). credit_utilization/delinquencies/active_loans "
        "require bureau + credit_card_balance aggregation."
    ),
    sentinels={"DAYS_EMPLOYED": (DAYS_EMPLOYED_SENTINEL,)},
    fields=(
        FieldSpec("age_years", A.DERIVED, _age_years, Unit.YEARS, "from -DAYS_BIRTH"),
        FieldSpec("annual_income", A.NATIVE, "AMT_INCOME_TOTAL", Unit.CURRENCY),
        FieldSpec("loan_amount", A.NATIVE, "AMT_CREDIT", Unit.CURRENCY),
        FieldSpec("installment", A.NATIVE, "AMT_ANNUITY", Unit.CURRENCY),
        FieldSpec(
            "credit_score",
            A.PROXY,
            "EXT_SOURCE_2",
            Unit.SCORE,
            "Normalised external score in [0,1], NOT a bureau score. Best-covered "
            "of EXT_SOURCE_1/2/3. Direction and scale differ from CIBIL/FICO — "
            "never compare its raw value across datasets.",
        ),
        FieldSpec("dti", A.DERIVED, _dti, Unit.RATIO, "AMT_ANNUITY / AMT_INCOME_TOTAL"),
        FieldSpec(
            "employment_tenure_years",
            A.DERIVED,
            _employment_tenure_years,
            Unit.YEARS,
            f"from -DAYS_EMPLOYED; sentinel {DAYS_EMPLOYED_SENTINEL} -> NaN",
        ),
        FieldSpec(
            "credit_utilization",
            A.ABSENT,
            None,
            Unit.RATIO,
            "Requires credit_card_balance.csv aggregation (AMT_BALANCE / "
            "AMT_CREDIT_LIMIT_ACTUAL). Phase 0 task.",
        ),
        FieldSpec(
            "delinquencies",
            A.NATIVE,
            "BUREAU_DELINQUENCY_COUNT",
            Unit.COUNT,
            "Count of bureau.csv credit lines with CREDIT_DAY_OVERDUE > 0 for "
            "this applicant; 0 if the applicant has no bureau.csv rows. See "
            "research/data/bureau_aggregates.py. Requires the caller to merge "
            "bureau aggregates onto application_train before calling "
            "build_bundle — this spec cannot do the merge itself.",
        ),
        FieldSpec(
            "active_loans",
            A.NATIVE,
            "BUREAU_ACTIVE_LOAN_COUNT",
            Unit.COUNT,
            "Count of bureau.csv credit lines with CREDIT_ACTIVE == 'Active'; "
            "0 if the applicant has no bureau.csv rows. See "
            "research/data/bureau_aggregates.py.",
        ),
        FieldSpec("gender", A.NATIVE, _gender, Unit.CATEGORY, "CODE_GENDER; XNA -> NA"),
        FieldSpec(
            "region",
            A.PROXY,
            "REGION_RATING_CLIENT",
            Unit.CATEGORY,
            "Ordinal 1-3 region rating; urbanicity proxy, not a geography.",
        ),
    ),
    target="TARGET",
    selected=None,  # not derivable from application_train alone — see module docstring
)
