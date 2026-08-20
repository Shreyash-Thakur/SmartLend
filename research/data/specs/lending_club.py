"""Canonical mapping for Lending Club.

Role: external validity, and — more importantly — the dataset with a genuinely
observable lender decision.

The selection story here needs care. The public data ships as two files with
*different schemas*:

  accepted_2007_to_2018Q4.csv  ~2.2M rows, 150 columns, has outcomes
  rejected_2007_to_2018Q4.csv  ~27M rows, 9 columns, no outcomes

The rejected file carries only: Amount Requested, Application Date, Loan Title,
Risk_Score, Debt-To-Income Ratio, Zip Code, State, Employment Length, Policy
Code. So the propensity model `P(approved | x)` can only use the **intersection**
of the two schemas — roughly amount, risk score, DTI, state, employment length.

That constraint is not a nuisance; it is the paper's setting. A lender's own
approval model saw far more than the intersection, which is precisely why
positivity fails: the covariates driving the historical decision are partly
unobserved in the reject file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.data.canonical import Availability as A
from research.data.canonical import DatasetSpec, FieldSpec, Unit

ACCEPTED_FILE = "data/raw/lending_club/accepted_2007_to_2018Q4.csv"
REJECTED_FILE = "data/raw/lending_club/rejected_2007_to_2018Q4.csv"

# Loan states that resolve to "bad". Current/Issued/In Grace Period are
# unresolved and must be EXCLUDED, not treated as good — including them biases
# the target toward 0 and is a common error in Lending Club papers.
BAD_STATUSES = frozenset(
    {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
)
GOOD_STATUSES = frozenset(
    {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
)

# Columns present in BOTH accepted and rejected files (after renaming). The
# propensity model is restricted to these.
PROPENSITY_INTERSECTION: tuple[str, ...] = (
    "loan_amount",
    "credit_score",
    "dti",
    "employment_tenure_years",
    "region",
)


def _employment_tenure_years(df: pd.DataFrame) -> pd.Series:
    """Parse emp_length: '10+ years', '< 1 year', '3 years', or NaN."""
    raw = df["emp_length"].astype("string")
    years = raw.str.extract(r"(\d+)", expand=False).astype("Float64")
    # '< 1 year' has no digit other than the 1 in the comparison, so handle it
    # explicitly before the generic extraction is trusted.
    years = years.mask(raw.str.contains("<", na=False), 0.5)
    return years.astype("float64")


def _credit_score(df: pd.DataFrame) -> pd.Series:
    """Midpoint of the reported FICO band."""
    return (df["fico_range_low"] + df["fico_range_high"]) / 2.0


def _credit_utilization(df: pd.DataFrame) -> pd.Series:
    """revol_util arrives as a percentage string ('45.3%') in some releases."""
    raw = df["revol_util"]
    if raw.dtype == object or str(raw.dtype) == "string":
        raw = raw.astype("string").str.rstrip("%").astype("Float64").astype("float64")
    return raw / 100.0


def _target(df: pd.DataFrame) -> pd.Series:
    """1 = bad, 0 = good, NaN = unresolved (caller must drop these)."""
    status = df["loan_status"].astype("string")
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out[status.isin(GOOD_STATUSES)] = 0.0
    out[status.isin(BAD_STATUSES)] = 1.0
    return out


SPEC = DatasetSpec(
    name="lending_club",
    notes=(
        "External validity + observable lender decision. Unresolved loan "
        "statuses (Current, In Grace Period, Late) yield NaN target and MUST be "
        "dropped, not coerced to 0. Propensity modelling is limited to the "
        "accepted/rejected schema intersection."
    ),
    fields=(
        FieldSpec(
            "age_years",
            A.ABSENT,
            None,
            Unit.YEARS,
            "Lending Club does not publish applicant age.",
        ),
        FieldSpec("annual_income", A.NATIVE, "annual_inc", Unit.CURRENCY),
        FieldSpec("loan_amount", A.NATIVE, "loan_amnt", Unit.CURRENCY),
        FieldSpec("installment", A.NATIVE, "installment", Unit.CURRENCY),
        FieldSpec(
            "credit_score",
            A.DERIVED,
            _credit_score,
            Unit.SCORE,
            "FICO band midpoint; scale differs from CIBIL and from EXT_SOURCE_2.",
        ),
        FieldSpec("dti", A.NATIVE, "dti", Unit.RATIO, "Reported as percent in raw data"),
        FieldSpec(
            "employment_tenure_years",
            A.DERIVED,
            _employment_tenure_years,
            Unit.YEARS,
            "emp_length parsed; '< 1 year' -> 0.5, '10+ years' -> 10 (censored)",
        ),
        FieldSpec("credit_utilization", A.DERIVED, _credit_utilization, Unit.RATIO),
        FieldSpec("delinquencies", A.NATIVE, "delinq_2yrs", Unit.COUNT, "2-year window"),
        FieldSpec("active_loans", A.NATIVE, "open_acc", Unit.COUNT),
        FieldSpec("gender", A.ABSENT, None, Unit.CATEGORY, "Not published."),
        FieldSpec(
            "region",
            A.PROXY,
            "addr_state",
            Unit.CATEGORY,
            "US state; a geography proxy only. Using it as a protected-attribute "
            "proxy requires an explicit, defended argument in the paper.",
        ),
    ),
    target=_target,
    # `grade`/`sub_grade` is the lender's risk assignment for ACCEPTED loans, not
    # an accept/reject flag. True selection requires joining the rejected file.
    selected=None,
)
