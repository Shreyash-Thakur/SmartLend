"""Canonical mapping for the in-repo synthetic Indian loan dataset.

This dataset's role has changed. It is no longer the primary evaluation set — it
is the *controlled testbed*, and the only dataset where the selection mechanism
is known by construction. That makes it the sole place the weighted-conformal
correction can be verified rather than merely applied.

Column names verified against `backend/synthetic_indian_loan_dataset.csv`.
"""

from __future__ import annotations

import pandas as pd

from research.data.canonical import Availability as A
from research.data.canonical import DatasetSpec, FieldSpec, Unit

SOURCE_FILE = "backend/synthetic_indian_loan_dataset.csv"


def _region(df: pd.DataFrame) -> pd.Series:
    # Urban / Semi-Urban / Rural — already an urbanicity axis.
    return df["region"].astype("string")


SPEC = DatasetSpec(
    name="synthetic",
    notes=(
        "Labels drawn from a hand-written logistic model with injected flips. "
        "Use ONLY as a controlled testbed with known ground truth, never as "
        "primary evidence. `loan_approved` is a real, observable selection "
        "indicator, which is what makes this dataset useful."
    ),
    fields=(
        FieldSpec("age_years", A.NATIVE, "age", Unit.YEARS),
        FieldSpec("annual_income", A.NATIVE, "annual_income", Unit.CURRENCY),
        FieldSpec("loan_amount", A.NATIVE, "loan_amount", Unit.CURRENCY),
        FieldSpec("installment", A.NATIVE, "emi", Unit.CURRENCY),
        FieldSpec("credit_score", A.NATIVE, "cibil_score", Unit.SCORE),
        FieldSpec("dti", A.NATIVE, "debt_to_income_ratio", Unit.RATIO),
        FieldSpec("employment_tenure_years", A.NATIVE, "years_employed", Unit.YEARS),
        FieldSpec("credit_utilization", A.NATIVE, "credit_utilization_ratio", Unit.RATIO),
        FieldSpec("delinquencies", A.NATIVE, "missed_payments", Unit.COUNT),
        FieldSpec("active_loans", A.NATIVE, "active_loans", Unit.COUNT),
        FieldSpec("gender", A.NATIVE, "gender", Unit.CATEGORY),
        FieldSpec("region", A.NATIVE, _region, Unit.CATEGORY),
    ),
    target="default_risk",
    selected="loan_approved",
)
