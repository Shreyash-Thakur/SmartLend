"""The canonical core schema.

CBES consumes only these fields, so they must be populatable from every dataset.
ML models consume full native features separately — see `adapters.load_dataset`.

The asymmetry is deliberate: CBES encodes a portable domain prior, the ML model
exploits dataset-specific patterns, and their disagreement therefore measures
"learned patterns contradict domain knowledge" rather than "trees differ from
linear models".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import pandas as pd


class Unit(str, Enum):
    """Unit of a canonical field. Mismatches here are silent and fatal."""

    CURRENCY = "currency"  # dataset-native currency; not cross-dataset comparable
    YEARS = "years"
    RATIO = "ratio"  # expected in [0, 1]
    SCORE = "score"  # credit score on the dataset's own scale
    COUNT = "count"
    CATEGORY = "category"


class Availability(str, Enum):
    """How a canonical field is obtained from a given dataset."""

    NATIVE = "native"  # a column holds it directly
    DERIVED = "derived"  # computed from other columns
    PROXY = "proxy"  # imperfect substitute; interpret with care
    ABSENT = "absent"  # unavailable in this dataset


@dataclass(frozen=True)
class CanonicalField:
    """Definition of a canonical core field, independent of any dataset."""

    name: str
    unit: Unit
    description: str
    required: bool = True


@dataclass(frozen=True)
class FieldSpec:
    """How one dataset supplies one canonical field.

    `source` is either a column name, or a callable taking the raw frame and
    returning a Series. Callables exist because most real mappings are
    derivations (`-DAYS_BIRTH / 365`), not renames.
    """

    canonical: str
    availability: Availability
    source: str | Callable[[pd.DataFrame], pd.Series] | None = None
    unit: Unit | None = None
    notes: str = ""

    def resolve(self, df: pd.DataFrame) -> pd.Series | None:
        """Extract this field from a raw frame, or None when unavailable.

        Missing values are preserved as NaN. Nothing is imputed here — that is a
        decision for the profiler and the caller, never for the mapping layer.
        """
        if self.availability is Availability.ABSENT or self.source is None:
            return None
        if callable(self.source):
            return self.source(df)
        if self.source not in df.columns:
            return None
        return df[self.source]


# The canonical core. Twelve fields, chosen because every target dataset can
# populate them natively, by derivation, or by a documented proxy.
CANONICAL_CORE: tuple[CanonicalField, ...] = (
    CanonicalField("age_years", Unit.YEARS, "Applicant age at application"),
    CanonicalField("annual_income", Unit.CURRENCY, "Gross annual income"),
    CanonicalField("loan_amount", Unit.CURRENCY, "Requested/granted principal"),
    CanonicalField("installment", Unit.CURRENCY, "Periodic repayment (EMI/annuity)"),
    CanonicalField("credit_score", Unit.SCORE, "Bureau score or documented proxy"),
    CanonicalField("dti", Unit.RATIO, "Debt-to-income ratio"),
    CanonicalField("employment_tenure_years", Unit.YEARS, "Years in current employment"),
    CanonicalField("credit_utilization", Unit.RATIO, "Revolving credit utilisation"),
    CanonicalField("delinquencies", Unit.COUNT, "Delinquency/missed-payment count"),
    CanonicalField("active_loans", Unit.COUNT, "Currently open credit lines"),
    # Protected attributes: optional because not every dataset records them, but
    # the fairness analysis needs whichever are present.
    CanonicalField("gender", Unit.CATEGORY, "Protected attribute", required=False),
    CanonicalField("region", Unit.CATEGORY, "Region/urbanicity proxy", required=False),
)

CORE_NAMES: tuple[str, ...] = tuple(f.name for f in CANONICAL_CORE)
CORE_BY_NAME: dict[str, CanonicalField] = {f.name: f for f in CANONICAL_CORE}

TARGET = "target"  # 1 = default / bad outcome, consistently across datasets
SELECTED = "selected"  # 1 = approved by historical policy (label observable)


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset's full mapping into canonical space."""

    name: str
    fields: tuple[FieldSpec, ...]
    target: str | Callable[[pd.DataFrame], pd.Series]
    # Historical approve/reject decision, where observable. Required for the
    # selection-bias work; None means the dataset cannot support it.
    selected: str | Callable[[pd.DataFrame], pd.Series] | None = None
    notes: str = ""
    sentinels: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def by_canonical(self) -> dict[str, FieldSpec]:
        return {f.canonical: f for f in self.fields}

    def coverage(self) -> dict[str, Availability]:
        """Which canonical fields this dataset can populate, and how.

        This is the report that makes cross-dataset claims auditable — it states
        up front where a comparison rests on a proxy.
        """
        mapping = self.by_canonical()
        return {
            name: mapping[name].availability if name in mapping else Availability.ABSENT
            for name in CORE_NAMES
        }

    def missing_required(self) -> tuple[str, ...]:
        """Required canonical fields this dataset cannot supply."""
        cov = self.coverage()
        return tuple(
            name
            for name in CORE_NAMES
            if CORE_BY_NAME[name].required and cov[name] is Availability.ABSENT
        )
