"""Tests for the canonical mapping layer.

The synthetic tests run against the real CSV in the repo, so a pass means the
spec matches actual columns rather than documentation. The Home Credit and
Lending Club specs are written from data dictionaries and cannot be verified
until those files are downloaded — `validate_spec` is the tool for that, and
`test_validate_spec_catches_missing_columns` proves it works.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.data.adapters import build_bundle, coverage_report, validate_spec
from research.data.canonical import CORE_NAMES, Availability
from research.data.specs import home_credit, lending_club, synthetic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CSV = PROJECT_ROOT / synthetic.SOURCE_FILE


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    if not SYNTHETIC_CSV.exists():
        pytest.skip(f"{SYNTHETIC_CSV} not present")
    return pd.read_csv(SYNTHETIC_CSV)


class TestSyntheticSpec:
    def test_spec_validates_against_real_file(self, synthetic_df):
        assert validate_spec(synthetic.SPEC, synthetic_df) == []

    def test_supplies_every_required_field(self):
        assert synthetic.SPEC.missing_required() == ()

    def test_has_observable_selection_indicator(self, synthetic_df):
        # This is what makes the synthetic set the controlled testbed: the
        # approve/reject decision is actually observed.
        bundle = build_bundle(synthetic.SPEC, synthetic_df)
        assert bundle.selected is not None
        assert set(bundle.selected.unique()) <= {0, 1}

    def test_bundle_shapes_line_up(self, synthetic_df):
        bundle = build_bundle(synthetic.SPEC, synthetic_df)
        assert bundle.n_rows == len(synthetic_df)
        assert list(bundle.core.columns) == list(CORE_NAMES)
        assert len(bundle.target) == len(synthetic_df)

    def test_core_values_are_real_not_placeholders(self, synthetic_df):
        bundle = build_bundle(synthetic.SPEC, synthetic_df)
        for column in ("age_years", "annual_income", "loan_amount", "credit_score"):
            assert bundle.core[column].notna().all(), f"{column} should be fully populated"

    def test_target_and_selection_excluded_from_native(self, synthetic_df):
        bundle = build_bundle(synthetic.SPEC, synthetic_df)
        assert "default_risk" not in bundle.native.columns
        assert "loan_approved" not in bundle.native.columns

    def test_categoricals_survive_into_native(self, synthetic_df):
        """The regression guard for the select_dtypes(number) bug.

        The legacy pipeline dropped every categorical, which handicapped the
        model and made protected-attribute analysis impossible.
        """
        bundle = build_bundle(synthetic.SPEC, synthetic_df)
        categoricals = set(bundle.categorical_columns())
        for expected in ("gender", "education", "employment_type", "region"):
            assert expected in categoricals, f"{expected} must survive into native features"

    def test_numeric_and_categorical_partition_native(self, synthetic_df):
        bundle = build_bundle(synthetic.SPEC, synthetic_df)
        overlap = set(bundle.numeric_columns()) & set(bundle.categorical_columns())
        assert overlap == set()


class TestCoverageReport:
    def test_reports_every_canonical_field(self):
        report = coverage_report(home_credit.SPEC)
        assert list(report["canonical"]) == list(CORE_NAMES)

    def test_home_credit_gaps_are_surfaced_not_hidden(self):
        """The three bureau-derived fields must show as ABSENT.

        Marking them available would let a silently-empty column reach CBES.
        """
        report = coverage_report(home_credit.SPEC).set_index("canonical")
        for field in ("credit_utilization", "delinquencies", "active_loans"):
            assert report.loc[field, "availability"] == Availability.ABSENT.value
        assert set(home_credit.SPEC.missing_required()) == {
            "credit_utilization",
            "delinquencies",
            "active_loans",
        }

    def test_proxy_fields_are_labelled_as_proxies(self):
        report = coverage_report(home_credit.SPEC).set_index("canonical")
        assert report.loc["credit_score", "availability"] == Availability.PROXY.value
        assert "NOT a bureau score" in report.loc["credit_score", "notes"]

    def test_lending_club_declares_missing_age_and_gender(self):
        report = coverage_report(lending_club.SPEC).set_index("canonical")
        assert report.loc["age_years", "availability"] == Availability.ABSENT.value
        assert report.loc["gender", "availability"] == Availability.ABSENT.value


class TestHomeCreditDerivations:
    @pytest.fixture
    def raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "DAYS_BIRTH": [-10_957, -14_610],  # ~30 and ~40 years
                "DAYS_EMPLOYED": [-1_825, home_credit.DAYS_EMPLOYED_SENTINEL],
                "AMT_INCOME_TOTAL": [200_000.0, 0.0],
                "AMT_ANNUITY": [24_000.0, 12_000.0],
                "AMT_CREDIT": [500_000.0, 300_000.0],
                "EXT_SOURCE_2": [0.5, 0.7],
                "CODE_GENDER": ["M", "XNA"],
                "REGION_RATING_CLIENT": [2, 3],
                "TARGET": [0, 1],
            }
        )

    def test_age_derived_from_days_birth(self, raw):
        bundle = build_bundle(home_credit.SPEC, raw)
        assert bundle.core["age_years"].iloc[0] == pytest.approx(30.0, abs=0.1)

    def test_employment_sentinel_becomes_nan_not_a_thousand_years(self, raw):
        bundle = build_bundle(home_credit.SPEC, raw)
        tenure = bundle.core["employment_tenure_years"]
        assert tenure.iloc[0] == pytest.approx(5.0, abs=0.1)
        assert pd.isna(tenure.iloc[1]), "365243 must not become ~1000 years"

    def test_zero_income_does_not_produce_infinite_dti(self, raw):
        bundle = build_bundle(home_credit.SPEC, raw)
        dti = bundle.core["dti"]
        assert dti.iloc[0] == pytest.approx(0.12)
        assert pd.isna(dti.iloc[1]), "division by zero income must yield NaN, not inf"
        assert not np.isinf(dti.to_numpy(dtype="float64")).any()

    def test_xna_gender_becomes_missing(self, raw):
        bundle = build_bundle(home_credit.SPEC, raw)
        assert bundle.core["gender"].iloc[0] == "M"
        assert pd.isna(bundle.core["gender"].iloc[1])

    def test_absent_fields_are_all_nan_not_fabricated(self, raw):
        bundle = build_bundle(home_credit.SPEC, raw)
        assert bundle.core["credit_utilization"].isna().all()
        assert bundle.core["active_loans"].isna().all()


class TestLendingClubDerivations:
    @pytest.fixture
    def raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "annual_inc": [60_000.0, 90_000.0, 45_000.0],
                "loan_amnt": [10_000.0, 20_000.0, 5_000.0],
                "installment": [325.0, 640.0, 180.0],
                "fico_range_low": [700, 660, 720],
                "fico_range_high": [704, 664, 724],
                "dti": [15.0, 22.0, 9.0],
                "emp_length": ["10+ years", "< 1 year", "3 years"],
                "revol_util": [45.3, 80.0, 12.5],
                "delinq_2yrs": [0, 1, 0],
                "open_acc": [8, 12, 4],
                "addr_state": ["CA", "NY", "TX"],
                "loan_status": ["Fully Paid", "Charged Off", "Current"],
            }
        )

    def test_emp_length_parsing_including_edge_cases(self, raw):
        bundle = build_bundle(lending_club.SPEC, raw)
        tenure = bundle.core["employment_tenure_years"]
        assert tenure.iloc[0] == pytest.approx(10.0)
        assert tenure.iloc[1] == pytest.approx(0.5), "'< 1 year' must not parse as 1"
        assert tenure.iloc[2] == pytest.approx(3.0)

    def test_fico_midpoint(self, raw):
        bundle = build_bundle(lending_club.SPEC, raw)
        assert bundle.core["credit_score"].iloc[0] == pytest.approx(702.0)

    def test_utilisation_converted_to_ratio(self, raw):
        bundle = build_bundle(lending_club.SPEC, raw)
        assert bundle.core["credit_utilization"].iloc[0] == pytest.approx(0.453)

    def test_unresolved_status_is_nan_not_zero(self, raw):
        """The trap: coercing Current loans to 0 biases the target."""
        bundle = build_bundle(lending_club.SPEC, raw)
        assert bundle.target.iloc[0] == 0.0
        assert bundle.target.iloc[1] == 1.0
        assert pd.isna(bundle.target.iloc[2])

    def test_drop_unresolved_removes_only_unresolved_rows(self, raw):
        bundle = build_bundle(lending_club.SPEC, raw).drop_unresolved_target()
        assert bundle.n_rows == 2
        assert bundle.target.tolist() == [0, 1]
        assert len(bundle.core) == len(bundle.native) == 2


class TestValidateSpec:
    def test_catches_missing_columns(self, synthetic_df):
        broken = synthetic_df.drop(columns=["cibil_score", "annual_income"])
        problems = validate_spec(synthetic.SPEC, broken)
        joined = " ".join(problems)
        assert "cibil_score" in joined
        assert "annual_income" in joined

    def test_catches_missing_target(self, synthetic_df):
        broken = synthetic_df.drop(columns=["default_risk"])
        assert any("target" in p for p in validate_spec(synthetic.SPEC, broken))

    def test_reports_required_gaps_for_home_credit(self):
        minimal = pd.DataFrame(
            {
                "DAYS_BIRTH": [-10_957],
                "DAYS_EMPLOYED": [-1_825],
                "AMT_INCOME_TOTAL": [200_000.0],
                "AMT_ANNUITY": [24_000.0],
                "AMT_CREDIT": [500_000.0],
                "EXT_SOURCE_2": [0.5],
                "CODE_GENDER": ["M"],
                "REGION_RATING_CLIENT": [2],
                "TARGET": [0],
            }
        )
        problems = validate_spec(home_credit.SPEC, minimal)
        assert any("required canonical fields unavailable" in p for p in problems)
