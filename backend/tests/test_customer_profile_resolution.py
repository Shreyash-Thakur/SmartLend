"""Regression tests for customer profile resolution and pipeline coverage.

Two bugs are pinned here.

1. The form used to be unusable on any machine without a ~180MB CSV that is not
   in the repo: no customer id resolved, so no application could be submitted.
   `test_profile_resolves_without_the_external_extract` fails if the seeded
   fixture stops being enough on its own.

2. The resolved payload used to reach `MLPredictor.predict_application` with
   camelCase names (`cibilScore`, `yearsOfEmployment`) that the pipeline's
   snake_case `feature_names` do not match, so most features silently became
   0.0 and applicants were scored on near-empty vectors.
   `test_resolved_payload_covers_every_profile_pipeline_feature` fails if any
   pipeline feature the profile should supply goes missing again.
"""

from __future__ import annotations

import joblib
import pytest

from backend.app.database import init_db
from backend.app.services import customer_profile_service as cps
from backend.app.services.customer_seed_service import SEED_FIXTURE_PATH, load_seed_records
from backend.app.services.ml_service import PIPELINE_PATH


@pytest.fixture(scope="module", autouse=True)
def _seeded_db() -> None:
    # init_db() creates customer_profiles and loads the committed fixture.
    init_db()


@pytest.fixture(scope="module")
def pipeline_feature_names() -> list[str]:
    return list(joblib.load(PIPELINE_PATH)["feature_names"])


@pytest.fixture(scope="module")
def seeded_customer_id() -> str:
    records = load_seed_records()
    assert records, f"seed fixture missing or empty at {SEED_FIXTURE_PATH}"
    return str(records[0]["customer_id"])


@pytest.fixture
def without_external_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a fresh clone: the big CSV does not exist anywhere."""
    monkeypatch.setenv(cps._ENV_VAR, str(SEED_FIXTURE_PATH.parent / "definitely-not-here.csv"))
    monkeypatch.setattr(cps, "_DEFAULT_PATHS", ())
    cps.reset_cache()
    yield
    cps.reset_cache()


# ---------------------------------------------------------------------------
# Problem 1 — the form must work with no external file
# ---------------------------------------------------------------------------


def test_seed_fixture_is_committed_and_populated() -> None:
    records = load_seed_records()
    assert SEED_FIXTURE_PATH.exists()
    assert len(records) >= 100, "fixture too small to demo a spread of decisions"
    # Stratified sampling must produce a real credit-score spread, not 500
    # near-identical approvals.
    scores = [r["ext_source_2"] for r in records if r.get("ext_source_2") is not None]
    assert min(scores) < 0.25 and max(scores) > 0.70


def test_seed_fixture_never_carries_the_default_outcome() -> None:
    # TARGET stratifies the sample; storing it would put an observed default
    # label one attribute-access away from a scoring payload.
    for record in load_seed_records():
        assert "TARGET" not in record
        assert "target" not in record


def test_profile_resolves_without_the_external_extract(
    without_external_extract: None, seeded_customer_id: str
) -> None:
    profile = cps.get_profile(seeded_customer_id)
    assert profile is not None, "a fresh clone must still resolve seeded customers"
    assert profile["customer_id"] == seeded_customer_id
    assert profile["found"] is True
    assert profile["age"] is not None


def test_unknown_id_still_returns_none(without_external_extract: None) -> None:
    assert cps.get_profile("999999999") is None
    assert cps.get_profile("not-a-number") is None


def test_sample_customers_endpoint_payload(without_external_extract: None) -> None:
    samples = cps.get_sample_customers(limit=10)
    assert len(samples) == 10
    for sample in samples:
        assert cps.get_profile(sample["customer_id"]) is not None
        assert sample["descriptor"]
        assert sample["expected_decision_hint"] in {"APPROVE", "REJECT", "DEFER"}
    # The panel exists so a demo shows more than one outcome.
    assert len({s["expected_decision_hint"] for s in samples}) >= 2


# ---------------------------------------------------------------------------
# Problem 2 — pipeline feature coverage
# ---------------------------------------------------------------------------


def test_profile_features_are_a_real_subset_of_the_pipeline(
    pipeline_feature_names: list[str],
) -> None:
    """The three provenance buckets must exactly partition `feature_names`."""
    declared = (
        set(cps.PROFILE_FEATURES)
        | set(cps.FORM_FEATURES)
        | set(cps.DERIVED_FEATURES)
        | set(cps.LEAKED_OUTPUT_FEATURES)
    )
    assert declared == set(pipeline_feature_names)


def test_get_profile_emits_every_profile_sourced_feature(seeded_customer_id: str) -> None:
    profile = cps.get_profile(seeded_customer_id)
    assert profile is not None
    missing = [f for f in cps.PROFILE_FEATURES if f not in profile]
    assert not missing, f"profile no longer supplies pipeline features: {missing}"


def test_get_profile_does_not_invent_form_or_output_features(seeded_customer_id: str) -> None:
    profile = cps.get_profile(seeded_customer_id)
    assert profile is not None
    # Loan terms and assets come from the applicant; `loan_approved` /
    # `confidence_score` are model outputs. The profile must supply none of them.
    for feature in (*cps.FORM_FEATURES, *cps.LEAKED_OUTPUT_FEATURES):
        assert feature not in profile


def test_resolved_payload_covers_every_profile_pipeline_feature(
    pipeline_feature_names: list[str], seeded_customer_id: str
) -> None:
    payload = cps.resolve_application_payload(
        {
            "customer_id": seeded_customer_id,
            "loan_amount": 500_000.0,
            "loan_tenure_months": 60,
            "interestRate": 11.5,
            "emi": 9_500.0,
            "residentialAssetsValue": 1_200_000.0,
            "commercialAssetsValue": 0.0,
            "bankBalance": 75_000.0,
        }
    )
    assert payload["profile_resolved"] is True

    expected = [
        f
        for f in pipeline_feature_names
        if f not in cps.LEAKED_OUTPUT_FEATURES  # model outputs; see LEAKED_OUTPUT_FEATURES
    ]
    missing = [f for f in expected if f not in payload]
    assert not missing, f"pipeline features absent from the resolved payload: {missing}"

    # Form values must survive unchanged, not be overwritten by the profile.
    assert payload["loan_amount"] == 500_000.0
    assert payload["loan_term"] == 60
    assert payload["interest_rate"] == 11.5
    assert payload["emi"] == 9_500.0

    # Derived fields are arithmetic over form + profile, nothing invented.
    assert payload["total_assets"] == pytest.approx(1_200_000.0)
    assert payload["emi_income_ratio"] == pytest.approx(
        9_500.0 / payload["monthly_income"], rel=1e-3
    )
    assert payload["loan_income_ratio"] == pytest.approx(
        500_000.0 / payload["annual_income"], rel=1e-3
    )
    assert payload["debt_to_income_ratio"] == payload["dti"]
    assert payload["cibil_score"] == payload["cibilScore"]
    assert payload["missed_payments"] == payload["delinquencies"]


def test_leaked_output_features_are_never_supplied(seeded_customer_id: str) -> None:
    """`loan_approved` / `confidence_score` are model outputs, not inputs.

    They are in the artifact's `feature_names` because the old synthetic
    training frame still contained the label and the model's own confidence.
    There is no honest inference-time value, so nothing may fabricate one.
    """
    payload = cps.resolve_application_payload(
        {"customer_id": seeded_customer_id, "loan_amount": 300_000.0}
    )
    for feature in cps.LEAKED_OUTPUT_FEATURES:
        assert payload.get(feature) is None
