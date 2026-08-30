from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.models import DeferredReview
from backend.app.services.deferred_review_service import (
    DEFAULT_EXPLORATION_RATE,
    OVERRIDE_TO_APPROVE,
    OVERRIDE_TO_REJECT,
    maybe_route_to_exploration,
    record_deferral,
    record_human_decision,
    record_outcome,
)


@dataclass
class FakeDecisionResult:
    decision: str
    decision_reason: str
    p_ml: float
    p_cbes: float
    p_blend: float
    disagreement: float
    confidence: float
    t_approve: float
    t_reject: float
    cbes_breakdown: dict = field(default_factory=dict)


def _memory_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _defer_result(p_blend: float = 0.50, **overrides) -> FakeDecisionResult:
    payload = dict(
        decision="DEFER",
        decision_reason="grey_zone",
        p_ml=0.52,
        p_cbes=0.44,
        p_blend=p_blend,
        disagreement=0.08,
        confidence=0.15,
        t_approve=0.55,
        t_reject=0.45,
        cbes_breakdown={"credit": 0.4, "capacity": 0.5, "behaviour": 0.6,
                        "stability": 0.7, "region": 0.5},
    )
    payload.update(overrides)
    return FakeDecisionResult(**payload)


# --------------------------------------------------------------------------
# Table + row creation
# --------------------------------------------------------------------------

def test_deferred_review_table_creates_and_accepts_minimal_row():
    session = _memory_session()
    row = DeferredReview(
        application_id="app-1",
        decision_reason="disagreement",
        p_ml=0.5,
        p_cbes=0.4,
        p_blend=0.475,
        disagreement=0.1,
        confidence=0.3,
        t_approve=0.55,
        t_reject=0.45,
        cbes_breakdown_json={"credit": 0.5},
        engine_version="v1",
        threshold_artifact_hash="abc123",
        t_base=0.5,
    )
    session.add(row)
    session.commit()
    assert row.review_id is not None
    assert row.exploration_flag is False
    assert row.outcome_censored is True
    assert row.human_decision is None
    session.close()


def test_record_deferral_persists_full_engine_state():
    session = _memory_session()
    row = record_deferral(
        session,
        _defer_result(),
        application_id="app-42",
        engine_version="v1",
        threshold_artifact_hash="hash-abc",
        t_base=0.50,
        applicant_segment={"age_band": "25-34"},
    )
    session.commit()

    assert row.review_id is not None
    assert row.application_id == "app-42"
    assert row.decision_reason == "grey_zone"
    assert row.p_ml == pytest.approx(0.52)
    assert row.p_cbes == pytest.approx(0.44)
    assert row.p_blend == pytest.approx(0.50)
    assert row.disagreement == pytest.approx(0.08)
    assert row.confidence == pytest.approx(0.15)
    assert row.t_approve == pytest.approx(0.55)
    assert row.t_reject == pytest.approx(0.45)
    assert row.t_base == pytest.approx(0.50)
    assert row.cbes_breakdown_json["credit"] == pytest.approx(0.4)
    assert row.engine_version == "v1"
    assert row.threshold_artifact_hash == "hash-abc"
    assert row.applicant_segment_json == {"age_band": "25-34"}
    assert row.exploration_flag is False
    assert row.outcome_censored is True
    assert row.human_decision is None
    assert row.agreed_with_engine is None
    session.close()


def test_record_deferral_rejects_non_defer_decisions():
    session = _memory_session()
    approved = _defer_result(decision="APPROVE", decision_reason="ml_approve", p_blend=0.8)
    with pytest.raises(ValueError):
        record_deferral(
            session, approved, application_id="app-43",
            engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
        )
    session.close()


def test_record_deferral_accepts_auto_decision_only_as_exploration_row():
    session = _memory_session()
    approved = _defer_result(decision="APPROVE", decision_reason="ml_approve", p_blend=0.8)
    row = record_deferral(
        session, approved, application_id="app-44",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
        exploration_flag=True,
    )
    session.commit()
    assert row.exploration_flag is True
    assert row.decision_reason == "ml_approve"
    session.close()


def test_record_deferral_refuses_to_flag_a_genuine_defer_as_exploration():
    session = _memory_session()
    with pytest.raises(ValueError):
        record_deferral(
            session, _defer_result(), application_id="app-45",
            engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
            exploration_flag=True,
        )
    session.close()


# --------------------------------------------------------------------------
# Human decision: agreed_with_engine / override_direction, both directions
# --------------------------------------------------------------------------

def test_human_decision_agreeing_with_engine_lean_sets_agreement():
    session = _memory_session()
    # p_blend above t_approve -> engine leaned APPROVE
    row = record_deferral(
        session, _defer_result(p_blend=0.60, decision_reason="disagreement"),
        application_id="app-50", engine_version="v1",
        threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    record_human_decision(
        session, row, human_decision="APPROVE", reviewer_id="rev-7",
        time_spent_seconds=42.5, human_reason_codes=["CB-01"],
        human_free_text="documents verified", reviewer_confidence=4,
    )
    session.commit()

    assert row.human_decision == "APPROVE"
    assert row.reviewer_id == "rev-7"
    assert row.reviewed_at is not None
    assert row.time_spent_seconds == pytest.approx(42.5)
    assert row.human_reason_codes == ["CB-01"]
    assert row.human_free_text == "documents verified"
    assert row.reviewer_confidence == 4
    assert row.agreed_with_engine is True
    assert row.override_direction is None
    session.close()


def test_human_override_of_approve_lean_records_downward_direction():
    session = _memory_session()
    row = record_deferral(
        session, _defer_result(p_blend=0.60), application_id="app-51",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    record_human_decision(session, row, human_decision="REJECT", reviewer_id="rev-7")
    session.commit()

    assert row.agreed_with_engine is False
    assert row.override_direction == OVERRIDE_TO_REJECT
    session.close()


def test_human_override_of_reject_lean_records_upward_direction():
    session = _memory_session()
    # p_blend at/below t_reject -> engine leaned REJECT
    row = record_deferral(
        session, _defer_result(p_blend=0.40), application_id="app-52",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    record_human_decision(session, row, human_decision="APPROVE", reviewer_id="rev-7")
    session.commit()

    assert row.agreed_with_engine is False
    assert row.override_direction == OVERRIDE_TO_APPROVE
    session.close()


def test_human_decision_agreeing_with_reject_lean_sets_agreement():
    session = _memory_session()
    row = record_deferral(
        session, _defer_result(p_blend=0.40), application_id="app-53",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    record_human_decision(session, row, human_decision="REJECT", reviewer_id="rev-7")
    session.commit()

    assert row.agreed_with_engine is True
    assert row.override_direction is None
    session.close()


def test_true_grey_zone_records_no_agreement_rather_than_a_fabricated_one():
    session = _memory_session()
    # t_reject < p_blend < t_approve -> engine had no lean at all
    row = record_deferral(
        session, _defer_result(p_blend=0.50), application_id="app-54",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    record_human_decision(session, row, human_decision="REJECT", reviewer_id="rev-7")
    session.commit()

    assert row.human_decision == "REJECT"
    assert row.agreed_with_engine is None
    assert row.override_direction is None
    session.close()


def test_invalid_human_decision_and_confidence_are_rejected():
    session = _memory_session()
    row = record_deferral(
        session, _defer_result(), application_id="app-55",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    with pytest.raises(ValueError):
        record_human_decision(session, row, human_decision="MAYBE", reviewer_id="rev-7")
    with pytest.raises(ValueError):
        record_human_decision(
            session, row, human_decision="APPROVE", reviewer_id="rev-7",
            reviewer_confidence=9,
        )
    session.close()


# --------------------------------------------------------------------------
# Outcome recording, including the censored case
# --------------------------------------------------------------------------

def test_record_outcome_stores_observed_default():
    session = _memory_session()
    row = record_deferral(
        session, _defer_result(), application_id="app-60",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    observed_at = datetime(2027, 1, 15, tzinfo=timezone.utc)
    record_outcome(session, row, realized_outcome=1, observed_at=observed_at)
    session.commit()

    assert row.realized_outcome == 1
    assert row.outcome_censored is False
    assert row.outcome_observed_at is not None
    assert row.outcome_observed_at.year == 2027
    session.close()


def test_record_outcome_defaults_observed_at_when_not_supplied():
    session = _memory_session()
    row = record_deferral(
        session, _defer_result(), application_id="app-61",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    record_outcome(session, row, realized_outcome=0)
    session.commit()

    assert row.realized_outcome == 0
    assert row.outcome_censored is False
    assert row.outcome_observed_at is not None
    session.close()


def test_record_outcome_censored_case_leaves_outcome_null():
    session = _memory_session()
    row = record_deferral(
        session, _defer_result(), application_id="app-62",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    record_outcome(session, row, realized_outcome=None)
    session.commit()

    assert row.realized_outcome is None
    assert row.outcome_observed_at is None
    assert row.outcome_censored is True
    session.close()


def test_record_outcome_can_recensor_a_previously_observed_row():
    session = _memory_session()
    row = record_deferral(
        session, _defer_result(), application_id="app-63",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    record_outcome(session, row, realized_outcome=1)
    record_outcome(session, row, censored=True)
    session.commit()

    assert row.realized_outcome is None
    assert row.outcome_observed_at is None
    assert row.outcome_censored is True
    session.close()


def test_record_outcome_rejects_invalid_outcome_and_inconsistent_censoring():
    session = _memory_session()
    row = record_deferral(
        session, _defer_result(), application_id="app-64",
        engine_version="v1", threshold_artifact_hash="hash-abc", t_base=0.50,
    )
    with pytest.raises(ValueError):
        record_outcome(session, row, realized_outcome=2)
    with pytest.raises(ValueError):
        record_outcome(session, row, realized_outcome=None, censored=False)
    session.close()


# --------------------------------------------------------------------------
# Exploration arm — deterministic under a seeded RNG
# --------------------------------------------------------------------------

def test_exploration_default_rate_is_three_percent_inside_the_spec_band():
    assert DEFAULT_EXPLORATION_RATE == pytest.approx(0.03)
    assert 0.02 <= DEFAULT_EXPLORATION_RATE <= 0.05


def test_exploration_sampling_hits_configured_rate_with_seeded_rng():
    trials = 20_000
    rng = random.Random(1234)
    hits = sum(1 for _ in range(trials) if maybe_route_to_exploration(rng=rng))
    observed = hits / trials
    assert 0.02 < observed < 0.04, f"observed exploration rate {observed}"


def test_exploration_sampling_is_reproducible_for_the_same_seed():
    def draw(seed: int) -> list[bool]:
        rng = random.Random(seed)
        return [maybe_route_to_exploration(rate=0.05, rng=rng) for _ in range(500)]

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)


def test_exploration_sampling_tracks_a_configured_non_default_rate():
    trials = 20_000
    rng = random.Random(99)
    hits = sum(1 for _ in range(trials) if maybe_route_to_exploration(rate=0.05, rng=rng))
    observed = hits / trials
    assert 0.04 < observed < 0.06, f"observed exploration rate {observed}"


def test_exploration_zero_rate_never_fires_and_full_rate_always_fires():
    rng = random.Random(0)
    assert all(not maybe_route_to_exploration(rate=0.0, rng=rng) for _ in range(1000))
    assert all(maybe_route_to_exploration(rate=1.0, rng=rng) for _ in range(1000))


def test_exploration_rate_outside_unit_interval_raises():
    with pytest.raises(ValueError):
        maybe_route_to_exploration(rate=-0.1)
    with pytest.raises(ValueError):
        maybe_route_to_exploration(rate=1.5)


# --------------------------------------------------------------------------
# The capture layer must stay capture-only
# --------------------------------------------------------------------------

def test_service_module_contains_no_training_or_retraining_hooks():
    from pathlib import Path

    import backend.app.services.deferred_review_service as service

    source = Path(service.__file__).read_text(encoding="utf-8").lower()
    body = source.split('"""', 2)[-1]  # skip the module docstring, which discusses them
    for forbidden in ("def train", "retrain", ".fit(", "reject_inference", "partial_fit"):
        assert forbidden not in body, f"capture layer must not contain {forbidden!r}"
