"""Tests for the LIVE wiring of the relearning loop.

The capture layer itself is covered by test_deferred_review_service.py. These
tests cover the three places it is now plugged into the decision path, and the
property that matters more than any of them: capture failing must never break a
lending decision.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.models import DeferredReview, LoanApplication
from backend.app.routers import applications as applications_router
from backend.app.schemas import ManualDecisionRequest
from backend.app.services.decision_engine import hybrid_decision


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()


FORM = {
    "firstName": "Test",
    "lastName": "Applicant",
    "loanAmount": 500000,
    "monthlyIncome": 90000,
    "age": 35,
    "region": "west",
    "gender": "female",
}


def _prediction(p_ml: float, p_cbes: float, tau_d: float, t_base: float = 0.50):
    """A real DecisionResult with the provenance ml_service attaches to it."""
    result = hybrid_decision(
        p_ml=p_ml,
        p_cbes=p_cbes,
        tau_d=tau_d,
        t_base=t_base,
        cbes_breakdown={"credit": 0.4, "capacity": 0.5, "behaviour": 0.6,
                        "stability": 0.7, "region": 0.5},
    )
    result.engineered_features = {}
    result.cbes_weights = {}
    result.cbes_components = result.cbes_breakdown
    result.selected_model = "TestModel"
    result.t_base = t_base
    result.tau_d = tau_d
    result.engine_version = "test-engine/v1"
    result.threshold_artifact_hash = "deadbeefcafe0000"
    return result


def _defer_prediction():
    # Disagreement gate: |0.90 - 0.10| = 0.80 > tau_d
    result = _prediction(p_ml=0.90, p_cbes=0.10, tau_d=0.10)
    assert result.decision == "DEFER"
    return result


def _approve_prediction():
    result = _prediction(p_ml=0.90, p_cbes=0.85, tau_d=0.30)
    assert result.decision == "APPROVE"
    return result


class _FakePredictor:
    def __init__(self, prediction):
        self._prediction = prediction

    def predict_application(self, _input_data):
        return self._prediction


@pytest.fixture
def wire(monkeypatch):
    """Install a fake predictor and pin the exploration coin flip."""

    def _wire(prediction, *, explore: bool = False):
        monkeypatch.setattr(
            applications_router, "get_predictor", lambda: _FakePredictor(prediction)
        )
        monkeypatch.setattr(
            applications_router, "maybe_route_to_exploration", lambda rate: explore
        )
        return prediction

    return _wire


def _create(session, form=None):
    return applications_router._create_application_record(dict(form or FORM), session)


# --------------------------------------------------------------------------
# 1. capture on DEFER
# --------------------------------------------------------------------------


def test_deferral_is_recorded_when_the_engine_defers(session, wire):
    prediction = wire(_defer_prediction())
    payload = _create(session)

    rows = session.query(DeferredReview).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.application_id == payload["id"]
    assert row.exploration_flag is False
    assert row.decision_reason == prediction.decision_reason
    assert row.p_ml == pytest.approx(prediction.p_ml)
    assert row.p_cbes == pytest.approx(prediction.p_cbes)
    assert row.p_blend == pytest.approx(prediction.p_blend)
    assert row.disagreement == pytest.approx(prediction.disagreement)
    assert row.confidence == pytest.approx(prediction.confidence)
    assert row.t_approve == pytest.approx(prediction.t_approve)
    assert row.t_reject == pytest.approx(prediction.t_reject)
    assert row.t_base == pytest.approx(0.50)
    assert row.cbes_breakdown_json == prediction.cbes_breakdown
    assert row.engine_version == "test-engine/v1"
    assert row.threshold_artifact_hash == "deadbeefcafe0000"
    assert row.outcome_censored is True
    assert row.human_decision is None
    # Segment is captured for the disparate-deferral check.
    assert row.applicant_segment_json["region"] == "west"
    assert row.applicant_segment_json["age_band"] == "30-39"
    assert payload["routedToHumanReview"] is True


def test_auto_decision_without_exploration_captures_nothing(session, wire):
    wire(_approve_prediction(), explore=False)
    payload = _create(session)

    assert session.query(DeferredReview).count() == 0
    assert payload["routedToHumanReview"] is False
    assert payload["explorationFlag"] is False


# --------------------------------------------------------------------------
# 2. exploration arm
# --------------------------------------------------------------------------


def test_exploration_arm_flags_a_would_be_auto_decision(session, wire):
    prediction = wire(_approve_prediction(), explore=True)
    payload = _create(session)

    row = session.query(DeferredReview).one()
    assert row.exploration_flag is True
    # The engine's own APPROVE is preserved on the row — that is what makes
    # this an un-selected label rather than a router-selected one.
    assert prediction.decision == "APPROVE"
    assert payload["finalDecision"] == "APPROVE"
    assert payload["explorationFlag"] is True
    assert payload["routedToHumanReview"] is True


def test_exploration_never_converts_a_defer_into_an_auto_decision(session, wire):
    prediction = wire(_defer_prediction(), explore=True)
    payload = _create(session)

    row = session.query(DeferredReview).one()
    # Even with the coin flip rigged to always fire, a DEFER stays a DEFER and
    # is never mislabelled as a control-arm sample.
    assert row.exploration_flag is False
    assert prediction.decision == "DEFER"
    assert payload["finalDecision"] == "DEFER"


def test_exploration_rate_comes_from_config_and_defaults_to_three_percent(monkeypatch):
    from backend.app import config

    monkeypatch.delenv(config.EXPLORATION_RATE_ENV, raising=False)
    assert config.exploration_rate() == pytest.approx(0.03)

    monkeypatch.setenv(config.EXPLORATION_RATE_ENV, "0.05")
    assert config.exploration_rate() == pytest.approx(0.05)

    # Out-of-band and malformed values are clamped / ignored, never trusted.
    monkeypatch.setenv(config.EXPLORATION_RATE_ENV, "0.9")
    assert config.exploration_rate() == pytest.approx(0.05)
    monkeypatch.setenv(config.EXPLORATION_RATE_ENV, "not-a-number")
    assert config.exploration_rate() == pytest.approx(0.03)


# --------------------------------------------------------------------------
# 3. reviewer decision
# --------------------------------------------------------------------------


def _review(session, application_id, **kwargs):
    payload = ManualDecisionRequest(**kwargs)
    return applications_router.update_manual_decision(application_id, payload, session)


def test_reviewer_decision_is_recorded_against_the_deferral_row(session, wire):
    wire(_defer_prediction())
    created = _create(session)

    response = _review(
        session,
        created["id"],
        status="approved",
        notes="Verified salary slips manually.",
        reviewerId="analyst-7",
        reviewerConfidence=4,
        timeSpentSeconds=182.5,
        reasonCodes=["CP-01", "CB-02"],
    )

    row = session.query(DeferredReview).one()
    assert row.human_decision == "APPROVE"
    assert row.reviewer_id == "analyst-7"
    assert row.reviewer_confidence == 4
    assert row.time_spent_seconds == pytest.approx(182.5)
    assert row.human_reason_codes == ["CP-01", "CB-02"]
    assert row.human_free_text == "Verified salary slips manually."
    assert row.reviewed_at is not None
    # p_blend 0.70 is above t_approve, so the engine leaned APPROVE and the
    # reviewer agreed — no override recorded.
    assert row.agreed_with_engine is True
    assert row.override_direction is None
    assert response["decisionCode"] == "APPROVE"


def test_reviewer_override_direction_is_recorded(session, wire):
    wire(_defer_prediction())
    created = _create(session)

    _review(session, created["id"], status="rejected", reviewerId="analyst-7")

    row = session.query(DeferredReview).one()
    assert row.human_decision == "REJECT"
    assert row.agreed_with_engine is False
    assert row.override_direction == "engine_approve_to_human_reject"


def test_reviewer_keeping_the_case_deferred_is_not_a_decision(session, wire):
    wire(_defer_prediction())
    created = _create(session)

    _review(session, created["id"], status="deferred", notes="Need more docs")

    row = session.query(DeferredReview).one()
    assert row.human_decision is None


def test_reviewer_decision_on_an_uncaptured_application_is_a_no_op(session, wire):
    wire(_approve_prediction(), explore=False)
    created = _create(session)

    response = _review(session, created["id"], status="approved")

    assert session.query(DeferredReview).count() == 0
    assert response["decisionCode"] == "APPROVE"


# --------------------------------------------------------------------------
# 4. failure isolation — the point of the whole wiring
# --------------------------------------------------------------------------


def _boom(*_args, **_kwargs):
    raise RuntimeError("simulated database failure during capture")


def test_capture_failure_does_not_break_the_decision(session, wire, monkeypatch, caplog):
    wire(_defer_prediction())
    monkeypatch.setattr(applications_router, "record_deferral", _boom)

    payload = _create(session)

    # The applicant still gets a decision...
    assert payload["responseStatus"] == "success"
    assert payload["finalDecision"] == "DEFER"
    # ...and it is durably persisted.
    assert session.query(LoanApplication).filter_by(id=payload["id"]).one() is not None
    # Only the research row is missing.
    assert session.query(DeferredReview).count() == 0


def test_exploration_sampling_failure_does_not_break_the_decision(session, wire, monkeypatch):
    wire(_approve_prediction())
    monkeypatch.setattr(applications_router, "maybe_route_to_exploration", _boom)

    payload = _create(session)

    assert payload["responseStatus"] == "success"
    assert payload["explorationFlag"] is False
    assert session.query(DeferredReview).count() == 0


def test_reviewer_capture_failure_does_not_break_the_manual_decision(session, wire, monkeypatch):
    wire(_defer_prediction())
    created = _create(session)
    monkeypatch.setattr(applications_router, "record_human_decision", _boom)

    response = _review(session, created["id"], status="approved", reviewerId="analyst-7")

    # The analyst's decision stands and is persisted.
    assert response["responseStatus"] == "success"
    assert response["decisionCode"] == "APPROVE"
    assert session.query(LoanApplication).filter_by(id=created["id"]).one().final_decision == "APPROVE"
    # The capture row is simply left un-updated.
    assert session.query(DeferredReview).one().human_decision is None


# --------------------------------------------------------------------------
# 5. relearning service + gate
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_predictions(tmp_path) -> Path:
    """A small predictions artifact so the gate runs fast in tests."""
    path = tmp_path / "predictions.csv"
    rows = ["y_true,best_model_prob,final_decision,approval_threshold"]
    for i in range(200):
        y = 1 if i % 10 else 0
        prob = 0.9 if y else 0.2
        decision = "DEFER" if i % 3 == 0 else ("APPROVE" if y else "REJECT")
        rows.append(f"{y},{prob},{decision},0.5")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_status_reports_counts_and_the_live_gate_verdict(session, wire, tiny_predictions):
    from backend.app.services import relearning_service

    wire(_defer_prediction())
    created = _create(session)
    _review(session, created["id"], status="rejected", reviewerId="analyst-7")

    status = relearning_service.get_relearning_status(
        session, predictions_path=tiny_predictions
    )

    assert status["rows_captured"] == 1
    assert status["reviewed_count"] == 1
    assert status["outcomes_observed"] == 0
    assert status["exploration_rows"] == 0
    assert status["override_rate"] == pytest.approx(1.0)
    assert status["exploration_rate"] == pytest.approx(0.03)
    assert status["verdict"] == "DO NOT OPEN THE LOOP"
    assert status["retraining_permitted"] is False
    assert len(status["gate"]["conditions"]) == 4
    assert status["gate"]["failing_conditions"]


def test_status_fails_closed_when_the_gate_cannot_be_evaluated(session, tmp_path):
    from backend.app.services import relearning_service

    status = relearning_service.get_relearning_status(
        session, predictions_path=tmp_path / "does-not-exist.csv"
    )

    assert status["gate"]["unavailable"] is True
    assert status["verdict"] == "DO NOT OPEN THE LOOP"
    assert status["retraining_permitted"] is False


def test_attempt_retrain_refuses_with_a_structured_reason(session, tiny_predictions):
    from backend.app.services import relearning_service

    result = relearning_service.attempt_retrain(
        session, requested_by="test", predictions_path=tiny_predictions
    )

    assert result["performed"] is False
    assert result["permitted"] is False
    assert result["reason_code"] == relearning_service.REFUSAL_GATE_CLOSED
    assert "runaway feedback loop" in result["message"]
    assert result["verdict"] == "DO NOT OPEN THE LOOP"


def test_attempt_retrain_refuses_when_the_gate_is_unavailable(session, tmp_path):
    from backend.app.services import relearning_service

    result = relearning_service.attempt_retrain(
        session, predictions_path=tmp_path / "missing.csv"
    )
    assert result["reason_code"] == relearning_service.REFUSAL_GATE_UNAVAILABLE
    assert result["permitted"] is False


def test_service_uses_the_real_gate_rather_than_a_fork():
    """The thresholds must come from research/relearning/gate.py, not a copy."""
    from backend.app.services import relearning_service
    from research.relearning import gate

    assert relearning_service.gate_module is gate

    source = Path(relearning_service.__file__).read_text(encoding="utf-8")
    for constant in ("Z_THRESHOLD", "MIN_EXPLORATION_LABELS", "RANDOM_TRIALS"):
        assert f"{constant} =" not in source, f"{constant} must not be redefined here"


def test_relearning_service_contains_no_training_code():
    """attempt_retrain checks the gate; it must never learn anything."""
    from backend.app.services import relearning_service

    source = Path(relearning_service.__file__).read_text(encoding="utf-8").lower()
    body = source.split('"""', 2)[-1]  # skip the module docstring, which discusses them
    for forbidden in ("def train", ".fit(", "partial_fit", "reject_inference", "import sklearn"):
        assert forbidden not in body, f"relearning service must not contain {forbidden!r}"


def test_status_endpoint_is_registered_and_answers(session, tiny_predictions, monkeypatch):
    from backend.app.routers import relearning as relearning_route

    paths = {route.path for route in relearning_route.router.routes}
    assert "/relearning/status" in paths

    payload = relearning_route.relearning_status(session)
    assert payload["verdict"] == "DO NOT OPEN THE LOOP"
    assert "gate" in payload


def test_no_http_route_can_trigger_retraining():
    from backend.app.routers import relearning as relearning_route

    for route in relearning_route.router.routes:
        assert "retrain" not in route.path.lower()
        assert "train" not in route.path.lower()
