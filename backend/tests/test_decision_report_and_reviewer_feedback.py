"""End-to-end proof that the reviewer-feedback loop closes.

The chain under test is the whole point of the capture layer, and until now it
was only ever asserted one link at a time:

    DEFER  ->  a `deferred_reviews` row exists
      ->  a reviewer decision carrying structured reason codes UPDATES that row
      ->  `GET /api/relearning/status` counts reflect it
      ->  `GET /api/applications/{id}/report` returns BOTH halves

`test_relearning_loop_closes_end_to_end` walks that chain in one test so the
evidence is concrete rather than inferred from four passing unit tests that
never meet.

The rest of the module covers the report endpoint's edges (no reviewer yet,
auto-decided application, unknown id) and the reason-code taxonomy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.services.decision_engine import _BLEND_ALPHA
from backend.app.models import DeferredReview, LoanApplication
from backend.app.routers import applications as applications_router
from backend.app.schemas import ManualDecisionRequest
from backend.app.services import relearning_service, review_reason_codes
from backend.app.services.decision_engine import hybrid_decision


# --------------------------------------------------------------------------
# fixtures — same wiring style as test_relearning_wiring.py
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
    "firstName": "Reviewer",
    "lastName": "Loop",
    "loanAmount": 500000,
    "monthlyIncome": 90000,
    "age": 35,
    "region": "west",
    "gender": "female",
}


def _prediction(p_ml: float, p_cbes: float, tau_d: float, t_base: float = 0.50):
    result = hybrid_decision(
        p_ml=p_ml,
        p_cbes=p_cbes,
        tau_d=tau_d,
        t_base=t_base,
        cbes_breakdown={"credit": 0.4, "capacity": 0.5, "behaviour": 0.6,
                        "stability": 0.7, "region": 0.5},
    )
    result.engineered_features = {}
    result.cbes_weights = {"credit": 0.3, "capacity": 0.3, "behaviour": 0.2,
                           "stability": 0.1, "region": 0.1}
    result.cbes_components = result.cbes_breakdown
    result.selected_model = "TestModel"
    result.t_base = t_base
    result.tau_d = tau_d
    result.engine_version = "test-engine/v1"
    result.threshold_artifact_hash = "deadbeefcafe0000"
    return result


def _defer_prediction():
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
    def _wire(prediction, *, explore: bool = False):
        monkeypatch.setattr(
            applications_router, "get_predictor", lambda: _FakePredictor(prediction)
        )
        monkeypatch.setattr(
            applications_router, "maybe_route_to_exploration", lambda rate: explore
        )
        return prediction

    return _wire


@pytest.fixture
def tiny_predictions(tmp_path) -> Path:
    """A small predictions artifact so the gate runs fast."""
    path = tmp_path / "predictions.csv"
    rows = ["y_true,best_model_prob,final_decision,approval_threshold"]
    for i in range(200):
        y = 1 if i % 10 else 0
        prob = 0.9 if y else 0.2
        decision = "DEFER" if i % 3 == 0 else ("APPROVE" if y else "REJECT")
        rows.append(f"{y},{prob},{decision},0.5")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _create(session, form=None):
    return applications_router._create_application_record(dict(form or FORM), session)


def _review(session, application_id, **kwargs):
    return applications_router.update_manual_decision(
        application_id, ManualDecisionRequest(**kwargs), session
    )


def _report(session, application_id):
    return applications_router.application_decision_report(application_id, session)


# ==========================================================================
# THE LOOP
# ==========================================================================


def test_relearning_loop_closes_end_to_end(session, wire, tiny_predictions):
    """DEFER -> capture row -> reviewer feedback -> status counts -> report."""
    prediction = wire(_defer_prediction())

    # --- link 1: a DEFER creates a capture row ----------------------------
    created = _create(session)
    application_id = created["id"]
    assert created["finalDecision"] == "DEFER"
    assert created["routedToHumanReview"] is True

    row = session.query(DeferredReview).one()
    assert row.application_id == application_id
    assert row.human_decision is None
    assert row.human_reason_codes is None

    # Before any human touches it, the report already answers — with the
    # engine half only. This is the state an auditor sees most often.
    pre = _report(session, application_id)
    assert pre["engine"]["decision"] == "DEFER"
    assert pre["humanReview"] is None

    # Status agrees: captured but not reviewed.
    before = relearning_service.get_relearning_status(session, predictions_path=tiny_predictions)
    assert before["rows_captured"] == 1
    assert before["reviewed_count"] == 0
    assert before["override_denominator"] == 0

    # --- link 2: the reviewer's structured feedback updates THAT row ------
    reason_codes = ["REJ-EMI-BURDEN", "REJ-DOCS", "GEN-MODEL-MISMATCH"]
    response = _review(
        session,
        application_id,
        status="rejected",
        notes="Salary credits do not reconcile with the submitted payslips.",
        reviewerId="analyst-11",
        reviewerConfidence=4,
        timeSpentSeconds=214.0,
        reasonCodes=reason_codes,
    )
    assert response["decisionCode"] == "REJECT"

    session.refresh(row)
    assert session.query(DeferredReview).count() == 1, "feedback must UPDATE, not insert"
    assert row.human_decision == "REJECT"
    assert row.human_reason_codes == reason_codes
    assert row.reviewer_id == "analyst-11"
    assert row.reviewer_confidence == 4
    assert row.time_spent_seconds == pytest.approx(214.0)
    assert row.human_free_text.startswith("Salary credits")
    # p_blend 0.70 sat above t_approve, so the engine leaned APPROVE and this
    # is a genuine override.
    assert row.agreed_with_engine is False
    assert row.override_direction == "engine_approve_to_human_reject"

    # --- link 3: /api/relearning/status reflects the new counts -----------
    after = relearning_service.get_relearning_status(session, predictions_path=tiny_predictions)
    assert after["rows_captured"] == 1
    assert after["reviewed_count"] == before["reviewed_count"] + 1
    assert after["override_denominator"] == 1
    assert after["override_count"] == 1
    assert after["override_rate"] == pytest.approx(1.0)
    # Capturing feedback must not unlock anything.
    assert after["retraining_permitted"] is False
    assert after["verdict"] == "DO NOT OPEN THE LOOP"

    # --- link 4: the report returns both halves --------------------------
    report = _report(session, application_id)

    engine = report["engine"]
    assert report["applicationId"] == application_id
    assert engine["pMl"] == pytest.approx(prediction.p_ml, abs=1e-4)
    assert engine["pCbes"] == pytest.approx(prediction.p_cbes, abs=1e-4)
    assert engine["pBlend"] == pytest.approx(prediction.p_blend, abs=1e-4)
    assert engine["pBlendSource"] == "captured"
    assert engine["confidence"] == pytest.approx(prediction.confidence, abs=1e-4)
    assert engine["thresholds"]["approve"] == pytest.approx(prediction.t_approve, abs=1e-4)
    assert engine["thresholds"]["reject"] == pytest.approx(prediction.t_reject, abs=1e-4)
    assert engine["thresholds"]["base"] == pytest.approx(0.50)
    assert engine["cbesBreakdown"]["credit"] == pytest.approx(0.4)
    assert engine["engineVersion"] == "test-engine/v1"
    assert engine["thresholdArtifactHash"] == "deadbeefcafe0000"
    assert isinstance(engine["topFactors"], list)

    human = report["humanReview"]
    assert human is not None
    assert human["reviewId"] == row.review_id
    assert human["reviewerId"] == "analyst-11"
    assert human["decision"] == "REJECT"
    assert human["reviewerConfidence"] == 4
    assert human["timeSpentSeconds"] == pytest.approx(214.0)
    assert human["agreedWithEngine"] is False
    assert human["overrideDirection"] == "engine_approve_to_human_reject"
    assert [entry["code"] for entry in human["reasonCodes"]] == reason_codes
    # Codes are expanded to readable labels for the audit reader.
    assert human["reasonCodes"][0]["label"] == "Income insufficient for EMI"
    assert human["reasonCodes"][0]["direction"] == "reject"
    assert human["reasonCodes"][2]["direction"] == "either"

    # The applicant's own data is part of the audit record.
    assert report["application"]["loanAmount"] == pytest.approx(500000)
    assert "_decision_meta" not in report["application"]["data"]


# ==========================================================================
# report endpoint edges
# ==========================================================================


def test_report_for_an_auto_decided_application_has_no_human_half(session, wire):
    wire(_approve_prediction(), explore=False)
    created = _create(session)

    report = _report(session, created["id"])

    assert session.query(DeferredReview).count() == 0
    assert report["humanReview"] is None
    assert report["engine"]["decision"] == "APPROVE"
    # With no capture row there is nothing recorded to read p_blend from, so it
    # is reconstructed from the engine's own blend weight and labelled as such.
    assert report["engine"]["pBlendSource"] == "derived"
    # Derive from the engine constant rather than hardcoding the weights:
    # this assertion previously pinned 0.75/0.25 and broke the moment the
    # CBES blend weight was retuned, even though the code was correct.
    expected_blend = (1.0 - _BLEND_ALPHA) * 0.90 + _BLEND_ALPHA * 0.85
    assert report["engine"]["pBlend"] == pytest.approx(expected_blend, abs=1e-4)
    assert report["engine"]["thresholds"]["approve"] is not None


def test_report_for_an_unknown_application_is_a_404(session):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _report(session, "app-does-not-exist")
    assert excinfo.value.status_code == 404


def test_report_endpoint_is_registered():
    paths = {route.path for route in applications_router.router.routes}
    assert "/applications/{application_id}/report" in paths
    assert "/review-reason-codes" in paths


def test_report_survives_an_application_with_no_decision_meta(session, wire):
    """Seeded and legacy rows have no `_decision_meta`; the report must not blow up."""
    wire(_defer_prediction())
    created = _create(session)
    item = session.query(LoanApplication).filter_by(id=created["id"]).one()
    item.input_data = {k: v for k, v in (item.input_data or {}).items() if k != "_decision_meta"}
    session.add(item)
    session.commit()

    report = _report(session, created["id"])
    assert report["engine"]["decision"] == "DEFER"
    assert report["engine"]["pMl"] is not None


# ==========================================================================
# reason-code taxonomy
# ==========================================================================


def test_taxonomy_covers_both_directions_and_a_neutral_group():
    directions = {entry["direction"] for entry in review_reason_codes.catalog()}
    assert directions == {"approve", "reject", "either"}
    for entry in review_reason_codes.catalog():
        assert entry["code"] and entry["label"] and entry["description"]


def test_unknown_reason_codes_are_surfaced_not_dropped():
    """A legacy or future code must still appear in the audit record."""
    described = review_reason_codes.describe_reason_codes(["CP-01", "APR-GUARANTOR"])
    assert [entry["code"] for entry in described] == ["CP-01", "APR-GUARANTOR"]
    assert described[0]["direction"] == "unknown"
    assert described[0]["label"] == "CP-01"
    assert described[1]["label"] == "Guarantor strength"


def test_reason_codes_are_not_validated_server_side(session, wire):
    """An off-taxonomy code must never cost a reviewer their decision."""
    wire(_defer_prediction())
    created = _create(session)

    response = _review(session, created["id"], status="approved", reasonCodes=["NOT-A-REAL-CODE"])

    assert response["decisionCode"] == "APPROVE"
    assert session.query(DeferredReview).one().human_reason_codes == ["NOT-A-REAL-CODE"]


def test_report_layer_contains_no_training_code():
    """Same boundary as the capture layer: this is an audit view, not a trainer."""
    from backend.app.services import decision_report_service

    for module in (decision_report_service, review_reason_codes):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        body = source.split('"""', 2)[-1]  # skip the docstring, which discusses them
        for forbidden in ("def train", "retrain", ".fit(", "partial_fit", "reject_inference"):
            assert forbidden not in body, f"{module.__name__} must not contain {forbidden!r}"
