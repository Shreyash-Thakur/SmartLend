"""A reviewer's verdict must not be shown as the model's.

Regression for the review page reading "Model Recommendation: Reject" on a case
the engine had deferred and a human rejected: apply_manual_decision overwrote
final_decision and modelRecommendation was derived from it.
"""

from types import SimpleNamespace

from backend.app.services.decision_service import (
    apply_manual_decision,
    build_application_response,
    engine_decision,
)


def _deferred_app(**input_extra):
    return SimpleNamespace(
        id="app-test", final_decision="DEFER", ml_prob=0.81, cbes_prob=0.54, confidence=0.5,
        applicant_id="cust-test", created_at=None, documents=[],
        input_data={"loanAmount": 200_000, "_decision_meta": {"decision_reason": "ml_uncertainty"}, **input_extra},
    )


def test_manual_decision_keeps_engine_recommendation():
    item = _deferred_app()
    payload = apply_manual_decision(item, "rejected", "EMIs too high")

    assert item.final_decision == "REJECT"
    assert item.input_data["_engine_decision"] == "DEFER"
    assert item.input_data["_manual_decided_at"]
    assert payload["status"] == "rejected"
    assert payload["modelRecommendation"] == "deferred"
    assert payload["manualDecisionApplied"] is True

    after = build_application_response(item)
    assert after["modelRecommendation"] == "deferred" and after["status"] == "rejected"


def test_rows_adjudicated_before_the_fix_recover_engine_decision_from_reason():
    item = _deferred_app(_manual_decision_applied=True)
    item.final_decision = "REJECT"  # overwritten by the old code path
    assert engine_decision(item) == "DEFER"

    item.input_data["_decision_meta"]["decision_reason"] = "ml_approve"
    assert engine_decision(item) == "APPROVE"


def test_auto_decided_rows_report_their_own_decision():
    item = _deferred_app()
    item.final_decision = "APPROVE"
    assert engine_decision(item) == "APPROVE"
