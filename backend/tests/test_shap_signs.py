"""SHAP factors keep their sign end to end.

Regression for the bug where ml_service returned |SHAP|, so every factor on an
APPROVE/DEFER read as a risk factor and the DEFER text called a CIBIL of 366
"generally favorable". Customer 105969 at Rs 2L/36m is a model DEFER whose top
factor (CIBIL 366) raises default risk and whose low DTI lowers it.
"""

from types import SimpleNamespace

from backend.app.routers.applications import _validate_payload
from backend.app.services.explainability_service import build_explainability_payload
from backend.app.services.ml_service import get_predictor


def _predict():
    payload = _validate_payload({"customer_id": "105969", "loan_amount": 200_000,
                                 "loan_tenure_months": 36, "end_use_declaration": True})
    return get_predictor().predict_application(payload)


def test_ml_service_keeps_shap_sign():
    result = _predict()
    impacts = {item["name"]: item["impact"] for item in result.shap_explanation}
    assert impacts["cibil_score"] > 0  # raises P(default)
    assert impacts["debt_to_income_ratio"] < 0  # lowers P(default)


def test_defer_text_matches_sign():
    result = _predict()
    assert result.decision == "DEFER"
    app_item = SimpleNamespace(
        id="test", final_decision="DEFER", ml_prob=result.p_ml, cbes_prob=0.5, confidence=0.5,
        input_data={"_decision_meta": {"shap_explanation": result.shap_explanation}},
    )
    payload = build_explainability_payload(app_item)
    risk = " ".join(payload["negativeFactors"])
    favourable = " ".join(payload["positiveFactors"])
    assert "CIBIL" in risk and "raised the risk" in risk
    assert "Debt To Income" in favourable and "favorable" in favourable


def test_decision_text_is_plain_and_jargon_moves_to_note():
    from backend.app.services.explainability_service import _plain_explanation

    result = _predict()
    app_item = SimpleNamespace(
        id="test", final_decision="DEFER", ml_prob=result.p_ml, cbes_prob=0.5, confidence=0.5,
        input_data={"_decision_meta": {"shap_explanation": result.shap_explanation,
                                       "decision_reason": "ml_uncertainty"}},
    )
    payload = build_explainability_payload(app_item)
    assert "uncertain range" in payload["explanation"]
    assert "SHAP" not in payload["explanation"] and "SHAP" in payload["explanationNote"]

    assert "approve automatically" in _plain_explanation("APPROVE", "ml_approve", 0.04)
    assert "4% chance" in _plain_explanation("APPROVE", "ml_approve", 0.04)
    assert "too high" in _plain_explanation("REJECT", "ml_reject", 0.62)
    # A policy reject must not also claim the model found the risk too high.
    assert "too high" not in _plain_explanation("REJECT", "affordability_reject", 0.02)


def test_application_response_keeps_economics_and_note():
    """LoanApplicationResponse used to strip these, so the economics card never showed."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.app.database import Base
    from backend.app.models import LoanApplication
    from backend.app.schemas import LoanApplicationResponse
    from backend.app.services.decision_service import build_application_response

    engine = create_engine("sqlite:///:memory:")  # never the real data/smartlend.db
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    result = _predict()
    item = LoanApplication(
        applicant_id="cust-test", ml_prob=result.p_ml, cbes_prob=0.5, final_decision="DEFER",
        confidence=0.5, documents=[],
        input_data={"loanAmount": 200_000, "_decision_meta": {
            "shap_explanation": result.shap_explanation, "decision_reason": "ml_uncertainty"}},
    )
    db.add(item)
    db.commit()

    decision = LoanApplicationResponse.model_validate(build_application_response(item)).model_dump()["decision"]
    assert decision["decisionEconomics"]["costFalseApproval"] == 0.45
    assert "SHAP" in decision["explanationNote"]
    assert decision["explanationSource"] == "shap"
