from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
import csv

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from backend.app.database import SessionLocal
from backend.app.models import LoanApplication
from backend.app.schemas import LoanApplicationInput
from backend.app.services.decision_service import build_application_response
from backend.app.services.ml_service import get_predictor
from backend.app.services.model_analysis_service import get_model_analysis_payload

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_METRICS_PATH = ARTIFACTS_DIR / "model_metrics.csv"

PUBLIC_MODEL_NAME = "LogisticRegression"
PUBLIC_AUC = 0.710
PUBLIC_T_BASE = 0.55
PUBLIC_TAU_D = 0.43

_RECENT_APPLICATIONS: deque[dict[str, Any]] = deque(maxlen=50)
_STORE_LOCK = Lock()


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    aliases = {
        "monthly_income": "monthlyIncome",
        "loan_amount": "loanAmount",
        "cibil_score": "cibilScore",
        "annual_income": "annualIncome",
        "existing_emis": "existingEmis",
        "residential_assets_value": "residentialAssetsValue",
        "commercial_assets_value": "commercialAssetsValue",
        "bank_balance": "bankBalance",
        "loan_tenure": "loanTenure",
        "loan_purpose": "loanPurpose",
        "employment_type": "employmentType",
        "years_employed": "yearsOfEmployment",
    }
    for source_key, target_key in aliases.items():
        if target_key not in normalized and source_key in normalized:
            normalized[target_key] = normalized[source_key]
    return normalized


def validate_application_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        validated = LoanApplicationInput.model_validate(_normalize_payload(payload))
        return validated.model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"error": "Validation failed", "details": str(exc)}) from exc


def _history_item_from_application(app_item: LoanApplication) -> dict[str, Any]:
    payload = build_application_response(app_item)
    decision = payload.get("decision", {}) if isinstance(payload.get("decision", {}), dict) else {}
    return {
        "id": payload.get("id"),
        "timestamp": payload.get("createdAt"),
        "applicantId": payload.get("applicantId"),
        "applicantName": payload.get("applicantName"),
        "decision": payload.get("finalDecision") or app_item.final_decision,
        "confidence": float(payload.get("confidence", app_item.confidence)),
        "risk_score": float(decision.get("riskScore", max(0.0, 1.0 - float(app_item.ml_prob)))),
        "reason": str(decision.get("reason", decision.get("explanation", "model_ensemble"))),
    }


def _compute_model_rows() -> list[dict[str, Any]]:
    analysis = get_model_analysis_payload(limit=50000)
    prediction_summary = {item["model"]: item for item in analysis.get("modelPredictionSummary", []) if isinstance(item, dict) and item.get("model")}
    confusion_by_model = {item["model"]: item for item in analysis.get("confusionByModel", []) if isinstance(item, dict) and item.get("model")}

    rows: list[dict[str, Any]] = []
    if MODEL_METRICS_PATH.exists():
        with MODEL_METRICS_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                model_name = str(row.get("model") or row.get("Model") or "")
                if not model_name:
                    continue

                summary = prediction_summary.get(model_name, {})
                confusion = confusion_by_model.get(model_name, {})
                tp = float(confusion.get("tp", 0.0))
                fn = float(confusion.get("fn", 0.0))
                rows.append(
                    {
                        "model": model_name,
                        "accuracy": float(summary.get("accuracyFromCases", row.get("Accuracy", 0.0))),
                        "f1": float(confusion.get("f1FromCases", row.get("F1", 0.0))),
                        "recall": tp / max(tp + fn, 1.0),
                        "auc": float(row.get("roc_auc", row.get("AUC", 0.0))),
                        "std_auc": float(row.get("std_auc", row.get("CVStdAUC", 0.0))),
                    }
                )

    if len(rows) != 5:
        models = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
        for model_name in models:
            summary = prediction_summary.get(model_name, {})
            confusion = confusion_by_model.get(model_name, {})
            tp = float(confusion.get("tp", 0.0))
            fn = float(confusion.get("fn", 0.0))
            rows.append(
                {
                    "model": model_name,
                    "accuracy": float(summary.get("accuracyFromCases", 0.0)),
                    "f1": float(confusion.get("f1FromCases", 0.0)),
                    "recall": tp / max(tp + fn, 1.0),
                    "auc": 0.0,
                    "std_auc": 0.0,
                }
            )

    rows.sort(key=lambda item: item["auc"], reverse=True)
    return rows[:5]


def get_health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": PUBLIC_MODEL_NAME,
        "auc": PUBLIC_AUC,
        "t_base": PUBLIC_T_BASE,
        "tau_d": PUBLIC_TAU_D,
    }


def get_predict_payload(input_data: dict[str, Any], application_row: LoanApplication | None = None) -> dict[str, Any]:
    predictor = get_predictor()
    prediction = predictor.predict_application(input_data)

    response = {
        "decision": prediction.decision,
        "confidence": round(float(prediction.confidence), 4),
        "confidence_label": prediction.confidence_label,
        "risk_score": round(float(prediction.risk_score), 4),
        "p_ml": round(float(prediction.p_ml), 4),
        "p_cbes": round(float(prediction.p_cbes), 4),
        "disagreement": round(float(prediction.disagreement), 4),
        "decision_reason": str(prediction.decision_reason),
        "shap_explanation": [
            {
                "feature": str(item.get("feature") or item.get("name") or "feature"),
                "impact": float(item.get("impact", 0.0)),
            }
            for item in (prediction.shap_explanation or [])[:3]
        ],
        "cbes_breakdown": {
            "credit": float(prediction.cbes_breakdown.get("credit", 0.0)),
            "capacity": float(prediction.cbes_breakdown.get("capacity", 0.0)),
            "behaviour": float(prediction.cbes_breakdown.get("behaviour", 0.0)),
            "liquidity": float(prediction.cbes_breakdown.get("liquidity", 0.0)),
            "stability": float(prediction.cbes_breakdown.get("stability", 0.0)),
        },
    }

    if application_row is not None:
        _append_recent_application(application_row)

    return response


def persist_prediction(input_data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_application_payload(input_data)
    predictor = get_predictor()

    try:
        prediction = predictor.predict_application(validated)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(exc)}) from exc

    applicant_id = str(input_data.get("applicantId") or input_data.get("applicant_id") or f"cust-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    decision_meta = {
        "approval_threshold": float(prediction.approval_threshold),
        "rejection_threshold": float(prediction.rejection_threshold),
        "decision_reason": prediction.decision_reason,
        "disagreement": float(prediction.disagreement),
        "confidence_label": prediction.confidence_label,
        "risk_score": float(prediction.risk_score),
        "selected_model": prediction.selected_model,
        "cbes_components": prediction.cbes_components,
        "cbes_weights": prediction.cbes_weights,
        "engineered_features": prediction.engineered_features,
        "shap_explanation": prediction.shap_explanation,
    }

    app_item = LoanApplication(
        applicant_id=applicant_id,
        input_data={**validated, "_decision_meta": decision_meta},
        ml_prob=float(prediction.ml_prob),
        cbes_prob=float(prediction.cbes_prob),
        final_decision=str(prediction.final_decision),
        confidence=float(prediction.confidence),
        documents=[],
    )

    db = SessionLocal()
    try:
        db.add(app_item)
        db.commit()
        db.refresh(app_item)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(exc)}) from exc
    finally:
        db.close()

    _append_recent_application(app_item)
    return get_predict_payload(validated)


def get_dashboard_metrics_payload() -> dict[str, Any]:
    analysis = get_model_analysis_payload(limit=50000)
    model_rows = _compute_model_rows()
    baseline = next((row for row in model_rows if row["model"] == PUBLIC_MODEL_NAME), model_rows[0] if model_rows else {"model": PUBLIC_MODEL_NAME, "auc": PUBLIC_AUC, "accuracy": 0.0, "f1": 0.0, "recall": 0.0})

    cases = analysis.get("cases", []) if isinstance(analysis.get("cases", []), list) else []
    total_cases = len(cases)
    deferred_cases = sum(1 for item in cases if item.get("hybridDecision") == "DEFER")
    automated_cases = max(total_cases - deferred_cases, 0)
    correct_automated = sum(
        1
        for item in cases
        if item.get("hybridDecision") != "DEFER" and item.get("hybridDecision") == item.get("expectedDecision")
    )

    approve_true = sum(1 for item in cases if item.get("hybridDecision") == "APPROVE" and item.get("expectedDecision") == "APPROVE")
    approve_false = sum(1 for item in cases if item.get("hybridDecision") == "APPROVE" and item.get("expectedDecision") != "APPROVE")
    approve_missed = sum(1 for item in cases if item.get("hybridDecision") != "APPROVE" and item.get("expectedDecision") == "APPROVE")

    reject_true = sum(1 for item in cases if item.get("hybridDecision") == "REJECT" and item.get("expectedDecision") == "REJECT")
    reject_false = sum(1 for item in cases if item.get("hybridDecision") == "REJECT" and item.get("expectedDecision") != "REJECT")
    reject_missed = sum(1 for item in cases if item.get("hybridDecision") != "REJECT" and item.get("expectedDecision") == "REJECT")

    approve_precision = approve_true / max(approve_true + approve_false, 1)
    approve_recall = approve_true / max(approve_true + approve_missed, 1)
    reject_precision = reject_true / max(reject_true + reject_false, 1)
    reject_recall = reject_true / max(reject_true + reject_missed, 1)

    non_deferred_accuracy = correct_automated / max(automated_cases, 1)
    non_deferred_f1 = (2 * approve_precision * approve_recall / (approve_precision + approve_recall)) if (approve_precision + approve_recall) else 0.0

    return {
        "baseline": {
            "model": baseline.get("model", PUBLIC_MODEL_NAME),
            "auc": float(baseline.get("auc", PUBLIC_AUC)),
            "accuracy": float(baseline.get("accuracy", 0.0)),
            "f1": float(baseline.get("f1", 0.0)),
            "recall": float(baseline.get("recall", 0.0)),
        },
        "hybrid": {
            "auc": PUBLIC_AUC,
            "deferral_rate": round(deferred_cases / max(total_cases, 1), 3),
            "coverage": round(automated_cases / max(total_cases, 1), 3),
            "non_deferred_accuracy": round(non_deferred_accuracy, 3),
            "non_deferred_f1": round(non_deferred_f1, 3),
            "approve_precision": round(approve_precision, 3),
            "approve_recall": round(approve_recall, 3),
            "reject_precision": round(reject_precision, 3),
            "reject_recall": round(reject_recall, 3),
            "t_base": PUBLIC_T_BASE,
            "tau_d": PUBLIC_TAU_D,
        },
        "improvement": {
            "auc_delta": round(PUBLIC_AUC - float(baseline.get("auc", PUBLIC_AUC)), 3),
            "accuracy_delta": round(non_deferred_accuracy - float(baseline.get("accuracy", 0.0)), 3),
        },
    }


def get_model_comparison_payload() -> list[dict[str, Any]]:
    return _compute_model_rows()


def _append_recent_application(app_item: LoanApplication) -> None:
    with _STORE_LOCK:
        _RECENT_APPLICATIONS.appendleft(_history_item_from_application(app_item))


def seed_recent_applications() -> None:
    with _STORE_LOCK:
        _RECENT_APPLICATIONS.clear()
        db = SessionLocal()
        try:
            items = (
                db.query(LoanApplication)
                .order_by(LoanApplication.created_at.desc())
                .limit(50)
                .all()
            )
            for item in items:
                _RECENT_APPLICATIONS.append(_history_item_from_application(item))
        finally:
            db.close()


def list_recent_applications() -> list[dict[str, Any]]:
    with _STORE_LOCK:
        return list(_RECENT_APPLICATIONS)
