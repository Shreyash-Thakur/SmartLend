from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import uuid
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import LoanApplication
from app.schemas import (
    ApplicationExplainResponse,
    DashboardMetricsResponse,
    DocumentUploadResponse,
    LoanApplicationInput,
    LoanApplicationResponse,
    ManualDecisionRequest,
    PublicMetricsResponse,
)
from app.services.decision_service import apply_manual_decision, build_application_response, build_dashboard_metrics
from app.services.explainability_service import build_explainability_payload
from app.services.ml_service import get_predictor
from app.services.parser_service import parse_document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["applications"])


def _error_payload(error: str, details: str) -> dict[str, str]:
    return {"error": error, "details": details}


def _validate_payload(form_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(form_data)

    if "monthlyIncome" not in payload and "monthly_income" in payload:
        payload["monthlyIncome"] = payload["monthly_income"]
    if "loanAmount" not in payload and "loan_amount" in payload:
        payload["loanAmount"] = payload["loan_amount"]
    if "cibilScore" not in payload and "cibil_score" in payload:
        payload["cibilScore"] = payload["cibil_score"]

    try:
        validated = LoanApplicationInput.model_validate(payload)
        return validated.model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_error_payload("Validation failed", str(exc))) from exc


def _create_application_record(form_data: dict[str, Any], db: Session, documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    validated_payload = _validate_payload(form_data)
    predictor = get_predictor()

    try:
        prediction = predictor.predict_application(validated_payload)
    except Exception as exc:
        logger.exception("Prediction failure")
        raise HTTPException(status_code=400, detail=_error_payload("Prediction failed", str(exc))) from exc

    applicant_id = str(form_data.get("applicantId") or form_data.get("applicant_id") or f"cust-{uuid.uuid4().hex[:8]}")
    app_documents = documents or list(form_data.get("documents", []))

    decision_meta = {
        "approval_threshold": prediction.approval_threshold,
        "rejection_threshold": prediction.rejection_threshold,
        "cbes_components": prediction.cbes_components,
        "engineered_features": prediction.engineered_features,
    }

    app_item = LoanApplication(
        applicant_id=applicant_id,
        input_data={**validated_payload, "documents": app_documents, "_decision_meta": decision_meta},
        ml_prob=prediction.ml_prob,
        cbes_prob=prediction.cbes_prob,
        final_decision=prediction.final_decision,
        confidence=prediction.confidence,
        documents=app_documents,
    )

    try:
        db.add(app_item)
        db.commit()
        db.refresh(app_item)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Database write failure")
        raise HTTPException(status_code=400, detail=_error_payload("Database operation failed", str(exc))) from exc

    logger.info(
        "Decision generated | application_id=%s applicant_id=%s decision=%s ml_prob=%.4f cbes_prob=%.4f approval_threshold=%.4f rejection_threshold=%.4f",
        app_item.id,
        app_item.applicant_id,
        app_item.final_decision,
        app_item.ml_prob,
        app_item.cbes_prob,
        prediction.approval_threshold,
        prediction.rejection_threshold,
    )

    payload = build_application_response(app_item)
    payload["status"] = payload.get("status", "submitted")
    payload["responseStatus"] = "success"
    payload["ml_prob"] = round(app_item.ml_prob, 4)
    payload["cbes_prob"] = round(app_item.cbes_prob, 4)
    payload["decisionCode"] = app_item.final_decision
    payload["finalDecision"] = app_item.final_decision
    payload["confidence"] = round(app_item.confidence, 4)
    payload["decisionMeta"] = decision_meta
    return payload


@router.post("/applications", response_model=LoanApplicationResponse)
def create_application(form_data: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, Any]:
    return _create_application_record(form_data, db)


@router.post("/upload-form", response_model=LoanApplicationResponse)
async def upload_form(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict[str, Any]:
    suffix = Path(file.filename or "uploaded-form").suffix or ".bin"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)

    try:
        parsed_document = parse_document(temp_path, file.filename)

        critical_fields = ["monthly_income", "loan_amount", "age", "cibil_score"]
        confidence_map = parsed_document.get("confidence", {})
        weak_critical = [field for field in critical_fields if float(confidence_map.get(field, 0.0)) < 0.4]
        if weak_critical:
            raise HTTPException(
                status_code=400,
                detail=_error_payload(
                    "Invalid document format",
                    f"Could not extract required fields with confidence: {', '.join(weak_critical)}",
                ),
            )

        payload = _create_application_record(
            parsed_document["mappedData"],
            db,
            documents=[
                {
                    "id": f"doc-{uuid.uuid4().hex[:12]}",
                    "fileName": parsed_document["fileName"],
                    "documentType": parsed_document["documentType"],
                    "uploadedAt": datetime.now(timezone.utc).isoformat(),
                    "fileSize": temp_path.stat().st_size,
                    "extractedData": parsed_document["extractedData"],
                    "mappedData": parsed_document["mappedData"],
                    "confidence": parsed_document.get("confidence", {}),
                    "lowConfidenceFields": parsed_document.get("lowConfidenceFields", []),
                    "defaultsApplied": parsed_document.get("defaultsApplied", []),
                }
            ],
        )
        payload["parsedDocument"] = {
            "fileName": parsed_document["fileName"],
            "documentType": parsed_document["documentType"],
            "rawText": parsed_document["rawText"][:2000],
            "extractedData": parsed_document["extractedData"],
            "mappedData": parsed_document["mappedData"],
            "confidence": parsed_document.get("confidence", {}),
            "lowConfidenceFields": parsed_document.get("lowConfidenceFields", []),
            "defaultsApplied": parsed_document.get("defaultsApplied", []),
        }
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload form pipeline failed")
        raise HTTPException(status_code=400, detail=_error_payload("Invalid document format", f"Could not extract required fields: {exc}")) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("/applications", response_model=list[LoanApplicationResponse])
def list_applications(
    scope: str = Query(default="all"),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    if scope not in {"all", "org", "customer"}:
        raise HTTPException(status_code=400, detail=_error_payload("Bad input", "Invalid scope"))

    items = db.query(LoanApplication).order_by(LoanApplication.created_at.desc()).all()
    return [build_application_response(item) for item in items]


@router.get("/applications/{application_id}", response_model=LoanApplicationResponse)
def get_application(application_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Application not found"))
    return build_application_response(item)


@router.post("/applications/{application_id}/decision", response_model=LoanApplicationResponse)
def update_manual_decision(
    application_id: str,
    payload: ManualDecisionRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Application not found"))

    updated_payload = apply_manual_decision(item, payload.status, payload.notes)

    try:
        db.add(item)
        db.commit()
        db.refresh(item)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Manual decision DB update failed")
        raise HTTPException(status_code=400, detail=_error_payload("Database operation failed", str(exc))) from exc

    updated_payload["responseStatus"] = "success"
    updated_payload["decisionCode"] = item.final_decision
    return updated_payload


@router.post("/applications/{application_id}/documents", response_model=DocumentUploadResponse)
async def upload_application_document(
    application_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Application not found"))

    suffix = Path(file.filename or "uploaded-document").suffix or ".bin"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)

    try:
        parsed_document = parse_document(temp_path, file.filename)
        document_record = {
            "id": f"doc-{uuid.uuid4().hex[:12]}",
            "fileName": parsed_document["fileName"],
            "documentType": parsed_document["documentType"],
            "uploadedAt": datetime.now(timezone.utc).isoformat(),
            "fileSize": temp_path.stat().st_size,
            "extractedData": parsed_document["extractedData"],
            "mappedData": parsed_document["mappedData"],
            "confidence": parsed_document.get("confidence", {}),
            "lowConfidenceFields": parsed_document.get("lowConfidenceFields", []),
            "defaultsApplied": parsed_document.get("defaultsApplied", []),
        }

        existing_documents = list(item.documents or [])
        existing_documents.append(document_record)
        item.documents = existing_documents

        merged_input = dict(item.input_data or {})
        merged_input.setdefault("documents", [])
        merged_input["documents"] = existing_documents
        merged_input["documentExtraction"] = {
            "fileName": parsed_document["fileName"],
            "documentType": parsed_document["documentType"],
            "extractedData": parsed_document["extractedData"],
            "mappedData": parsed_document["mappedData"],
            "confidence": parsed_document.get("confidence", {}),
            "lowConfidenceFields": parsed_document.get("lowConfidenceFields", []),
            "defaultsApplied": parsed_document.get("defaultsApplied", []),
        }
        item.input_data = merged_input

        try:
            db.add(item)
            db.commit()
        except SQLAlchemyError as exc:
            db.rollback()
            logger.exception("Document DB update failed")
            raise HTTPException(status_code=400, detail=_error_payload("Database operation failed", str(exc))) from exc

        logger.info("Document uploaded | application_id=%s file=%s", application_id, parsed_document["fileName"])

        return {
            "fileName": parsed_document["fileName"],
            "documentType": parsed_document["documentType"],
            "uploadedAt": document_record["uploadedAt"],
            "extractedData": {
                **parsed_document["extractedData"],
                "confidence": parsed_document.get("confidence", {}),
                "lowConfidenceFields": parsed_document.get("lowConfidenceFields", []),
            },
            "mappedData": parsed_document["mappedData"],
            "fileSize": document_record["fileSize"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document upload parsing failed")
        raise HTTPException(status_code=400, detail=_error_payload("Invalid document format", f"Could not extract required fields: {exc}")) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("/applications/{application_id}/explain", response_model=ApplicationExplainResponse)
def explain_application(application_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Application not found"))

    payload = build_explainability_payload(item)
    payload["responseStatus"] = "success"
    return payload


@router.get("/dashboard-metrics", response_model=DashboardMetricsResponse)
def dashboard_metrics(db: Session = Depends(get_db)) -> dict[str, int]:
    items = db.query(LoanApplication).all()
    return build_dashboard_metrics(items)


@router.get("/public-metrics", response_model=PublicMetricsResponse)
def public_metrics(db: Session = Depends(get_db)) -> dict[str, int | float]:
    items = db.query(LoanApplication).all()
    metrics = build_dashboard_metrics(items)

    return {
        "applicationsProcessed": metrics["totalApplications"],
        "approvalSpeedup": 2.3,
        "accuracy": 94.2,
        "automationRate": metrics["automationRate"],
    }


@router.get("/trends")
def trends(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    items = db.query(LoanApplication).all()
    if not items:
        return []

    now = datetime.now(timezone.utc)
    windows = [
        ("Week 1", now - timedelta(days=28), now - timedelta(days=21)),
        ("Week 2", now - timedelta(days=21), now - timedelta(days=14)),
        ("Week 3", now - timedelta(days=14), now - timedelta(days=7)),
        ("Week 4", now - timedelta(days=7), now + timedelta(days=1)),
    ]

    points: list[dict[str, Any]] = []
    for label, start, end in windows:
        bucket = [
            app_item
            for app_item in items
            if app_item.created_at and start <= app_item.created_at.replace(tzinfo=timezone.utc) < end
        ]
        points.append(
            {
                "date": label,
                "count": len(bucket),
                "approved": sum(1 for app_item in bucket if app_item.final_decision == "APPROVE"),
                "rejected": sum(1 for app_item in bucket if app_item.final_decision == "REJECT"),
                "deferred": sum(1 for app_item in bucket if app_item.final_decision == "DEFER"),
            }
        )

    return points


@router.get("/metrics")
def decision_metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    items = db.query(LoanApplication).all()
    if not items:
        return {
            "responseStatus": "success",
            "total": 0,
            "approval_rate": 0,
            "rejection_rate": 0,
            "deferral_rate": 0,
            "avg_ml_prob": 0,
            "avg_cbes_prob": 0,
        }

    total = len(items)
    approved = sum(1 for item in items if item.final_decision == "APPROVE")
    rejected = sum(1 for item in items if item.final_decision == "REJECT")
    deferred = sum(1 for item in items if item.final_decision == "DEFER")

    return {
        "responseStatus": "success",
        "total": total,
        "approval_rate": round(approved / total, 4),
        "rejection_rate": round(rejected / total, 4),
        "deferral_rate": round(deferred / total, 4),
        "avg_ml_prob": round(sum(item.ml_prob for item in items) / total, 4),
        "avg_cbes_prob": round(sum(item.cbes_prob for item in items) / total, 4),
    }
