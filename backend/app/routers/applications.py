from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import re
import uuid
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.config import exploration_rate
from backend.app.database import get_db
from backend.app.models import DeferredReview, LoanApplication
from backend.app.schemas import (
    ApplicationExplainResponse,
    DashboardMetricsResponse,
    DecisionReportResponse,
    DocumentUploadResponse,
    LoanApplicationInput,
    LoanApplicationResponse,
    ManualDecisionRequest,
    ModelAnalysisResponse,
    PublicMetricsResponse,
    ReasonCodeCatalogResponse,
    StatsResponse,
)
from backend.app.services.customer_profile_service import resolve_application_payload
from backend.app.services.decision_report_service import build_decision_report, latest_review_for
from backend.app.services.decision_service import apply_manual_decision, build_application_response, build_dashboard_metrics
from backend.app.services.deferred_review_service import (
    maybe_route_to_exploration,
    record_deferral,
    record_human_decision,
)
from backend.app.services.explainability_service import build_explainability_payload
from backend.app.services.ml_service import get_predictor
from backend.app.services.model_analysis_service import get_model_analysis_payload
from backend.app.services.parser_service import parse_document
from backend.app.services.review_reason_codes import catalog as reason_code_catalog
from backend.app.services.training_data_service import get_training_application_by_id, get_training_applications

logger = logging.getLogger(__name__)

router = APIRouter(tags=["applications"])


CITY_TO_REGION: dict[str, str] = {
    "mumbai": "West",
    "pune": "West",
    "nagpur": "West",
    "ahmedabad": "West",
    "surat": "West",
    "jaipur": "North",
    "delhi": "North",
    "new delhi": "North",
    "lucknow": "North",
    "chandigarh": "North",
    "kolkata": "East",
    "bhubaneswar": "East",
    "patna": "East",
    "guwahati": "East",
    "chennai": "South",
    "bengaluru": "South",
    "bangalore": "South",
    "hyderabad": "South",
    "kochi": "South",
    "thiruvananthapuram": "South",
    "bhopal": "Central",
    "raipur": "Central",
    "indore": "Central",
    "kanpur": "North",
    "varanasi": "East",
    "mohali": "North",
    "noida": "North",
    "gurugram": "North",
    "gurgaon": "North",
    "visakhapatnam": "South",
    "vishakhapatnam": "South",
    "mysuru": "South",
    "mangalore": "South",
    "kozhikode": "South",
    "madurai": "South",
    "coimbatore": "South",
    "thane": "West",
    "nashik": "West",
    "vadodara": "West",
    "rajkot": "West",
    "siliguri": "East",
    "ranchi": "East",
    "jamshedpur": "East",
}

CITY_TO_STATE: dict[str, str] = {
    "mumbai": "Maharashtra",
    "pune": "Maharashtra",
    "nagpur": "Maharashtra",
    "ahmedabad": "Gujarat",
    "surat": "Gujarat",
    "jaipur": "Rajasthan",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "lucknow": "Uttar Pradesh",
    "chandigarh": "Chandigarh",
    "kolkata": "West Bengal",
    "bhubaneswar": "Odisha",
    "patna": "Bihar",
    "guwahati": "Assam",
    "chennai": "Tamil Nadu",
    "bengaluru": "Karnataka",
    "bangalore": "Karnataka",
    "hyderabad": "Telangana",
    "kochi": "Kerala",
    "thiruvananthapuram": "Kerala",
    "bhopal": "Madhya Pradesh",
    "raipur": "Chhattisgarh",
    "indore": "Madhya Pradesh",
    "kanpur": "Uttar Pradesh",
    "varanasi": "Uttar Pradesh",
    "mohali": "Punjab",
    "noida": "Uttar Pradesh",
    "gurugram": "Haryana",
    "gurgaon": "Haryana",
    "visakhapatnam": "Andhra Pradesh",
    "vishakhapatnam": "Andhra Pradesh",
    "mysuru": "Karnataka",
    "mangalore": "Karnataka",
    "kozhikode": "Kerala",
    "madurai": "Tamil Nadu",
    "coimbatore": "Tamil Nadu",
    "thane": "Maharashtra",
    "nashik": "Maharashtra",
    "vadodara": "Gujarat",
    "rajkot": "Gujarat",
    "siliguri": "West Bengal",
    "ranchi": "Jharkhand",
    "jamshedpur": "Jharkhand",
}

STATE_TO_REGION: dict[str, str] = {
    "andaman and nicobar islands": "South",
    "andaman and nicobar": "South",
    "andhra pradesh": "South",
    "arunachal pradesh": "East",
    "assam": "East",
    "bihar": "East",
    "chandigarh": "North",
    "chhattisgarh": "Central",
    "dadra and nagar haveli": "West",
    "dadra and nagar haveli and daman and diu": "West",
    "daman and diu": "West",
    "delhi": "North",
    "goa": "West",
    "gujarat": "West",
    "haryana": "North",
    "himachal pradesh": "North",
    "jammu and kashmir": "North",
    "jharkhand": "East",
    "karnataka": "South",
    "kerala": "South",
    "ladakh": "North",
    "lakshadweep": "South",
    "madhya pradesh": "Central",
    "maharashtra": "West",
    "manipur": "East",
    "meghalaya": "East",
    "mizoram": "East",
    "nagaland": "East",
    "odisha": "East",
    "orissa": "East",
    "puducherry": "South",
    "punjab": "North",
    "rajasthan": "North",
    "sikkim": "East",
    "tamil nadu": "South",
    "telangana": "South",
    "tripura": "East",
    "uttar pradesh": "North",
    "uttarakhand": "North",
    "west bengal": "East",
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

REGION_ALIASES: dict[str, str] = {
    "north": "North",
    "northern": "North",
    "south": "South",
    "southern": "South",
    "east": "East",
    "eastern": "East",
    "west": "West",
    "western": "West",
    "central": "Central",
    "centre": "Central",
}


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

    # Short-form submissions carry only a customer_id plus the ~14 questions the
    # bank cannot answer itself. Merge the on-file demographic + bureau block in
    # BEFORE validation, so the scoring engine still receives every input it
    # expects and the legacy camelCase fields are populated for stored rows.
    # A payload without a customer_id (legacy full-form callers, seeded rows,
    # parsed documents) passes through untouched.
    if payload.get("customer_id") or payload.get("customerId"):
        payload = resolve_application_payload(payload)
        if not payload.get("profile_resolved"):
            raise HTTPException(
                status_code=404,
                detail=_error_payload(
                    "Unknown customer",
                    f"No customer profile found for id {payload.get('customer_id') or payload.get('customerId')!r}. "
                    "The application cannot be scored without the bank's own demographic and bureau data.",
                ),
            )

    try:
        validated = LoanApplicationInput.model_validate(payload)
        return validated.model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_error_payload("Validation failed", str(exc))) from exc


# ===========================================================================
# Relearning-loop capture wiring
# ===========================================================================
# Three call sites, all of them WRITE-ONLY with respect to model behaviour:
#   1. `_capture_deferral`      - on DEFER, and on exploration-arm samples
#   2. `_capture_human_decision`- when an analyst approves/rejects
#   3. `_should_explore`        - the 3% control arm
#
# FAILURE ISOLATION IS THE LOAD-BEARING PROPERTY HERE. Capture is a research
# instrument bolted onto a lending decision. Every one of these helpers catches
# BaseException-minus-the-unrecoverables (i.e. `Exception`), rolls back its own
# transaction, logs a warning, and returns. A capture failure must degrade to a
# missing row in a research table, never to a customer who did not get their
# decision. That is why none of them re-raise, and why each is called AFTER the
# decision itself has been committed.
#
# Design rationale, gate conditions, and what a future developer must do before
# any of this may feed a trainer: docs/RELEARNING-LOOP.md.
# ===========================================================================

_SEGMENT_KEYS = ("region", "city", "gender", "maritalStatus", "employmentType")


def _applicant_segment(input_data: dict[str, Any]) -> dict[str, Any]:
    """Coarse segment attributes stored alongside a deferral.

    Spec section 2: uncertainty-based deferral is not automatically neutral and
    can disproportionately affect under-represented groups ("Unequal
    Uncertainty"), so segment is logged at defer time to make the
    disparate-deferral check possible later. Coarse only — an age band, not an
    age — because this table exists to be analysed, not to re-identify anyone.
    """
    segment: dict[str, Any] = {
        key: str(input_data.get(key)) for key in _SEGMENT_KEYS if input_data.get(key) is not None
    }
    age = input_data.get("age")
    if isinstance(age, (int, float)):
        segment["age_band"] = f"{int(age) // 10 * 10}-{int(age) // 10 * 10 + 9}"
    return segment


def _should_explore() -> bool:
    """Control arm: does this would-be-AUTO decision get a human look anyway?

    Only ever *adds* a review. `maybe_route_to_exploration` is a pure coin
    flip with no side effects, and this helper is called exclusively on the
    non-DEFER branch below, so by construction it cannot turn a DEFER into an
    auto-decision. A failure here means "no exploration this time", never a
    changed decision.
    """
    try:
        return maybe_route_to_exploration(rate=exploration_rate())
    except Exception:  # pragma: no cover - defensive; rate is clamped in config
        logger.warning("Exploration-arm sampling failed; treating as not sampled", exc_info=True)
        return False


def _capture_deferral(
    db: Session,
    prediction: Any,
    app_item: LoanApplication,
    *,
    exploration_flag: bool,
) -> None:
    """Write one `deferred_reviews` row for an application routed to a human.

    Called after the LoanApplication row is committed, so the decision is
    already durable before capture is attempted. Any failure — DB down, schema
    drift, a validation error in the capture layer — is swallowed with a
    warning: the application keeps its decision and the research table simply
    misses a row.
    """
    try:
        record_deferral(
            session=db,
            decision_result=prediction,
            application_id=app_item.id,
            engine_version=str(getattr(prediction, "engine_version", "unknown")),
            threshold_artifact_hash=str(getattr(prediction, "threshold_artifact_hash", "unknown")),
            t_base=float(getattr(prediction, "t_base", 0.50)),
            applicant_segment=_applicant_segment(app_item.input_data or {}),
            exploration_flag=exploration_flag,
        )
        db.commit()
        logger.info(
            "Relearning capture | application_id=%s decision=%s exploration=%s",
            app_item.id,
            getattr(prediction, "decision", "?"),
            exploration_flag,
        )
    except Exception:
        # NEVER re-raise. See the failure-isolation note at the top of this block.
        try:
            db.rollback()
        except Exception:
            logger.warning("Rollback after failed relearning capture also failed", exc_info=True)
        logger.warning(
            "Relearning capture failed for application_id=%s; decision is unaffected",
            app_item.id,
            exc_info=True,
        )


_HUMAN_DECISION_MAP = {"approved": "APPROVE", "rejected": "REJECT"}


def _capture_human_decision(db: Session, application_id: str, payload: ManualDecisionRequest) -> None:
    """Attach the analyst's verdict to the open capture row for this application.

    "deferred" is not a terminal reviewer verdict — the case is still under
    review — so it is skipped rather than recorded as a decision. Likewise an
    application with no open capture row (auto-decided, not sampled into the
    exploration arm) is a normal, expected miss and is logged at debug level.

    As with `_capture_deferral`, this runs after the manual decision has been
    committed and never re-raises.
    """
    try:
        human_decision = _HUMAN_DECISION_MAP.get(str(payload.status).lower())
        if human_decision is None:
            return

        review = (
            db.query(DeferredReview)
            .filter(
                DeferredReview.application_id == application_id,
                DeferredReview.human_decision.is_(None),
            )
            .order_by(DeferredReview.created_at.desc())
            .first()
        )
        if review is None:
            logger.debug(
                "No open deferred_reviews row for application_id=%s; nothing to capture",
                application_id,
            )
            return

        record_human_decision(
            session=db,
            review=review,
            human_decision=human_decision,
            reviewer_id=payload.reviewerId or "analyst-unattributed",
            time_spent_seconds=payload.timeSpentSeconds,
            human_reason_codes=payload.reasonCodes,
            human_free_text=payload.notes or None,
            reviewer_confidence=payload.reviewerConfidence,
        )
        db.commit()
        logger.info(
            "Relearning capture | reviewer decision recorded | application_id=%s decision=%s",
            application_id,
            human_decision,
        )
    except Exception:
        # NEVER re-raise: the analyst's decision is already committed.
        try:
            db.rollback()
        except Exception:
            logger.warning("Rollback after failed reviewer capture also failed", exc_info=True)
        logger.warning(
            "Reviewer capture failed for application_id=%s; the manual decision stands",
            application_id,
            exc_info=True,
        )


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
        "decision_reason": prediction.decision_reason,
        "disagreement": prediction.disagreement,
        "confidence_label": prediction.confidence_label,
        "risk_score": prediction.risk_score,
        "selected_model": prediction.selected_model,
        "cbes_components": prediction.cbes_components,
        "cbes_weights": prediction.cbes_weights,
        "engineered_features": prediction.engineered_features,
        "shap_explanation": prediction.shap_explanation,
    }

    # --- relearning loop: decide routing BEFORE persisting -----------------
    # A DEFER is already a human review. For everything else the exploration
    # arm gets a 3% coin flip; when it lands, the application is additionally
    # routed to a human even though the engine reached an AUTO decision. Note
    # the asymmetry that makes this safe: `_should_explore()` is only consulted
    # on the non-DEFER branch, so exploration can only ever ADD a review. It
    # can never convert a DEFER into an auto-decision, and it does not touch
    # `prediction.decision` — the engine's APPROVE/REJECT is preserved on the
    # row, which is precisely what makes these labels *un-selected* by the
    # router (spec section 3, the escape hatch from the selective-labels trap).
    is_defer = str(getattr(prediction, "decision", "")).upper() == "DEFER"
    exploration_flag = False if is_defer else _should_explore()
    routed_to_human_review = is_defer or exploration_flag
    decision_meta["exploration_flag"] = exploration_flag
    decision_meta["routed_to_human_review"] = routed_to_human_review

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

    # Capture happens only after the decision is durably committed above, and
    # cannot fail the request — see the failure-isolation block near the top of
    # this module.
    if routed_to_human_review:
        _capture_deferral(db, prediction, app_item, exploration_flag=exploration_flag)

    payload = build_application_response(app_item)
    payload["status"] = payload.get("status", "submitted")
    payload["responseStatus"] = "success"
    payload["ml_prob"] = round(app_item.ml_prob, 4)
    payload["cbes_prob"] = round(app_item.cbes_prob, 4)
    payload["cbes_score"] = round(app_item.cbes_prob, 4)
    payload["decisionCode"] = app_item.final_decision
    payload["finalDecision"] = app_item.final_decision
    payload["confidence"] = round(app_item.confidence, 4)
    payload["decisionMeta"] = decision_meta
    payload["explorationFlag"] = exploration_flag
    payload["routedToHumanReview"] = routed_to_human_review
    return payload


def _sort_applications(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(item: dict[str, Any]) -> datetime:
        created_at = item.get("createdAt")
        if isinstance(created_at, datetime):
            return created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
        if isinstance(created_at, str):
            try:
                parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return datetime.min.replace(tzinfo=timezone.utc)
        return datetime.min.replace(tzinfo=timezone.utc)

    return sorted(items, key=_sort_key, reverse=True)


def _normalize_region(input_data: dict[str, Any]) -> str:
    raw_region = str(input_data.get("region") or "").strip().lower()
    if raw_region in REGION_ALIASES:
        return REGION_ALIASES[raw_region]

    city = str(input_data.get("city") or "").strip().lower()
    if city in CITY_TO_REGION:
        return CITY_TO_REGION[city]

    state = _normalize_key(str(input_data.get("state") or input_data.get("province") or ""))
    if state in STATE_TO_REGION:
        return STATE_TO_REGION[state]

    return "Unknown"


def _normalize_city(input_data: dict[str, Any]) -> str:
    city = str(input_data.get("city") or "").strip()
    if not city:
        return "Unknown"
    return city.title()


def _normalize_state(input_data: dict[str, Any]) -> str:
    state = str(input_data.get("state") or input_data.get("province") or "").strip()
    if state:
        normalized = _normalize_key(state)
        if normalized in STATE_TO_REGION:
            return {
                "andaman and nicobar islands": "Andaman and Nicobar Islands",
                "andaman and nicobar": "Andaman and Nicobar",
                "arunachal pradesh": "Arunachal Pradesh",
                "dadra and nagar haveli": "Dadra and Nagar Haveli",
                "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
                "daman and diu": "Daman and Diu",
                "himachal pradesh": "Himachal Pradesh",
                "jammu and kashmir": "Jammu and Kashmir",
                "madhya pradesh": "Madhya Pradesh",
                "tamil nadu": "Tamil Nadu",
                "uttar pradesh": "Uttar Pradesh",
                "west bengal": "West Bengal",
            }[normalized]
        return state.title()

    city = str(input_data.get("city") or "").strip().lower()
    if city in CITY_TO_STATE:
        return CITY_TO_STATE[city]

    return "Unknown"


def _apply_decision_counters(bucket: dict[str, float | int], decision: str) -> None:
    decision_code = (decision or "").upper()
    if decision_code == "APPROVE":
        bucket["approved"] += 1
    elif decision_code == "REJECT":
        bucket["rejected"] += 1
    elif decision_code == "DEFER":
        bucket["deferred"] += 1


def _finalize_geo_bucket(bucket: dict[str, float | int]) -> None:
    applications = int(bucket["applications"])
    if applications <= 0:
        return

    bucket["approvalRate"] = round((int(bucket["approved"]) / applications) * 100, 2)
    bucket["rejectionRate"] = round((int(bucket["rejected"]) / applications) * 100, 2)
    bucket["deferralRate"] = round((int(bucket["deferred"]) / applications) * 100, 2)


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
    applicant_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    if scope not in {"all", "org", "customer"}:
        raise HTTPException(status_code=400, detail=_error_payload("Bad input", "Invalid scope"))

    query = db.query(LoanApplication).order_by(LoanApplication.created_at.desc())
    if scope == "customer" and applicant_id:
        query = query.filter(LoanApplication.applicant_id == applicant_id)

    db_items = query.all()
    api_items = [build_application_response(item) for item in db_items]

    for item in api_items:
        item["cbes_score"] = item.get("cbes_prob")

    if scope == "customer":
        return api_items

    training_items = get_training_applications()
    return _sort_applications([*api_items, *training_items])


@router.get("/applications/{application_id}", response_model=LoanApplicationResponse)
def get_application(application_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    training_item = get_training_application_by_id(application_id)
    if training_item is not None:
        return training_item

    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Application not found"))
    payload = build_application_response(item)
    payload["cbes_score"] = payload.get("cbes_prob")
    return payload


@router.post("/applications/{application_id}/decision", response_model=LoanApplicationResponse)
def update_manual_decision(
    application_id: str,
    payload: ManualDecisionRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    # Training-data applications are not persisted in the DB — handle them gracefully
    if application_id.startswith("train-"):
        training_item = get_training_application_by_id(application_id)
        if training_item is None:
            raise HTTPException(status_code=404, detail=_error_payload("Not found", "Training application not found"))
        # Map analyst decision to a decision code
        decision_map = {"approved": "APPROVE", "rejected": "REJECT", "deferred": "DEFER"}
        decision_code = decision_map.get(str(payload.status).lower(), "DEFER")
        status_out = str(payload.status).lower()
        response = dict(training_item)
        response["status"] = status_out
        response["finalDecision"] = decision_code
        response["decisionCode"] = decision_code
        response["responseStatus"] = "success"
        if isinstance(response.get("decision"), dict):
            response["decision"] = dict(response["decision"])
            response["decision"]["status"] = status_out
            response["decision"]["decidedBy"] = "analyst"
            response["decision"]["notes"] = payload.notes or ""
        return response

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

    # Relearning loop: the reviewer's APPROVE/REJECT, reason codes, confidence
    # and time-spent are attached to the original deferral row here — after the
    # decision is committed, and isolated from it. `human_decision` is captured
    # for override-rate monitoring (SR 11-7) and reviewer-consistency modelling
    # ONLY; no code path reads it as a training label. See docs/RELEARNING-LOOP.md.
    _capture_human_decision(db, application_id, payload)

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

    file_bytes = await file.read()
    file_name = file.filename or "uploaded-document"
    suffix = Path(file_name).suffix.lower() or ".bin"
    file_size = len(file_bytes)

    import base64
    content_type_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    mime = content_type_map.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(file_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    document_record = {
        "id": f"doc-{uuid.uuid4().hex[:12]}",
        "fileName": file_name,
        "documentType": suffix.lstrip("."),
        "uploadedAt": datetime.now(timezone.utc).isoformat(),
        "fileSize": file_size,
        "mimeType": mime,
        "dataUrl": data_url,
        "extractedData": {},
        "mappedData": {},
        "confidence": {},
        "lowConfidenceFields": [],
        "defaultsApplied": [],
    }

    existing_documents = list(item.documents or [])
    existing_documents.append(document_record)
    item.documents = existing_documents

    merged_input = dict(item.input_data or {})
    merged_input["documents"] = existing_documents
    item.input_data = merged_input

    try:
        db.add(item)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Document DB update failed")
        raise HTTPException(status_code=400, detail=_error_payload("Database operation failed", str(exc))) from exc

    logger.info("Document stored | application_id=%s file=%s size=%d", application_id, file_name, file_size)

    return {
        "fileName": file_name,
        "documentType": suffix.lstrip("."),
        "uploadedAt": document_record["uploadedAt"],
        "extractedData": {"confidence": {}, "lowConfidenceFields": []},
        "mappedData": {},
        "fileSize": file_size,
    }




@router.delete("/applications/{application_id}/documents/{document_id}")
def delete_application_document(
    application_id: str,
    document_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Application not found"))

    existing = list(item.documents or [])
    updated = [d for d in existing if d.get("id") != document_id]
    if len(updated) == len(existing):
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Document not found"))

    item.documents = updated
    merged_input = dict(item.input_data or {})
    merged_input["documents"] = updated
    item.input_data = merged_input

    try:
        db.add(item)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=_error_payload("Database error", str(exc))) from exc

    return {"responseStatus": "success", "deletedId": document_id}


@router.get("/applications/{application_id}/explain", response_model=ApplicationExplainResponse)
def explain_application(application_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Application not found"))

    payload = build_explainability_payload(item)
    payload["responseStatus"] = "success"
    return payload


@router.get("/applications/{application_id}/report", response_model=DecisionReportResponse)
def application_decision_report(application_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Complete audit record for one application: engine half + human half.

    An application that has not been reviewed — or was auto-decided and never
    captured at all — returns `humanReview: null` with a 200. Only a genuinely
    unknown application id is a 404. The report is exactly where an auditor
    looks *because* a case is unresolved, so "no reviewer yet" must never be an
    error response.
    """
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Application not found"))

    return build_decision_report(item, latest_review_for(db, application_id))


@router.get("/review-reason-codes", response_model=ReasonCodeCatalogResponse)
def review_reason_codes() -> dict[str, Any]:
    """The reviewer reason-code taxonomy.

    Served so the review screen and this backend cannot drift apart on the code
    strings that end up in `deferred_reviews.human_reason_codes`. The frontend
    ships the same list as a static fallback, so a failure here degrades to
    stale labels rather than a review screen with no checkboxes.
    """
    return {"reasonCodes": reason_code_catalog()}


@router.get("/dashboard-metrics", response_model=DashboardMetricsResponse)
def dashboard_metrics(db: Session = Depends(get_db)) -> dict[str, int]:
    items = db.query(LoanApplication).all()
    return build_dashboard_metrics(items)


@router.get("/public-metrics", response_model=PublicMetricsResponse)
def public_metrics(db: Session = Depends(get_db)) -> dict[str, int | float]:
    db_items = [build_application_response(item) for item in db.query(LoanApplication).all()]
    for item in db_items:
        item["cbes_score"] = item.get("cbes_prob")
    all_items = [*db_items, *get_training_applications()]
    total = len(all_items)
    deferred = sum(1 for item in all_items if item.get("finalDecision") == "DEFER")
    automation_rate = round(((total - deferred) / total) * 100) if total else 0

    analysis_payload = get_model_analysis_payload(limit=100)
    summary = analysis_payload.get("summary", {}) if isinstance(analysis_payload, dict) else {}
    automated_accuracy = float(summary.get("automatedAccuracy", 0.0) or 0.0)
    deferral_rate = float(summary.get("deferralRate", 0.0) or 0.0)

    # Hybrid quality assumes deferred cases receive analyst adjudication.
    analyst_resolution_quality = 92.0
    hybrid_quality = automated_accuracy + ((deferral_rate / 100.0) * max(0.0, analyst_resolution_quality - automated_accuracy))
    quality_score = round(hybrid_quality, 2) if total else 0.0

    return {
        "applicationsProcessed": total,
        "approvalSpeedup": round(1 + (automation_rate / 100), 2),
        "accuracy": quality_score,
        "automationRate": automation_rate,
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


@router.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)) -> dict[str, int | float]:
    db_items = [build_application_response(item) for item in db.query(LoanApplication).all()]
    for item in db_items:
        item["cbes_score"] = item.get("cbes_prob")

    applications = [*db_items, *get_training_applications()]
    total = len(applications)
    if total == 0:
        return {
            "totalApplications": 0,
            "approved": 0,
            "rejected": 0,
            "deferred": 0,
            "approvalRate": 0.0,
            "rejectionRate": 0.0,
            "deferralRate": 0.0,
            "averageCBES": 0.0,
            "averageMLProbability": 0.0,
        }

    approved = sum(1 for item in applications if item.get("finalDecision") == "APPROVE")
    rejected = sum(1 for item in applications if item.get("finalDecision") == "REJECT")
    deferred = sum(1 for item in applications if item.get("finalDecision") == "DEFER")
    avg_cbes = sum(float(item.get("cbes_score", 0.0) or 0.0) for item in applications) / total
    avg_ml = sum(float(item.get("ml_prob", 0.0) or 0.0) for item in applications) / total

    return {
        "totalApplications": total,
        "approved": approved,
        "rejected": rejected,
        "deferred": deferred,
        "approvalRate": round((approved / total) * 100, 2),
        "rejectionRate": round((rejected / total) * 100, 2),
        "deferralRate": round((deferred / total) * 100, 2),
        "averageCBES": round(avg_cbes, 4),
        "averageMLProbability": round(avg_ml, 4),
    }


@router.get("/region-metrics")
def region_metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    metrics: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "applications": 0,
            "approved": 0,
            "rejected": 0,
            "deferred": 0,
            "approvalRate": 0.0,
            "rejectionRate": 0.0,
            "deferralRate": 0.0,
        }
    )

    items = db.query(LoanApplication).all()
    for app_item in items:
        input_data = app_item.input_data or {}
        region = _normalize_region(input_data)
        bucket = metrics[region]
        bucket["applications"] += 1

        _apply_decision_counters(bucket, app_item.final_decision)

    training_items = get_training_applications()
    for training_item in training_items:
        input_data = dict(training_item.get("applicationData") or {})
        region = _normalize_region(input_data)
        bucket = metrics[region]
        bucket["applications"] += 1
        _apply_decision_counters(bucket, str(training_item.get("finalDecision") or ""))

    for bucket in metrics.values():
        _finalize_geo_bucket(bucket)

    return {
        "regions": dict(sorted(metrics.items())),
        "totalApplications": len(items) + len(training_items),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/location-metrics")
def location_metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    template = {
        "applications": 0,
        "approved": 0,
        "rejected": 0,
        "deferred": 0,
        "approvalRate": 0.0,
        "rejectionRate": 0.0,
        "deferralRate": 0.0,
    }
    areas: dict[str, dict[str, float | int]] = defaultdict(lambda: dict(template))
    states: dict[str, dict[str, float | int]] = defaultdict(lambda: dict(template))
    cities: dict[str, dict[str, float | int]] = defaultdict(lambda: dict(template))

    db_items = db.query(LoanApplication).all()
    for app_item in db_items:
        input_data = app_item.input_data or {}
        area = _normalize_region(input_data)
        state = _normalize_state(input_data)
        city = _normalize_city(input_data)

        for bucket in (areas[area], states[state], cities[city]):
            bucket["applications"] += 1
            _apply_decision_counters(bucket, app_item.final_decision)

    training_items = get_training_applications()
    for training_item in training_items:
        input_data = dict(training_item.get("applicationData") or {})
        area = _normalize_region(input_data)
        state = _normalize_state(input_data)
        city = _normalize_city(input_data)
        decision_code = str(training_item.get("finalDecision") or "")

        for bucket in (areas[area], states[state], cities[city]):
            bucket["applications"] += 1
            _apply_decision_counters(bucket, decision_code)

    for collection in (areas, states, cities):
        for bucket in collection.values():
            _finalize_geo_bucket(bucket)

    def _sorted(payload: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
        return dict(
            sorted(
                payload.items(),
                key=lambda item: int(item[1].get("applications", 0)),
                reverse=True,
            )
        )

    return {
        "areas": _sorted(areas),
        "states": _sorted(states),
        "cities": _sorted(cities),
        "totalApplications": len(db_items) + len(training_items),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/model-analysis", response_model=ModelAnalysisResponse)
def model_analysis(limit: int = Query(default=300, ge=1, le=50000)) -> dict[str, Any]:
    payload = get_model_analysis_payload(limit=limit)
    if not payload["models"] and not payload["cases"]:
        raise HTTPException(status_code=404, detail=_error_payload("Not found", "Model analysis artifacts are unavailable"))
    return payload


class ActiveModelRequest(BaseModel):
    model_name: str

@router.post("/model-analysis/active")
def set_active_model(request: ActiveModelRequest) -> dict[str, Any]:
    from backend.app.services.ml_service import ARTIFACTS_DIR
    active_model_file = ARTIFACTS_DIR / "active_model.txt"
    active_model_file.write_text(request.model_name)
    return {"status": "ok", "active_model": request.model_name}

@router.get("/model-analysis/active")
def get_active_model() -> dict[str, Any]:
    from backend.app.services.ml_service import ARTIFACTS_DIR
    active_model_file = ARTIFACTS_DIR / "active_model.txt"
    if active_model_file.exists():
        return {"active_model": active_model_file.read_text().strip()}
    return {"active_model": "LogisticRegression"}

