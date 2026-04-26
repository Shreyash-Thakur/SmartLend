from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.public_api_service import (
    get_dashboard_metrics_payload,
    get_health_payload,
    get_model_comparison_payload,
    list_recent_applications,
    persist_prediction,
    validate_application_payload,
)

router = APIRouter(tags=["public"])


@router.get("/health")
def health() -> dict[str, object]:
    return get_health_payload()


@router.post("/predict")
def predict(payload: dict[str, object]) -> dict[str, object]:
    validated = validate_application_payload(payload)
    return persist_prediction(validated)


@router.get("/dashboard/metrics")
def dashboard_metrics() -> dict[str, object]:
    return get_dashboard_metrics_payload()


@router.get("/dashboard/model-comparison")
def dashboard_model_comparison() -> list[dict[str, object]]:
    return get_model_comparison_payload()


@router.get("/applications")
def recent_applications() -> list[dict[str, object]]:
    return list_recent_applications()
