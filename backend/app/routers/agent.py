"""Routes for the optional reviewer-briefing agent.

Same contract philosophy as the voice module: a missing key or an unreachable
API is a 503 with instructions, never a 500 — the agent is an optional layer on
top of a system that must keep deciding loans without it.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import LoanApplication
from backend.app.services import underwriting_agent_service
from backend.app.services.decision_report_service import build_decision_report, latest_review_for
from backend.app.services.underwriting_agent_service import (
    AgentBriefingError,
    AgentUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])


@router.get("/agent/status")
def agent_status() -> dict[str, Any]:
    return {
        "configured": underwriting_agent_service.configured(),
        "model": underwriting_agent_service.MODEL,
        "detail": (
            "ready"
            if underwriting_agent_service.configured()
            else "ANTHROPIC_API_KEY is not set; add it to .env to enable reviewer briefings."
        ),
    }


@router.post("/applications/{application_id}/agent-briefing")
def agent_briefing(application_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Generate (and cache) a reviewer briefing for one application.

    The briefing is stored next to the decision in `_decision_meta`, so a page
    reload shows the existing briefing instead of paying for a regeneration;
    POST always regenerates, GET the application report to read the stored one.
    """
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail={"error": "Not found", "details": "Application not found"})

    report = build_decision_report(item, latest_review_for(db, application_id))
    try:
        briefing = underwriting_agent_service.generate_briefing(report)
    except AgentUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "Agent unavailable", "details": str(exc)},
        ) from exc
    except AgentBriefingError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "Agent produced no briefing", "details": str(exc)},
        ) from exc

    briefing["generatedAt"] = datetime.now(timezone.utc).isoformat()

    # Persist next to — never inside — the decision. A failed write degrades to
    # an uncached briefing; it must not fail the request that already has it.
    try:
        merged_input = dict(item.input_data or {})
        meta = dict(merged_input.get("_decision_meta") or {})
        meta["agent_briefing"] = briefing
        merged_input["_decision_meta"] = meta
        item.input_data = merged_input
        db.add(item)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.warning(
            "Could not persist agent briefing for application_id=%s", application_id, exc_info=True
        )

    return {"responseStatus": "success", "applicationId": application_id, "briefing": briefing}
