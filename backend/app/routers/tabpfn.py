"""Routes for the optional Colab-hosted TabPFN second-opinion endpoint.

Everything here is advisory. The engine's decision is already made and
persisted before any of these routes can run; a remote TabPFN score is stored
*next to* the decision (``_decision_meta["tabpfn_second_opinion"]``), never
inside it. TabPFN-2.5's licence makes its outputs non-commercial
(models/README.md), which is a second, independent reason it must stay off the
decision path.
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
from backend.app.services import remote_tabpfn_service
from backend.app.services.ml_service import get_predictor
from backend.app.services.remote_tabpfn_service import RemoteTabPFNError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tabpfn"])


@router.get("/tabpfn/status")
def tabpfn_status() -> dict[str, Any]:
    """Is the Colab endpoint configured, reachable, and what is it serving?"""
    return remote_tabpfn_service.remote_status()


def _feature_row_for(item: LoanApplication) -> dict[str, Any]:
    """Rebuild the exact feature row the serving model saw for this application.

    ``predict_application`` stores its sanitised row as
    ``_decision_meta.engineered_features`` — the post-imputation values in the
    artifact's vocabulary. Preferring that stored row means the second opinion
    scores the same numbers the decision was made on, not a re-derivation that
    could silently drift. Older rows without it fall back to reading
    ``feature_names`` straight off the raw payload.
    """
    input_data = dict(item.input_data or {})
    meta = dict(input_data.get("_decision_meta") or {})
    engineered = meta.get("engineered_features")
    if isinstance(engineered, dict) and engineered:
        return dict(engineered)

    predictor = get_predictor()
    return {name: input_data.get(name) for name in predictor.feature_names}


@router.post("/applications/{application_id}/tabpfn-second-opinion")
def tabpfn_second_opinion(application_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Score one existing application on the remote TabPFN and record the result.

    Returns 503 (not 500) when the tunnel is down or unconfigured: the caller
    asked for an optional instrument that is currently absent, and the UI
    should say so rather than report a server fault.
    """
    item = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail={"error": "Not found", "details": "Application not found"})

    row = _feature_row_for(item)
    try:
        prediction = remote_tabpfn_service.score_rows([row])[0]
    except RemoteTabPFNError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "Remote TabPFN unavailable", "details": str(exc)},
        ) from exc

    second_opinion = {
        "model": "TabPFN-2.5 (remote, Colab)",
        "pApprove": round(prediction["p_approve"], 6),
        "pDefault": round(prediction["p_default"], 6),
        "pMlAtDecision": round(float(item.ml_prob), 6) if item.ml_prob is not None else None,
        "delta": (
            round(prediction["p_approve"] - float(item.ml_prob), 6)
            if item.ml_prob is not None
            else None
        ),
        "scoredAt": datetime.now(timezone.utc).isoformat(),
        "advisoryOnly": True,
        "licence": "TabPFN-2.5 outputs are non-commercial; research/demo use only.",
    }

    # Persist next to — never inside — the decision. A failed write degrades to
    # an unrecorded opinion; it must not fail the request that already has it.
    try:
        merged_input = dict(item.input_data or {})
        meta = dict(merged_input.get("_decision_meta") or {})
        meta["tabpfn_second_opinion"] = second_opinion
        merged_input["_decision_meta"] = meta
        item.input_data = merged_input
        db.add(item)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.warning(
            "Could not persist TabPFN second opinion for application_id=%s", application_id, exc_info=True
        )

    return {
        "responseStatus": "success",
        "applicationId": application_id,
        "engineDecision": item.final_decision,
        "secondOpinion": second_opinion,
    }
