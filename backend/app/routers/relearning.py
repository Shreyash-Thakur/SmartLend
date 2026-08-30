"""Read-only observability for the relearning loop.

One endpoint, `GET /api/relearning/status`. It reports what the capture layer
has collected and the live verdict from the real gate
(`research/relearning/gate.py`).

There is deliberately no endpoint that triggers retraining. `attempt_retrain`
in the service layer exists to refuse, and it is not reachable over HTTP.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.services.relearning_service import get_relearning_status

router = APIRouter(tags=["relearning"])


@router.get("/relearning/status")
def relearning_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Capture counts, override rate, and the current four-condition gate verdict.

    `get_relearning_status` never raises and fails closed, so this endpoint
    always answers, and always answers "DO NOT OPEN THE LOOP" unless every gate
    condition genuinely passed.
    """
    return get_relearning_status(db)
