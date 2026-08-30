"""Read-only status and gate enforcement for the relearning loop.

This module is the single honest answer to the question "can we retrain on the
human-reviewed cases yet?". It counts what the capture layer has collected, it
runs the REAL gate (imported from ``research.relearning.gate`` — the thresholds
are never reimplemented or forked here), and it refuses.

WHAT THIS MODULE IS NOT
-----------------------
There is no training code here, and there must never be. ``attempt_retrain``
exists specifically so that a future implementer who wants to open the loop has
one obvious place to go, finds the gate already checked, and cannot bypass it by
accident. It returns a structured refusal; it does not fit anything.

The reasoning behind the refusal is in
``docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md``
section 3 and restated in ``docs/RELEARNING-LOOP.md``: retraining on cases the
router selected is a runaway feedback loop (Ensign et al.), outcomes are only
observed for the approved subset (selective labels — Lakkaraju et al., KDD
2017), and reviewer judgments carry their own bias (Madras et al., NeurIPS
2018). This project's deferral rule is currently documented as *worse* than
random, which makes the first of those an active hazard rather than a
theoretical one.

FAIL-CLOSED
-----------
Every failure mode here resolves to "DO NOT OPEN THE LOOP". A missing
prediction artifact, an unreadable database, an exception inside the gate —
none of them produce a permissive verdict. Refusal is the default, not the
error path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.config import exploration_rate
from backend.app.models import DeferredReview

# The real gate. Imported, never reimplemented: the four conditions and their
# thresholds (Z_THRESHOLD, MIN_EXPLORATION_LABELS, the Tasche bound) live in
# exactly one place so a copy here cannot drift loose from the CLI verdict.
from research.relearning import gate as gate_module

logger = logging.getLogger(__name__)

CLOSED_VERDICT = "DO NOT OPEN THE LOOP"

# Structured refusal codes returned by attempt_retrain().
REFUSAL_GATE_CLOSED = "GATE_CLOSED"
REFUSAL_GATE_UNAVAILABLE = "GATE_UNAVAILABLE"
NOT_IMPLEMENTED_BY_DESIGN = "GATE_OPEN_BUT_NO_RETRAINING_IMPLEMENTATION"


# ---------------------------------------------------------------------------
# capture-table counts
# ---------------------------------------------------------------------------


def _capture_counts(session: Session) -> dict[str, Any]:
    """Counts over ``deferred_reviews``. Pure aggregation, no row contents."""
    total = session.query(func.count(DeferredReview.review_id)).scalar() or 0
    reviewed = (
        session.query(func.count(DeferredReview.review_id))
        .filter(DeferredReview.human_decision.isnot(None))
        .scalar()
        or 0
    )
    outcomes_observed = (
        session.query(func.count(DeferredReview.review_id))
        .filter(DeferredReview.outcome_censored.is_(False))
        .filter(DeferredReview.realized_outcome.isnot(None))
        .scalar()
        or 0
    )
    exploration_rows = (
        session.query(func.count(DeferredReview.review_id))
        .filter(DeferredReview.exploration_flag.is_(True))
        .scalar()
        or 0
    )
    # Gate condition 3 counts UN-SELECTED labels: exploration rows that have an
    # actually-observed outcome. An exploration row with a censored outcome is
    # not yet a label, and a deferred row's outcome is router-selected, so
    # neither counts here.
    exploration_labels = (
        session.query(func.count(DeferredReview.review_id))
        .filter(DeferredReview.exploration_flag.is_(True))
        .filter(DeferredReview.outcome_censored.is_(False))
        .filter(DeferredReview.realized_outcome.isnot(None))
        .scalar()
        or 0
    )

    # Override rate (SR 11-7 monitoring). Denominator excludes rows where the
    # engine had no lean (true grey zone, agreed_with_engine IS NULL) —
    # counting those as agreements or overrides would both be fabrications.
    evaluable = (
        session.query(func.count(DeferredReview.review_id))
        .filter(DeferredReview.agreed_with_engine.isnot(None))
        .scalar()
        or 0
    )
    overrides = (
        session.query(func.count(DeferredReview.review_id))
        .filter(DeferredReview.agreed_with_engine.is_(False))
        .scalar()
        or 0
    )

    return {
        "rows_captured": int(total),
        "reviewed_count": int(reviewed),
        "outcomes_observed": int(outcomes_observed),
        "exploration_rows": int(exploration_rows),
        "exploration_labels": int(exploration_labels),
        "override_rate": (float(overrides) / evaluable) if evaluable else None,
        "override_denominator": int(evaluable),
        "override_count": int(overrides),
    }


def _empty_counts() -> dict[str, Any]:
    return {
        "rows_captured": 0,
        "reviewed_count": 0,
        "outcomes_observed": 0,
        "exploration_rows": 0,
        "exploration_labels": 0,
        "override_rate": None,
        "override_denominator": 0,
        "override_count": 0,
        "counts_unavailable": True,
    }


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def _evaluate_gate(exploration_labels: int, predictions_path: Path | None) -> dict[str, Any]:
    """Run the real four-condition gate, or fail closed with a reason."""
    path = predictions_path or gate_module.DEFAULT_PREDICTIONS
    try:
        frame = gate_module.load_predictions(path)
        return gate_module.evaluate_gate(frame, exploration_labels=exploration_labels)
    except Exception as exc:
        logger.warning("Gate evaluation unavailable (%s); failing closed", exc, exc_info=True)
        return {
            "spec": gate_module.evaluate_gate.__module__,
            "n_rows": 0,
            "conditions": [],
            "failing_conditions": [1, 2, 3, 4],
            "all_conditions_pass": False,
            "verdict": CLOSED_VERDICT,
            "unavailable": True,
            "unavailable_reason": f"{type(exc).__name__}: {exc}",
        }


def _condition_summaries(gate_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Trim the gate's full metric dump to a status line per condition."""
    return [
        {
            "condition": cond["condition"],
            "name": cond["name"],
            "status": cond["status"],
            "reason": cond["reason"],
        }
        for cond in gate_result.get("conditions", [])
    ]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def get_relearning_status(
    session: Session | None = None,
    *,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    """What the loop has captured, and whether it may be opened.

    Never raises: a broken database yields zeroed counts flagged
    ``counts_unavailable``, and a broken gate yields the closed verdict flagged
    ``unavailable``. Callers get a verdict in every case, and that verdict is
    ``DO NOT OPEN THE LOOP`` unless all four conditions genuinely passed.
    """
    if session is None:
        counts = _empty_counts()
    else:
        try:
            counts = _capture_counts(session)
        except Exception:
            logger.warning("Could not read deferred_reviews counts", exc_info=True)
            counts = _empty_counts()

    gate_result = _evaluate_gate(counts["exploration_labels"], predictions_path)
    permitted = bool(gate_result.get("all_conditions_pass", False))

    return {
        **counts,
        "exploration_rate": exploration_rate(),
        "gate": {
            "verdict": gate_result.get("verdict", CLOSED_VERDICT),
            "all_conditions_pass": permitted,
            "failing_conditions": gate_result.get("failing_conditions", [1, 2, 3, 4]),
            "conditions": _condition_summaries(gate_result),
            "rows_evaluated": gate_result.get("n_rows", 0),
            "unavailable": bool(gate_result.get("unavailable", False)),
            "unavailable_reason": gate_result.get("unavailable_reason"),
            "spec": (
                "docs/superpowers/specs/"
                "2026-08-30-cbes-research-and-explanation-design.md section 3"
            ),
        },
        "verdict": gate_result.get("verdict", CLOSED_VERDICT),
        "retraining_permitted": permitted,
        "docs": "docs/RELEARNING-LOOP.md",
    }


def attempt_retrain(
    session: Session | None = None,
    *,
    requested_by: str = "unknown",
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    """Check the gate and refuse, with a reason.

    THIS FUNCTION CONTAINS NO TRAINING CODE AND MUST NEVER CONTAIN ANY. It is a
    checkpoint, not an entry point. If you are here because you want to open the
    loop: the gate below is the thing you must satisfy first, in the real world,
    with real numbers — not by editing this function.

    Returns a structured result. ``permitted`` is False whenever any of the four
    gate conditions fails or the gate could not be evaluated at all. Even when
    the gate passes, this returns ``performed: False`` with
    ``GATE_OPEN_BUT_NO_RETRAINING_IMPLEMENTATION``: passing the gate authorises a
    *written, reviewed* scorecard redevelopment (spec section 3, "rebuild
    periodically as a versioned scorecard redevelopment — never continuously"),
    not an automatic one triggered from a web request.
    """
    status = get_relearning_status(session, predictions_path=predictions_path)
    gate_info = status["gate"]

    if gate_info["unavailable"]:
        reason_code = REFUSAL_GATE_UNAVAILABLE
        message = (
            "The relearning gate could not be evaluated "
            f"({gate_info['unavailable_reason']}). Refusing: the loop fails closed, "
            "so an unevaluable gate is treated exactly like a failed one."
        )
    elif not gate_info["all_conditions_pass"]:
        reason_code = REFUSAL_GATE_CLOSED
        failing = ", ".join(str(n) for n in gate_info["failing_conditions"])
        message = (
            f"Gate conditions {failing} do not hold. No retraining on "
            "human-reviewed or deferred cases is permitted. Retraining on the "
            "cases a router selected — while that router is documented as worse "
            "than random at selecting hard cases — is the runaway feedback loop "
            "(Ensign et al.), and outcomes observed only for the approved subset "
            "are the selective-labels trap (Lakkaraju et al., KDD 2017)."
        )
    else:
        reason_code = NOT_IMPLEMENTED_BY_DESIGN
        message = (
            "All four gate conditions pass. That authorises a written, reviewed "
            "scorecard redevelopment that models the selection mechanism and "
            "reviewer bias — not an automatic retrain triggered from here. See "
            "docs/RELEARNING-LOOP.md, 'Opening the loop safely'."
        )

    logger.warning(
        "attempt_retrain refused | requested_by=%s reason=%s verdict=%s",
        requested_by,
        reason_code,
        status["verdict"],
    )

    return {
        "performed": False,
        "permitted": gate_info["all_conditions_pass"],
        "reason_code": reason_code,
        "message": message,
        "requested_by": requested_by,
        "verdict": status["verdict"],
        "gate": gate_info,
        "docs": "docs/RELEARNING-LOOP.md",
    }
