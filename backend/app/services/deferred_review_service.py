"""Capture layer for the relearning loop: records human-review cases and the
small random exploration arm. WRITES ONLY — it never trains, and it never
changes what the engine decided.

WHY RETRAINING IS GATED (read before wiring anything here into a trainer)
------------------------------------------------------------------------
Per docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md
section 3, nothing may be retrained on these rows until the deferral rule is
independently validated as better-than-random at selecting hard cases. This
project's deferral rule is currently documented as *worse* than random, and
three separate failure modes make `human_decision` unusable as a label today:

1. Runaway feedback loop. A system whose own router decides which cases get
   new labels, then retrains on those labels, reinforces its initial bias
   regardless of the true underlying rate (Ensign et al., "Runaway Feedback
   Loops in Predictive Policing"; FAccT 2023, "A Classification of Feedback
   Loops ... in Automated Decision-Making Systems"). Retraining on the output
   of a router already known to be worse than random would amplify an
   actively harmful selection policy.
2. Selective labels. Outcomes are observed only for the approved subset, so
   training or evaluating on them yields biased risk estimates over the full
   population (Lakkaraju et al., "The Selective Labels Problem", KDD 2017;
   Kleinberg, Lakkaraju et al., "Human Decisions and Machine Predictions",
   QJE). The exploration arm below is the only source of *un-selected* labels
   and the intended escape hatch — it is not yet large enough to be one.
3. Human labels carry their own bias. Feeding reviewer judgments back as
   training labels can cause non-convergent training and amplify human bias
   when treated as ground truth (Madras et al., "Predict Responsibly",
   NeurIPS 2018; "Designing Closed Human-in-the-loop Deferral Pipelines").

Therefore this module deliberately does NOT contain, and must not grow:

* any automatic retraining trigger, scheduled or volume-based;
* any code path that reads `human_decision` (or `realized_outcome`) as a
  training label;
* any reject-inference imputation of outcomes for deferred/rejected cases;
* any online/continuous learning;
* any CBES weight update driven by deferral outcomes (spec section 1.11).

The four gate conditions that must ALL hold before that changes are listed in
spec section 3, "Gate to open the loop". If you are here to connect this table
to a trainer: check the gate first, in writing, with the diagnostics named
there. There is no innocent way to wire this up.

SCOPE OF EFFECTS
----------------
Every function here either writes a capture row or returns a boolean. The one
function that can influence routing, `maybe_route_to_exploration`, only ever
converts a would-be-auto decision into a human review — never the reverse —
and is not wired into the live decision path (future work per the plan).
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from backend.app.models import DeferredReview

# Spec section 3: "Route a small random 2-5% of would-be-auto-decided
# applications into human review anyway." 3% is the midpoint default.
DEFAULT_EXPLORATION_RATE: float = 0.03
EXPLORATION_RATE_BOUNDS: tuple[float, float] = (0.02, 0.05)

# Process-wide RNG used when no rng is injected. Tests inject a seeded
# random.Random so exploration sampling is deterministic.
_DEFAULT_RNG = random.Random()

OVERRIDE_TO_APPROVE = "engine_reject_to_human_approve"
OVERRIDE_TO_REJECT = "engine_approve_to_human_reject"

_VALID_HUMAN_DECISIONS = {"APPROVE", "REJECT"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def engine_implied_decision(
    p_blend: float, t_approve: float, t_reject: float
) -> str | None:
    """The direction the engine was leaning when it deferred.

    `DecisionResult` records a DEFER, not an APPROVE/REJECT, so the baseline
    for override monitoring is derived from the blended probability against
    the two thresholds: at or above `t_approve` the engine leaned APPROVE, at
    or below `t_reject` it leaned REJECT, and in between (the grey zone) it
    genuinely had no lean — returns None, and no agreement is recorded rather
    than a fabricated one.
    """
    if p_blend >= t_approve:
        return "APPROVE"
    if p_blend <= t_reject:
        return "REJECT"
    return None


def record_deferral(
    session: Session,
    decision_result: Any,
    application_id: str,
    engine_version: str,
    threshold_artifact_hash: str,
    t_base: float,
    applicant_segment: dict | None = None,
    exploration_flag: bool = False,
) -> DeferredReview:
    """Write one capture row for a decision routed to a human.

    Normally called only for `decision == "DEFER"`. The one exception is the
    exploration arm (`exploration_flag=True`), where a would-be-auto-decided
    application is routed to review anyway — those rows carry the engine's
    APPROVE/REJECT decision, which is exactly what makes them un-selected
    labels. Any other combination raises `ValueError` so this can never
    silently log a case that was not actually reviewed by a human.

    `t_base` is not exposed on `DecisionResult` (`hybrid_decision` derives
    `t_approve`/`t_reject` from it but does not return it), so the caller —
    which already holds it — threads it through explicitly rather than having
    it approximated from the shifted thresholds.

    The row is added and flushed, not committed: the caller owns the
    transaction.
    """
    decision = getattr(decision_result, "decision", None)
    if decision != "DEFER" and not exploration_flag:
        raise ValueError(
            f"record_deferral called with decision={decision!r} and "
            "exploration_flag=False; only DEFER decisions (or exploration-arm "
            "samples) are recorded here."
        )
    if decision == "DEFER" and exploration_flag:
        raise ValueError(
            "record_deferral called with decision='DEFER' and "
            "exploration_flag=True; the exploration arm samples would-be-auto "
            "decisions only, so flagging a genuine DEFER would corrupt the "
            "control arm."
        )

    row = DeferredReview(
        application_id=application_id,
        decision_reason=decision_result.decision_reason,
        p_ml=float(decision_result.p_ml),
        p_cbes=float(decision_result.p_cbes),
        p_blend=float(decision_result.p_blend),
        disagreement=float(decision_result.disagreement),
        confidence=float(decision_result.confidence),
        t_approve=float(decision_result.t_approve),
        t_reject=float(decision_result.t_reject),
        cbes_breakdown_json=dict(decision_result.cbes_breakdown),
        engine_version=engine_version,
        threshold_artifact_hash=threshold_artifact_hash,
        t_base=float(t_base),
        applicant_segment_json=dict(applicant_segment) if applicant_segment else None,
        exploration_flag=bool(exploration_flag),
        outcome_censored=True,
    )
    session.add(row)
    session.flush()
    return row


# Name used by the implementation plan; kept as an alias so either name works.
record_deferred_review = record_deferral


def record_human_decision(
    session: Session,
    review: DeferredReview,
    human_decision: str,
    reviewer_id: str,
    reviewed_at: datetime | None = None,
    time_spent_seconds: float | None = None,
    human_reason_codes: list | None = None,
    human_free_text: str | None = None,
    reviewer_confidence: int | None = None,
) -> DeferredReview:
    """Attach the reviewer's decision to an existing capture row.

    Sets `agreed_with_engine` and `override_direction` for override-rate
    monitoring (SR 11-7). This is monitoring metadata only — see the module
    docstring: `human_decision` is never read as a training label.
    """
    if human_decision not in _VALID_HUMAN_DECISIONS:
        raise ValueError(
            f"human_decision={human_decision!r}; expected one of {sorted(_VALID_HUMAN_DECISIONS)}"
        )
    if reviewer_confidence is not None and not 1 <= int(reviewer_confidence) <= 5:
        raise ValueError("reviewer_confidence must be an integer 1-5 (Madras et al. scale)")

    review.human_decision = human_decision
    review.reviewer_id = reviewer_id
    review.reviewed_at = reviewed_at or _now()
    review.time_spent_seconds = (
        float(time_spent_seconds) if time_spent_seconds is not None else None
    )
    review.human_reason_codes = list(human_reason_codes) if human_reason_codes else None
    review.human_free_text = human_free_text
    review.reviewer_confidence = (
        int(reviewer_confidence) if reviewer_confidence is not None else None
    )

    implied = engine_implied_decision(review.p_blend, review.t_approve, review.t_reject)
    if implied is None:
        # True grey zone: the engine had no lean, so there is nothing to agree
        # or disagree with. Recording False here would inflate the override rate.
        review.agreed_with_engine = None
        review.override_direction = None
    elif implied == human_decision:
        review.agreed_with_engine = True
        review.override_direction = None
    else:
        review.agreed_with_engine = False
        review.override_direction = (
            OVERRIDE_TO_APPROVE if human_decision == "APPROVE" else OVERRIDE_TO_REJECT
        )

    session.add(review)
    session.flush()
    return review


def record_outcome(
    session: Session,
    review: DeferredReview,
    realized_outcome: int | None = None,
    observed_at: datetime | None = None,
    censored: bool | None = None,
) -> DeferredReview:
    """Record the realized repayment outcome (1 = default/bad, 0 = good).

    Pass `realized_outcome=None` (or `censored=True`) for the censored case —
    no outcome observed yet, or never observable because the application was
    not funded. Censoring is stored explicitly rather than imputed: the
    missingness is MNAR and any imputation here would be reject inference,
    which spec section 3 forbids.
    """
    if realized_outcome is not None and int(realized_outcome) not in (0, 1):
        raise ValueError("realized_outcome must be 0, 1, or None (censored)")

    is_censored = realized_outcome is None if censored is None else bool(censored)
    if is_censored:
        review.realized_outcome = None
        review.outcome_observed_at = None
        review.outcome_censored = True
    else:
        if realized_outcome is None:
            raise ValueError("censored=False requires a realized_outcome of 0 or 1")
        review.realized_outcome = int(realized_outcome)
        review.outcome_observed_at = observed_at or _now()
        review.outcome_censored = False

    session.add(review)
    session.flush()
    return review


def maybe_route_to_exploration(
    rate: float = DEFAULT_EXPLORATION_RATE,
    rng: random.Random | None = None,
) -> bool:
    """Control arm: should this would-be-auto-decided application be routed to
    a human anyway?

    Returns True with probability `rate` (default 3%, the midpoint of the
    spec's 2-5% band). Pure function with no side effects — the caller decides
    what True means, and by construction True can only ever turn an auto
    decision INTO a review, never turn a review into an auto decision.

    `rng` accepts an injected `random.Random`, so a seeded instance makes
    sampling deterministic in tests; the process-wide RNG is used otherwise.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"exploration rate must be in [0, 1], got {rate}")
    generator = rng if rng is not None else _DEFAULT_RNG
    return generator.random() < rate
