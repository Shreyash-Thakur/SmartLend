"""The structured reason-code taxonomy a human reviewer ticks when deciding a
deferred case.

WHY A FIXED TAXONOMY AND NOT FREE TEXT
--------------------------------------
`deferred_reviews.human_free_text` already exists and is kept, but prose is not
analysable. The two things the capture layer is actually for — the
disparate-deferral check and reviewer-consistency modelling (Madras et al.,
*Predict Responsibly*, NeurIPS 2018) — both need the reviewer's *stated grounds*
as a categorical variable that can be counted across reviewers and segments. A
decision recorded with no reason code is a row that can never contribute to
either analysis, which is why the UI blocks submission until at least one box is
ticked (see docs/RELEARNING-LOOP.md §3).

Codes are split by the direction they support so the review screen can show the
reviewer only the boxes that make sense for the verdict they picked, plus the
direction-neutral ones. That split is presentation guidance, NOT validation:
`ManualDecisionRequest.reasonCodes` is deliberately un-validated server-side so
that (a) an older client, a bulk action, or a future taxonomy revision can never
have a reviewer's decision rejected because of a bookkeeping mismatch, and
(b) codes recorded under an earlier taxonomy stay readable. Unknown codes are
surfaced verbatim by the report builder rather than dropped.

NOT A TRAINING SIGNAL. Like every other column on `deferred_reviews`, reason
codes are captured for audit and monitoring only. No code path reads them as a
label; the gate in `research/relearning/gate.py` decides when, if ever, any of
this may inform a model.
"""

from __future__ import annotations

from typing import Any

# Direction the code argues *for*. "either" codes are direction-neutral and are
# always offered.
DIRECTION_APPROVE = "approve"
DIRECTION_REJECT = "reject"
DIRECTION_EITHER = "either"

# code -> (label, direction, helper text)
REASON_CODES: dict[str, dict[str, str]] = {
    # --- supporting APPROVE ------------------------------------------------
    "APR-EMP-STABLE": {
        "label": "Stable employment",
        "direction": DIRECTION_APPROVE,
        "description": "Continuous employment / business vintage supports servicing ability.",
    },
    "APR-REPAY-STRONG": {
        "label": "Strong repayment history",
        "direction": DIRECTION_APPROVE,
        "description": "Existing and closed credit lines were serviced without delinquency.",
    },
    "APR-COLLATERAL": {
        "label": "Adequate collateral",
        "direction": DIRECTION_APPROVE,
        "description": "Security offered covers the exposure at an acceptable margin.",
    },
    "APR-INCOME-HEADROOM": {
        "label": "Sufficient income headroom",
        "direction": DIRECTION_APPROVE,
        "description": "Residual income after the proposed EMI leaves comfortable margin.",
    },
    "APR-RELATIONSHIP": {
        "label": "Long-standing customer",
        "direction": DIRECTION_APPROVE,
        "description": "Established relationship with observed conduct on prior facilities.",
    },
    "APR-GUARANTOR": {
        "label": "Guarantor strength",
        "direction": DIRECTION_APPROVE,
        "description": "Guarantor's standing materially reduces the residual risk.",
    },
    # --- supporting REJECT -------------------------------------------------
    "REJ-OBLIGATIONS": {
        "label": "High existing obligations",
        "direction": DIRECTION_REJECT,
        "description": "Current debt service already absorbs too much of income.",
    },
    "REJ-THIN-FILE": {
        "label": "Thin or no credit file",
        "direction": DIRECTION_REJECT,
        "description": "Too little bureau history to evidence repayment behaviour.",
    },
    "REJ-DELINQUENCY": {
        "label": "Recent delinquency",
        "direction": DIRECTION_REJECT,
        "description": "Overdue or written-off accounts within the recent window.",
    },
    "REJ-EMI-BURDEN": {
        "label": "Income insufficient for EMI",
        "direction": DIRECTION_REJECT,
        "description": "Proposed instalment exceeds demonstrable repayment capacity.",
    },
    "REJ-DOCS": {
        "label": "Unverifiable documents",
        "direction": DIRECTION_REJECT,
        "description": "Submitted proofs are missing, inconsistent, or could not be verified.",
    },
    "REJ-UTILISATION": {
        "label": "High credit utilisation",
        "direction": DIRECTION_REJECT,
        "description": "Revolving lines are drawn close to their sanctioned limits.",
    },
    # --- direction-neutral -------------------------------------------------
    "GEN-MODEL-MISMATCH": {
        "label": "Model score inconsistent with the file",
        "direction": DIRECTION_EITHER,
        "description": "The engine's score does not match what the documents show.",
    },
    "GEN-POLICY-EXCEPTION": {
        "label": "Policy exception applied",
        "direction": DIRECTION_EITHER,
        "description": "Decision rests on an approved deviation from standard policy.",
    },
    "GEN-OTHER": {
        "label": "Other",
        "direction": DIRECTION_EITHER,
        "description": "Grounds not covered above — describe them in the notes.",
    },
}

# Human decision (as stored on `deferred_reviews.human_decision`) -> the
# direction whose codes are relevant. Used by the UI to filter; never enforced.
DECISION_TO_DIRECTION = {
    "APPROVE": DIRECTION_APPROVE,
    "REJECT": DIRECTION_REJECT,
}


def describe_reason_code(code: str) -> dict[str, str]:
    """Expand a stored code into `{code, label, direction, description}`.

    An unrecognised code — a legacy value, or one from a newer taxonomy — is
    returned verbatim with `direction = "unknown"` rather than discarded. The
    audit record must never quietly lose a reason the human actually recorded.
    """
    entry = REASON_CODES.get(code)
    if entry is None:
        return {
            "code": code,
            "label": code,
            "direction": "unknown",
            "description": "Reason code not present in the current taxonomy.",
        }
    return {"code": code, **entry}


def describe_reason_codes(codes: Any) -> list[dict[str, str]]:
    """Expand a stored `human_reason_codes` JSON value. Tolerates None/garbage."""
    if not isinstance(codes, (list, tuple)):
        return []
    return [describe_reason_code(str(code)) for code in codes]


def catalog() -> list[dict[str, str]]:
    """The full taxonomy as a flat, ordered list — the shape the UI renders."""
    return [{"code": code, **entry} for code, entry in REASON_CODES.items()]
