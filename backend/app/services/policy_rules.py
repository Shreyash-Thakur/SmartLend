"""Affordability gate: the bank-policy check that runs AFTER the model.

Why this exists
---------------
The serving model never sees whether the *requested* loan is affordable. In
the Home Credit training data larger credit went to safer clients, so the
model learned "bigger loan -> lower risk" and approves more applicants as the
requested amount grows (measured on the 510 seeded customers: 65% approve at
Rs 1L/36m, 89% at Rs 25L/24m, 0.4% reject at every size). Real lenders do not
leave this to a scorecard; they apply a FOIR (fixed obligations to income
ratio) policy. This module is that policy, kept deliberately lenient:

  FOIR > 2.0          -> REJECT  (EMIs would be more than twice the income:
                                  cannot be repaid, no model score overrides it)
  1.0 < FOIR <= 2.0   -> an APPROVE becomes DEFER (a human checks income the
                                  bank cannot see); DEFER/REJECT unchanged
  otherwise           -> no effect

The thresholds were picked so deferrals do not shrink at ordinary loan sizes.
Measured on the 510 seeded customers, % approve/defer/reject before -> after
(reports/affordability_gate_impact.json):
  Rs 2L / 36m : A67 D33 R0.4 -> A55 D43 R2
  Rs 5L / 60m : A71 D29 R0.4 -> A30 D61 R9
  Rs 25L / 24m: A89 D11 R0.2 -> R100

FOIR = existing debt-to-income (profile `debt_to_income_ratio`)
     + 12 * (new EMI + EMIs at other lenders) / (annual income + 12 * additional monthly income)

The model itself, its thresholds and the deferral router are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FOIR_REJECT = 2.0
FOIR_REVIEW = 1.0
DEFAULT_ANNUAL_RATE_PCT = 12.0  # schema default for `interestRate`

REASON_REJECT = "affordability_reject"
REASON_REVIEW = "affordability_review"


@dataclass(frozen=True)
class AffordabilityCheck:
    foir: float
    new_emi: float
    action: str | None  # "REJECT", "DEFER" or None


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def monthly_emi(principal: float, months: int, annual_rate_pct: float) -> float:
    """Standard reducing-balance EMI."""
    rate = annual_rate_pct / 1200.0
    if rate <= 0:
        return principal / months
    growth = (1.0 + rate) ** months
    return principal * rate * growth / (growth - 1.0)


def check_affordability(payload: dict[str, Any]) -> AffordabilityCheck | None:
    """FOIR for the requested loan, or None when the inputs are not all present.

    A missing input means no verdict: the gate never invents an income.
    """
    loan = _num(payload.get("loan_amount") or payload.get("loanAmount"))
    months = _num(payload.get("loan_tenure_months") or payload.get("loanTenure"))
    annual_income = _num(payload.get("annual_income"))
    if not loan or not months or not annual_income or loan <= 0 or months <= 0 or annual_income <= 0:
        return None

    rate = _num(payload.get("interestRate")) or DEFAULT_ANNUAL_RATE_PCT
    other_emis = _num(payload.get("obligations_other_banks")) or 0.0
    extra_income = _num(payload.get("additional_income")) or 0.0
    existing_dti = _num(payload.get("debt_to_income_ratio")) or 0.0

    new_emi = monthly_emi(loan, int(months), rate)
    income = annual_income + 12.0 * max(extra_income, 0.0)
    foir = existing_dti + 12.0 * (new_emi + max(other_emis, 0.0)) / income

    if foir > FOIR_REJECT:
        action = "REJECT"
    elif foir > FOIR_REVIEW:
        action = "DEFER"
    else:
        action = None
    return AffordabilityCheck(foir=round(foir, 4), new_emi=round(new_emi, 2), action=action)


def apply_affordability_gate(decision: str, reason: str, check: AffordabilityCheck | None) -> tuple[str, str]:
    """Final (decision, reason) after the policy. Only ever makes a decision stricter."""
    if check is None or check.action is None:
        return decision, reason
    if check.action == "REJECT":
        return "REJECT", REASON_REJECT
    if decision == "APPROVE":  # review band: approvals go to a human
        return "DEFER", REASON_REVIEW
    return decision, reason
