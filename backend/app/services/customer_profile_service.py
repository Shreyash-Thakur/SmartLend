"""Customer profile lookup — the data the bank *already holds*.

Premise (docs/FORM-REDESIGN.md): a loan applicant is an existing customer.
Their KYC, demographics, salary credits and bureau pull are already on file, so
the application form must not ask for them again. This module is the stand-in
for "core banking profile + bureau pull": given a customer id it returns that
block, and `resolve_application_payload()` merges it with the short form into
the full field set the scoring engine expects.

Data source
-----------
The Home Credit dataset *is* the existing-customer database for this demo:
``SK_ID_CURR`` is the customer id. We read a pre-merged extract that already
carries the bureau aggregates (``active_credits`` / ``overdue_credits`` / ...),
so no join is needed at request time.

Caching
-------
The extract is ~307k rows and ~130 columns. Loading it at import time would
make `import backend.app.main` slow and would pull a large CSV into every
process that merely imports a schema; loading it per request would be far
worse. So:

  * load is **lazy** — nothing touches disk until the first lookup;
  * only the ~18 columns this module actually reads are parsed (``usecols``),
    which is what keeps the resident set small rather than the row count;
  * the result is cached for the process lifetime as a single DataFrame
    indexed by ``SK_ID_CURR`` (a hash index, so ``.loc`` is O(1)). A DataFrame
    beats a dict-of-dicts here purely on memory: 307k small Python dicts cost
    an order of magnitude more than 18 typed numpy columns.
  * a failed/missing load is cached too (as an empty frame) so we do not retry
    a multi-second CSV read on every request when the file is absent.

Unknown ids return ``None``. We never synthesise a profile: inventing
demographics for an id the bank does not recognise would be worse than
refusing, because the fabricated values would flow straight into a credit
decision.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Env override first so deployments/tests can point at their own extract;
# otherwise the local Home Credit merge, then a repo-relative fallback.
_ENV_VAR = "SMARTLEND_CUSTOMER_DATA"
_DEFAULT_PATHS: tuple[Path, ...] = (
    Path(r"C:\Users\shrey\Downloads\creddefer_full_merged.csv"),
    PROJECT_ROOT / "data" / "raw" / "creddefer_full_merged.csv",
)

CUSTOMER_ID_COLUMN = "SK_ID_CURR"

# Only these columns are parsed. Everything else in the 130-column extract is
# irrelevant to the profile block and would only cost memory.
_USED_COLUMNS: tuple[str, ...] = (
    CUSTOMER_ID_COLUMN,
    "CODE_GENDER",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "EXT_SOURCE_2",
    "REGION_RATING_CLIENT",
    "total_prev_credits",
    "active_credits",
    "closed_credits",
    "overdue_credits",
    "total_credit_sum",
    "total_credit_debt",
)

# DAYS_EMPLOYED == 365243 encodes "pensioner / not employed", not 1000 years of
# service. Same sentinel handling as research/data/specs/home_credit.py.
_DAYS_EMPLOYED_SENTINEL = 365243

_frame: pd.DataFrame | None = None
_frame_lock = Lock()


def _resolve_source_path() -> Path | None:
    override = os.environ.get(_ENV_VAR, "").strip()
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    for candidate in _DEFAULT_PATHS:
        if candidate.exists():
            return candidate
    return None


def _load_frame() -> pd.DataFrame:
    """Read the extract once and cache it, indexed by customer id."""
    global _frame
    if _frame is not None:
        return _frame
    with _frame_lock:
        # Re-check inside the lock: two requests can race to the first lookup.
        if _frame is not None:
            return _frame
        path = _resolve_source_path()
        if path is None:
            # Cache the miss: an empty frame means "no customer database
            # available", and every lookup then cleanly returns None instead of
            # re-stat-ing the filesystem on each call.
            _frame = pd.DataFrame(columns=list(_USED_COLUMNS)).set_index(CUSTOMER_ID_COLUMN)
            return _frame
        header = pd.read_csv(path, nrows=0)
        available = [column for column in _USED_COLUMNS if column in header.columns]
        loaded = pd.read_csv(path, usecols=available, low_memory=False)
        _frame = loaded.set_index(CUSTOMER_ID_COLUMN, drop=False)
        return _frame


def reset_cache() -> None:
    """Drop the cached extract. Used by tests that repoint the data source."""
    global _frame
    with _frame_lock:
        _frame = None


def _clean(value: Any) -> Any:
    """NaN/NaT -> None, numpy scalars -> plain Python, so this is JSON-safe."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _num(value: Any, default: float | None = None) -> float | None:
    cleaned = _clean(value)
    if cleaned is None:
        return default
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return default


def _gender_label(code: Any) -> str:
    # CODE_GENDER carries a handful of 'XNA' rows; surface them as "other"
    # rather than silently folding them into male or female.
    mapping = {"M": "male", "F": "female"}
    return mapping.get(str(_clean(code) or "").strip().upper(), "other")


def _region_label(rating: float | None) -> str:
    # REGION_RATING_CLIENT is ordinal urbanicity (1 = best/most urban,
    # 3 = worst). Mapped to the app's existing region vocabulary for display.
    if rating is None:
        return "semi_urban"
    if rating <= 1:
        return "urban"
    if rating >= 3:
        return "rural"
    return "semi_urban"


def _display_credit_score(ext_source_2: float | None) -> int | None:
    """Rescale EXT_SOURCE_2 (a normalised [0,1] external score) onto 300-900.

    Display and legacy-field compatibility only. This is NOT a CIBIL score and
    must never be presented as one: the underlying value is Home Credit's
    anonymised external score, whose scale and direction are not comparable to
    a real bureau score. The scoring engine consumes the raw `credit_score`
    below, never this rescaled number.
    """
    if ext_source_2 is None:
        return None
    return int(round(max(300.0, min(900.0, 300.0 + 600.0 * ext_source_2))))


def get_profile(customer_id: Any) -> dict[str, Any] | None:
    """Return the demographic + bureau block for `customer_id`, else None.

    Returns None for an unknown or malformed id. Callers must handle that;
    this function never invents a profile.
    """
    try:
        key = int(str(customer_id).strip())
    except (TypeError, ValueError):
        return None

    frame = _load_frame()
    if frame.empty or key not in frame.index:
        return None

    row = frame.loc[key]
    if isinstance(row, pd.DataFrame):  # duplicate ids: take the first record
        row = row.iloc[0]

    days_birth = _num(row.get("DAYS_BIRTH"))
    age = int(round(-days_birth / 365.25)) if days_birth is not None else None

    days_employed = _num(row.get("DAYS_EMPLOYED"))
    if days_employed is None or days_employed == _DAYS_EMPLOYED_SENTINEL:
        tenure_years = None
    else:
        tenure_years = round(-days_employed / 365.25, 2)

    annual_income = _num(row.get("AMT_INCOME_TOTAL"))
    annuity = _num(row.get("AMT_ANNUITY"))
    ext_source_2 = _num(row.get("EXT_SOURCE_2"))
    region_rating = _num(row.get("REGION_RATING_CLIENT"))

    # Home Credit has no native DTI; annuity/income is the standard stand-in.
    dti = None
    if annuity is not None and annual_income:
        dti = round(annuity / annual_income, 4)

    # Credit utilisation is not in application_train either. Outstanding bureau
    # debt over total sanctioned bureau credit is the closest available proxy.
    total_credit_sum = _num(row.get("total_credit_sum"))
    total_credit_debt = _num(row.get("total_credit_debt"))
    utilisation = None
    if total_credit_sum:
        utilisation = round(max(0.0, min(1.0, (total_credit_debt or 0.0) / total_credit_sum)), 4)

    active_loans = _num(row.get("active_credits"), 0.0) or 0.0
    closed_loans = _num(row.get("closed_credits"), 0.0) or 0.0
    total_loans = _num(row.get("total_prev_credits"), 0.0) or 0.0
    delinquencies = _num(row.get("overdue_credits"), 0.0) or 0.0

    return {
        "customer_id": str(key),
        "found": True,
        # --- core banking / KYC ---
        "age": age,
        "gender": _gender_label(row.get("CODE_GENDER")),
        "dependents": int(_num(row.get("CNT_CHILDREN"), 0.0) or 0.0),
        "marital_status": str(_clean(row.get("NAME_FAMILY_STATUS")) or "unknown"),
        "region": _region_label(region_rating),
        "region_rating": int(region_rating) if region_rating is not None else None,
        "employment_type": str(_clean(row.get("NAME_INCOME_TYPE")) or "unknown"),
        "employment_tenure_years": tenure_years,
        # --- account history ---
        "annual_income": annual_income,
        "monthly_income": round(annual_income / 12.0, 2) if annual_income else None,
        "existing_emis": annuity,
        "dti": dti,
        # --- bureau pull ---
        "credit_score": ext_source_2,
        "credit_score_display": _display_credit_score(ext_source_2),
        "credit_score_basis": "EXT_SOURCE_2 (normalised external score, not CIBIL)",
        "delinquencies": int(delinquencies),
        "active_loans": int(active_loans),
        "closed_loans": int(closed_loans),
        "total_loans": int(total_loans),
        "credit_utilisation": utilisation,
    }


# ---------------------------------------------------------------------------
# Short form + profile -> full scoring payload
# ---------------------------------------------------------------------------

# Keys the CBES engine reads (backend/app/services/cbes_engine.py). Every one of
# them now comes from the bank's own systems except loan_amount, which is the
# only risk input the applicant actually types.
CBES_KEYS = (
    "credit_score",
    "delinquencies",
    "active_loans",
    "dti",
    "employment_tenure_years",
    "annual_income",
    "loan_amount",
    "region",
)


def resolve_application_payload(form_data: dict[str, Any]) -> dict[str, Any]:
    """Merge the short form with the customer's on-file profile.

    Returns a payload carrying three vocabularies at once:

      1. the short form's own fields (loan_amount, collateral_type, ...);
      2. the snake_case keys CBES scores on (see CBES_KEYS);
      3. the legacy camelCase fields (`age`, `monthlyIncome`, `cibilScore`, ...)
         that `LoanApplicationInput`, `build_application_response()` and every
         stored row still expect.

    (3) is the backward-compatibility bridge: the applicant no longer types
    those values, but nothing downstream had to change to stop reading them.

    A payload with no `customer_id`, or with one the bank does not recognise,
    is returned unchanged apart from a `profile_resolved` flag. Callers decide
    whether that is an error; we do not fabricate the missing block.
    """
    payload = dict(form_data)
    customer_id = payload.get("customer_id") or payload.get("customerId")

    profile = get_profile(customer_id) if customer_id else None
    if profile is None:
        payload["profile_resolved"] = False
        return payload

    payload["customer_id"] = profile["customer_id"]
    payload["profile_resolved"] = True
    payload["customer_profile"] = profile

    # --- CBES vocabulary (snake_case) ---
    # `region` for CBES is the ordinal rating, not the label; the engine treats
    # it as 1=best..3=worst.
    scoring = {
        "credit_score": profile["credit_score"],
        "delinquencies": profile["delinquencies"],
        "active_loans": profile["active_loans"],
        "dti": profile["dti"],
        "employment_tenure_years": profile["employment_tenure_years"],
        "annual_income": profile["annual_income"],
        "region": profile["region_rating"],
    }
    for key, value in scoring.items():
        # The form never supplies these, but an explicit caller-provided value
        # (e.g. a what-if simulation) wins over the profile.
        if payload.get(key) is None and value is not None:
            payload[key] = value

    # --- legacy camelCase vocabulary ---
    legacy = {
        "age": profile["age"],
        "gender": profile["gender"],
        "dependents": profile["dependents"],
        "region": profile["region_rating"],  # numeric for CBES; label below
        "regionLabel": profile["region"],
        "employmentType": profile["employment_type"],
        "yearsOfEmployment": (
            int(profile["employment_tenure_years"])
            if profile["employment_tenure_years"] is not None
            else 0
        ),
        "monthlyIncome": profile["monthly_income"],
        "annualIncome": profile["annual_income"],
        "existingEmis": profile["existing_emis"],
        "cibilScore": profile["credit_score_display"],
        "totalLoans": profile["total_loans"],
        "activeLoans": profile["active_loans"],
        "closedLoans": profile["closed_loans"],
        "missedPayments": profile["delinquencies"],
        "creditUtilizationRatio": profile["credit_utilisation"],
        "debtToIncomeRatio": profile["dti"],
    }
    for key, value in legacy.items():
        if payload.get(key) is None and value is not None:
            payload[key] = value

    return payload
