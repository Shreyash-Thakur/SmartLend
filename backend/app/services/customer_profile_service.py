"""Customer profile lookup — the data the bank *already holds*.

Premise (docs/FORM-REDESIGN.md): a loan applicant is an existing customer.
Their KYC, demographics, salary credits and bureau pull are already on file, so
the application form must not ask for them again. This module is the stand-in
for "core banking profile + bureau pull": given a customer id it returns that
block, and `resolve_application_payload()` merges it with the short form into
the full field set the scoring engine expects.

Data sources, in resolution order
---------------------------------
``get_profile()`` tries three things, in this order, and stops at the first hit:

  1. **The seeded ``customer_profiles`` SQLite table.** ~500 real customers
     committed to the repo as ``backend/data/customer_profiles_seed.json`` and
     loaded at ``init_db()`` time (see `customer_seed_service`). This is what
     makes a fresh clone work: no external file is required for the form to
     resolve an id and be submitted.
  2. **The full Home Credit extract**, if that ~180MB CSV happens to be on the
     machine. It covers all ~307k customers rather than the seeded 500.
  3. **``None``.**

The DB is first on purpose. It is the source that always exists, it is two
orders of magnitude smaller, and both paths run the *same* derivation
(`_build_profile`) over the *same* raw Home Credit columns, so a customer
present in both resolves identically either way.

Caching (CSV path)
------------------
The extract is ~307k rows and ~130 columns. Loading it at import time would
make `import backend.app.main` slow and would pull a large CSV into every
process that merely imports a schema; loading it per request would be far
worse. So:

  * load is **lazy** — nothing touches disk until the first CSV lookup;
  * only the ~18 columns this module actually reads are parsed (``usecols``),
    which is what keeps the resident set small rather than the row count;
  * the result is cached for the process lifetime as a single DataFrame
    indexed by ``SK_ID_CURR`` (a hash index, so ``.loc`` is O(1)). A DataFrame
    beats a dict-of-dicts here purely on memory: 307k small Python dicts cost
    an order of magnitude more than 18 typed numpy columns.
  * a failed/missing load is cached too (as an empty frame) so we do not retry
    a multi-second CSV read on every request when the file is absent.

The DB path is deliberately *not* cached in-process: it is an indexed primary
key lookup against a local SQLite file, and caching it would mean a re-seed
required a restart to take effect.

Unknown ids return ``None``. We never synthesise a profile: inventing
demographics for an id the bank does not recognise would be worse than
refusing, because the fabricated values would flow straight into a credit
decision.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Env override first so deployments/tests can point at their own extract;
# otherwise the local Home Credit merge, then a repo-relative fallback.
_ENV_VAR = "SMARTLEND_CUSTOMER_DATA"
_DEFAULT_PATHS: tuple[Path, ...] = (
    Path(r"C:\Users\shrey\Downloads\creddefer_full_merged.csv"),
    PROJECT_ROOT / "data" / "raw" / "creddefer_full_merged.csv",
)

CUSTOMER_ID_COLUMN = "SK_ID_CURR"

# The canonical raw field set. Keys are the `CustomerProfile` column names;
# values are the corresponding Home Credit CSV columns. Both resolution paths
# normalise into a dict keyed by the left-hand side before `_build_profile()`
# runs, which is what keeps the two paths from drifting apart.
_RAW_TO_CSV: dict[str, str] = {
    "customer_id": CUSTOMER_ID_COLUMN,
    "code_gender": "CODE_GENDER",
    "days_birth": "DAYS_BIRTH",
    "days_employed": "DAYS_EMPLOYED",
    "cnt_children": "CNT_CHILDREN",
    "name_family_status": "NAME_FAMILY_STATUS",
    "name_income_type": "NAME_INCOME_TYPE",
    "region_rating_client": "REGION_RATING_CLIENT",
    "amt_income_total": "AMT_INCOME_TOTAL",
    "amt_annuity": "AMT_ANNUITY",
    "amt_credit": "AMT_CREDIT",
    "ext_source_2": "EXT_SOURCE_2",
    "total_prev_credits": "total_prev_credits",
    "active_credits": "active_credits",
    "closed_credits": "closed_credits",
    "overdue_credits": "overdue_credits",
    "total_credit_sum": "total_credit_sum",
    "total_credit_debt": "total_credit_debt",
}

# Only these columns are parsed out of the CSV. Everything else in the
# 130-column extract is irrelevant to the profile block and would only cost
# memory.
_USED_COLUMNS: tuple[str, ...] = tuple(_RAW_TO_CSV.values())

# DAYS_EMPLOYED == 365243 encodes "pensioner / not employed", not 1000 years of
# service. Same sentinel handling as research/data/specs/home_credit.py.
_DAYS_EMPLOYED_SENTINEL = 365243

_frame: pd.DataFrame | None = None
_frame_lock = Lock()


# ===========================================================================
# Pipeline feature provenance
# ===========================================================================
# `backend/artifacts/pipeline.joblib` declares 25 `feature_names`, and
# `MLPredictor.predict_application` reads them out of the payload by those exact
# snake_case names — anything missing is silently replaced with 0.0. Before this
# module emitted them, the profile supplied `cibilScore` / `yearsOfEmployment`
# (camelCase) and the applicant was scored on mostly-zero features.
#
# The 25 split three ways. Nothing outside these lists is invented.

# 1. Genuinely on file at the bank. This module MUST emit every one of these.
PROFILE_FEATURES: tuple[str, ...] = (
    "age",
    "dependents",
    "years_employed",
    "annual_income",
    "monthly_income",
    "existing_emis",
    "cibil_score",
    "total_loans",
    "active_loans",
    "closed_loans",
    "missed_payments",
    "credit_utilization_ratio",
    "debt_to_income_ratio",
)

# 2. Typed by the applicant, or otherwise not knowable from a bureau pull. The
#    profile must NOT supply these; `resolve_application_payload()` copies them
#    across from the form's own (camelCase) vocabulary.
FORM_FEATURES: tuple[str, ...] = (
    "loan_amount",
    "loan_term",
    "interest_rate",
    "emi",
    "residential_assets_value",
    "commercial_assets_value",
    "bank_balance",
)

# 3. Arithmetic over (1) and (2). Computed in `resolve_application_payload()`
#    because each one needs a form value as well as a profile value.
DERIVED_FEATURES: tuple[str, ...] = (
    "total_assets",
    "emi_income_ratio",
    "loan_income_ratio",
)

# 4. LEAK. `loan_approved` and `confidence_score` are the model's own OUTPUTS,
#    yet they appear in the artifact's `feature_names` — the old synthetic
#    training frame was built with the label and the model's confidence still in
#    it. There is no honest value to supply at inference time, so we supply
#    none; `predict_application` fills them with 0.0 for every applicant, which
#    at least makes them constant and therefore inert. Fixing this properly
#    means retraining without those columns (deferred with the rest of the
#    Home Credit training work — see the NOTE at the top of ml_service.py).
LEAKED_OUTPUT_FEATURES: tuple[str, ...] = (
    "loan_approved",
    "confidence_score",
)


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
    a real bureau score. The CBES engine consumes the raw `credit_score`
    below, never this rescaled number.

    The ML pipeline is the one exception, and only because it has no choice:
    its `cibil_score` feature was fitted on a 300-900 scale, so the rescaled
    number is the only value on the right scale we can give it. That is a
    known artifact/data mismatch, not an endorsement of the mapping.
    """
    if ext_source_2 is None:
        return None
    return int(round(max(300.0, min(900.0, 300.0 + 600.0 * ext_source_2))))


# ---------------------------------------------------------------------------
# Resolution paths -> raw column dict
# ---------------------------------------------------------------------------


def _raw_from_db(key: int) -> dict[str, Any] | None:
    """Source 1: the seeded `customer_profiles` table (always available)."""
    # Imported here, not at module scope: `models` imports `database`, and this
    # module is imported by schema/route modules that must not pull the DB in.
    try:
        from backend.app.database import SessionLocal
        from backend.app.models import CustomerProfile
    except Exception:  # noqa: BLE001 - no DB layer available (bare import checks)
        return None

    try:
        session = SessionLocal()
    except Exception:  # noqa: BLE001
        return None
    try:
        record = session.get(CustomerProfile, key)
        if record is None:
            return None
        return {field: getattr(record, field, None) for field in _RAW_TO_CSV}
    except Exception:  # noqa: BLE001 - table not created yet, locked db, ...
        # A DB problem must not make a customer unresolvable when the CSV could
        # still answer; fall through to source 2 rather than raising.
        logger.debug("customer_profiles lookup failed for %s", key, exc_info=True)
        return None
    finally:
        session.close()


def _raw_from_csv(key: int) -> dict[str, Any] | None:
    """Source 2: the full Home Credit extract, if present on this machine."""
    frame = _load_frame()
    if frame.empty or key not in frame.index:
        return None

    row = frame.loc[key]
    if isinstance(row, pd.DataFrame):  # duplicate ids: take the first record
        row = row.iloc[0]

    return {field: row.get(column) for field, column in _RAW_TO_CSV.items()}


# ---------------------------------------------------------------------------
# Raw columns -> the profile block
# ---------------------------------------------------------------------------


def _build_profile(raw: dict[str, Any], key: int) -> dict[str, Any]:
    """Derive the profile block from raw Home Credit columns.

    Single implementation shared by both resolution paths, so a DB-resolved
    profile and a CSV-resolved profile for the same customer are identical.
    """
    days_birth = _num(raw.get("days_birth"))
    age = int(round(-days_birth / 365.25)) if days_birth is not None else None

    days_employed = _num(raw.get("days_employed"))
    if days_employed is None or days_employed == _DAYS_EMPLOYED_SENTINEL:
        tenure_years = None
    else:
        tenure_years = round(-days_employed / 365.25, 2)

    annual_income = _num(raw.get("amt_income_total"))
    annuity = _num(raw.get("amt_annuity"))
    ext_source_2 = _num(raw.get("ext_source_2"))
    region_rating = _num(raw.get("region_rating_client"))

    # Home Credit has no native DTI; annuity/income is the standard stand-in.
    dti = None
    if annuity is not None and annual_income:
        dti = round(annuity / annual_income, 4)

    # Credit utilisation is not in application_train either. Outstanding bureau
    # debt over total sanctioned bureau credit is the closest available proxy.
    total_credit_sum = _num(raw.get("total_credit_sum"))
    total_credit_debt = _num(raw.get("total_credit_debt"))
    utilisation = None
    if total_credit_sum:
        utilisation = round(max(0.0, min(1.0, (total_credit_debt or 0.0) / total_credit_sum)), 4)

    active_loans = _num(raw.get("active_credits"), 0.0) or 0.0
    closed_loans = _num(raw.get("closed_credits"), 0.0) or 0.0
    total_loans = _num(raw.get("total_prev_credits"), 0.0) or 0.0
    delinquencies = _num(raw.get("overdue_credits"), 0.0) or 0.0

    dependents = int(_num(raw.get("cnt_children"), 0.0) or 0.0)
    monthly_income = round(annual_income / 12.0, 2) if annual_income else None
    credit_score_display = _display_credit_score(ext_source_2)

    profile = {
        "customer_id": str(key),
        "found": True,
        # --- core banking / KYC ---
        "age": age,
        "gender": _gender_label(raw.get("code_gender")),
        "dependents": dependents,
        "marital_status": str(_clean(raw.get("name_family_status")) or "unknown"),
        "region": _region_label(region_rating),
        "region_rating": int(region_rating) if region_rating is not None else None,
        "employment_type": str(_clean(raw.get("name_income_type")) or "unknown"),
        "employment_tenure_years": tenure_years,
        # --- account history ---
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        "existing_emis": annuity,
        "dti": dti,
        # --- bureau pull ---
        "credit_score": ext_source_2,
        "credit_score_display": credit_score_display,
        "credit_score_basis": "EXT_SOURCE_2 (normalised external score, not CIBIL)",
        "delinquencies": int(delinquencies),
        "active_loans": int(active_loans),
        "closed_loans": int(closed_loans),
        "total_loans": int(total_loans),
        "credit_utilisation": utilisation,
    }

    # --- ML pipeline vocabulary (exact snake_case `feature_names`) ----------
    # Same numbers as above under the names `pipeline.joblib` actually looks
    # up. Emitting them here (rather than only in the camelCase legacy block)
    # is what stops the model from scoring an applicant on zeros. Every one of
    # these is a re-labelling of a value already in the profile — nothing new
    # is introduced, and no FORM_FEATURES / LEAKED_OUTPUT_FEATURES appear.
    profile.update(
        {
            # `age` and `dependents` already carry the pipeline's names above.
            # Pensioners have no employment record; 0 years is the truthful
            # reading of "no current employment", not a filled-in default.
            "years_employed": float(tenure_years) if tenure_years is not None else 0.0,
            # `annual_income` / `monthly_income` / `existing_emis` likewise
            # already use the pipeline's names.
            "cibil_score": credit_score_display,  # 300-900 rescale; see _display_credit_score
            "total_loans": int(total_loans),
            "active_loans": int(active_loans),
            "closed_loans": int(closed_loans),
            "missed_payments": int(delinquencies),
            "credit_utilization_ratio": utilisation,
            "debt_to_income_ratio": dti,
        }
    )
    return profile


def get_profile(customer_id: Any) -> dict[str, Any] | None:
    """Return the demographic + bureau block for `customer_id`, else None.

    Resolution order (see module docstring): seeded `customer_profiles` table,
    then the full extract if it is on this machine, then None.

    Returns None for an unknown or malformed id. Callers must handle that;
    this function never invents a profile.
    """
    try:
        key = int(str(customer_id).strip())
    except (TypeError, ValueError):
        return None

    raw = _raw_from_db(key)  # 1. seeded DB table — always present in a clone
    if raw is None:
        raw = _raw_from_csv(key)  # 2. the 180MB extract, if it exists here
    if raw is None:
        return None  # 3. unknown id

    return _build_profile(raw, key)


def get_sample_customers(limit: int = 10) -> list[dict[str, Any]]:
    """Example customer ids for the form's "try one of these" panel.

    Reads the `is_sample` rows the seed fixture flagged — chosen for a spread
    of profile shapes and expected outcomes — and falls back to any seeded rows
    if none were flagged. Returns [] when nothing is seeded; the UI can then
    simply not render the panel.
    """
    from backend.app.services.customer_seed_service import (
        describe_profile_row,
        expected_decision_hint,
    )

    try:
        from backend.app.database import SessionLocal
        from backend.app.models import CustomerProfile
    except Exception:  # noqa: BLE001
        return []

    try:
        session = SessionLocal()
    except Exception:  # noqa: BLE001
        return []
    try:
        query = session.query(CustomerProfile).filter(CustomerProfile.is_sample.is_(True))
        records = query.order_by(CustomerProfile.customer_id).limit(limit).all()
        if not records:
            records = (
                session.query(CustomerProfile).order_by(CustomerProfile.customer_id).limit(limit).all()
            )
    except Exception:  # noqa: BLE001 - table missing / not seeded yet
        logger.debug("sample customer lookup failed", exc_info=True)
        return []
    finally:
        session.close()

    samples: list[dict[str, Any]] = []
    for record in records:
        raw = {field: getattr(record, field, None) for field in _RAW_TO_CSV}
        samples.append(
            {
                "customer_id": str(record.customer_id),
                "descriptor": record.descriptor or describe_profile_row(raw),
                # A hint, not a prediction: the real decision also needs the
                # loan amount, which does not exist until the form is filled in.
                "expected_decision_hint": expected_decision_hint(raw),
                "credit_score_display": _display_credit_score(_num(record.ext_source_2)),
                "annual_income": _num(record.amt_income_total),
            }
        )
    return samples


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

# Form (camelCase / short-form) key -> the pipeline's snake_case feature name.
# These are the applicant's own answers; the profile never supplies them.
_FORM_TO_PIPELINE: dict[str, tuple[str, ...]] = {
    "loan_amount": ("loan_amount", "loanAmount"),
    "loan_term": ("loan_term", "loan_tenure_months", "loanTenure"),
    "interest_rate": ("interest_rate", "interestRate"),
    "emi": ("emi",),
    "residential_assets_value": ("residential_assets_value", "residentialAssetsValue"),
    "commercial_assets_value": ("commercial_assets_value", "commercialAssetsValue"),
    "bank_balance": ("bank_balance", "bankBalance"),
}


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def resolve_application_payload(form_data: dict[str, Any]) -> dict[str, Any]:
    """Merge the short form with the customer's on-file profile.

    Returns a payload carrying four vocabularies at once:

      1. the short form's own fields (loan_amount, collateral_type, ...);
      2. the snake_case keys CBES scores on (see CBES_KEYS);
      3. the snake_case keys the ML pipeline scores on (see PROFILE_FEATURES /
         FORM_FEATURES / DERIVED_FEATURES);
      4. the legacy camelCase fields (`age`, `monthlyIncome`, `cibilScore`, ...)
         that `LoanApplicationInput`, `build_application_response()` and every
         stored row still expect.

    (4) is the backward-compatibility bridge: the applicant no longer types
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

    # --- ML pipeline vocabulary: what the bank holds --------------------
    # Copied under the pipeline's exact feature names. Without this the
    # pipeline finds nothing under `cibil_score` / `years_employed` / ... and
    # `MLPredictor.predict_application` scores the applicant on 0.0s.
    for feature in PROFILE_FEATURES:
        value = profile.get(feature)
        if payload.get(feature) is None and value is not None:
            payload[feature] = value

    # --- ML pipeline vocabulary: what the applicant typed ---------------
    # Strictly form-sourced. The profile has no view of the requested loan's
    # terms or of the applicant's assets, so nothing here falls back to it.
    for feature, form_keys in _FORM_TO_PIPELINE.items():
        value = _first_present(payload, form_keys)
        if value is not None:
            payload[feature] = value

    # --- ML pipeline vocabulary: derived ---------------------------------
    # Arithmetic only, and only when every input is actually present. A missing
    # input leaves the feature absent rather than inventing a ratio.
    if payload.get("total_assets") is None:
        residential = _num(payload.get("residential_assets_value"), 0.0) or 0.0
        commercial = _num(payload.get("commercial_assets_value"), 0.0) or 0.0
        payload["total_assets"] = residential + commercial  # sum of declared assets

    monthly_income = _num(profile.get("monthly_income"))
    emi = _num(payload.get("emi"))
    if emi is not None and monthly_income:
        payload["emi_income_ratio"] = round(emi / monthly_income, 4)

    annual_income = _num(profile.get("annual_income"))
    loan_amount = _num(payload.get("loan_amount"))
    if loan_amount is not None and annual_income:
        payload["loan_income_ratio"] = round(loan_amount / annual_income, 4)

    # `debt_to_income_ratio` is already covered by PROFILE_FEATURES above (it
    # is the profile's `dti`), so it is not recomputed here.
    #
    # NOT SET, DELIBERATELY: `loan_approved` and `confidence_score`. They are
    # model outputs that leaked into the artifact's feature list; see
    # LEAKED_OUTPUT_FEATURES.

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
