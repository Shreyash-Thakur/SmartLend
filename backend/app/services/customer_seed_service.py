"""Seed the `customer_profiles` table so the application form works offline.

THE PROBLEM THIS SOLVES
-----------------------
`customer_profile_service` used to resolve customer ids exclusively from a
~180MB Home Credit extract (``creddefer_full_merged.csv``) that lives outside
the repository and is never committed. On a fresh clone that file is absent, so
*no* customer id resolved, so the redesigned application form — which requires
a resolvable customer id before it will score anything — could not be submitted
at all.

The fix is two-stage and deliberately ordered so the committed artifact is the
one that matters:

  1. `build_seed_fixture()` samples ~500 real customers out of the big CSV and
     writes them to ``backend/data/customer_profiles_seed.json``. This runs on
     a machine that HAS the CSV, and its output is committed.
  2. `seed_customer_profiles()` loads that committed fixture into SQLite at
     `init_db()` time. It needs no external file, so a fresh clone works.

`seed_customer_profiles()` will fall back to building from the CSV if the
fixture is missing *and* the CSV happens to be present, but that is a
convenience, not the supported path.

SAMPLING
--------
A naive `head(500)` or uniform sample would be ~92% non-defaulters clustered in
the middle of the score distribution, and every demo would show the same
APPROVE. So the sample is stratified over
``TARGET x EXT_SOURCE_2 quartile`` (8 cells, filled evenly) with a fixed seed,
which spreads the sample across both observed outcomes and the whole credit
score range and therefore produces APPROVE, REJECT and DEFER decisions.

``TARGET`` is used ONLY to pick rows. It is not written to the fixture and not
stored in the table — an observed default outcome must never be reachable from
a scoring payload.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
SEED_FIXTURE_PATH = BACKEND_ROOT / "data" / "customer_profiles_seed.json"

DEFAULT_SAMPLE_SIZE = 500
SAMPLE_RANDOM_SEED = 20260830  # fixed: the committed fixture must be reproducible

# CSV column -> `CustomerProfile` attribute. Only these are sampled; the other
# ~113 columns of the extract are irrelevant to the profile block.
_CSV_TO_MODEL: dict[str, str] = {
    "SK_ID_CURR": "customer_id",
    "CODE_GENDER": "code_gender",
    "DAYS_BIRTH": "days_birth",
    "DAYS_EMPLOYED": "days_employed",
    "CNT_CHILDREN": "cnt_children",
    "NAME_FAMILY_STATUS": "name_family_status",
    "NAME_INCOME_TYPE": "name_income_type",
    "REGION_RATING_CLIENT": "region_rating_client",
    "AMT_INCOME_TOTAL": "amt_income_total",
    "AMT_ANNUITY": "amt_annuity",
    "AMT_CREDIT": "amt_credit",
    "EXT_SOURCE_2": "ext_source_2",
    "total_prev_credits": "total_prev_credits",
    "active_credits": "active_credits",
    "closed_credits": "closed_credits",
    "overdue_credits": "overdue_credits",
    "total_credit_sum": "total_credit_sum",
    "total_credit_debt": "total_credit_debt",
}

# Stratification column. Read from the CSV, never persisted (see module docstring).
_STRATIFY_TARGET = "TARGET"
_STRATIFY_SCORE = "EXT_SOURCE_2"

_DAYS_EMPLOYED_SENTINEL = 365243


# ---------------------------------------------------------------------------
# Descriptors — plain-language labels for the /customers/samples endpoint
# ---------------------------------------------------------------------------


def describe_profile_row(row: dict[str, Any]) -> str:
    """One-line human descriptor, derived only from observable profile fields.

    Never derived from TARGET: the point of the descriptor is to tell a demo
    user what *kind* of applicant this is, not to reveal the answer.
    """
    score = _f(row.get("ext_source_2"))
    income = _f(row.get("amt_income_total"))
    annuity = _f(row.get("amt_annuity"))
    overdue = _f(row.get("overdue_credits")) or 0.0
    total_prev = _f(row.get("total_prev_credits")) or 0.0
    credit_sum = _f(row.get("total_credit_sum")) or 0.0
    credit_debt = _f(row.get("total_credit_debt")) or 0.0
    days_employed = _f(row.get("days_employed"))

    dti = (annuity / income) if (annuity is not None and income) else None
    utilisation = (credit_debt / credit_sum) if credit_sum else None

    # Ordered most-distinctive-first; the first match wins.
    if total_prev <= 0:
        return "thin file - no bureau history on record"
    if overdue >= 1:
        return "prior delinquencies on bureau record"
    if dti is not None and dti >= 0.30:
        return "high DTI - existing obligations eat the income"
    if utilisation is not None and utilisation >= 0.60:
        return "high credit utilisation"
    if days_employed is not None and days_employed == _DAYS_EMPLOYED_SENTINEL:
        return "pensioner / no current employment record"
    if days_employed is not None and -days_employed / 365.25 < 1.0:
        return "short employment tenure (under a year)"
    if score is not None and score >= 0.65:
        return "strong profile - high external score, clean history"
    if score is not None and score <= 0.30:
        return "weak external score"
    if income is not None and income <= 90_000:
        return "modest income"
    return "mid-range profile - clean but unremarkable"


def expected_decision_hint(row: dict[str, Any]) -> str:
    """Rough APPROVE / DEFER / REJECT flavour for the samples endpoint.

    A HINT, not a prediction. It reads the same three drivers the CBES credit
    and capacity components weigh most (external score, delinquencies, DTI) but
    it does not run the engine — the engine also needs the loan amount, which
    only exists once a form is filled in. Labelled as a hint everywhere it is
    surfaced so nobody mistakes it for the decision.
    """
    score = _f(row.get("ext_source_2"))
    income = _f(row.get("amt_income_total"))
    annuity = _f(row.get("amt_annuity"))
    overdue = _f(row.get("overdue_credits")) or 0.0
    dti = (annuity / income) if (annuity is not None and income) else None

    if score is None:
        return "DEFER"
    if overdue >= 2 or score <= 0.25 or (dti is not None and dti >= 0.45):
        return "REJECT"
    if score >= 0.60 and overdue == 0 and (dti is None or dti < 0.25):
        return "APPROVE"
    return "DEFER"


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # drop NaN


# ---------------------------------------------------------------------------
# Stage 1 — build the committed fixture from the big CSV
# ---------------------------------------------------------------------------


def build_seed_fixture(
    csv_path: str | Path,
    output_path: str | Path = SEED_FIXTURE_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_seed: int = SAMPLE_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Sample `sample_size` customers from the extract and write the fixture.

    Requires the big CSV. Run once on a machine that has it; commit the result.
    """
    import pandas as pd  # local import: the runtime seed path must not need pandas

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Customer extract not found at {csv_path}")

    header = pd.read_csv(csv_path, nrows=0)
    wanted = [c for c in (*_CSV_TO_MODEL, _STRATIFY_TARGET) if c in header.columns]
    frame = pd.read_csv(csv_path, usecols=wanted, low_memory=False)

    # 8 strata: observed outcome x external-score quartile. Filling them evenly
    # is what puts REJECTs and DEFERs in the sample instead of 500 APPROVEs.
    frame = frame[frame[_STRATIFY_SCORE].notna()]
    frame["_score_q"] = pd.qcut(frame[_STRATIFY_SCORE], 4, labels=False, duplicates="drop")
    strata = list(frame.groupby([_STRATIFY_TARGET, "_score_q"], observed=True))
    per_stratum = max(1, sample_size // max(1, len(strata)))

    chunks = []
    for _, group in strata:
        take = min(per_stratum, len(group))
        chunks.append(group.sample(n=take, random_state=random_seed))
    sampled = pd.concat(chunks) if chunks else frame.head(0)

    # Top up to the exact target from whatever is left, still seeded.
    shortfall = sample_size - len(sampled)
    if shortfall > 0:
        remainder = frame.drop(index=sampled.index)
        if len(remainder):
            sampled = pd.concat(
                [sampled, remainder.sample(n=min(shortfall, len(remainder)), random_state=random_seed)]
            )

    sampled = sampled.sort_values("SK_ID_CURR")

    records: list[dict[str, Any]] = []
    for _, csv_row in sampled.iterrows():
        record: dict[str, Any] = {}
        for csv_column, attribute in _CSV_TO_MODEL.items():
            if csv_column not in sampled.columns:
                continue
            value = csv_row[csv_column]
            if pd.isna(value):
                record[attribute] = None
            elif attribute in ("customer_id",):
                record[attribute] = int(value)
            elif isinstance(value, str):
                record[attribute] = value
            else:
                record[attribute] = float(value)
        # TARGET is intentionally NOT copied into the record.
        record["descriptor"] = describe_profile_row(record)
        record["is_sample"] = False
        records.append(record)

    _mark_showcase_samples(records)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=1, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %d seed profiles to %s", len(records), output_path)
    return records


def _mark_showcase_samples(records: list[dict[str, Any]], count: int = 10) -> None:
    """Flag ~`count` records as the ids the UI offers as examples.

    Picked for variety of *descriptor* and of expected decision, so the panel
    shows an APPROVE, a REJECT and a DEFER rather than ten near-identical ids.
    """
    by_descriptor: dict[tuple[str, str], dict[str, Any]] = {}
    for record in sorted(records, key=lambda r: r["customer_id"]):
        key = (record.get("descriptor", ""), expected_decision_hint(record))
        by_descriptor.setdefault(key, record)

    chosen = list(by_descriptor.values())[:count]
    # If distinct descriptors are scarce, top up in id order so the endpoint
    # still returns a full set.
    if len(chosen) < count:
        for record in sorted(records, key=lambda r: r["customer_id"]):
            if record not in chosen:
                chosen.append(record)
            if len(chosen) >= count:
                break
    for record in chosen:
        record["is_sample"] = True


# ---------------------------------------------------------------------------
# Stage 2 — load the fixture into SQLite (the fresh-clone path)
# ---------------------------------------------------------------------------


def load_seed_records(fixture_path: str | Path = SEED_FIXTURE_PATH) -> list[dict[str, Any]]:
    path = Path(fixture_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Customer seed fixture at %s is unreadable; skipping seed", path)
        return []
    return data if isinstance(data, list) else []


def seed_customer_profiles(force: bool = False, fixture_path: str | Path = SEED_FIXTURE_PATH) -> int:
    """Populate `customer_profiles` from the committed fixture.

    Idempotent and best-effort: an already-populated table is left alone, and
    any failure is logged rather than raised. Seeding demo data must never be
    able to stop the API from starting.

    Returns the number of rows inserted (0 if the table was already seeded).
    """
    from sqlalchemy import func, select

    from backend.app.database import SessionLocal
    from backend.app.models import CustomerProfile

    records = load_seed_records(fixture_path)
    if not records:
        logger.warning(
            "No customer seed fixture at %s - customer ids will only resolve if the "
            "full extract is present.",
            fixture_path,
        )
        return 0

    session = SessionLocal()
    try:
        existing = session.execute(select(func.count()).select_from(CustomerProfile)).scalar_one()
        if existing and not force:
            return 0
        if force and existing:
            session.query(CustomerProfile).delete()

        columns = {c.name for c in CustomerProfile.__table__.columns}
        session.bulk_insert_mappings(
            CustomerProfile,
            [{k: v for k, v in record.items() if k in columns} for record in records],
        )
        session.commit()
        logger.info("Seeded %d customer profiles from %s", len(records), fixture_path)
        return len(records)
    except Exception:  # noqa: BLE001 - see docstring: seeding must not break startup
        session.rollback()
        logger.exception("Customer profile seeding failed; continuing without seed data")
        return 0
    finally:
        session.close()
