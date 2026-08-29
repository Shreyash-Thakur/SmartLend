# Home Credit Data Layer + CBES Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the synthetic-dataset-derived CBES rule engine with one built on real Home Credit fields, fill the two bureau-derived canonical gaps (`delinquencies`, `active_loans`), and retire the synthetic generator — with **no model training and no SHAP** in this pass.

**Architecture:** Extend the existing `research/data/` canonical-mapping layer with a new `bureau_aggregates.py` module that turns `bureau.csv` into two native columns Home Credit's spec can consume directly. Rewrite `backend/app/services/cbes_engine.py` around the resulting 7-field vocabulary, with thresholds computed by a new percentile-calibration script (`backend/app/services/cbes_calibration.py`) instead of hand-picked constants. Delete the synthetic generator and its CSV; leave everything that depended on them (training scripts, live API schema, frontend) marked with pointer comments rather than fixed, since fixing them requires the deferred training work.

**Tech Stack:** Python 3.13, pandas, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-29-home-credit-swap-design.md` (sections 2a, 3.1, 3.2, 3.4 — this plan does not implement 3.3, see "Descoped" below)

## Global Constraints

- No model is trained in this pass (spec §2a). No SHAP.
- `credit_utilization` stays `Availability.ABSENT` — `credit_card_balance.csv` was not fetched (spec decision 3).
- CBES rule set is exactly 7 fields: `credit_score`, `delinquencies`, `active_loans`, `dti`, `employment_tenure_years`, an income/loan-amount affordability check, optional low-weight `region` (spec decision 4).
- Nothing is imputed to a fabricated value; missing stays `NaN`/explicit (canonical.py's existing rule, `research/data/canonical.py:69-70`).
- Delete `synthetic_indian_loan_dataset.csv` and `backend/generate_indian_loan_dataset.py` now (spec decision 5, accepted trade-off against the 2026-08-18 spec's C4 phase).

## Descoped from this plan (do not implement)

Spec §3.3 (interim API/frontend schema rebuild) is **not** part of this plan. Reason found during planning: `public_api_service.get_predict_payload()` calls `get_predictor().predict_application()` (`backend/app/services/ml_service.py`), which requires a trained model artifact. Since training is deferred, the live predict endpoint cannot functionally run regardless of schema changes — rewriting `schemas.py`/`public_api_service.py`/the frontend form now would be undoable end-to-end and likely redone once a model exists. This will be its own plan once a HuggingFace model is chosen. Flag this to the user if it wasn't already surfaced.

---

## File Structure

- **Create** `research/data/bureau_aggregates.py` — bureau.csv → per-applicant counts.
- **Create** `research/tests/test_bureau_aggregates.py` — unit tests for the above.
- **Modify** `research/data/specs/home_credit.py` — `delinquencies`/`active_loans` become `NATIVE` (sourced from the merged bureau columns) instead of `ABSENT`.
- **Modify** `research/tests/test_adapters.py` — update the two tests that currently assert these fields are `ABSENT`.
- **Modify** `research/data/cli.py` — merge bureau aggregates into the frame before validating/profiling the `home_credit` dataset.
- **Rewrite** `backend/app/services/cbes_engine.py` — 7-field vocabulary, loads calibrated thresholds.
- **Create** `backend/app/services/cbes_calibration.py` — computes percentile thresholds from real data, writes `backend/artifacts/cbes_thresholds.json`.
- **Create** `backend/tests/test_cbes_engine.py` — unit tests for the new engine (may already exist as `backend/tests/`; check before creating, extend if so).
- **Create** `backend/tests/test_cbes_calibration.py` — unit tests for the calibration script.
- **Delete** `backend/generate_indian_loan_dataset.py`, `synthetic_indian_loan_dataset.csv` (repo root, if present) / any other path holding it — check first (`git ls-files | grep synthetic_indian_loan_dataset`).
- **Modify** (pointer-note only, one comment block each): `backend/retrain_pipeline_v2.py`, `backend/run_evaluation.py`, `backend/compute_baselines.py`, `backend/training_comparison.py`, `backend/run_calibration_report.py`, `backend/app/services/ml_service.py` (top of `train_pipeline`).

---

### Task 1: Bureau aggregation module

**Files:**
- Create: `research/data/bureau_aggregates.py`
- Test: `research/tests/test_bureau_aggregates.py`

**Interfaces:**
- Produces: `compute_bureau_aggregates(bureau: pd.DataFrame) -> pd.DataFrame` — indexed by `SK_ID_CURR`, columns `BUREAU_ACTIVE_LOAN_COUNT` (int), `BUREAU_DELINQUENCY_COUNT` (int).
- Produces: `attach_bureau_aggregates(application_df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame` — left-merges the above onto `application_df` on `SK_ID_CURR`, filling absent-from-bureau applicants with `0` for both counts (documented as "no bureau record found" rather than "known zero", see docstring below).

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/test_bureau_aggregates.py
from __future__ import annotations

import pandas as pd
import pytest

from research.data.bureau_aggregates import (
    attach_bureau_aggregates,
    compute_bureau_aggregates,
)


@pytest.fixture
def bureau() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2, 2, 3],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed", "Closed", "Active"],
            "CREDIT_DAY_OVERDUE": [0, 5, 0, 0, 12, 0],
        }
    )


def test_active_loan_count_per_applicant(bureau):
    agg = compute_bureau_aggregates(bureau)
    assert agg.loc[1, "BUREAU_ACTIVE_LOAN_COUNT"] == 2
    assert agg.loc[2, "BUREAU_ACTIVE_LOAN_COUNT"] == 0
    assert agg.loc[3, "BUREAU_ACTIVE_LOAN_COUNT"] == 1


def test_delinquency_count_counts_overdue_rows_not_days(bureau):
    agg = compute_bureau_aggregates(bureau)
    # applicant 1 has one row with CREDIT_DAY_OVERDUE=5 -> one delinquent credit line
    assert agg.loc[1, "BUREAU_DELINQUENCY_COUNT"] == 1
    assert agg.loc[2, "BUREAU_DELINQUENCY_COUNT"] == 1


def test_attach_merges_onto_application_frame(bureau):
    application_df = pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4]})
    merged = attach_bureau_aggregates(application_df, bureau)
    assert list(merged["BUREAU_ACTIVE_LOAN_COUNT"]) == [2, 0, 1, 0]
    assert list(merged["BUREAU_DELINQUENCY_COUNT"]) == [1, 1, 0, 0]


def test_attach_does_not_mutate_input(bureau):
    application_df = pd.DataFrame({"SK_ID_CURR": [1, 2]})
    original_columns = list(application_df.columns)
    attach_bureau_aggregates(application_df, bureau)
    assert list(application_df.columns) == original_columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest research/tests/test_bureau_aggregates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.data.bureau_aggregates'`

- [ ] **Step 3: Write the implementation**

```python
# research/data/bureau_aggregates.py
"""Aggregate bureau.csv into per-applicant counts for the canonical core.

bureau.csv is many-rows-per-applicant (one row per prior credit line at
another institution). The canonical schema needs one row per SK_ID_CURR, so
this module reduces it before `research/data/specs/home_credit.py` can treat
`delinquencies`/`active_loans` as NATIVE fields instead of ABSENT.

An applicant with no rows in bureau.csv gets 0 for both counts. This is a
judgment call, not a neutral default: it reads "no bureau record" as "no
active loans / no delinquencies found", which is the standard convention for
this field in the Home Credit competition, not a claim that we know their
true bureau history. Documented here so it is not mistaken for imputation of
an unknown value.
"""

from __future__ import annotations

import pandas as pd

ACTIVE_LOAN_COL = "BUREAU_ACTIVE_LOAN_COUNT"
DELINQUENCY_COL = "BUREAU_DELINQUENCY_COUNT"


def compute_bureau_aggregates(bureau: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR: active loan count and delinquency count."""
    grouped = bureau.groupby("SK_ID_CURR")
    active = grouped["CREDIT_ACTIVE"].apply(lambda s: int((s == "Active").sum()))
    delinquent = grouped["CREDIT_DAY_OVERDUE"].apply(lambda s: int((s > 0).sum()))
    return pd.DataFrame({ACTIVE_LOAN_COL: active, DELINQUENCY_COL: delinquent})


def attach_bureau_aggregates(
    application_df: pd.DataFrame, bureau_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-merge bureau aggregates onto an application frame by SK_ID_CURR."""
    aggregates = compute_bureau_aggregates(bureau_df)
    merged = application_df.merge(
        aggregates, how="left", left_on="SK_ID_CURR", right_index=True
    )
    merged[ACTIVE_LOAN_COL] = merged[ACTIVE_LOAN_COL].fillna(0).astype(int)
    merged[DELINQUENCY_COL] = merged[DELINQUENCY_COL].fillna(0).astype(int)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest research/tests/test_bureau_aggregates.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add research/data/bureau_aggregates.py research/tests/test_bureau_aggregates.py
git commit -m "feat: aggregate bureau.csv into per-applicant active-loan/delinquency counts"
```

---

### Task 2: Wire bureau aggregates into the Home Credit canonical spec

**Files:**
- Modify: `research/data/specs/home_credit.py`
- Modify: `research/tests/test_adapters.py:85-97, 149-152`

**Interfaces:**
- Consumes: `BUREAU_ACTIVE_LOAN_COUNT`, `BUREAU_DELINQUENCY_COUNT` from Task 1 (must already be present as columns on any frame passed to `build_bundle(home_credit.SPEC, df)` — Task 3 is what actually merges them onto the real file).

- [ ] **Step 1: Update the failing/changing tests first**

In `research/tests/test_adapters.py`, replace `test_home_credit_gaps_are_surfaced_not_hidden` (only `credit_utilization` remains absent now) and `test_absent_fields_are_all_nan_not_fabricated` (only assert `credit_utilization`, and add a positive test for the two now-native fields):

```python
    def test_home_credit_gaps_are_surfaced_not_hidden(self):
        """credit_utilization must show as ABSENT; delinquencies/active_loans
        are now NATIVE via bureau aggregation (see bureau_aggregates.py)."""
        report = coverage_report(home_credit.SPEC).set_index("canonical")
        assert report.loc["credit_utilization", "availability"] == Availability.ABSENT.value
        assert report.loc["delinquencies", "availability"] == Availability.NATIVE.value
        assert report.loc["active_loans", "availability"] == Availability.NATIVE.value
        assert set(home_credit.SPEC.missing_required()) == {"credit_utilization"}
```

And in `TestHomeCreditDerivations`, replace `test_absent_fields_are_all_nan_not_fabricated` with:

```python
    def test_credit_utilization_still_absent(self, raw):
        bundle = build_bundle(home_credit.SPEC, raw)
        assert bundle.core["credit_utilization"].isna().all()

    def test_bureau_derived_fields_populate_when_columns_present(self, raw):
        raw = raw.assign(
            BUREAU_ACTIVE_LOAN_COUNT=[1, 0],
            BUREAU_DELINQUENCY_COUNT=[0, 2],
        )
        bundle = build_bundle(home_credit.SPEC, raw)
        assert list(bundle.core["active_loans"]) == [1, 0]
        assert list(bundle.core["delinquencies"]) == [0, 2]
```

- [ ] **Step 2: Run to verify these fail**

Run: `.venv/Scripts/python -m pytest research/tests/test_adapters.py -v -k "gaps_are_surfaced or bureau_derived or credit_utilization_still_absent"`
Expected: FAIL — `active_loans`/`delinquencies` still report `ABSENT` in the current spec, and the new columns are unmapped so `bundle.core["active_loans"]` is all-NaN, not `[1, 0]`.

- [ ] **Step 3: Update the spec**

In `research/data/specs/home_credit.py`, replace the two `FieldSpec` entries and the module docstring's fact #2:

```python
        FieldSpec(
            "delinquencies",
            A.NATIVE,
            "BUREAU_DELINQUENCY_COUNT",
            Unit.COUNT,
            "Count of bureau.csv credit lines with CREDIT_DAY_OVERDUE > 0 for "
            "this applicant; 0 if the applicant has no bureau.csv rows. See "
            "research/data/bureau_aggregates.py. Requires the caller to merge "
            "bureau aggregates onto application_train before calling "
            "build_bundle — this spec cannot do the merge itself.",
        ),
        FieldSpec(
            "active_loans",
            A.NATIVE,
            "BUREAU_ACTIVE_LOAN_COUNT",
            Unit.COUNT,
            "Count of bureau.csv credit lines with CREDIT_ACTIVE == 'Active'; "
            "0 if the applicant has no bureau.csv rows. See "
            "research/data/bureau_aggregates.py.",
        ),
```

Update the module docstring (lines 6-19) to drop `delinquencies`/`active_loans` from the "NOT in application_train" list, keeping only `credit_utilization`:

```python
2. One canonical field (credit_utilization) is NOT in application_train and
   requires aggregating credit_card_balance.csv, which has not been fetched.
   It is marked ABSENT deliberately so `missing_required()` reports the gap
   rather than hiding it. delinquencies/active_loans ARE available, via
   bureau.csv aggregation (research/data/bureau_aggregates.py) — the caller
   must merge those columns onto application_train before calling
   build_bundle(); the spec only declares where the values live once merged.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest research/tests/test_adapters.py research/tests/test_bureau_aggregates.py -v`
Expected: PASS (all tests, including the previously-passing ones — check none broke)

- [ ] **Step 5: Commit**

```bash
git add research/data/specs/home_credit.py research/tests/test_adapters.py
git commit -m "feat: mark delinquencies/active_loans NATIVE via bureau aggregation"
```

---

### Task 3: Merge bureau aggregates in the CLI pipeline and re-validate against real data

**Files:**
- Modify: `research/data/cli.py:34-54`

**Interfaces:**
- Consumes: `attach_bureau_aggregates` (Task 1), `home_credit.SPEC` (Task 2).

- [ ] **Step 1: Modify `main()` to merge bureau data for the `home_credit` dataset**

In `research/data/cli.py`, after the existing `df = pd.read_csv(path, ...)` line (currently line 53), insert a home-credit-specific bureau merge:

```python
    print(f"Reading {path} ...")
    df = pd.read_csv(path, nrows=args.nrows, low_memory=False)
    print(f"  {len(df):,} rows x {len(df.columns)} columns\n")

    if args.dataset == "home_credit":
        bureau_path = path.parent / "bureau.csv"
        if bureau_path.exists():
            print(f"Merging bureau aggregates from {bureau_path} ...")
            bureau_df = pd.read_csv(bureau_path, low_memory=False)
            df = attach_bureau_aggregates(df, bureau_df)
            print(f"  merged; now {len(df.columns)} columns\n")
        else:
            print(f"  ! {bureau_path} not found — delinquencies/active_loans will be NaN\n")
```

Add the import at the top of the file:

```python
from research.data.bureau_aggregates import attach_bureau_aggregates
```

- [ ] **Step 2: Run against the real files and confirm the gap closes**

Run: `.venv/Scripts/python -m research.data.cli home_credit --out reports/`
Expected: `=== SPEC VALIDATION ===` now shows only `credit_utilization` under "required canonical fields unavailable" (down from three), and `=== CANONICAL COVERAGE ===` shows `active_loans`/`delinquencies` as `native` with a non-empty `source`.

- [ ] **Step 3: Run the full test suite once more**

Run: `.venv/Scripts/python -m pytest research/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add research/data/cli.py reports/home_credit_coverage.csv reports/home_credit_missingness.csv
git commit -m "feat: merge bureau aggregates into the home_credit CLI validation pass"
```

---

### Task 4: CBES engine redesign (7-field vocabulary)

**Files:**
- Rewrite: `backend/app/services/cbes_engine.py`
- Create: `backend/tests/test_cbes_engine.py` (check `backend/tests/` first for an existing file to extend instead)

**Interfaces:**
- Consumes: `backend/app/services/cbes_calibration.py::load_thresholds()` (Task 5) — but only **lazily**, on first real use, never at import time. Task 5 does not exist yet when this task starts; the engine must not crash on import before Task 5 provides a real artifact. See the lazy-loading pattern in Step 3.
- Produces: `compute_cbes(data: dict[str, Any]) -> tuple[float, dict[str, float]]` — **same return shape as before** (probability, breakdown dict) so `decision_engine.hybrid_decision` and any other caller of `p_cbes`/`cbes_breakdown` keeps working unchanged. Breakdown keys change from `{credit, capacity, behaviour, liquidity, stability}` to `{credit, capacity, behaviour, stability, region}`.
- New input vocabulary (replaces the old 15-key dict): `credit_score` (raw `EXT_SOURCE_2`, `[0,1]`), `delinquencies` (count), `active_loans` (count), `dti` (ratio), `employment_tenure_years` (float, `NaN` allowed), `annual_income` (currency), `loan_amount` (currency), `region` (`1`/`2`/`3`, optional).

This task's tests stub the thresholds via monkeypatch (Step 1), so they never trigger a real `load_thresholds()` call — Task 4 is fully testable before Task 5 exists, and Task 5 (run later) is what makes a real, non-stubbed call succeed.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_cbes_engine.py
from __future__ import annotations

import math

import pytest

from backend.app.services import cbes_engine


FAKE_THRESHOLDS = {
    # percentile breakpoints: 5 edges -> 4 bands, mapped to [0, 0.25, 0.5, 0.75, 1.0]
    "credit_score": [0.10, 0.30, 0.50, 0.70, 0.90],
    "delinquencies": [0, 0, 1, 2, 5],
    "active_loans": [0, 1, 2, 4, 8],
    "dti": [0.05, 0.15, 0.25, 0.40, 0.80],
    "employment_tenure_years": [0.0, 1.0, 3.0, 7.0, 15.0],
    "loan_to_income": [0.5, 1.5, 3.0, 5.0, 10.0],
}


@pytest.fixture(autouse=True)
def stub_thresholds(monkeypatch):
    monkeypatch.setattr(cbes_engine, "_THRESHOLDS", FAKE_THRESHOLDS)


def test_returns_probability_and_five_component_breakdown():
    p_cbes, breakdown = cbes_engine.compute_cbes(
        {
            "credit_score": 0.75,
            "delinquencies": 0,
            "active_loans": 1,
            "dti": 0.10,
            "employment_tenure_years": 5.0,
            "annual_income": 500_000.0,
            "loan_amount": 800_000.0,
            "region": 1,
        }
    )
    assert 0.0 <= p_cbes <= 1.0
    assert set(breakdown) == {"credit", "capacity", "behaviour", "stability", "region"}
    assert all(0.0 <= v <= 1.0 for v in breakdown.values())


def test_strong_profile_scores_higher_than_weak_profile():
    strong = cbes_engine.compute_cbes(
        {
            "credit_score": 0.85,
            "delinquencies": 0,
            "active_loans": 1,
            "dti": 0.08,
            "employment_tenure_years": 10.0,
            "annual_income": 900_000.0,
            "loan_amount": 500_000.0,
            "region": 1,
        }
    )[0]
    weak = cbes_engine.compute_cbes(
        {
            "credit_score": 0.15,
            "delinquencies": 4,
            "active_loans": 6,
            "dti": 0.60,
            "employment_tenure_years": 0.2,
            "annual_income": 150_000.0,
            "loan_amount": 900_000.0,
            "region": 3,
        }
    )[0]
    assert strong > weak


def test_missing_fields_use_conservative_defaults_not_crash():
    p_cbes, breakdown = cbes_engine.compute_cbes({})
    assert 0.0 <= p_cbes <= 1.0
    assert not any(math.isnan(v) for v in breakdown.values())


def test_nan_employment_tenure_does_not_crash():
    # DAYS_EMPLOYED sentinel -> NaN, per research/data/specs/home_credit.py
    p_cbes, _ = cbes_engine.compute_cbes(
        {
            "credit_score": 0.5,
            "delinquencies": 0,
            "active_loans": 0,
            "dti": 0.2,
            "employment_tenure_years": float("nan"),
            "annual_income": 300_000.0,
            "loan_amount": 300_000.0,
        }
    )
    assert 0.0 <= p_cbes <= 1.0


def test_zero_income_does_not_produce_infinite_ratio():
    p_cbes, _ = cbes_engine.compute_cbes(
        {
            "credit_score": 0.5,
            "delinquencies": 0,
            "active_loans": 0,
            "dti": 0.2,
            "employment_tenure_years": 2.0,
            "annual_income": 0.0,
            "loan_amount": 300_000.0,
        }
    )
    assert 0.0 <= p_cbes <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest backend/tests/test_cbes_engine.py -v`
Expected: FAIL — old `compute_cbes` signature/keys don't match (`KeyError`/`AttributeError` on `_THRESHOLDS`, wrong breakdown keys).

- [ ] **Step 3: Rewrite the engine**

```python
# backend/app/services/cbes_engine.py
"""CBES: hand-designed heuristic risk score, redesigned for Home Credit fields.

Replaces the prior 15-key India-specific vocabulary (cibil_score,
residential_assets_value, ...), which has no Home Credit equivalent. This
version consumes exactly the 7 fields the canonical data layer
(research/data/canonical.py) can actually populate for Home Credit:
credit_score (EXT_SOURCE_2 proxy), delinquencies, active_loans (both from
bureau.csv via research/data/bureau_aggregates.py), dti, employment_tenure_years,
an income/loan-amount affordability ratio, and an optional region rule.

Thresholds are percentile breakpoints computed from the real training
distribution by cbes_calibration.py — see load_thresholds() — not hand-picked
bank conventions. EXT_SOURCE_2 has no established "prime/subprime" cutoff the
way a real bureau score does, so treating it as a real-world scale would be
scientifically dishonest; percentile bands make the rule set legible instead
("bottom 20% of applicants by this dataset's own score distribution").
"""

from __future__ import annotations

import math
from typing import Any

from backend.app.services.cbes_calibration import load_thresholds

# Conservative fallback defaults: as bad as observed data in this dataset gets,
# so a missing field never masks risk as neutral.
DEFAULTS: dict[str, float] = {
    "credit_score": 0.0,
    "delinquencies": 10.0,
    "active_loans": 10.0,
    "dti": 1.0,
    "employment_tenure_years": 0.0,
    "annual_income": 1.0,
    "loan_amount": 10_000_000.0,
    "region": 3.0,
}

# Loaded lazily, not at import time: Task 5 (cbes_calibration.py) may not have
# produced a real artifact yet when this module is first imported (e.g. during
# Task 4's own test collection), and this module must not crash on import
# because of that. Tests stub this via monkeypatch before compute_cbes runs;
# real callers get it filled in on first use.
_THRESHOLDS: dict[str, list[float]] | None = None


def _get_thresholds() -> dict[str, list[float]]:
    global _THRESHOLDS
    if _THRESHOLDS is None:
        _THRESHOLDS = load_thresholds()
    return _THRESHOLDS


def _safe_float(val: Any, default: float) -> float:
    if val is None:
        return default
    try:
        f = float(val)
    except (ValueError, TypeError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _percentile_score(value: float, breakpoints: list[float], higher_is_better: bool) -> float:
    """Map a raw value onto [0, 1] via 5 percentile breakpoints (p10/p30/p50/p70/p90).

    Values below breakpoints[0] or above breakpoints[-1] clip to 0.0/1.0.
    `higher_is_better=False` inverts the scale (e.g. more delinquencies = worse).
    """
    edges = breakpoints
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = float(_interp(value, edges, positions))
    return score if higher_is_better else 1.0 - score


def _interp(value: float, edges: list[float], positions: list[float]) -> float:
    if value <= edges[0]:
        return positions[0]
    if value >= edges[-1]:
        return positions[-1]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= value <= hi:
            if hi == lo:
                return positions[i]
            frac = (value - lo) / (hi - lo)
            return positions[i] + frac * (positions[i + 1] - positions[i])
    return positions[-1]


def component_sigmoid(x: float) -> float:
    """k=4: softer curve, output spans [0.27, 0.73] rather than [0.02, 0.98]."""
    return 1.0 / (1.0 + math.exp(-4.0 * (x - 0.5)))


def compute_cbes(data: dict[str, Any]) -> tuple[float, dict[str, float]]:
    thresholds = _get_thresholds()
    credit_score = _safe_float(data.get("credit_score"), DEFAULTS["credit_score"])
    delinquencies = _safe_float(data.get("delinquencies"), DEFAULTS["delinquencies"])
    active_loans = _safe_float(data.get("active_loans"), DEFAULTS["active_loans"])
    dti = _safe_float(data.get("dti"), DEFAULTS["dti"])
    tenure = _safe_float(data.get("employment_tenure_years"), DEFAULTS["employment_tenure_years"])
    income = max(_safe_float(data.get("annual_income"), DEFAULTS["annual_income"]), 1.0)
    loan_amount = max(_safe_float(data.get("loan_amount"), DEFAULTS["loan_amount"]), 0.0)
    region = _safe_float(data.get("region"), DEFAULTS["region"])

    loan_to_income = loan_amount / income

    # 1. CREDIT (w=0.35): external score + delinquency history
    credit_raw = 0.70 * _percentile_score(
        credit_score, thresholds["credit_score"], higher_is_better=True
    ) + 0.30 * _percentile_score(
        delinquencies, thresholds["delinquencies"], higher_is_better=False
    )
    credit_final = component_sigmoid(credit_raw)

    # 2. CAPACITY (w=0.30): debt-to-income + loan-to-income affordability
    capacity_raw = 0.60 * _percentile_score(
        dti, thresholds["dti"], higher_is_better=False
    ) + 0.40 * _percentile_score(
        loan_to_income, thresholds["loan_to_income"], higher_is_better=False
    )
    capacity_final = component_sigmoid(capacity_raw)

    # 3. BEHAVIOUR (w=0.20): concurrent active credit lines
    behaviour_raw = _percentile_score(
        active_loans, thresholds["active_loans"], higher_is_better=False
    )
    behaviour_final = component_sigmoid(behaviour_raw)

    # 4. STABILITY (w=0.10): employment tenure
    stability_raw = _percentile_score(
        tenure, thresholds["employment_tenure_years"], higher_is_better=True
    )
    stability_final = component_sigmoid(stability_raw)

    # 5. REGION (w=0.05): urbanicity proxy, low weight, deliberately not a
    # geography claim (REGION_RATING_CLIENT is ordinal 1=best, 3=worst).
    region_raw = 1.0 - _percentile_score(region, [1, 1, 2, 3, 3], higher_is_better=True)
    region_final = component_sigmoid(region_raw)

    CBES_raw = (
        0.35 * credit_final
        + 0.30 * capacity_final
        + 0.20 * behaviour_final
        + 0.10 * stability_final
        + 0.05 * region_final
    )
    p_cbes = 1.0 / (1.0 + math.exp(-5.0 * (CBES_raw - 0.5)))

    breakdown = {
        "credit": float(credit_final),
        "capacity": float(capacity_final),
        "behaviour": float(behaviour_final),
        "stability": float(stability_final),
        "region": float(region_final),
    }
    return float(p_cbes), breakdown
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest backend/tests/test_cbes_engine.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/cbes_engine.py backend/tests/test_cbes_engine.py
git commit -m "feat: redesign CBES engine around 7 real Home Credit fields"
```

---

### Task 5: Percentile threshold calibration script

**Files:**
- Create: `backend/app/services/cbes_calibration.py`
- Create: `backend/tests/test_cbes_calibration.py`

**Interfaces:**
- Produces: `load_thresholds(path: Path | None = None) -> dict[str, list[float]]` — used by Task 4's `cbes_engine.py`.
- Produces: `compute_thresholds(core_df: pd.DataFrame) -> dict[str, list[float]]` — takes a canonical-core-shaped DataFrame (columns: `credit_score`, `delinquencies`, `active_loans`, `dti`, `employment_tenure_years`, `loan_to_income`) and returns 5-point percentile breakpoints (p10/p30/p50/p70/p90) per column, NaN-safe.
- Produces: a `__main__` block runnable as `python -m backend.app.services.cbes_calibration`, which loads the real Home Credit data via `research.data.adapters.build_bundle` + `research.data.bureau_aggregates.attach_bureau_aggregates`, computes thresholds, and writes them to `backend/artifacts/cbes_thresholds.json`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_cbes_calibration.py
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.app.services.cbes_calibration import compute_thresholds, load_thresholds


def test_compute_thresholds_returns_five_point_breakpoints_per_column():
    core_df = pd.DataFrame(
        {
            "credit_score": np.linspace(0.0, 1.0, 100),
            "delinquencies": np.arange(100),
            "active_loans": np.arange(100),
            "dti": np.linspace(0.0, 1.0, 100),
            "employment_tenure_years": np.linspace(0.0, 20.0, 100),
            "loan_to_income": np.linspace(0.1, 10.0, 100),
        }
    )
    thresholds = compute_thresholds(core_df)
    assert set(thresholds) == {
        "credit_score",
        "delinquencies",
        "active_loans",
        "dti",
        "employment_tenure_years",
        "loan_to_income",
    }
    for column, edges in thresholds.items():
        assert len(edges) == 5, column
        assert edges == sorted(edges), f"{column} breakpoints must be non-decreasing"


def test_compute_thresholds_ignores_nan(): 
    core_df = pd.DataFrame(
        {
            "credit_score": [0.1, 0.5, np.nan, 0.9] * 25,
            "delinquencies": [0, 1, 2, 3] * 25,
            "active_loans": [0, 1, 2, 3] * 25,
            "dti": [0.1, 0.2, 0.3, 0.4] * 25,
            "employment_tenure_years": [1.0, 2.0, np.nan, 4.0] * 25,
            "loan_to_income": [1.0, 2.0, 3.0, 4.0] * 25,
        }
    )
    thresholds = compute_thresholds(core_df)
    assert all(np.isfinite(e) for e in thresholds["credit_score"])
    assert all(np.isfinite(e) for e in thresholds["employment_tenure_years"])


def test_load_thresholds_round_trips_through_json(tmp_path):
    payload = {"credit_score": [0.1, 0.3, 0.5, 0.7, 0.9]}
    path = tmp_path / "cbes_thresholds.json"
    path.write_text(json.dumps(payload))
    loaded = load_thresholds(path)
    assert loaded == payload


def test_load_thresholds_missing_file_raises_actionable_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="cbes_calibration"):
        load_thresholds(missing)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest backend/tests/test_cbes_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.app.services.cbes_calibration'`

- [ ] **Step 3: Write the implementation**

```python
# backend/app/services/cbes_calibration.py
"""Compute CBES's percentile-band thresholds from the real Home Credit data.

CBES needs a "high/low" scale for fields with no established real-world
convention (EXT_SOURCE_2 has no bank-standard prime/subprime cutoff the way a
CIBIL score does). Rather than invent one, this script derives 5-point
percentile breakpoints (p10/p30/p50/p70/p90) from the training distribution
itself, and documents them as dataset-derived — not a claimed banking
standard. Run this whenever the underlying data changes; cbes_engine.py loads
the result at import time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
THRESHOLDS_PATH = ARTIFACTS_DIR / "cbes_thresholds.json"

_PERCENTILES = (10, 30, 50, 70, 90)

_COLUMNS = (
    "credit_score",
    "delinquencies",
    "active_loans",
    "dti",
    "employment_tenure_years",
    "loan_to_income",
)


def compute_thresholds(core_df: pd.DataFrame) -> dict[str, list[float]]:
    """5-point percentile breakpoints per CBES input column, NaN-safe."""
    thresholds: dict[str, list[float]] = {}
    for column in _COLUMNS:
        values = core_df[column].to_numpy(dtype="float64")
        values = values[~np.isnan(values)]
        edges = [float(np.percentile(values, p)) for p in _PERCENTILES]
        # Guarantee non-decreasing edges even if a column is near-constant.
        for i in range(1, len(edges)):
            if edges[i] < edges[i - 1]:
                edges[i] = edges[i - 1]
        thresholds[column] = edges
    return thresholds


def load_thresholds(path: Path | None = None) -> dict[str, list[float]]:
    target = path or THRESHOLDS_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Run "
            "`python -m backend.app.services.cbes_calibration` first to "
            "compute CBES's percentile thresholds from real data."
        )
    return json.loads(target.read_text())


def _build_core_frame() -> pd.DataFrame:
    """Load Home Credit, attach bureau aggregates, return the canonical core
    plus the derived loan_to_income ratio CBES needs."""
    from research.data.adapters import build_bundle
    from research.data.bureau_aggregates import attach_bureau_aggregates
    from research.data.specs import home_credit

    project_root = Path(__file__).resolve().parents[3]
    application_df = pd.read_csv(project_root / home_credit.SOURCE_FILE, low_memory=False)
    bureau_df = pd.read_csv(
        project_root / "data/raw/home_credit/bureau.csv", low_memory=False
    )
    merged = attach_bureau_aggregates(application_df, bureau_df)
    bundle = build_bundle(home_credit.SPEC, merged)
    core = bundle.core.copy()
    core["loan_to_income"] = core["loan_amount"] / core["annual_income"].replace(0, np.nan)
    return core


def main() -> int:
    core_df = _build_core_frame()
    thresholds = compute_thresholds(core_df)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))
    print(f"Wrote {THRESHOLDS_PATH}")
    for column, edges in thresholds.items():
        print(f"  {column}: {edges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `.venv/Scripts/python -m pytest backend/tests/test_cbes_calibration.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the real calibration and confirm Task 4's engine loads it**

Run: `.venv/Scripts/python -m backend.app.services.cbes_calibration`
Expected: prints 6 threshold lists and writes `backend/artifacts/cbes_thresholds.json`.

Run: `.venv/Scripts/python -m pytest backend/tests/test_cbes_engine.py backend/tests/test_cbes_calibration.py -v`
Expected: PASS — unchanged from Step 4/earlier, since `test_cbes_engine.py`'s fixture still stubs `_THRESHOLDS` directly. This step exists to confirm a *real*, non-stubbed `compute_cbes` call (e.g. from Task 7's integration test) would now succeed via `_get_thresholds()` lazily calling `load_thresholds()` against the real artifact, instead of raising `FileNotFoundError`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/cbes_calibration.py backend/tests/test_cbes_calibration.py backend/artifacts/cbes_thresholds.json
git commit -m "feat: calibrate CBES thresholds from real Home Credit percentiles"
```

---

### Task 6: Retire the synthetic dataset

**Files:**
- Delete: `backend/generate_indian_loan_dataset.py`
- Delete: the synthetic CSV (locate exact path first — likely repo root or `backend/`, confirm with `git ls-files | grep synthetic_indian_loan_dataset`)
- Modify (pointer-note comment only, ~4 lines each, at the top of the file below any module docstring): `backend/retrain_pipeline_v2.py`, `backend/run_evaluation.py`, `backend/compute_baselines.py`, `backend/training_comparison.py`, `backend/run_calibration_report.py`, `backend/app/services/ml_service.py`

**Interfaces:** none — this task has no runtime interface, only file removal and comments.

- [ ] **Step 1: Confirm exact paths before deleting**

Run: `git ls-files | grep -i synthetic_indian_loan_dataset`
Expected: lists `backend/generate_indian_loan_dataset.py` and the CSV's actual tracked path (confirm whether it's committed at all — `backend/training.py:30` uses a relative path `synthetic_indian_loan_dataset.csv`, so check both repo root and `backend/`).

- [ ] **Step 2: Delete the files**

```bash
git rm backend/generate_indian_loan_dataset.py
git rm <exact CSV path found in Step 1>
```

- [ ] **Step 3: Add pointer-note comments to now-broken scripts**

For each of `backend/retrain_pipeline_v2.py`, `backend/run_evaluation.py`, `backend/compute_baselines.py`, `backend/training_comparison.py`, `backend/run_calibration_report.py`, insert this block immediately after the module's existing docstring/imports (adapt the file name in the first line):

```python
# NOTE (2026-08-30): this script's data source (synthetic_indian_loan_dataset.csv)
# has been deleted — see docs/superpowers/specs/2026-08-29-home-credit-swap-design.md
# section 3.4. It will not run until the deferred training work (same spec,
# section 2a) rebuilds a Home Credit-based training pipeline. Left in place,
# not fixed, so this history isn't silently lost.
```

In `backend/app/services/ml_service.py`, add the same note immediately above the `train_pipeline` function definition (find it with `grep -n "def train_pipeline" backend/app/services/ml_service.py` first).

- [ ] **Step 4: Confirm nothing else in the currently-working test suite imports the deleted files**

Run: `.venv/Scripts/python -m pytest backend/ research/ -v 2>&1 | tail -60`
Expected: no `ModuleNotFoundError`/`FileNotFoundError` for the deleted files from tests that were passing before this task (pre-existing failures in scripts that already depended on synthetic data, e.g. any test that imports `retrain_pipeline_v2`, are expected and out of scope — note them, don't fix them).

- [ ] **Step 5: Commit**

```bash
git add backend/retrain_pipeline_v2.py backend/run_evaluation.py backend/compute_baselines.py backend/training_comparison.py backend/run_calibration_report.py backend/app/services/ml_service.py
git commit -m "chore: retire synthetic dataset generator, note now-broken training scripts"
```

---

### Task 7: End-to-end integration check (CBES only, no model)

**Files:**
- Create: `backend/tests/test_cbes_home_credit_integration.py`

**Interfaces:**
- Consumes: `research.data.adapters.build_bundle`, `research.data.bureau_aggregates.attach_bureau_aggregates`, `research.data.specs.home_credit`, `backend.app.services.cbes_engine.compute_cbes`, `backend.app.services.decision_engine.hybrid_decision`.

- [ ] **Step 1: Write the integration test**

```python
# backend/tests/test_cbes_home_credit_integration.py
"""Proves the new CBES engine runs end-to-end on real Home Credit rows,
through to a hybrid decision, with the ML side stubbed (no model this pass)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.app.services.cbes_engine import compute_cbes
from backend.app.services.decision_engine import hybrid_decision
from research.data.adapters import build_bundle
from research.data.bureau_aggregates import attach_bureau_aggregates
from research.data.specs import home_credit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPLICATION_CSV = PROJECT_ROOT / home_credit.SOURCE_FILE
BUREAU_CSV = PROJECT_ROOT / "data/raw/home_credit/bureau.csv"


@pytest.fixture(scope="module")
def sample_core() -> pd.DataFrame:
    if not APPLICATION_CSV.exists() or not BUREAU_CSV.exists():
        pytest.skip("Home Credit raw data not present in data/raw/home_credit/")
    application_df = pd.read_csv(APPLICATION_CSV, nrows=200, low_memory=False)
    bureau_df = pd.read_csv(BUREAU_CSV, low_memory=False)
    merged = attach_bureau_aggregates(application_df, bureau_df)
    bundle = build_bundle(home_credit.SPEC, merged)
    core = bundle.core.copy()
    core["loan_to_income"] = core["loan_amount"] / core["annual_income"].replace(0, float("nan"))
    return core


def test_every_sample_row_produces_a_decision_without_key_errors(sample_core):
    for _, row in sample_core.iterrows():
        p_cbes, breakdown = compute_cbes(row.to_dict())
        assert 0.0 <= p_cbes <= 1.0
        result = hybrid_decision(p_ml=0.5, p_cbes=p_cbes, tau_d=0.43, cbes_breakdown=breakdown)
        assert result.decision in {"APPROVE", "REJECT", "DEFER"}
```

- [ ] **Step 2: Run to verify it fails first (before Task 5's real artifact exists, or if the artifact path is wrong)**

Run: `.venv/Scripts/python -m pytest backend/tests/test_cbes_home_credit_integration.py -v`
Expected: at this point in the plan (after Tasks 1-6 are already done) this should PASS immediately — if it fails, the most likely cause is `cbes_thresholds.json` missing (re-run Task 5 Step 5) or a column name mismatch between `sample_core` and `_COLUMNS` in `cbes_calibration.py`.

- [ ] **Step 3: Fix any mismatch found, then confirm pass**

Run: `.venv/Scripts/python -m pytest backend/tests/test_cbes_home_credit_integration.py -v`
Expected: PASS

- [ ] **Step 4: Run the full test suite one last time**

Run: `.venv/Scripts/python -m pytest backend/ research/ -v`
Expected: PASS, except pre-existing/known-broken scripts noted in Task 6 Step 4.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_cbes_home_credit_integration.py
git commit -m "test: integration check CBES against real Home Credit rows end-to-end"
```

---

## After this plan

- Spec §3.3 (interim API/frontend schema) and all model/SHAP work remain open, tracked in `docs/superpowers/specs/2026-08-29-home-credit-swap-design.md` §2a and the "Descoped" note above — not part of this plan.
- `credit_card_balance.csv` / `credit_utilization`, `previous_application.csv`, and the C4 synthetic-regeneration need (spec §4) remain deferred, as already logged in the spec.
