# CBES Diagnostics, Customer Explanation Module & Relearning-Loop Capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three independent, buildable-now pieces from the CBES research spec — (1) read-only diagnostic scripts validating CBES's weights against real data, (2) a customer-facing explanation module for APPROVE/REJECT/DEFER decisions, (3) a `deferred_review` data-capture table for the future relearning loop — none of which require a trained model, SHAP, or touching the live API.

**Architecture:** Diagnostics reuse `cbes_calibration._build_core_frame()` (already builds the canonical core + `loan_to_income` from real Home Credit data) rather than duplicating that logic. The explanation module reads `cbes_engine.py`'s component weights (exported as a new constant, not duplicated) and `decision_engine.DecisionResult`'s actual fields. The capture table is a plain SQLAlchemy model auto-created by the existing `init_db()` pattern, with a service module that only ever writes rows — no code path anywhere reads `human_decision` as a training label.

**Tech Stack:** Python 3.13, pandas, numpy, pytest, SQLAlchemy (existing stack — no new dependencies; VIF and WOE/IV are implemented directly with numpy/pandas rather than adding `statsmodels`).

**Spec:** `docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md` (primary), `docs/superpowers/specs/2026-08-29-home-credit-swap-design.md` (prior CBES context)

## Global Constraints

- No model training, no SHAP, no live API/frontend schema changes in this plan (still descoped per the prior spec).
- Diagnostics are read-only: they must NOT change `cbes_engine.py`'s production weights (35/30/20/10/5) — per the spec, no weight change is evidence-backed, only the export-a-constant refactor in Task 3.
- The explanation module is a new, independently testable module — not wired into `public_api_service.py` or the live predict endpoint.
- The `deferred_review` capture table is capture-only: no retraining trigger, no code path treats `human_decision` as a training label (spec §3, "Explicitly do not build yet").
- No new third-party dependencies.

---

## File Structure

- **Create** `backend/app/services/cbes_diagnostics.py` — VIF, WOE/IV, and weight-sensitivity functions, plus a `main()` that runs all three against real data and writes a report.
- **Create** `backend/tests/test_cbes_diagnostics.py` — unit tests for the three diagnostic functions on synthetic fixtures.
- **Create** `reports/cbes_diagnostics_report.md` — generated output from running `main()` against real Home Credit data (committed, like the existing `reports/home_credit_*.csv` files).
- **Modify** `backend/app/services/cbes_engine.py` — export `COMPONENT_WEIGHTS` as a module-level constant instead of inline literals in `compute_cbes`, so the explanation module doesn't duplicate the weights.
- **Create** `backend/app/services/explanation_service.py` — reason-code catalog, ranking rule, three per-decision templates.
- **Create** `backend/tests/test_explanation_service.py` — unit tests proving each template's shape and content rules.
- **Modify** `backend/app/models.py` — add `DeferredReview` SQLAlchemy model.
- **Create** `backend/app/services/deferred_review_service.py` — `record_deferred_review()` and `maybe_route_to_exploration()`.
- **Create** `backend/tests/test_deferred_review_service.py` — unit tests using a temp SQLite DB.

---

### Task 1: VIF (multicollinearity) check

**Files:**
- Create: `backend/app/services/cbes_diagnostics.py`
- Test: `backend/tests/test_cbes_diagnostics.py`

**Interfaces:**
- Produces: `compute_vif(df: pd.DataFrame, columns: list[str]) -> dict[str, float]` — VIF per column, computed via OLS (numpy, no `statsmodels`): for each column `i`, regress it on all other listed columns (with an intercept) and compute `VIF_i = 1 / (1 - R²_i)`. Rows with any NaN across `columns` are dropped before computing (documented, not imputed).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_cbes_diagnostics.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.services.cbes_diagnostics import compute_vif


def test_independent_columns_have_low_vif():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=1000),
            "b": rng.normal(size=1000),
            "c": rng.normal(size=1000),
        }
    )
    vif = compute_vif(df, ["a", "b", "c"])
    assert set(vif) == {"a", "b", "c"}
    assert all(v < 2.0 for v in vif.values()), vif


def test_perfectly_correlated_columns_have_high_vif():
    rng = np.random.default_rng(42)
    base = rng.normal(size=1000)
    df = pd.DataFrame({"x": base, "y": base * 2.0 + 1.0, "z": rng.normal(size=1000)})
    vif = compute_vif(df, ["x", "y", "z"])
    assert vif["x"] > 50.0
    assert vif["y"] > 50.0
    assert vif["z"] < 2.0


def test_nan_rows_are_dropped_not_imputed():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, np.nan, 14.0, 16.0],
        }
    )
    # Should not raise, and should compute over the 6 fully-observed rows.
    vif = compute_vif(df, ["a", "b"])
    assert set(vif) == {"a", "b"}
    assert all(np.isfinite(v) for v in vif.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_cbes_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.app.services.cbes_diagnostics'`

- [ ] **Step 3: Write the implementation**

```python
# backend/app/services/cbes_diagnostics.py
"""Read-only diagnostics validating CBES's inputs/weights against real data.

Per docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md
section 1: these are diagnostics, not a retuning mechanism. No finding in
that research endorses changing CBES's 35/30/20/10/5 weights, so nothing
here writes back into cbes_engine.py's production constants — it only
reports numbers for a human to read and, if warranted, document.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vif(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Variance Inflation Factor per column via OLS (numpy, no statsmodels).

    VIF_i = 1 / (1 - R²_i), where R²_i comes from regressing column i on all
    other listed columns (with an intercept). Rows with any NaN across
    `columns` are dropped before computing — documented, not imputed, per the
    project's missingness discipline (research/data/canonical.py).
    """
    clean = df[columns].dropna()
    n = len(clean)
    vif: dict[str, float] = {}
    for target in columns:
        predictors = [c for c in columns if c != target]
        y = clean[target].to_numpy(dtype="float64")
        X = clean[predictors].to_numpy(dtype="float64")
        X_with_intercept = np.column_stack([np.ones(n), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        y_pred = X_with_intercept @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif[target] = 1.0 / max(1.0 - r_squared, 1e-9)
    return vif
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_cbes_diagnostics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/cbes_diagnostics.py backend/tests/test_cbes_diagnostics.py
git commit -m "feat: add VIF multicollinearity diagnostic for CBES inputs"
```

---

### Task 2: WOE/IV comparison against Home Credit's TARGET

**Files:**
- Modify: `backend/app/services/cbes_diagnostics.py`
- Modify: `backend/tests/test_cbes_diagnostics.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent function in the same file).
- Produces: `compute_woe_iv(df: pd.DataFrame, column: str, target: pd.Series, n_bins: int = 10) -> float` — Information Value for one column against a binary target, via decile (or fewer, if the column has few unique values) binning and the standard WOE/IV formula. `compute_iv_ranking(df: pd.DataFrame, columns: list[str], target: pd.Series) -> dict[str, float]` — IV per column, for comparing against CBES's weight ordering.

- [ ] **Step 1: Write the failing tests**

```python
# append to backend/tests/test_cbes_diagnostics.py
from backend.app.services.cbes_diagnostics import compute_iv_ranking, compute_woe_iv


def test_woe_iv_is_zero_for_uninformative_column():
    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({"noise": rng.normal(size=n)})
    # Target independent of "noise"
    target = pd.Series(rng.integers(0, 2, size=n))
    iv = compute_woe_iv(df, "noise", target)
    assert iv < 0.05, f"expected near-zero IV for an uninformative column, got {iv}"


def test_woe_iv_is_high_for_perfectly_separating_column():
    n = 2000
    # "score" perfectly separates target: low score -> target=1, high score -> target=0
    df = pd.DataFrame({"score": list(range(n))})
    target = pd.Series([1] * (n // 2) + [0] * (n // 2))
    iv = compute_woe_iv(df, "score", target)
    assert iv > 0.5, f"expected high IV for a perfectly-separating column, got {iv}"


def test_iv_ranking_covers_all_requested_columns():
    rng = np.random.default_rng(3)
    n = 1000
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    target = pd.Series(rng.integers(0, 2, size=n))
    ranking = compute_iv_ranking(df, ["a", "b"], target)
    assert set(ranking) == {"a", "b"}
    assert all(v >= 0.0 for v in ranking.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_cbes_diagnostics.py -v -k woe_iv or iv_ranking`
Expected: FAIL with `ImportError: cannot import name 'compute_woe_iv'`

- [ ] **Step 3: Add the implementation**

```python
# append to backend/app/services/cbes_diagnostics.py

def compute_woe_iv(df: pd.DataFrame, column: str, target: pd.Series, n_bins: int = 10) -> float:
    """Information Value of one column against a binary target (1=bad/default).

    Standard credit-scoring WOE/IV: bin the column into n_bins (or fewer,
    for low-cardinality columns), compute the log-odds of good-vs-bad per
    bin (WOE), and sum the weighted WOE differences (IV). Rows with NaN in
    either the column or the target are dropped.
    """
    frame = pd.DataFrame({"value": df[column], "target": target.to_numpy()}).dropna()
    try:
        frame["bin"] = pd.qcut(frame["value"], q=n_bins, duplicates="drop")
    except ValueError:
        frame["bin"] = frame["value"]

    total_good = float((frame["target"] == 0).sum())
    total_bad = float((frame["target"] == 1).sum())
    if total_good == 0 or total_bad == 0:
        return 0.0

    iv = 0.0
    for _, group in frame.groupby("bin", observed=True):
        good = float((group["target"] == 0).sum())
        bad = float((group["target"] == 1).sum())
        pct_good = max(good / total_good, 1e-6)
        pct_bad = max(bad / total_bad, 1e-6)
        woe = np.log(pct_good / pct_bad)
        iv += (pct_good - pct_bad) * woe
    return float(iv)


def compute_iv_ranking(df: pd.DataFrame, columns: list[str], target: pd.Series) -> dict[str, float]:
    """IV per column, for comparing against CBES's hand-assigned weight ordering."""
    return {column: compute_woe_iv(df, column, target) for column in columns}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_cbes_diagnostics.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/cbes_diagnostics.py backend/tests/test_cbes_diagnostics.py
git commit -m "feat: add WOE/IV diagnostic comparing CBES inputs against TARGET"
```

---

### Task 3: Export CBES component weights as a named constant

**Files:**
- Modify: `backend/app/services/cbes_engine.py`
- Modify: `backend/tests/test_cbes_engine.py`

**Interfaces:**
- Produces: `COMPONENT_WEIGHTS: dict[str, float] = {"credit": 0.35, "capacity": 0.30, "behaviour": 0.20, "stability": 0.10, "region": 0.05}` — module-level constant.
- Modifies: `compute_cbes(data: dict[str, Any], weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]` — adds an optional `weights` parameter (defaults to `COMPONENT_WEIGHTS`) so Task 4's sensitivity sweep can pass perturbed weights without duplicating `compute_cbes`'s logic. Existing callers that don't pass `weights` see no behavior change.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_cbes_engine.py
from backend.app.services.cbes_engine import COMPONENT_WEIGHTS, compute_cbes


def test_component_weights_constant_matches_production_values():
    assert COMPONENT_WEIGHTS == {
        "credit": 0.35,
        "capacity": 0.30,
        "behaviour": 0.20,
        "stability": 0.10,
        "region": 0.05,
    }


def test_compute_cbes_accepts_weight_override_without_changing_default_behavior():
    payload = {
        "credit_score": 0.6, "delinquencies": 1, "active_loans": 2,
        "dti": 0.2, "employment_tenure_years": 4.0,
        "annual_income": 400_000.0, "loan_amount": 600_000.0, "region": 1,
    }
    default_p, _ = compute_cbes(payload)
    explicit_p, _ = compute_cbes(payload, weights=COMPONENT_WEIGHTS)
    assert default_p == pytest.approx(explicit_p)

    overridden_weights = dict(COMPONENT_WEIGHTS)
    overridden_weights["credit"] = 0.50
    overridden_weights["region"] = 0.0
    overridden_p, _ = compute_cbes(payload, weights=overridden_weights)
    assert overridden_p != pytest.approx(default_p)
```

Note: this test file already stubs `_THRESHOLDS` via an autouse fixture (from the prior plan) — no changes needed there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_cbes_engine.py -v -k "component_weights or weight_override"`
Expected: FAIL with `ImportError: cannot import name 'COMPONENT_WEIGHTS'`

- [ ] **Step 3: Modify `cbes_engine.py`**

Add the constant near the top (after `DEFAULTS`):

```python
# Component weights, exported so diagnostics (cbes_diagnostics.py) and the
# explanation module (explanation_service.py) don't duplicate these values.
COMPONENT_WEIGHTS: dict[str, float] = {
    "credit": 0.35,
    "capacity": 0.30,
    "behaviour": 0.20,
    "stability": 0.10,
    "region": 0.05,
}
```

Change `compute_cbes`'s signature and its `CBES_raw` computation:

```python
def compute_cbes(
    data: dict[str, Any], weights: dict[str, float] | None = None
) -> tuple[float, dict[str, float]]:
    w = weights or COMPONENT_WEIGHTS
    thresholds = _get_thresholds()
    # ... (unchanged extraction/derivation lines above CBES_raw) ...

    CBES_raw = (
        w["credit"] * credit_final
        + w["capacity"] * capacity_final
        + w["behaviour"] * behaviour_final
        + w["stability"] * stability_final
        + w["region"] * region_final
    )
    # ... (unchanged from here down) ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_cbes_engine.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/cbes_engine.py backend/tests/test_cbes_engine.py
git commit -m "refactor: export CBES component weights as a named constant"
```

---

### Task 4: Weight sensitivity sweep + real-data report

**Files:**
- Modify: `backend/app/services/cbes_diagnostics.py`
- Modify: `backend/tests/test_cbes_diagnostics.py`
- Create: `reports/cbes_diagnostics_report.md`

**Interfaces:**
- Consumes: `backend.app.services.cbes_engine.compute_cbes` and `COMPONENT_WEIGHTS` (Task 3 — `compute_cbes` now accepts an optional `weights` override), `backend.app.services.cbes_calibration._build_core_frame` (existing, from the prior plan — builds the canonical core + `loan_to_income` from real Home Credit data).
- Produces: `run_weight_sensitivity(core_rows: list[dict], base_weights: dict[str, float], perturbation_points: tuple[int, ...] = (5, 10)) -> list[dict]` — for each component and each ± perturbation point (renormalized so weights still sum to 1.0), recomputes `compute_cbes` over all `core_rows` with those weights and reports the mean absolute change in `p_cbes` versus the baseline. Returns a list of `{"component": str, "delta_points": int, "mean_abs_p_cbes_shift": float}` dicts.
- Produces: `main() -> int` — loads real Home Credit data via `cbes_calibration._build_core_frame()`, runs `compute_vif`, `compute_iv_ranking`, and `run_weight_sensitivity`, writes `reports/cbes_diagnostics_report.md`.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_cbes_diagnostics.py
from backend.app.services.cbes_diagnostics import run_weight_sensitivity
from backend.app.services.cbes_engine import COMPONENT_WEIGHTS


def test_weight_sensitivity_reports_shift_per_component_and_perturbation():
    core_rows = [
        {
            "credit_score": 0.6, "delinquencies": 1, "active_loans": 2,
            "dti": 0.2, "employment_tenure_years": 4.0,
            "annual_income": 400_000.0, "loan_amount": 600_000.0, "region": 1,
        },
        {
            "credit_score": 0.2, "delinquencies": 5, "active_loans": 6,
            "dti": 0.5, "employment_tenure_years": 0.5,
            "annual_income": 200_000.0, "loan_amount": 900_000.0, "region": 3,
        },
    ]
    report = run_weight_sensitivity(core_rows, COMPONENT_WEIGHTS, perturbation_points=(5, 10))
    components = {"credit", "capacity", "behaviour", "stability", "region"}
    seen_components = {row["component"] for row in report}
    assert seen_components == components
    seen_deltas = {row["delta_points"] for row in report}
    assert seen_deltas == {-10, -5, 5, 10}
    assert all(row["mean_abs_p_cbes_shift"] >= 0.0 for row in report)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_cbes_diagnostics.py -v -k weight_sensitivity`
Expected: FAIL with `ImportError: cannot import name 'run_weight_sensitivity'`

- [ ] **Step 3: Add the implementation**

```python
# append to backend/app/services/cbes_diagnostics.py

from backend.app.services.cbes_engine import compute_cbes


def _renormalize(weights: dict[str, float], component: str, delta_points: int) -> dict[str, float]:
    """Shift one component's weight by delta_points (out of 100), renormalizing the rest proportionally."""
    delta = delta_points / 100.0
    adjusted = dict(weights)
    adjusted[component] = max(adjusted[component] + delta, 0.0)
    other_components = [c for c in weights if c != component]
    other_total_before = sum(weights[c] for c in other_components)
    other_total_after = max(1.0 - adjusted[component], 0.0)
    if other_total_before > 0:
        for c in other_components:
            adjusted[c] = weights[c] / other_total_before * other_total_after
    return adjusted


def run_weight_sensitivity(
    core_rows: list[dict],
    base_weights: dict[str, float],
    perturbation_points: tuple[int, ...] = (5, 10),
) -> list[dict]:
    """For each component and each +/- perturbation, report mean |p_cbes shift| vs baseline."""
    baseline = [compute_cbes(row, weights=base_weights)[0] for row in core_rows]
    report: list[dict] = []
    for component in base_weights:
        for magnitude in perturbation_points:
            for sign in (1, -1):
                delta_points = sign * magnitude
                perturbed_weights = _renormalize(base_weights, component, delta_points)
                perturbed = [compute_cbes(row, weights=perturbed_weights)[0] for row in core_rows]
                shift = float(np.mean([abs(a - b) for a, b in zip(baseline, perturbed)]))
                report.append(
                    {
                        "component": component,
                        "delta_points": delta_points,
                        "mean_abs_p_cbes_shift": shift,
                    }
                )
    return report


def main() -> int:
    from pathlib import Path

    from backend.app.services.cbes_calibration import _build_core_frame
    from backend.app.services.cbes_engine import COMPONENT_WEIGHTS
    from research.data.adapters import build_bundle
    from research.data.specs import home_credit

    project_root = Path(__file__).resolve().parents[3]
    core_df = _build_core_frame()

    application_df = pd.read_csv(project_root / home_credit.SOURCE_FILE, low_memory=False)
    target = build_bundle(home_credit.SPEC, application_df).target

    diagnostic_columns = [
        "credit_score", "delinquencies", "active_loans",
        "dti", "employment_tenure_years", "loan_to_income",
    ]
    vif = compute_vif(core_df, diagnostic_columns)
    iv_ranking = compute_iv_ranking(core_df, diagnostic_columns, target)

    core_rows = core_df.fillna(
        {
            "credit_score": 0.5, "delinquencies": 0, "active_loans": 0,
            "dti": 0.3, "employment_tenure_years": 3.0,
            "loan_to_income": 2.0, "region": 2,
        }
    ).to_dict("records")
    sensitivity = run_weight_sensitivity(core_rows, COMPONENT_WEIGHTS)

    lines = [
        "# CBES Diagnostics Report",
        "",
        "Generated by `backend/app/services/cbes_diagnostics.py` against real Home Credit data.",
        "Per docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md section 1 —",
        "these are diagnostics only; no weight changes are made based on these numbers.",
        "",
        "## VIF (multicollinearity)",
        "",
        "| Column | VIF |",
        "|---|---|",
    ]
    for column, value in sorted(vif.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {column} | {value:.3f} |")

    lines += ["", "## Information Value vs. CBES weight ordering", "", "| Column | IV | CBES weight (component) |", "|---|---|---|"]
    weight_by_input = {
        "credit_score": COMPONENT_WEIGHTS["credit"],
        "delinquencies": COMPONENT_WEIGHTS["credit"],
        "active_loans": COMPONENT_WEIGHTS["behaviour"],
        "dti": COMPONENT_WEIGHTS["capacity"],
        "loan_to_income": COMPONENT_WEIGHTS["capacity"],
        "employment_tenure_years": COMPONENT_WEIGHTS["stability"],
    }
    for column, iv in sorted(iv_ranking.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {column} | {iv:.4f} | {weight_by_input.get(column, 0.0):.2f} |")

    lines += ["", "## Weight sensitivity (mean |p_cbes shift|)", "", "| Component | Delta (pts) | Mean abs shift |", "|---|---|---|"]
    for row in sensitivity:
        lines.append(f"| {row['component']} | {row['delta_points']:+d} | {row['mean_abs_p_cbes_shift']:.4f} |")

    report_path = project_root / "reports" / "cbes_diagnostics_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_cbes_diagnostics.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Run the real report generation against Home Credit data**

Run: `.venv/Scripts/python.exe -m backend.app.services.cbes_diagnostics`
Expected: prints `Wrote .../reports/cbes_diagnostics_report.md`; the file contains real VIF/IV/sensitivity numbers, not placeholders. Read the generated file afterward and sanity-check: VIF for `dti`/`loan_to_income` should be elevated (per the spec's named risk) — if it isn't, that's a real finding too, note it, don't force the report to match the hypothesis.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/cbes_diagnostics.py backend/tests/test_cbes_diagnostics.py reports/cbes_diagnostics_report.md
git commit -m "feat: run CBES weight-sensitivity sweep and generate diagnostics report from real data"
```

---

### Task 5: Customer explanation module

**Files:**
- Create: `backend/app/services/explanation_service.py`
- Test: `backend/tests/test_explanation_service.py`

**Interfaces:**
- Consumes: `backend.app.services.cbes_engine.COMPONENT_WEIGHTS` (Task 3). `backend.app.services.decision_engine.DecisionResult` fields (existing, unchanged): `.decision` (`"APPROVE"|"REJECT"|"DEFER"`), `.decision_reason` (`"ml_approve"|"ml_reject"|"cbes_fallback_approve"|"cbes_fallback_reject"|"disagreement"|"low_confidence"|"grey_zone"`), `.cbes_breakdown` (`dict[str, float]` with keys `credit/capacity/behaviour/stability/region`).
- Produces: `generate_explanation(decision: str, decision_reason: str, cbes_breakdown: dict[str, float]) -> dict` — returns `{"outcome": str, "reasons": list[dict], "rights_notice": str | None}`. Each reason dict: `{"code": str, "component": str, "weight": float, "component_score": float, "shortfall": float, "statement": str}`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_explanation_service.py
from __future__ import annotations

import pytest

from backend.app.services.explanation_service import generate_explanation

WEAK_BREAKDOWN = {"credit": 0.2, "capacity": 0.3, "behaviour": 0.6, "stability": 0.7, "region": 0.5}
STRONG_BREAKDOWN = {"credit": 0.9, "capacity": 0.85, "behaviour": 0.8, "stability": 0.9, "region": 0.9}


def test_reject_gives_top_four_reasons_ranked_by_shortfall():
    result = generate_explanation("REJECT", "ml_reject", WEAK_BREAKDOWN)
    assert result["outcome"].upper().startswith("REJECT") or "not approved" in result["outcome"].lower()
    assert len(result["reasons"]) == 4
    # weight * (1 - score): credit=0.35*0.8=0.28, capacity=0.30*0.7=0.21,
    # region=0.05*0.5=0.025, behaviour=0.20*0.4=0.08, stability=0.10*0.3=0.03
    # ranked: credit(0.28) > capacity(0.21) > behaviour(0.08) > stability(0.03) > region(0.025)
    # top 4 drops region (lowest shortfall)
    components_shown = [r["component"] for r in result["reasons"]]
    assert "region" not in components_shown
    assert components_shown[0] == "credit"
    assert components_shown[1] == "capacity"


def test_reject_includes_gdpr_rights_notice():
    result = generate_explanation("REJECT", "cbes_fallback_reject", WEAK_BREAKDOWN)
    assert result["rights_notice"] is not None
    assert "human" in result["rights_notice"].lower()


def test_reject_reasons_never_expose_raw_probabilities_or_thresholds():
    result = generate_explanation("REJECT", "ml_reject", WEAK_BREAKDOWN)
    serialized = str(result)
    for forbidden in ("p_ml", "p_cbes", "p_blend", "t_approve", "t_reject"):
        assert forbidden not in serialized


def test_approve_gives_two_weakest_components_as_informational():
    result = generate_explanation("APPROVE", "ml_approve", STRONG_BREAKDOWN)
    assert result["outcome"].upper().startswith("APPROVE") or "approved" in result["outcome"].lower()
    assert len(result["reasons"]) == 2
    assert result["rights_notice"] is None


def test_defer_never_exposes_internal_reason_strings():
    for internal_reason in ("disagreement", "low_confidence", "grey_zone"):
        result = generate_explanation("DEFER", internal_reason, WEAK_BREAKDOWN)
        serialized = str(result).lower()
        assert "disagreement" not in serialized
        assert "low_confidence" not in serialized
        assert "grey_zone" not in serialized
        assert "under human review" in serialized or "additional review" in serialized
        assert result["reasons"] == []


def test_unknown_decision_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        generate_explanation("MAYBE", "ml_approve", WEAK_BREAKDOWN)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_explanation_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.app.services.explanation_service'`

- [ ] **Step 3: Write the implementation**

```python
# backend/app/services/explanation_service.py
"""Customer-facing explanation for APPROVE/REJECT/DEFER decisions.

Built from CBES's component breakdown only — no SHAP, no trained model.
Reason-code catalog and per-decision templates are specified in
docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md
section 2. Ranking rule: shortfall_i = weight_i * (1 - component_score_i),
sorted descending; REJECT discloses the top 4 of 5 (Reg B commentary treats
more than 4 reasons as unhelpful to the applicant), APPROVE discloses the
2 weakest as informational, DEFER discloses none — it is not an outcome.
"""

from __future__ import annotations

from backend.app.services.cbes_engine import COMPONENT_WEIGHTS

REASON_CATALOG: dict[str, dict[str, str]] = {
    "credit": {"code": "CB-01", "statement": "Your credit-bureau risk score is in the lower range compared with other applicants."},
    "capacity": {"code": "CP-01", "statement": "Your existing debt is high relative to your income."},
    "behaviour": {"code": "BH-01", "statement": "You currently have more open credit accounts than most applicants."},
    "stability": {"code": "ST-01", "statement": "Your length of employment is shorter than most applicants."},
    "region": {"code": "RG-01", "statement": "Regional risk rating recorded for your application."},
}

DEFER_STATEMENT = "Your application needs additional review by a member of our team before a decision is made."
RIGHTS_NOTICE = (
    "You have the right to request human review of this decision, to express your view, "
    "and to contest it."
)

_VALID_DECISIONS = {"APPROVE", "REJECT", "DEFER"}


def _ranked_reasons(cbes_breakdown: dict[str, float]) -> list[dict]:
    ranked = []
    for component, score in cbes_breakdown.items():
        weight = COMPONENT_WEIGHTS.get(component, 0.0)
        shortfall = weight * (1.0 - score)
        catalog_entry = REASON_CATALOG.get(component, {"code": component.upper(), "statement": f"{component} factor."})
        ranked.append(
            {
                "code": catalog_entry["code"],
                "component": component,
                "weight": weight,
                "component_score": score,
                "shortfall": shortfall,
                "statement": catalog_entry["statement"],
            }
        )
    ranked.sort(key=lambda r: r["shortfall"], reverse=True)
    return ranked


def generate_explanation(decision: str, decision_reason: str, cbes_breakdown: dict[str, float]) -> dict:
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"Unknown decision {decision!r}; expected one of {_VALID_DECISIONS}")

    if decision == "DEFER":
        return {"outcome": "Application under human review.", "reasons": [], "rights_notice": None}

    ranked = _ranked_reasons(cbes_breakdown)

    if decision == "REJECT":
        return {
            "outcome": "Your application was not approved.",
            "reasons": ranked[:4],
            "rights_notice": RIGHTS_NOTICE,
        }

    # APPROVE: two weakest components, informational only.
    weakest_first = list(reversed(ranked))
    return {
        "outcome": "Your application was approved.",
        "reasons": weakest_first[:2],
        "rights_notice": None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_explanation_service.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/explanation_service.py backend/tests/test_explanation_service.py
git commit -m "feat: add customer-facing explanation module for CBES decisions"
```

---

### Task 6: `DeferredReview` capture table

**Files:**
- Modify: `backend/app/models.py`
- Test: `backend/tests/test_deferred_review_service.py` (created in Task 7; this task just adds the model and confirms it creates cleanly)

**Interfaces:**
- Produces: `DeferredReview` SQLAlchemy model (table `deferred_reviews`) with the fields from spec section 3's schema table.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_deferred_review_service.py
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.models import DeferredReview


def _memory_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_deferred_review_table_creates_and_accepts_minimal_row():
    session = _memory_session()
    row = DeferredReview(
        application_id="app-1",
        decision_reason="disagreement",
        p_ml=0.5,
        p_cbes=0.4,
        p_blend=0.475,
        disagreement=0.1,
        confidence=0.3,
        t_approve=0.55,
        t_reject=0.45,
        cbes_breakdown_json={"credit": 0.5},
        engine_version="v1",
        threshold_artifact_hash="abc123",
        t_base=0.5,
        exploration_flag=False,
    )
    session.add(row)
    session.commit()
    assert row.review_id is not None
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_deferred_review_service.py -v`
Expected: FAIL with `ImportError: cannot import name 'DeferredReview'`

- [ ] **Step 3: Add the model**

Append to `backend/app/models.py`:

```python
class DeferredReview(Base):
    """One row per DEFER decision routed to a human reviewer.

    Capture-only: this table records the engine state and (later) the human
    decision, so a future relearning pass has data to work with. Per
    docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md
    section 3, no code anywhere reads human_decision as a training label,
    and there is no retraining trigger — building that requires first
    validating the deferral rule is better-than-random (same spec, gate
    conditions).
    """

    __tablename__ = "deferred_reviews"

    review_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"rev-{uuid.uuid4().hex[:12]}")
    application_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    decision_reason: Mapped[str] = mapped_column(String, nullable=False)
    p_ml: Mapped[float] = mapped_column(Float, nullable=False)
    p_cbes: Mapped[float] = mapped_column(Float, nullable=False)
    p_blend: Mapped[float] = mapped_column(Float, nullable=False)
    disagreement: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    t_approve: Mapped[float] = mapped_column(Float, nullable=False)
    t_reject: Mapped[float] = mapped_column(Float, nullable=False)
    cbes_breakdown_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    engine_version: Mapped[str] = mapped_column(String, nullable=False)
    threshold_artifact_hash: Mapped[str] = mapped_column(String, nullable=False)
    t_base: Mapped[float] = mapped_column(Float, nullable=False)

    reviewer_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_spent_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    human_decision: Mapped[str | None] = mapped_column(String, nullable=True)
    human_reason_codes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    human_free_text: Mapped[str | None] = mapped_column(String, nullable=True)

    agreed_with_engine: Mapped[bool | None] = mapped_column(nullable=True)
    override_direction: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewer_confidence: Mapped[int | None] = mapped_column(nullable=True)

    applicant_segment_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    realized_outcome: Mapped[int | None] = mapped_column(nullable=True)
    outcome_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    outcome_censored: Mapped[bool] = mapped_column(default=True, nullable=False)

    exploration_flag: Mapped[bool] = mapped_column(default=False, nullable=False)
```

Add the needed import to the top of `backend/app/models.py`: `Boolean, Integer` from `sqlalchemy` if `Mapped[bool]`/`Mapped[int]` require explicit column types in this SQLAlchemy version — check the existing `LoanApplication` model's imports first; if `Mapped[bool]`/`Mapped[int]` without an explicit `mapped_column(Boolean)`/`mapped_column(Integer)` fails to map, add explicit types (`mapped_column(Boolean, ...)`, `mapped_column(Integer, ...)`) to match this project's SQLAlchemy version.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_deferred_review_service.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add backend/app/models.py backend/tests/test_deferred_review_service.py
git commit -m "feat: add DeferredReview capture table for future relearning loop"
```

---

### Task 7: Deferred-review recording service + exploration arm

**Files:**
- Create: `backend/app/services/deferred_review_service.py`
- Modify: `backend/tests/test_deferred_review_service.py`

**Interfaces:**
- Consumes: `DeferredReview` (Task 6), `backend.app.database.SessionLocal` (existing), `backend.app.services.decision_engine.DecisionResult` fields (existing).
- Produces: `record_deferred_review(session, decision_result, application_id: str, engine_version: str, threshold_artifact_hash: str, t_base: float) -> DeferredReview` — builds and commits a row from a `DecisionResult` (only called when `decision_result.decision == "DEFER"`; raises `ValueError` otherwise, so this can never accidentally log a non-deferred case). `t_base` is threaded through explicitly by the caller (the same value it passed to `hybrid_decision()`), since `DecisionResult` doesn't expose it separately from the derived `t_approve`/`t_reject`.
- Produces: `maybe_route_to_exploration(rate: float = 0.03) -> bool` — pure function: returns `True` with probability `rate` (default 3%, within the spec's 2-5% range), `False` otherwise. Takes no side effects; the caller decides what to do with `True`.

- [ ] **Step 1: Write the failing tests**

```python
# append to backend/tests/test_deferred_review_service.py
from dataclasses import dataclass

from backend.app.services.deferred_review_service import (
    maybe_route_to_exploration,
    record_deferred_review,
)


@dataclass
class FakeDecisionResult:
    decision: str
    decision_reason: str
    p_ml: float
    p_cbes: float
    p_blend: float
    disagreement: float
    confidence: float
    t_approve: float
    t_reject: float
    cbes_breakdown: dict


def test_record_deferred_review_persists_engine_state():
    session = _memory_session()
    result = FakeDecisionResult(
        decision="DEFER",
        decision_reason="grey_zone",
        p_ml=0.5, p_cbes=0.45, p_blend=0.4875,
        disagreement=0.05, confidence=0.15,
        t_approve=0.55, t_reject=0.45,
        cbes_breakdown={"credit": 0.4, "capacity": 0.5},
    )
    row = record_deferred_review(
        session, result, application_id="app-42",
        engine_version="v1", threshold_artifact_hash="hash-abc",
        t_base=0.50,
    )
    session.commit()
    assert row.review_id is not None
    assert row.decision_reason == "grey_zone"
    assert row.cbes_breakdown_json == {"credit": 0.4, "capacity": 0.5}
    assert row.t_base == pytest.approx(0.50)
    assert row.human_decision is None
    assert row.exploration_flag is False


def test_record_deferred_review_rejects_non_defer_decisions():
    session = _memory_session()
    result = FakeDecisionResult(
        decision="APPROVE", decision_reason="ml_approve",
        p_ml=0.8, p_cbes=0.7, p_blend=0.775,
        disagreement=0.1, confidence=0.6,
        t_approve=0.55, t_reject=0.45,
        cbes_breakdown={"credit": 0.7},
    )
    with pytest.raises(ValueError):
        record_deferred_review(
            session, result, application_id="app-43",
            engine_version="v1", threshold_artifact_hash="hash-abc",
            t_base=0.50,
        )


def test_maybe_route_to_exploration_rate_is_approximately_correct():
    trials = 20_000
    hits = sum(1 for _ in range(trials) if maybe_route_to_exploration(rate=0.03))
    observed_rate = hits / trials
    assert 0.02 < observed_rate < 0.04, f"observed exploration rate {observed_rate} outside expected band"


def test_maybe_route_to_exploration_zero_rate_never_fires():
    assert all(not maybe_route_to_exploration(rate=0.0) for _ in range(1000))


def test_maybe_route_to_exploration_full_rate_always_fires():
    assert all(maybe_route_to_exploration(rate=1.0) for _ in range(1000))
```

Add `import pytest` to the top of `backend/tests/test_deferred_review_service.py` if not already present from Task 6.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_deferred_review_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.app.services.deferred_review_service'`

- [ ] **Step 3: Write the implementation**

```python
# backend/app/services/deferred_review_service.py
"""Records DEFER decisions for future human review, and a small random
exploration arm — capture only.

Per docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md
section 3: no function here reads human_decision as a training label, and
there is no retraining trigger. This module ONLY writes rows.
"""

from __future__ import annotations

import random
from typing import Any

from sqlalchemy.orm import Session

from backend.app.models import DeferredReview


def record_deferred_review(
    session: Session,
    decision_result: Any,
    application_id: str,
    engine_version: str,
    threshold_artifact_hash: str,
    t_base: float,
) -> DeferredReview:
    """`t_base` is not exposed on `DecisionResult` (`decision_engine.py`'s
    `hybrid_decision` derives `t_approve`/`t_reject` from it but doesn't
    return the original value) — the caller already has it, since it's the
    same `t_base` argument passed into `hybrid_decision()`, so it's threaded
    through explicitly here rather than approximated from `t_approve`/
    `t_reject` (which are shifted, not equal to it)."""
    if decision_result.decision != "DEFER":
        raise ValueError(
            f"record_deferred_review called with decision={decision_result.decision!r}; "
            "only DEFER decisions are recorded here."
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
        exploration_flag=False,
    )
    session.add(row)
    return row


def maybe_route_to_exploration(rate: float = 0.03) -> bool:
    """Pure coin-flip at `rate` probability. Caller decides what routing to a
    True result means; this function has no side effects."""
    return random.random() < rate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest backend/tests/test_deferred_review_service.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest backend/ research/ -q`
Expected: PASS, no regressions from prior work.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/deferred_review_service.py backend/tests/test_deferred_review_service.py
git commit -m "feat: add deferred-review recording service and exploration-arm sampling"
```

---

## After this plan

- The explanation module and deferred-review capture are not wired into `public_api_service.py` or any live route — that remains descoped until a model is chosen (per the prior spec).
- `reports/cbes_diagnostics_report.md`'s actual numbers (especially the VIF check) may or may not confirm the spec's named risk (Capacity's `dti`/`loan_to_income` correlation) — read the real output and update the research spec's §1.5 with the actual finding once available, rather than leaving the hypothesis unconfirmed.
- The relearning loop's retraining trigger, the 4-condition gate, and wiring `maybe_route_to_exploration` into the live decision path all remain future work per spec §3.
