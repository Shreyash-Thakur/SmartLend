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
