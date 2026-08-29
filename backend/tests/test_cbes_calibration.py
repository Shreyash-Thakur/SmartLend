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
