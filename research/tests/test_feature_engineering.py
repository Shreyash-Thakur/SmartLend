"""Unit tests for the v2 feature derivations — the pure logic, on toy frames.

The expensive build is exercised by running the module; what must be provably
right here is the arithmetic and the honesty rules: sentinel cleaning, safe
division, thin-file handling (absence stays NaN, never zero), and the
aggregate definitions the research claims will quote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.features.engineer import (
    DAYS_EMPLOYED_SENTINEL,
    _safe_div,
    application_features,
    bureau_features,
)


def _toy_application() -> pd.DataFrame:
    base = {
        "AMT_INCOME_TOTAL": [100000.0, 200000.0],
        "AMT_CREDIT": [500000.0, 400000.0],
        "AMT_ANNUITY": [25000.0, np.nan],
        "AMT_GOODS_PRICE": [450000.0, 400000.0],
        "CNT_FAM_MEMBERS": [2.0, 4.0],
        "CNT_CHILDREN": [0, 2],
        "DAYS_BIRTH": [-12000, -18000],
        "DAYS_EMPLOYED": [-2000, DAYS_EMPLOYED_SENTINEL],
        "OWN_CAR_AGE": [5.0, np.nan],
        "DAYS_REGISTRATION": [-4000.0, -9000.0],
        "DAYS_ID_PUBLISH": [-3000, -5000],
        "DAYS_LAST_PHONE_CHANGE": [-100.0, -2000.0],
        "EXT_SOURCE_1": [0.5, np.nan],
        "EXT_SOURCE_2": [0.6, 0.4],
        "EXT_SOURCE_3": [0.7, np.nan],
        "REGION_RATING_CLIENT": [1, 3],
        "DEF_30_CNT_SOCIAL_CIRCLE": [1.0, 0.0],
        "OBS_30_CNT_SOCIAL_CIRCLE": [4.0, 0.0],
        "FLAG_MOBIL": [1, 1], "FLAG_EMP_PHONE": [1, 0], "FLAG_WORK_PHONE": [0, 0],
        "FLAG_CONT_MOBILE": [1, 1], "FLAG_PHONE": [0, 1], "FLAG_EMAIL": [1, 0],
        "FLAG_DOCUMENT_3": [1, 0], "FLAG_DOCUMENT_6": [0, 0], "FLAG_DOCUMENT_8": [1, 1],
    }
    return pd.DataFrame(base)


class TestSafeDiv:
    def test_zero_denominator_is_nan_not_inf(self):
        out = _safe_div(pd.Series([1.0, 2.0]), pd.Series([0.0, 4.0]))
        assert np.isnan(out.iloc[0]) and out.iloc[1] == 0.5


class TestApplicationFeatures:
    def test_sentinel_days_employed_becomes_nan(self):
        df = application_features(_toy_application())
        assert np.isnan(df.loc[1, "DAYS_EMPLOYED"])
        # ...and therefore does not poison the ratio.
        assert np.isnan(df.loc[1, "APP_EMPLOYED_TO_AGE"])

    def test_burden_ratios(self):
        df = application_features(_toy_application())
        assert df.loc[0, "APP_ANNUITY_TO_INCOME"] == pytest.approx(0.25)
        assert df.loc[0, "APP_CREDIT_TO_ANNUITY"] == pytest.approx(20.0)  # term proxy
        assert np.isnan(df.loc[1, "APP_ANNUITY_TO_INCOME"])  # missing annuity stays missing

    def test_ext_source_combinations(self):
        df = application_features(_toy_application())
        assert df.loc[0, "APP_EXT_MEAN"] == pytest.approx((0.5 + 0.6 + 0.7) / 3)
        assert df.loc[1, "APP_EXT_MISSING"] == 2
        # Products require ALL factors present (min_count), not NaN-as-1.
        assert df.loc[0, "APP_EXT_PROD"] == pytest.approx(0.30)
        assert np.isnan(df.loc[1, "APP_EXT_PROD"])

    def test_flag_summaries(self):
        df = application_features(_toy_application())
        assert df.loc[0, "APP_DOC_COUNT"] == 2
        assert df.loc[1, "APP_CONTACT_COUNT"] == 3
        # 0/0 in the social-circle ratio is NaN, not 0 and not inf.
        assert np.isnan(df.loc[1, "APP_DEF_30_TO_OBS_30"])


class TestBureauFeatures:
    def _toy_bureau(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SK_ID_CURR": [1, 1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12, 20],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed"],
            "CREDIT_TYPE": ["Consumer credit", "Credit card", "Consumer credit", "Car loan"],
            "DAYS_CREDIT": [-100, -900, -400, -2000],
            "DAYS_CREDIT_ENDDATE": [300.0, -10.0, 500.0, -500.0],
            "DAYS_CREDIT_UPDATE": [-5, -300, -20, -1000],
            "CREDIT_DAY_OVERDUE": [0, 0, 30, 0],
            "CNT_CREDIT_PROLONG": [0, 1, 0, 0],
            "AMT_CREDIT_SUM": [100000.0, 50000.0, 200000.0, 80000.0],
            "AMT_CREDIT_SUM_DEBT": [60000.0, 0.0, 150000.0, np.nan],
            "AMT_CREDIT_SUM_LIMIT": [0.0, 0.0, 0.0, 0.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 5000.0, 0.0],
            "AMT_CREDIT_MAX_OVERDUE": [np.nan, 1000.0, 5000.0, 0.0],
            "AMT_ANNUITY": [5000.0, np.nan, 10000.0, 2000.0],
        })

    def test_counts_and_recency(self):
        agg = bureau_features(self._toy_bureau()).set_index("SK_ID_CURR")
        assert agg.loc[1, "BURO_loan_count"] == 3
        assert agg.loc[1, "BURO_active_count"] == 2
        assert agg.loc[1, "BURO_recent_12m_count"] == 1  # only DAYS_CREDIT -100
        assert agg.loc[1, "BURO_credit_type_nunique"] == 2
        assert agg.loc[1, "BURO_days_credit_max"] == -100  # most recent

    def test_active_only_exposure(self):
        agg = bureau_features(self._toy_bureau()).set_index("SK_ID_CURR")
        assert agg.loc[1, "BURO_active_credit_sum"] == pytest.approx(300000.0)
        assert agg.loc[1, "BURO_active_debt_sum"] == pytest.approx(210000.0)
        # Applicant 2 has no active loans: active-only aggregates are NaN, not 0.
        assert np.isnan(agg.loc[2, "BURO_active_credit_sum"])

    def test_portfolio_ratios(self):
        agg = bureau_features(self._toy_bureau()).set_index("SK_ID_CURR")
        assert agg.loc[1, "BURO_active_share"] == pytest.approx(2 / 3)
        assert agg.loc[1, "BURO_total_debt_to_credit"] == pytest.approx(210000.0 / 350000.0)
        assert agg.loc[1, "BURO_overdue_to_debt"] == pytest.approx(5000.0 / 210000.0)
