"""Tests for the missingness profiler.

Fixtures deliberately reproduce the real Home Credit patterns the detectors
exist to catch, so a passing test means the detector would fire on the actual
data rather than on a toy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.data.profile import (
    Missingness,
    add_missingness_indicators,
    apply_sentinels,
    detect_blocks,
    detect_sentinels,
    detect_structural,
    profile_frame,
    score_informativeness,
    to_frame,
)

RNG = np.random.default_rng(7)
N = 4_000
SENTINEL = 365243


@pytest.fixture
def home_credit_like() -> pd.DataFrame:
    """A frame carrying each Home Credit missingness pathology."""
    days_employed = -RNG.integers(100, 8_000, N).astype(float)
    # ~18% sentinel-coded, matching the real ~55k/307k share.
    sentinel_mask = RNG.random(N) < 0.18
    days_employed[sentinel_mask] = SENTINEL

    owns_car = RNG.random(N) < 0.34
    own_car_age = np.where(owns_car, RNG.integers(1, 25, N).astype(float), np.nan)

    # Three columns that go missing together (the building-characteristics block).
    block_missing = RNG.random(N) < 0.6
    building = {
        f"building_{i}": np.where(block_missing, np.nan, RNG.normal(size=N))
        for i in range(3)
    }

    # Missingness that predicts the target: the informative case.
    informative_missing = RNG.random(N) < 0.3
    ext_source = np.where(informative_missing, np.nan, RNG.random(N))
    target = np.where(
        informative_missing,
        (RNG.random(N) < 0.30).astype(int),  # 30% default when missing
        (RNG.random(N) < 0.06).astype(int),  # 6% when present
    )

    frame = pd.DataFrame(
        {
            "DAYS_EMPLOYED": days_employed,
            "FLAG_OWN_CAR": np.where(owns_car, "Y", "N"),
            "OWN_CAR_AGE": own_car_age,
            "EXT_SOURCE_1": ext_source,
            "AMT_INCOME_TOTAL": RNG.lognormal(11, 0.5, N),
            "CNT_CHILDREN": RNG.integers(0, 4, N),
            "TARGET": target,
            **building,
        }
    )
    # A single-row gap: the genuinely-MAR trace case.
    frame.loc[0, "AMT_INCOME_TOTAL"] = np.nan
    return frame


class TestDetectSentinels:
    def test_finds_home_credit_days_employed_sentinel(self, home_credit_like):
        assert detect_sentinels(home_credit_like["DAYS_EMPLOYED"]) == (float(SENTINEL),)

    def test_no_false_positive_on_ordinary_distribution(self, home_credit_like):
        assert detect_sentinels(home_credit_like["AMT_INCOME_TOTAL"]) == ()

    def test_no_false_positive_on_low_cardinality_counts(self, home_credit_like):
        # CNT_CHILDREN's max holds real mass but is not a coded value.
        assert detect_sentinels(home_credit_like["CNT_CHILDREN"]) == ()

    def test_ignores_non_numeric(self, home_credit_like):
        assert detect_sentinels(home_credit_like["FLAG_OWN_CAR"]) == ()

    def test_ignores_zero_spread(self):
        assert detect_sentinels(pd.Series([5.0] * 100)) == ()

    def test_requires_minimum_mass(self):
        # One extreme outlier is an outlier, not a sentinel.
        series = pd.Series(list(RNG.normal(size=1000)) + [1e9])
        assert detect_sentinels(series) == ()

    def test_finds_sentinels_in_zero_inflated_counts(self):
        """Regression: Give-Me-Some-Credit's 96/98 delinquency codes.

        The earlier IQR-based detector returned nothing here, because a
        zero-inflated count column has q1 == median == q3 == 0 and therefore
        zero IQR. Real credit delinquency columns are always shaped like this,
        so the miss would have carried straight into Home Credit.
        """
        values = list(RNG.integers(0, 14, 149_731)) + [96] * 5 + [98] * 264
        assert detect_sentinels(pd.Series(values)) == (96.0, 98.0)

    def test_finds_sentinel_at_the_low_end(self):
        values = list(RNG.integers(20, 60, 5_000)) + [-999] * 40
        assert detect_sentinels(pd.Series(values)) == (-999.0,)

    def test_does_not_flag_a_legitimate_second_cluster(self):
        # Two dense populations are bimodality, not coding.
        values = list(RNG.integers(0, 50, 3_000)) + list(RNG.integers(900, 950, 3_000))
        assert detect_sentinels(pd.Series(values)) == ()


class TestDetectStructural:
    def test_finds_deterministic_predictor(self, home_credit_like):
        # OWN_CAR_AGE is missing exactly when FLAG_OWN_CAR == 'N'.
        assert (
            detect_structural(home_credit_like, "OWN_CAR_AGE") == "FLAG_OWN_CAR"
        )

    def test_returns_none_for_random_missingness(self, home_credit_like):
        assert detect_structural(home_credit_like, "EXT_SOURCE_1") is None

    def test_returns_none_when_nothing_missing(self, home_credit_like):
        assert detect_structural(home_credit_like, "CNT_CHILDREN") is None

    def test_ignores_high_cardinality_predictors(self):
        # A unique-per-row id can "predict" anything; it must not count.
        frame = pd.DataFrame(
            {"id": range(200), "value": [np.nan if i % 2 else 1.0 for i in range(200)]}
        )
        assert detect_structural(frame, "value", max_predictor_cardinality=50) is None


class TestDetectBlocks:
    def test_groups_co_missing_columns(self, home_credit_like):
        blocks = detect_blocks(
            home_credit_like,
            ["building_0", "building_1", "building_2", "EXT_SOURCE_1", "OWN_CAR_AGE"],
        )
        building_ids = {blocks[f"building_{i}"] for i in range(3)}
        assert len(building_ids) == 1, "the three building columns form one block"
        # Independently-missing columns must not join that block.
        assert blocks.get("EXT_SOURCE_1") not in building_ids
        assert blocks.get("OWN_CAR_AGE") not in building_ids

    def test_returns_empty_without_shared_pattern(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
        assert detect_blocks(frame, ["a", "b"]) == {}


class TestScoreInformativeness:
    def test_detects_informative_missingness(self, home_credit_like):
        auc, rate_missing, rate_present = score_informativeness(
            home_credit_like["EXT_SOURCE_1"].isna(), home_credit_like["TARGET"]
        )
        assert auc is not None and auc > 0.6
        assert rate_missing > rate_present

    def test_uninformative_missingness_scores_near_half(self):
        target = pd.Series(RNG.integers(0, 2, 3_000))
        is_missing = pd.Series(RNG.random(3_000) < 0.3)
        auc, _, _ = score_informativeness(is_missing, target)
        assert auc == pytest.approx(0.5, abs=0.05)

    def test_perfect_separation_scores_one(self):
        target = pd.Series([0, 0, 1, 1])
        is_missing = pd.Series([False, False, True, True])
        auc, _, _ = score_informativeness(is_missing, target)
        assert auc == pytest.approx(1.0)

    def test_returns_none_for_constant_input(self):
        auc, _, _ = score_informativeness(
            pd.Series([False] * 10), pd.Series([0, 1] * 5)
        )
        assert auc is None


class TestProfileFrame:
    @pytest.fixture
    def profiles(self, home_credit_like):
        return {
            p.column: p
            for p in profile_frame(home_credit_like, target="TARGET", sample_size=None)
        }

    def test_classifies_each_mechanism(self, profiles):
        assert profiles["DAYS_EMPLOYED"].classification is Missingness.SENTINEL_CODED
        assert profiles["OWN_CAR_AGE"].classification is Missingness.STRUCTURAL
        assert profiles["EXT_SOURCE_1"].classification is Missingness.INFORMATIVE
        assert profiles["AMT_INCOME_TOTAL"].classification is Missingness.MAR_TRACE
        assert profiles["CNT_CHILDREN"].classification is Missingness.COMPLETE

    def test_target_column_excluded(self, profiles):
        assert "TARGET" not in profiles

    def test_sentinel_rate_reported_separately_from_missing_rate(self, profiles):
        p = profiles["DAYS_EMPLOYED"]
        # The sentinel is a value, not a gap, so missing_rate stays 0...
        assert p.missing_rate == 0.0
        # ...but the effective rate exposes the truth.
        assert p.sentinel_rate > 0.15
        assert p.effective_missing_rate > 0.15

    def test_every_profile_carries_a_recommendation(self, profiles):
        assert all(p.recommendation for p in profiles.values())

    def test_report_frame_puts_sentinels_first(self, home_credit_like):
        report = to_frame(
            profile_frame(home_credit_like, target="TARGET", sample_size=None)
        )
        assert report.iloc[0]["classification"] == Missingness.SENTINEL_CODED.value


class TestTransformations:
    def test_apply_sentinels_converts_to_nan(self, home_credit_like):
        profiles = profile_frame(home_credit_like, target="TARGET", sample_size=None)
        cleaned = apply_sentinels(home_credit_like, profiles)

        assert (cleaned["DAYS_EMPLOYED"] == SENTINEL).sum() == 0
        assert cleaned["DAYS_EMPLOYED"].isna().sum() > 0
        # Original must be untouched.
        assert (home_credit_like["DAYS_EMPLOYED"] == SENTINEL).sum() > 0

    def test_sentinel_removal_makes_statistics_sane(self, home_credit_like):
        profiles = profile_frame(home_credit_like, target="TARGET", sample_size=None)
        cleaned = apply_sentinels(home_credit_like, profiles)
        # Before: a bogus ~1000-year maximum. After: bounded by real tenure.
        assert home_credit_like["DAYS_EMPLOYED"].max() == SENTINEL
        assert cleaned["DAYS_EMPLOYED"].max() < 0

    def test_indicators_collapse_blocks(self, home_credit_like):
        profiles = profile_frame(home_credit_like, target="TARGET", sample_size=None)
        out = add_missingness_indicators(home_credit_like, profiles)

        building_indicators = [
            c for c in out.columns if c.startswith("building_") and c.endswith("_isna")
        ]
        assert building_indicators == [], "block members must not emit individual flags"
        assert len([c for c in out.columns if c.startswith("block") and c.endswith("_isna")]) == 1

    def test_indicators_added_for_informative_and_structural(self, home_credit_like):
        profiles = profile_frame(home_credit_like, target="TARGET", sample_size=None)
        out = add_missingness_indicators(home_credit_like, profiles)

        assert "EXT_SOURCE_1_isna" in out.columns
        assert "OWN_CAR_AGE_isna" in out.columns
        # MAR trace is imputable; no indicator needed.
        assert "AMT_INCOME_TOTAL_isna" not in out.columns

    def test_indicators_are_int8(self, home_credit_like):
        profiles = profile_frame(home_credit_like, target="TARGET", sample_size=None)
        out = add_missingness_indicators(home_credit_like, profiles)
        assert out["EXT_SOURCE_1_isna"].dtype == "int8"
