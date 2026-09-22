"""Feature engineering v2 — the measured path to more accuracy.

Every accuracy avenue except this one is measured and exhausted: tuning found
nothing (reports/tuning_cpu.json), no roster hybrid beats XGBoost
(reports/complementarity.json — CBES, tree family, TabPFN, TabFM all KILL),
and the learning curve says same-kind data still helps but we have all of it
(reports/convergence.json). What the project has NOT used is most of the
information on disk: the current frame carries only 9 coarse aggregates from
bureau.csv, and none of the interaction/ratio features that every published
top solution to this exact dataset leaned on.

Grounding (public Home Credit solutions; the open-solution repo reached
private LB 0.798 with feature diversity + stacking, winners ~0.805 using ALL
side tables — we hold application_train + bureau only, so ~0.78–0.79 is the
realistic ceiling for this feature set):

* application ratios — annuity/income (payment burden), credit/annuity
  (term proxy), credit/income, credit/goods_price, income per family member,
  employed/age; EXT_SOURCE mean/std/min/max/product and age-weighted combos;
  document-flag counts; row-missingness count.
* bureau aggregates — history depth and recency (DAYS_CREDIT stats), active
  vs closed counts, debt/credit/overdue sums and ratios, prolongation counts,
  credit-type diversity, recent-12-months activity, plus the same core sums
  restricted to ACTIVE loans.

Honesty rules baked in:
* Only application_train + bureau are used (the files actually on disk);
  no leakage from anything derived from TARGET.
* The Home Credit sentinel DAYS_EMPLOYED == 365243 becomes NaN, never a value.
* Missing stays missing (NaN) for the tree models; "no bureau file" is also
  expressed as an explicit BURO_has_bureau flag, consistent with the
  project's no-imputing-away-thin-files rule.
* This feeds the RESEARCH reference models only. The serving artifact keeps
  its 15-feature train/serve contract untouched.

Run:  python -m research.features.engineer
Cache: data/processed/features_v2.pkl (gitignored — derived from Kaggle data)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
APPLICATION_CSV = REPO / "data" / "raw" / "home_credit" / "application_train.csv"
BUREAU_CSV = REPO / "data" / "raw" / "home_credit" / "bureau.csv"
CACHE = REPO / "data" / "processed" / "features_v2.pkl"

DAYS_EMPLOYED_SENTINEL = 365243


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise a/b with 0-denominators and infs mapped to NaN."""
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Domain ratios + EXT_SOURCE combinations on the application table.

    Returns a copy with engineered columns added (prefix APP_) and the
    DAYS_EMPLOYED sentinel cleaned.
    """
    df = app.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)

    # Payment burden and loan-shape ratios — the classic top-importance set.
    df["APP_ANNUITY_TO_INCOME"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["APP_CREDIT_TO_INCOME"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["APP_CREDIT_TO_ANNUITY"] = _safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])  # ~term in months
    df["APP_CREDIT_TO_GOODS"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
    df["APP_INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["APP_INCOME_PER_CHILD"] = _safe_div(df["AMT_INCOME_TOTAL"], 1 + df["CNT_CHILDREN"])

    # Life-stage ratios (all DAYS_* are negative day counts from application).
    df["APP_EMPLOYED_TO_AGE"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["APP_CAR_TO_AGE"] = _safe_div(df["OWN_CAR_AGE"], -df["DAYS_BIRTH"] / 365.25)
    df["APP_REGISTRATION_TO_AGE"] = _safe_div(df["DAYS_REGISTRATION"], df["DAYS_BIRTH"])
    df["APP_ID_PUBLISH_TO_AGE"] = _safe_div(df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"])
    df["APP_PHONE_CHANGE_TO_AGE"] = _safe_div(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"])

    # EXT_SOURCE combinations — the strongest signals in every public solution.
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["APP_EXT_MEAN"] = ext.mean(axis=1)
    df["APP_EXT_STD"] = ext.std(axis=1)
    df["APP_EXT_MIN"] = ext.min(axis=1)
    df["APP_EXT_MAX"] = ext.max(axis=1)
    df["APP_EXT_MISSING"] = ext.isna().sum(axis=1)
    df["APP_EXT_PROD"] = ext[["EXT_SOURCE_1", "EXT_SOURCE_2"]].prod(min_count=2, axis=1)
    df["APP_EXT_PROD3"] = ext.prod(min_count=3, axis=1)
    df["APP_EXT1_TO_AGE"] = _safe_div(df["EXT_SOURCE_1"], -df["DAYS_BIRTH"] / 365.25)
    df["APP_EXT2_X_EXT3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    # Document and contact-flag summaries instead of 20 near-empty binaries.
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["APP_DOC_COUNT"] = df[doc_cols].sum(axis=1)
    contact_cols = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                    "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
    df["APP_CONTACT_COUNT"] = df[contact_cols].sum(axis=1)

    # Row missingness is itself informative on this dataset (thin files).
    df["APP_MISSING_COUNT"] = df.isna().sum(axis=1)

    # Region/social interactions.
    df["APP_REGION_RATING_X_INCOME"] = df["REGION_RATING_CLIENT"] * np.log1p(df["AMT_INCOME_TOTAL"])
    df["APP_DEF_30_TO_OBS_30"] = _safe_div(df["DEF_30_CNT_SOCIAL_CIRCLE"], df["OBS_30_CNT_SOCIAL_CIRCLE"])
    return df


def bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Per-applicant aggregates over the full bureau table (prefix BURO_).

    One row per SK_ID_CURR present in bureau. Applicants absent from bureau
    (the 14.3% thin-file population) simply won't join — the caller keeps them
    as NaN + BURO_has_bureau = 0, never zero-filled.
    """
    b = bureau.copy()
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["IS_CLOSED"] = (b["CREDIT_ACTIVE"] == "Closed").astype(int)
    b["IS_RECENT_12M"] = (b["DAYS_CREDIT"] >= -365).astype(int)
    b["DEBT_TO_CREDIT"] = _safe_div(b["AMT_CREDIT_SUM_DEBT"], b["AMT_CREDIT_SUM"])
    b["ENDDATE_IN_FUTURE"] = (b["DAYS_CREDIT_ENDDATE"] > 0).astype(int)

    g = b.groupby("SK_ID_CURR")
    agg = pd.DataFrame({
        "BURO_loan_count": g.size(),
        "BURO_active_count": g["IS_ACTIVE"].sum(),
        "BURO_closed_count": g["IS_CLOSED"].sum(),
        "BURO_recent_12m_count": g["IS_RECENT_12M"].sum(),
        "BURO_credit_type_nunique": g["CREDIT_TYPE"].nunique(),
        "BURO_prolong_sum": g["CNT_CREDIT_PROLONG"].sum(),
        # History depth and recency (DAYS_CREDIT is negative; max = most recent).
        "BURO_days_credit_min": g["DAYS_CREDIT"].min(),
        "BURO_days_credit_max": g["DAYS_CREDIT"].max(),
        "BURO_days_credit_mean": g["DAYS_CREDIT"].mean(),
        "BURO_days_credit_std": g["DAYS_CREDIT"].std(),
        "BURO_days_update_max": g["DAYS_CREDIT_UPDATE"].max(),
        "BURO_enddate_future_sum": g["ENDDATE_IN_FUTURE"].sum(),
        # Exposure.
        "BURO_credit_sum_sum": g["AMT_CREDIT_SUM"].sum(),
        "BURO_credit_sum_mean": g["AMT_CREDIT_SUM"].mean(),
        "BURO_credit_sum_max": g["AMT_CREDIT_SUM"].max(),
        "BURO_debt_sum": g["AMT_CREDIT_SUM_DEBT"].sum(),
        "BURO_debt_mean": g["AMT_CREDIT_SUM_DEBT"].mean(),
        "BURO_limit_sum": g["AMT_CREDIT_SUM_LIMIT"].sum(),
        "BURO_annuity_sum": g["AMT_ANNUITY"].sum(),
        # Distress.
        "BURO_overdue_sum": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "BURO_overdue_max": g["AMT_CREDIT_SUM_OVERDUE"].max(),
        "BURO_max_overdue_max": g["AMT_CREDIT_MAX_OVERDUE"].max(),
        "BURO_day_overdue_max": g["CREDIT_DAY_OVERDUE"].max(),
        "BURO_day_overdue_mean": g["CREDIT_DAY_OVERDUE"].mean(),
        "BURO_debt_to_credit_mean": g["DEBT_TO_CREDIT"].mean(),
        "BURO_debt_to_credit_max": g["DEBT_TO_CREDIT"].max(),
    })

    # The same core exposure numbers restricted to ACTIVE loans — what the
    # applicant is carrying right now, not lifetime history.
    active = b[b["IS_ACTIVE"] == 1].groupby("SK_ID_CURR")
    agg["BURO_active_credit_sum"] = active["AMT_CREDIT_SUM"].sum()
    agg["BURO_active_debt_sum"] = active["AMT_CREDIT_SUM_DEBT"].sum()
    agg["BURO_active_annuity_sum"] = active["AMT_ANNUITY"].sum()
    agg["BURO_active_days_credit_max"] = active["DAYS_CREDIT"].max()

    # Portfolio-level ratios computed AFTER aggregation.
    agg["BURO_active_share"] = _safe_div(agg["BURO_active_count"], agg["BURO_loan_count"])
    agg["BURO_total_debt_to_credit"] = _safe_div(agg["BURO_debt_sum"], agg["BURO_credit_sum_sum"])
    agg["BURO_overdue_to_debt"] = _safe_div(agg["BURO_overdue_sum"], agg["BURO_debt_sum"])
    return agg.reset_index()


def build_feature_matrix(use_cache: bool = True):
    """(X: float32 DataFrame, y: ndarray, ids: ndarray) for the v2 feature set.

    Object columns are one-hot encoded (dummy_na so missingness stays visible);
    everything numeric is kept, NaNs preserved for the tree models.
    """
    if use_cache and CACHE.exists():
        X, y, ids = pd.read_pickle(CACHE)
        print(f"[features-v2] cache hit: {X.shape[0]:,} rows x {X.shape[1]} features")
        return X, y, ids

    started = time.time()
    app = pd.read_csv(APPLICATION_CSV, low_memory=False)
    print(f"[features-v2] application_train: {app.shape}")
    app = application_features(app)

    bureau = pd.read_csv(BUREAU_CSV, low_memory=False)
    print(f"[features-v2] bureau: {bureau.shape}")
    buro = bureau_features(bureau)
    app = app.merge(buro, on="SK_ID_CURR", how="left")
    app["BURO_has_bureau"] = app["SK_ID_CURR"].isin(buro["SK_ID_CURR"]).astype(int)

    y = app["TARGET"].astype(int).to_numpy()
    ids = app["SK_ID_CURR"].to_numpy()
    X = app.drop(columns=["TARGET", "SK_ID_CURR"])

    obj_cols = X.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(X, columns=obj_cols, dummy_na=True)
    # LightGBM rejects JSON-special characters in feature names, and dummy
    # levels carry commas/colons/spaces. Sanitize and de-duplicate.
    clean = X.columns.str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True).str.strip("_")
    X.columns = pd.io.common.dedup_names(clean, is_potential_multiindex=False)
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((X, y, ids), CACHE)
    print(f"[features-v2] built {X.shape[0]:,} rows x {X.shape[1]} features "
          f"in {time.time() - started:.1f}s -> {CACHE}")
    return X, y, ids


if __name__ == "__main__":
    build_feature_matrix(use_cache=False)
