"""FE-v3: do the four remaining Home Credit side tables extend the measured
features-beat-models result? (pre-registered follow-up to features_vs_fusion)

CLAIM RULE (fixed here, before results): v3 counts as a gain over v2 only if
the fold-paired mean AUC delta on the IDENTICAL folds exceeds +0.0036 and
every fold's delta is positive -- the same rule as features_vs_fusion. The
v2 side of the comparison is the RECORDED fold AUCs in
reports/features_vs_fusion.json (regenerating them is unnecessary: folds,
seeds and the fitting function are identical by import).

New features: 26 per-applicant aggregates over the four side tables
(previous_application, installments_payments, POS_CASH_balance,
credit_card_balance), designed from domain knowledge before any training:
application history (counts, refused/approved shares, decision recency,
credit-to-application ratio, instalment counts, down-payment rate),
repayment behaviour (late-payment share and magnitudes, under-payment
share, payment ratios), point-of-sale delinquency (DPD means and shares)
and revolving utilisation (balance-to-limit mean/max, drawings,
minimum-payment shortfall share).

Training protocol imported unchanged from features_vs_fusion:
StratifiedKFold(5, shuffle, seed 42) on the same row order; CatBoost and
XGBoost with 2000 trees, lr 0.05, early stopping 100 on an inner 10% split
of the training folds only (seed 42+fold).

Run: ``python -m research.analysis.features_v3_side_tables``
Writes ``reports/features_v3_side_tables.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from research.analysis.features_vs_fusion import GBM_PARAMS, oof_run  # noqa: F401
from research.features.engineer import build_feature_matrix

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "home_credit"
V2_REPORT = REPO / "reports" / "features_vs_fusion.json"
OUT_JSON = REPO / "reports" / "features_v3_side_tables.json"

SEED = 42
N_FOLDS = 5
NOISE_FLOOR = 0.0036
MODELS = ["CatBoost", "XGBoost"]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def side_table_features() -> pd.DataFrame:
    frames = []

    prev = pd.read_csv(RAW / "previous_application.csv", usecols=[
        "SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION", "AMT_CREDIT",
        "AMT_APPLICATION", "CNT_PAYMENT", "RATE_DOWN_PAYMENT"])
    prev["REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(float)
    prev["APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(float)
    prev["CREDIT_TO_APP"] = _safe_div(prev["AMT_CREDIT"], prev["AMT_APPLICATION"])
    g = prev.groupby("SK_ID_CURR")
    frames.append(pd.DataFrame({
        "PREV_COUNT": g.size(),
        "PREV_REFUSED_SHARE": g["REFUSED"].mean(),
        "PREV_APPROVED_SHARE": g["APPROVED"].mean(),
        "PREV_MEAN_DAYS_DECISION": g["DAYS_DECISION"].mean(),
        "PREV_LAST_DAYS_DECISION": g["DAYS_DECISION"].max(),
        "PREV_CREDIT_TO_APP_MEAN": g["CREDIT_TO_APP"].mean(),
        "PREV_MEAN_CNT_PAYMENT": g["CNT_PAYMENT"].mean(),
        "PREV_MEAN_DOWN_PAYMENT_RATE": g["RATE_DOWN_PAYMENT"].mean(),
    }))
    del prev

    inst = pd.read_csv(RAW / "installments_payments.csv", usecols=[
        "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT", "AMT_PAYMENT"])
    inst["DAYS_LATE"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["LATE"] = (inst["DAYS_LATE"] > 0).astype(float)
    inst["PAY_RATIO"] = _safe_div(inst["AMT_PAYMENT"], inst["AMT_INSTALMENT"])
    inst["UNDERPAID"] = (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"]).astype(float)
    g = inst.groupby("SK_ID_CURR")
    frames.append(pd.DataFrame({
        "INST_COUNT": g.size(),
        "INST_LATE_SHARE": g["LATE"].mean(),
        "INST_MEAN_DAYS_LATE": g["DAYS_LATE"].mean(),
        "INST_MAX_DAYS_LATE": g["DAYS_LATE"].max(),
        "INST_PAY_RATIO_MEAN": g["PAY_RATIO"].mean(),
        "INST_PAY_RATIO_MIN": g["PAY_RATIO"].min(),
        "INST_UNDERPAID_SHARE": g["UNDERPAID"].mean(),
    }))
    del inst

    pos = pd.read_csv(RAW / "POS_CASH_balance.csv", usecols=[
        "SK_ID_CURR", "SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT_FUTURE"])
    pos["DPD_POS"] = (pos["SK_DPD"] > 0).astype(float)
    g = pos.groupby("SK_ID_CURR")
    frames.append(pd.DataFrame({
        "POS_COUNT": g.size(),
        "POS_MEAN_SK_DPD": g["SK_DPD"].mean(),
        "POS_DPD_POS_SHARE": g["DPD_POS"].mean(),
        "POS_MEAN_SK_DPD_DEF": g["SK_DPD_DEF"].mean(),
        "POS_MEAN_CNT_INST_FUTURE": g["CNT_INSTALMENT_FUTURE"].mean(),
    }))
    del pos

    cc = pd.read_csv(RAW / "credit_card_balance.csv", usecols=[
        "SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD",
        "AMT_DRAWINGS_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_INST_MIN_REGULARITY"])
    cc["UTIL"] = _safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["MIN_SHORTFALL"] = (
        cc["AMT_PAYMENT_TOTAL_CURRENT"] < cc["AMT_INST_MIN_REGULARITY"]
    ).astype(float)
    g = cc.groupby("SK_ID_CURR")
    frames.append(pd.DataFrame({
        "CC_COUNT": g.size(),
        "CC_UTIL_MEAN": g["UTIL"].mean(),
        "CC_UTIL_MAX": g["UTIL"].max(),
        "CC_MEAN_SK_DPD": g["SK_DPD"].mean(),
        "CC_DRAWINGS_MEAN": g["AMT_DRAWINGS_CURRENT"].mean(),
        "CC_MIN_SHORTFALL_SHARE": g["MIN_SHORTFALL"].mean(),
    }))
    del cc

    out = pd.concat(frames, axis=1)
    out.index.name = "SK_ID_CURR"
    return out.astype(np.float32)


def main() -> int:
    started = time.time()
    X2, y, ids = build_feature_matrix()
    side = side_table_features()
    print(f"[fe-v3] side-table features: {side.shape[1]} columns, "
          f"{len(side):,} applicants", flush=True)
    X3 = pd.concat(
        [X2.reset_index(drop=True),
         side.reindex(ids).reset_index(drop=True)], axis=1)
    print(f"[fe-v3] v3 frame: {X3.shape[1]} features (v2 {X2.shape[1]})", flush=True)

    v2 = json.loads(V2_REPORT.read_text())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X2, y))

    results = {}
    for name in MODELS:
        print(f"[fe-v3] {name} on v3 features ...", flush=True)
        _, fold_aucs_v3 = oof_run(name, X3, y, folds)
        fold_aucs_v2 = v2["models"][name]["v2"]["fold_aucs"]
        deltas = [a3 - a2 for a3, a2 in zip(fold_aucs_v3, fold_aucs_v2)]
        results[name] = {
            "fold_aucs_v3": [round(a, 4) for a in fold_aucs_v3],
            "fold_aucs_v2_recorded": fold_aucs_v2,
            "oof_auc_v3_mean_of_folds": round(float(np.mean(fold_aucs_v3)), 4),
            "fold_paired_delta_v3_minus_v2": [round(d, 4) for d in deltas],
            "mean_delta": round(float(np.mean(deltas)), 4),
            "all_folds_positive": bool(all(d > 0 for d in deltas)),
            "feature_claim": bool(np.mean(deltas) > NOISE_FLOOR and all(d > 0 for d in deltas)),
        }
        r = results[name]
        print(f"[fe-v3] {name}: mean delta {r['mean_delta']:+.4f} "
              f"all-positive {r['all_folds_positive']} claim {r['feature_claim']}", flush=True)

    report = {
        "protocol": {
            "claim_rule": "fold-paired mean delta > +0.0036 AND every fold positive (same as features_vs_fusion)",
            "folds": f"StratifiedKFold({N_FOLDS}, shuffle, seed {SEED}) on the identical row order",
            "fit": "imported unchanged from features_vs_fusion (_fit_predict_fold via oof_run)",
            "v2_baseline": "recorded fold AUCs in reports/features_vs_fusion.json",
            "n_new_features": int(side.shape[1]),
            "side_tables": ["previous_application", "installments_payments",
                             "POS_CASH_balance", "credit_card_balance"],
        },
        "models": results,
        "generated_seconds": round(time.time() - started, 1),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
