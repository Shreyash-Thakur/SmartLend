"""Subgroup fairness evaluation of the deployed decision artifact (reviewer item).

A credit decisioning paper cannot leave fairness unmeasured. This script
evaluates the CURRENT decision artifact (``prediction_outputs.csv``, the
alpha=0.10 blend policy at 22.5% deferral) across applicant subgroups:

- gender (CODE_GENDER: F / M / XNA)
- age band (derived from DAYS_BIRTH: <30, 30-44, 45-59, 60+)
- region rating (REGION_RATING_CLIENT: 1 best .. 3 worst)
- bureau-file status (thin-file = no bureau.csv record vs has record)

Per group it reports: size, observed default rate, decision mix (approve /
reject / defer), approval rate among actual GOOD customers (equal-opportunity
view), approval rate among actual DEFAULTERS (false-approval view), ranking
AUC of the primary model, and 10-bin expected calibration error of the
approval probability. Gaps are descriptive measurements of the deployed
policy, not a compliance assessment.

LABEL DIRECTIONS: y_true == 1 means GOOD; best_model_prob is an APPROVAL
probability; calibration is measured as P(good) vs observed good share.

Run: ``python -m research.analysis.fairness_subgroups``
Writes ``reports/fairness_subgroups.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw" / "home_credit"
REPORT_PATH = REPO_ROOT / "reports" / "fairness_subgroups.json"

AGE_BINS = [0, 30, 45, 60, 200]
AGE_LABELS = ["<30", "30-44", "45-59", "60+"]


def ece_10bin(p_good: np.ndarray, y_good: np.ndarray) -> float:
    """Expected calibration error over 10 equal-width probability bins."""
    bins = np.clip((p_good * 10).astype(int), 0, 9)
    ece = 0.0
    n = len(p_good)
    for b in range(10):
        mask = bins == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(p_good[mask].mean() - y_good[mask].mean())
    return float(ece)


def group_metrics(g: pd.DataFrame) -> dict[str, Any]:
    y = g["y_true"].to_numpy(dtype=int)
    p = g["best_model_prob"].to_numpy(dtype=float)
    dec = g["final_decision"].astype(str).str.upper()
    approve = (dec == "APPROVE").to_numpy()
    defer = (dec == "DEFER").to_numpy()
    reject = (dec == "REJECT").to_numpy()
    good = y == 1
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return {
        "n": int(len(g)),
        "default_rate": float(1.0 - y.mean()),
        "approve_rate": float(approve.mean()),
        "reject_rate": float(reject.mean()),
        "defer_rate": float(defer.mean()),
        # equal-opportunity view: how often a genuinely good customer is approved
        "approve_rate_given_good": float(approve[good].mean()) if good.any() else None,
        # false-approval view: how often an eventual defaulter is approved
        "approve_rate_given_default": float(approve[~good].mean()) if (~good).any() else None,
        "defer_rate_given_good": float(defer[good].mean()) if good.any() else None,
        "defer_rate_given_default": float(defer[~good].mean()) if (~good).any() else None,
        "auc": auc,
        "ece_10bin": ece_10bin(p, y.astype(float)),
    }


def axis_report(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    groups = {str(k): group_metrics(g) for k, g in frame.groupby(col, observed=True)}
    # headline gaps across groups with n >= 500 (tiny groups make noisy gaps)
    big = {k: v for k, v in groups.items() if v["n"] >= 500}
    gaps = {}
    for metric in ["approve_rate", "defer_rate", "approve_rate_given_good", "auc", "ece_10bin"]:
        vals = {k: v[metric] for k, v in big.items() if v[metric] is not None}
        if len(vals) >= 2:
            hi, lo = max(vals, key=vals.get), min(vals, key=vals.get)
            gaps[metric] = {
                "max_gap": float(vals[hi] - vals[lo]),
                "highest": hi,
                "lowest": lo,
            }
    return {"groups": groups, "max_gaps_groups_n_ge_500": gaps}


def main() -> int:
    pred = pd.read_csv(REPO_ROOT / "backend" / "artifacts" / "prediction_outputs.csv")
    pred["SK_ID_CURR"] = pred["applicant_id"].astype(str).str.removeprefix("HC").astype(int)

    app = pd.read_csv(
        RAW / "application_train.csv",
        usecols=["SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH", "REGION_RATING_CLIENT"],
    )
    bureau_ids = set(
        pd.read_csv(RAW / "bureau.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"].unique()
    )

    frame = pred.merge(app, on="SK_ID_CURR", how="inner", validate="one_to_one")
    assert len(frame) == len(pred), "every scored row must join to application_train"

    frame["age_years"] = (-frame["DAYS_BIRTH"] / 365.25).round(1)
    frame["age_band"] = pd.cut(frame["age_years"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    frame["bureau_file"] = np.where(
        frame["SK_ID_CURR"].isin(bureau_ids), "has_bureau_record", "thin_file_no_bureau"
    )
    frame["region_rating"] = "rating_" + frame["REGION_RATING_CLIENT"].astype(str)

    report = {
        "protocol": {
            "data": "backend/artifacts/prediction_outputs.csv joined to application_train.csv on SK_ID_CURR",
            "n_rows": int(len(frame)),
            "policy_evaluated": "deployed decision artifact (alpha=0.10 blend, 22.5% capacity deferral)",
            "label_semantics": "y_true==1 GOOD; best_model_prob is an approval probability",
            "scope_note": (
                "Descriptive subgroup measurement of the deployed policy's decision "
                "mix, ranking and calibration. It is NOT a regulatory fairness "
                "compliance assessment, and gender/age are NOT model inputs to the "
                "15-field serving contract; they are joined here for measurement only."
            ),
        },
        "overall": group_metrics(frame),
        "by_gender": axis_report(frame, "CODE_GENDER"),
        "by_age_band": axis_report(frame, "age_band"),
        "by_region_rating": axis_report(frame, "region_rating"),
        "by_bureau_file": axis_report(frame, "bureau_file"),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for axis in ["by_gender", "by_age_band", "by_region_rating", "by_bureau_file"]:
        print(f"== {axis}")
        for k, v in report[axis]["groups"].items():
            print(
                f"  {k:<22} n={v['n']:>7} default={v['default_rate']:.3f} "
                f"approve={v['approve_rate']:.3f} defer={v['defer_rate']:.3f} "
                f"auc={v['auc'] if v['auc'] is None else round(v['auc'], 4)} ece={v['ece_10bin']:.4f}"
            )
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
