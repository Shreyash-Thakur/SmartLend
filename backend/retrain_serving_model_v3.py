"""Retrain the *serving* model on real Home Credit data, without the leak.

Why this script exists
----------------------
`backend/artifacts/pipeline.joblib` and `pipeline_v2.joblib` were fitted in
April on a synthetic dataset that has since been deleted. Two of their 25
`feature_names` are `loan_approved` and `confidence_score` — the model's own
OUTPUTS, which had leaked into the training frame as inputs (see
`customer_profile_service.LEAKED_OUTPUT_FEATURES`). Any performance those
artifacts reported is therefore meaningless.

What this script does NOT do
----------------------------
It does not change the serving contract to the full 129 Home Credit columns.
The form, `customer_profile_service` and the scoring path all speak the
25-name snake_case vocabulary; switching would break all of them. We keep the
vocabulary, change the data, and drop the leak.

Feature provenance
------------------
The training frame is built by calling `customer_profile_service._build_profile`
on each Home Credit row — the *same* function both serving resolution paths
use — so a feature seen in training is constructed exactly as it is at
inference. Nothing is re-derived here with a second, divergent formula.

Of the 25 - 2 = 23 non-leaked serving features, 8 have no Home Credit source
and are DROPPED rather than invented (see `DROPPED` below). The saved artifact's
`feature_names` reflects the drop; `MLPredictor.predict_application` builds its
row from `feature_names`, so a shorter list simply means fewer columns are
read out of the payload — nothing breaks.

Run:  python -m backend.retrain_serving_model_v3
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.services import customer_profile_service as cps
from backend.app.services.cbes_engine import DEFAULTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
# NEW artifact. pipeline.joblib / pipeline_v2.joblib are left untouched so the
# leaky-but-known-quantity models remain available for rollback.
OUT_PATH = ARTIFACTS_DIR / "pipeline_v3_real.joblib"
REPORT_PATH = REPORTS_DIR / "serving_model_retrain.json"

TARGET_COLUMN = "TARGET"  # 1 = defaulted

# ---------------------------------------------------------------------------
# The serving contract, minus the leak, minus what Home Credit cannot supply
# ---------------------------------------------------------------------------

BEFORE = list(
    cps.PROFILE_FEATURES + cps.FORM_FEATURES + cps.DERIVED_FEATURES + cps.LEAKED_OUTPUT_FEATURES
)

# Reason strings land verbatim in reports/serving_model_retrain.json.
DROPPED: dict[str, str] = {
    "loan_approved": (
        "TARGET LEAK — the model's own approve/reject output was present as an "
        "input column in the deleted synthetic training frame."
    ),
    "confidence_score": (
        "TARGET LEAK — the model's own confidence output was present as an input "
        "column in the deleted synthetic training frame."
    ),
    "loan_term": "No Home Credit equivalent; refused to invent a tenure.",
    "interest_rate": "No Home Credit equivalent; refused to invent a rate.",
    "emi": (
        "No Home Credit equivalent left. AMT_ANNUITY is already consumed as "
        "`existing_emis` by the serving mapping (customer_profile_service."
        "_build_profile); reusing it here would duplicate one column under two "
        "feature names."
    ),
    "residential_assets_value": "Not collected by Home Credit.",
    "commercial_assets_value": "Not collected by Home Credit.",
    "bank_balance": "Not collected by Home Credit.",
    "total_assets": "Derived from the two asset values above, both dropped.",
    "emi_income_ratio": "Derived from `emi`, dropped above.",
}

# What survives: the 13 PROFILE_FEATURES, plus the two FORM/DERIVED features
# Home Credit genuinely carries.
#   loan_amount       <- AMT_CREDIT, the credit amount of the application being
#                        scored, i.e. exactly what the form's loan_amount means.
#   loan_income_ratio <- loan_amount / annual_income, the same arithmetic (and
#                        the same rounding) resolve_application_payload() uses.
FEATURES: list[str] = list(cps.PROFILE_FEATURES) + ["loan_amount", "loan_income_ratio"]


def build_training_frame(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Home Credit rows -> the serving feature frame, via the serving mapping."""
    usecols = list(cps._USED_COLUMNS) + [TARGET_COLUMN, "AMT_CREDIT"]
    header = pd.read_csv(csv_path, nrows=0)
    usecols = sorted({c for c in usecols if c in header.columns})
    raw = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    print(f"[data] loaded {len(raw):,} rows x {raw.shape[1]} columns from {csv_path.name}")

    y = raw[TARGET_COLUMN].astype(int).to_numpy()

    # Rename CSV columns -> the raw keys `_build_profile` expects, exactly as
    # `_raw_from_csv` does.
    inverse = {csv: field for field, csv in cps._RAW_TO_CSV.items() if csv in raw.columns}
    records = raw.rename(columns=inverse).to_dict("records")
    amt_credit = raw["AMT_CREDIT"].to_numpy()

    rows: list[dict[str, float]] = []
    for record, credit in zip(records, amt_credit):
        key = int(record["customer_id"])
        # THE shared derivation. Not a copy of it.
        profile = cps._build_profile(record, key)

        row = {name: profile.get(name) for name in cps.PROFILE_FEATURES}

        loan_amount = cps._num(credit)
        row["loan_amount"] = loan_amount
        annual_income = cps._num(profile.get("annual_income"))
        row["loan_income_ratio"] = (
            round(loan_amount / annual_income, 4)
            if loan_amount is not None and annual_income
            else None
        )
        rows.append(row)

    X = pd.DataFrame(rows, columns=FEATURES)

    # Impute exactly the way `MLPredictor.predict_application` does, so a
    # missing value means the same number at fit time and at score time.
    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(DEFAULTS.get(column, 0.0))
    X = X.replace([np.inf, -np.inf], 0.0).astype(float)
    return X, y


def main() -> None:
    csv_path = cps._resolve_source_path()
    if csv_path is None:
        raise SystemExit("Home Credit extract not found; set SMARTLEND_CUSTOMER_DATA.")

    started = time.time()
    X, y = build_training_frame(csv_path)

    # Same split discipline as the rest of the project.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LogisticRegression + StandardScaler keeps the artifact shape the SHAP
    # LinearExplainer in ml_service already knows how to read.
    def make_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                # Same estimator spec train_pipeline() uses for this slot.
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    # Calibrated: decision_engine treats p_ml as a probability, not a score.
    calibrator = CalibratedClassifierCV(make_pipeline(), method="isotonic", cv=5)
    calibrator.fit(X_train, y_train)

    p_default_test = calibrator.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_default_test)),
        "pr_auc": float(average_precision_score(y_test, p_default_test)),
        "brier": float(brier_score_loss(y_test, p_default_test)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_default_rate": float(y_test.mean()),
    }
    print("[metrics]", json.dumps(metrics, indent=2))

    # F1-optimal T_base on the held-out split. p_ml = P(approval) = 1 - P(default),
    # and the engine predicts default when p_ml < t — same sweep as train_pipeline.
    p_ml_test = 1.0 - p_default_test
    thresholds = np.arange(0.30, 0.70, 0.01)
    f1_scores = [f1_score(y_test, (p_ml_test < t).astype(int), zero_division=0) for t in thresholds]
    t_base = float(round(thresholds[int(np.argmax(f1_scores))], 4))
    best_f1 = float(max(f1_scores))
    # The sweep is inherited from train_pipeline() unchanged, but be honest
    # about it: with a calibrated 8%-base-rate model, p_ml concentrates near
    # 0.92, so almost nothing falls below any threshold in [0.30, 0.70] and the
    # F1 at the argmax is near zero. The winning t_base is therefore the top of
    # a very flat curve, not a sharply identified optimum. Reported, not tuned.
    print(f"[t_base] F1-optimal = {t_base:.4f} (F1 = {best_f1:.4f})")

    # A plain fitted pipeline is still needed: ml_service unpacks `scaler` and
    # `model` off it for SHAP.
    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)
    background_data = pipeline.named_steps["scaler"].transform(
        X_train.sample(min(100, len(X_train)), random_state=42)
    )

    payload = {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "feature_names": FEATURES,
        "model_name": "LogisticRegression",
        "background_data": background_data,
        "t_base": t_base,
        "tau_d": 0.30,
        # Provenance — nothing in the scoring path reads these back.
        "trained_on": "home_credit_real",
        "leak_removed": list(cps.LEAKED_OUTPUT_FEATURES),
        "test_metrics": metrics,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUT_PATH)
    print(f"[artifact] wrote {OUT_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "artifact": str(OUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "supersedes": ["backend/artifacts/pipeline.joblib", "backend/artifacts/pipeline_v2.joblib"],
                "data_source": str(csv_path),
                "target": f"{TARGET_COLUMN} (1 = defaulted)",
                "split": "train_test_split(test_size=0.2, random_state=42, stratify=y)",
                "model": "StandardScaler + LogisticRegression(max_iter=1000), CalibratedClassifierCV(isotonic, cv=5)",
                "features_before": BEFORE,
                "features_after": FEATURES,
                "n_features_before": len(BEFORE),
                "n_features_after": len(FEATURES),
                "dropped": DROPPED,
                "held_out_metrics": metrics,
                "t_base": t_base,
                "t_base_f1": best_f1,
                "t_base_note": (
                    "Same F1 sweep over [0.30, 0.70) that train_pipeline() used. With a "
                    "calibrated model on an 8% base rate, p_ml = 1 - P(default) concentrates "
                    "near 0.92, so the sweep flags almost nothing and the F1 curve is flat and "
                    "near zero. t_base is the argmax of that flat curve, not a sharp optimum."
                ),
                "tau_d": 0.30,
                "note": (
                    "Metrics are LOWER than the April synthetic artifacts reported. That is "
                    "the point: those numbers came from a frame containing the model's own "
                    "outputs as features. Nothing was tuned to recover the difference."
                ),
                "generated_seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[report] wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
