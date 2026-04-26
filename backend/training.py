from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.app.services.analysis import generate_analysis_plots
from backend.app.services.calibrate import calibrate_tau_d
from backend.app.services.cbes_engine import COMPONENT_WEIGHTS, compute_cbes_probability
from backend.app.services.decision_engine import hybrid_decision


RANDOM_STATE = 42
DATA_PATH = Path("synthetic_indian_loan_dataset.csv")
ARTIFACT_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACT_DIR / "plots"
PIPELINE_PATH = ARTIFACT_DIR / "pipeline.joblib"
PIPELINE_HASH_PATH = ARTIFACT_DIR / "pipeline.sha256"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(number) or np.isinf(number):
        return default
    return number


def build_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    for col in [
        "monthly_income",
        "annual_income",
        "loan_amount",
        "existing_emis",
        "emi",
        "bank_balance",
        "total_assets",
        "active_loans",
        "total_loans",
        "closed_loans",
        "missed_payments",
        "years_employed",
        "age",
        "cibil_score",
        "credit_utilization_ratio",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "annual_income" not in df.columns and "monthly_income" in df.columns:
        df["annual_income"] = df["monthly_income"] * 12

    if "total_assets" not in df.columns:
        residential = pd.to_numeric(df.get("residential_assets_value", 0), errors="coerce").fillna(0)
        commercial = pd.to_numeric(df.get("commercial_assets_value", 0), errors="coerce").fillna(0)
        balance = pd.to_numeric(df.get("bank_balance", 0), errors="coerce").fillna(0)
        df["total_assets"] = residential + commercial + balance

    df["emi_income_ratio"] = pd.to_numeric(df.get("emi", 0), errors="coerce").fillna(0) / (
        pd.to_numeric(df.get("monthly_income", 0), errors="coerce").fillna(0) + 1
    )
    df["debt_to_income_ratio"] = (
        pd.to_numeric(df.get("existing_emis", 0), errors="coerce").fillna(0) * 12
        + pd.to_numeric(df.get("loan_amount", 0), errors="coerce").fillna(0)
    ) / (pd.to_numeric(df.get("annual_income", 0), errors="coerce").fillna(0) + 1)
    df["loan_income_ratio"] = pd.to_numeric(df.get("loan_amount", 0), errors="coerce").fillna(0) / (
        pd.to_numeric(df.get("annual_income", 0), errors="coerce").fillna(0) + 1
    )
    df["asset_coverage"] = pd.to_numeric(df.get("total_assets", 0), errors="coerce").fillna(0) / (
        pd.to_numeric(df.get("loan_amount", 0), errors="coerce").fillna(0) + 1
    )
    df["liquidity_ratio"] = pd.to_numeric(df.get("bank_balance", 0), errors="coerce").fillna(0) / (
        pd.to_numeric(df.get("loan_amount", 0), errors="coerce").fillna(0) + 1
    )
    df["loan_activity_ratio"] = pd.to_numeric(df.get("active_loans", 0), errors="coerce").fillna(0) / (
        pd.to_numeric(df.get("total_loans", 0), errors="coerce").fillna(0) + 1
    )
    df["repayment_score"] = pd.to_numeric(df.get("closed_loans", 0), errors="coerce").fillna(0) / (
        pd.to_numeric(df.get("total_loans", 0), errors="coerce").fillna(0) + 1
    )
    df["missed_payment_ratio"] = pd.to_numeric(df.get("missed_payments", 0), errors="coerce").fillna(0) / (
        pd.to_numeric(df.get("total_loans", 0), errors="coerce").fillna(0) + 1
    )
    df["employment_stability"] = pd.to_numeric(df.get("years_employed", 0), errors="coerce").fillna(0) / (
        pd.to_numeric(df.get("age", 0), errors="coerce").fillna(0) + 1
    )
    df["repayments_on_time_last_12"] = 12 - np.minimum(
        pd.to_numeric(df.get("missed_payments", 0), errors="coerce").fillna(0),
        12,
    )
    df["gross_monthly_income"] = pd.to_numeric(df.get("monthly_income", 0), errors="coerce").fillna(0)
    df["net_monthly_income"] = pd.to_numeric(df.get("monthly_income", 0), errors="coerce").fillna(0)
    df["monthly_emi"] = pd.to_numeric(df.get("emi", 0), errors="coerce").fillna(0)
    df["total_monthly_debt"] = pd.to_numeric(df.get("existing_emis", 0), errors="coerce").fillna(0)
    df["credit_utilization"] = pd.to_numeric(df.get("credit_utilization_ratio", 0), errors="coerce").fillna(0)

    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(float(df[col].median()) if not df[col].dropna().empty else 0.0)
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(str(mode.iloc[0]) if not mode.empty else "unknown")

    return df


def to_model_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.get_dummies(feature_df, drop_first=True, dtype=np.float32)
    return matrix


def row_to_cbes_input(row: pd.Series) -> dict[str, float]:
    return {
        "cibil_score": _safe_float(row.get("cibil_score"), 620.0),
        "missed_payment_ratio": _safe_float(row.get("missed_payment_ratio"), 0.2),
        "credit_utilization": _safe_float(row.get("credit_utilization"), 0.8),
        "gross_monthly_income": max(_safe_float(row.get("gross_monthly_income"), 30000.0), 1.0),
        "net_monthly_income": max(_safe_float(row.get("net_monthly_income"), 30000.0), 1.0),
        "total_monthly_debt": max(_safe_float(row.get("total_monthly_debt"), 15000.0), 0.0),
        "monthly_emi": max(_safe_float(row.get("monthly_emi"), 10000.0), 0.0),
        "repayments_on_time_last_12": _safe_float(row.get("repayments_on_time_last_12"), 8.0),
        "active_loans": max(_safe_float(row.get("active_loans"), 3.0), 0.0),
        "total_loans": max(_safe_float(row.get("total_loans"), 5.0), 0.0),
        "bank_balance": max(_safe_float(row.get("bank_balance"), 25000.0), 0.0),
        "loan_amount": max(_safe_float(row.get("loan_amount"), 300000.0), 1.0),
        "total_assets": max(_safe_float(row.get("total_assets"), 300000.0), 0.0),
        "years_employed": max(_safe_float(row.get("years_employed"), 2.0), 0.0),
        "age": max(_safe_float(row.get("age"), 30.0), 18.0),
    }


def compute_cv_auc_std(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []
    for train_idx, valid_idx in kfold.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        model = clone(estimator)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)[:, 1]
        aucs.append(float(roc_auc_score(y_va, probs)))
    return float(np.std(aucs))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH, engine="python")
    print("Dataset:", raw_df.shape)

    y = raw_df["loan_approved"].astype(int)
    feature_base = raw_df.drop(columns=[c for c in ["loan_approved", "applicant_id"] if c in raw_df.columns]).copy()
    feature_base = build_feature_frame(feature_base)

    defaults = {}
    for col in feature_base.columns:
        if feature_base[col].dtype.kind in "biufc":
            defaults[col] = float(feature_base[col].median())
        else:
            mode = feature_base[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else "unknown"

    X_full = to_model_matrix(feature_base)
    feature_columns = list(X_full.columns)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_full,
        y,
        raw_df.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_configs = {
        "Logistic Regression": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=260,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.2,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=340,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    }

    fitted_models: dict[str, Any] = {}
    fitted_calibrators: dict[str, Any] = {}
    model_metrics: list[dict[str, float | str]] = []

    calibration_method = "isotonic" if len(X_train) > 1000 else "sigmoid"

    for model_name, estimator in model_configs.items():
        cv_std_auc = compute_cv_auc_std(estimator, X_train, y_train)

        base_model = clone(estimator)
        base_model.fit(X_train, y_train)

        calibrator = CalibratedClassifierCV(estimator=clone(estimator), method=calibration_method, cv=3)
        calibrator.fit(X_train, y_train)

        probs = calibrator.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        accuracy = float(accuracy_score(y_test, preds))
        precision = float(precision_score(y_test, preds, zero_division=0))
        recall = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        auc_score = float(roc_auc_score(y_test, probs))

        score = auc_score + (0.20 * recall) - (0.10 * cv_std_auc)

        model_metrics.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": auc_score,
                "CVStdAUC": cv_std_auc,
                "SelectionScore": score,
            }
        )

        fitted_models[model_name] = base_model
        fitted_calibrators[model_name] = calibrator

        print(f"\n{model_name}")
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print("AUC:", round(auc_score, 4))
        print("CV std AUC:", round(cv_std_auc, 4))
        print("Selection score:", round(score, 4))

    metrics_df = pd.DataFrame(model_metrics).sort_values("SelectionScore", ascending=False).reset_index(drop=True)
    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_model = fitted_models[best_model_name]
    best_calibrator = fitted_calibrators[best_model_name]

    # Refit best model and calibrator on full data for deployment parity.
    best_model_full = clone(model_configs[best_model_name])
    best_model_full.fit(X_full, y)
    best_calibrator_full = CalibratedClassifierCV(estimator=clone(model_configs[best_model_name]), method=calibration_method, cv=3)
    best_calibrator_full.fit(X_full, y)

    p_ml_test = best_calibrator.predict_proba(X_test)[:, 1]

    cbes_probs_all: list[float] = []
    cbes_breakdowns: list[dict[str, float]] = []
    for _, row in feature_base.iterrows():
        p_cbes, breakdown = compute_cbes_probability(row_to_cbes_input(row), custom_weights=COMPONENT_WEIGHTS)
        cbes_probs_all.append(p_cbes)
        cbes_breakdowns.append(breakdown)

    p_cbes_all = np.asarray(cbes_probs_all, dtype=float)
    p_cbes_test = p_cbes_all[idx_test.to_numpy()]

    tau_result = calibrate_tau_d(p_ml_test, p_cbes_test, y_test.to_numpy())
    tau_d = tau_result.tau_d

    p_ml_all = best_calibrator_full.predict_proba(X_full)[:, 1]

    decisions = []
    confidences = []
    t_approves = []
    t_rejects = []
    disagreements = []
    decision_reasons = []

    for p_ml, p_cbes in zip(p_ml_all, p_cbes_all):
        d = hybrid_decision(float(p_ml), float(p_cbes), tau_d=tau_d)
        decisions.append(d.decision)
        confidences.append(d.confidence)
        t_approves.append(d.t_approve)
        t_rejects.append(d.t_reject)
        disagreements.append(d.disagreement)
        decision_reasons.append(d.decision_reason)

    decision_array = np.asarray(decisions)
    confidence_array = np.asarray(confidences, dtype=float)

    model_probabilities_all: dict[str, np.ndarray] = {}
    for model_name, calibrator in fitted_calibrators.items():
        full_calibrator = CalibratedClassifierCV(estimator=clone(model_configs[model_name]), method=calibration_method, cv=3)
        full_calibrator.fit(X_full, y)
        model_probabilities_all[model_name] = full_calibrator.predict_proba(X_full)[:, 1]

    model_metrics_export = metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "CVStdAUC", "SelectionScore"]]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_payload = {
        "version": "v3-research-hybrid",
        "model_name": best_model_name,
        "threshold": 0.5,
        "tau_d": tau_d,
        "cbes_weights": COMPONENT_WEIGHTS,
        "calibration_method": calibration_method,
        "feature_columns": feature_columns,
        "feature_defaults": defaults,
        "base_feature_columns": list(feature_base.columns),
        "best_model": best_model_full,
        "calibrator": best_calibrator_full,
    }

    joblib.dump(pipeline_payload, PIPELINE_PATH)
    pipeline_hash = hashlib.sha256(PIPELINE_PATH.read_bytes()).hexdigest()
    PIPELINE_HASH_PATH.write_text(pipeline_hash, encoding="utf-8")

    model_metrics_export.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            "applicant_id": raw_df["applicant_id"].astype(str),
            "y_true": y.to_numpy(),
            "cbes_prob": p_cbes_all,
            "best_model_prob": p_ml_all,
            "final_decision": decisions,
            "confidence": confidence_array,
            "approval_threshold": t_approves,
            "rejection_threshold": t_rejects,
            "disagreement": disagreements,
            "decision_reason": decision_reasons,
        }
    )

    for model_name, probs in model_probabilities_all.items():
        prediction_df[f"prob_{model_name}"] = probs
    prediction_df["prob_CBES"] = p_cbes_all
    prediction_df.to_csv(ARTIFACT_DIR / "prediction_outputs.csv", index=False)

    summary_payload = {
        "best_model": best_model_name,
        "selection_metric": "auc + 0.2*recall - 0.1*std_auc",
        "selected_alpha": 0.0,
        "tau_d": tau_d,
        "deferral_rate": float(np.mean(decision_array == "DEFER")),
        "accuracy_non_deferred": float(
            np.mean((decision_array[decision_array != "DEFER"] == "APPROVE").astype(int) == y.to_numpy()[decision_array != "DEFER"])
        ) if np.any(decision_array != "DEFER") else 0.0,
        "decision_distribution": {
            "APPROVE": int(np.sum(decision_array == "APPROVE")),
            "REJECT": int(np.sum(decision_array == "REJECT")),
            "DEFER": int(np.sum(decision_array == "DEFER")),
        },
        "cbes_weights": COMPONENT_WEIGHTS,
        "pipeline_sha256": pipeline_hash,
        "artifacts": {
            "pipeline_joblib": str(PIPELINE_PATH),
            "pipeline_hash": str(PIPELINE_HASH_PATH),
            "metrics_csv": str(ARTIFACT_DIR / "model_metrics.csv"),
            "predictions_csv": str(ARTIFACT_DIR / "prediction_outputs.csv"),
        },
    }

    with (ARTIFACT_DIR / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    cbes_breakdown_df = pd.DataFrame(cbes_breakdowns)
    generate_analysis_plots(
        output_dir=PLOTS_DIR,
        y_true=y.to_numpy(),
        model_probabilities=model_probabilities_all,
        p_ml=p_ml_all,
        p_cbes=p_cbes_all,
        tau_d=tau_d,
        decisions=decision_array,
        confidence_scores=confidence_array,
        cbes_breakdown_df=cbes_breakdown_df,
    )

    # Deferral tradeoff export for auditability.
    pd.DataFrame(tau_result.curve).to_csv(ARTIFACT_DIR / "tau_calibration_curve.csv", index=False)

    print("\n===== PIPELINE SUMMARY =====")
    print("Best model:", best_model_name)
    print("Selected TAU_D:", round(tau_d, 4))
    print("Pipeline hash:", pipeline_hash)
    print("Artifacts saved in:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
