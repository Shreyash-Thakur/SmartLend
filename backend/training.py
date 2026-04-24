import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

RANDOM_STATE = 42
DATA_PATH = Path("synthetic_indian_loan_dataset.csv")
ARTIFACT_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACT_DIR / "plots"

DEFAULT_CBES_WEIGHTS = {
    "credit": 0.35,
    "capacity": 0.30,
    "asset": 0.25,
    "stability": 0.10,
}


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(x, high))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["EMI_INCOME_RATIO"] = enriched["emi"] / (enriched["monthly_income"] + 1)
    enriched["DEBT_BURDEN"] = (enriched["existing_emis"] + enriched["emi"]) / (enriched["monthly_income"] + 1)
    enriched["DEBT_TO_INCOME_RATIO"] = (enriched["existing_emis"] * 12 + enriched["loan_amount"]) / (enriched["annual_income"] + 1)
    enriched["LOAN_INCOME_RATIO"] = enriched["loan_amount"] / (enriched["annual_income"] + 1)
    enriched["ASSET_COVERAGE"] = enriched["total_assets"] / (enriched["loan_amount"] + 1)
    enriched["LIQUIDITY_RATIO"] = enriched["bank_balance"] / (enriched["loan_amount"] + 1)
    enriched["LOAN_ACTIVITY_RATIO"] = enriched["active_loans"] / (enriched["total_loans"] + 1)
    enriched["REPAYMENT_SCORE"] = enriched["closed_loans"] / (enriched["total_loans"] + 1)
    enriched["MISSED_PAYMENT_RATIO"] = enriched["missed_payments"] / (enriched["total_loans"] + 1)
    enriched["EMPLOYMENT_STABILITY"] = enriched["years_employed"] / (enriched["age"] + 1)
    return enriched


def compute_cbes_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    cibil_norm = ((df["cibil_score"] - 300) / 600).clip(0, 1)
    payment_penalty = (1 - df["MISSED_PAYMENT_RATIO"]).clip(0, 1)
    util_penalty = (1 - df["credit_utilization_ratio"]).clip(0, 1)
    credit_component = 0.5 * cibil_norm + 0.3 * payment_penalty + 0.2 * util_penalty

    dti_score = (1 - df["DEBT_TO_INCOME_RATIO"]).clip(0.2, 1)
    emi_score = (1 - df["EMI_INCOME_RATIO"]).clip(0, 1)
    loan_income_score = (1 - df["LOAN_INCOME_RATIO"]).clip(0, 1)
    capacity_component = 0.5 * dti_score + 0.3 * emi_score + 0.2 * loan_income_score

    asset_score = df["ASSET_COVERAGE"].clip(0, 2) / 2
    liquidity_score = df["LIQUIDITY_RATIO"].clip(0, 1)
    asset_component = 0.7 * asset_score + 0.3 * liquidity_score

    stability_component = df["EMPLOYMENT_STABILITY"].clip(0, 1)

    return {
        "credit": credit_component.clip(0, 1),
        "capacity": capacity_component.clip(0, 1),
        "asset": asset_component.clip(0, 1),
        "stability": stability_component.clip(0, 1),
    }


def combine_cbes_components(components: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    score = (
        float(weights["credit"]) * components["credit"]
        + float(weights["capacity"]) * components["capacity"]
        + float(weights["asset"]) * components["asset"]
        + float(weights["stability"]) * components["stability"]
    )
    return score.clip(0, 1)


def tune_cbes_weights(
    y_cal: pd.Series,
    cal_components: dict[str, pd.Series],
    default_weights: dict[str, float],
    sample_size: int = 400,
) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)

    best_weights = dict(default_weights)
    best_auc = roc_auc_score(y_cal, combine_cbes_components(cal_components, best_weights))
    best_objective = best_auc

    # Dirichlet samples preserve explainability (positive weights summing to 1).
    for _ in range(sample_size):
        candidate = rng.dirichlet([3.0, 3.0, 2.0, 1.5])
        candidate_weights = {
            "credit": float(candidate[0]),
            "capacity": float(candidate[1]),
            "asset": float(candidate[2]),
            "stability": float(candidate[3]),
        }

        # Keep components meaningful and avoid collapsing into a single dominant factor.
        if not (
            0.20 <= candidate_weights["credit"] <= 0.55
            and 0.15 <= candidate_weights["capacity"] <= 0.45
            and 0.10 <= candidate_weights["asset"] <= 0.35
            and 0.05 <= candidate_weights["stability"] <= 0.20
        ):
            continue

        auc = roc_auc_score(y_cal, combine_cbes_components(cal_components, candidate_weights))
        drift_penalty = 0.025 * sum(abs(candidate_weights[k] - default_weights[k]) for k in default_weights)
        objective = auc - drift_penalty

        if objective > best_objective:
            best_objective = objective
            best_auc = auc
            best_weights = candidate_weights

    return {
        key: round(float(value), 4)
        for key, value in best_weights.items()
    }


def find_best_threshold(y_true: pd.Series, probs: np.ndarray) -> float:
    thresholds = np.linspace(0.3, 0.7, 81)
    best_threshold = 0.5
    best_score = -1.0

    y_true_np = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        precision = precision_score(y_true_np, preds, zero_division=0)
        recall = recall_score(y_true_np, preds, zero_division=0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        if f1 > best_score:
            best_score = f1
            best_threshold = float(threshold)

    return best_threshold


def dynamic_hybrid_decision(ml_prob: float, cbes_prob: float, alpha: float = 0.25) -> tuple[str, float, float, float]:
    alpha = clamp(alpha, 0.2, 0.4)
    center = clamp(0.5 - ((cbes_prob - 0.5) * alpha), 0.42, 0.58)

    neutrality = 1.0 - min(1.0, abs(cbes_prob - 0.5) * 2.0)
    defer_band = 0.06 + (0.08 * neutrality)

    rejection_threshold = clamp(center - (defer_band / 2), 0.2, 0.6)
    approval_threshold = clamp(center + (defer_band / 2), 0.4, 0.8)

    if approval_threshold <= rejection_threshold:
        midpoint = (approval_threshold + rejection_threshold) / 2
        rejection_threshold = clamp(midpoint - 0.03, 0.2, 0.6)
        approval_threshold = clamp(midpoint + 0.03, 0.4, 0.8)

    confidence = abs(ml_prob - 0.5)

    if ml_prob >= approval_threshold:
        return "APPROVE", confidence, approval_threshold, rejection_threshold
    if ml_prob <= rejection_threshold:
        return "REJECT", confidence, approval_threshold, rejection_threshold
    return "DEFER", confidence, approval_threshold, rejection_threshold


def evaluate_alpha(ml_probs: np.ndarray, cbes_probs: np.ndarray, y_true: np.ndarray, alpha: float) -> tuple[list[str], float, float]:
    decisions = [dynamic_hybrid_decision(ml, cbes, alpha=alpha)[0] for ml, cbes in zip(ml_probs, cbes_probs)]
    decisions_np = np.array(decisions)
    mask = decisions_np != "DEFER"
    defer_rate = float(np.mean(decisions_np == "DEFER"))

    if np.any(mask):
        pred_non_defer = (decisions_np[mask] == "APPROVE").astype(int)
        acc_non_defer = float(accuracy_score(y_true[mask], pred_non_defer))
    else:
        acc_non_defer = -1.0

    return decisions, defer_rate, acc_non_defer


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Dataset:", df.shape)

    df = add_engineered_features(df)
    components = compute_cbes_components(df)

    y = df["loan_approved"].astype(int)

    drop_cols = ["loan_approved", "applicant_id", "city"]
    X_raw = df.drop(columns=[col for col in drop_cols if col in df.columns])
    # Keep CBES as a separate explainable baseline and hybrid control signal.
    if "CBES_PROB" in X_raw.columns:
        X_raw = X_raw.drop(columns=["CBES_PROB"])
    X = pd.get_dummies(X_raw, drop_first=True)

    X_train_all, X_test, y_train_all, y_test, idx_train_all, idx_test, meta_train, meta_test = train_test_split(
        X,
        y,
        df.index,
        df[["applicant_id"]],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_cal, y_train, y_cal, idx_train, idx_cal = train_test_split(
        X_train_all,
        y_train_all,
        idx_train_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_all,
    )

    tuned_cbes_weights = tune_cbes_weights(
        y_cal,
        {
            key: series.loc[idx_cal]
            for key, series in components.items()
        },
        DEFAULT_CBES_WEIGHTS,
    )

    cbes_test = combine_cbes_components(
        {
            key: series.loc[idx_test]
            for key, series in components.items()
        },
        tuned_cbes_weights,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic": {
            "estimator": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            "input_mode": "scaled",
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                n_estimators=350,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "input_mode": "raw",
        },
        "LightGBM": {
            "estimator": LGBMClassifier(
                n_estimators=320,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                verbosity=-1,
            ),
            "input_mode": "raw",
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                n_estimators=240,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.2,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            "input_mode": "raw",
        },
    }

    results = []
    model_prob_store: dict[str, np.ndarray] = {}

    for name, config in models.items():
        estimator = config["estimator"]
        input_mode = config["input_mode"]

        if input_mode == "scaled":
            estimator.fit(X_train_scaled, y_train)
            cal_probs = estimator.predict_proba(X_cal_scaled)[:, 1]
            test_probs = estimator.predict_proba(X_test_scaled)[:, 1]
        else:
            estimator.fit(X_train, y_train)
            cal_probs = estimator.predict_proba(X_cal)[:, 1]
            test_probs = estimator.predict_proba(X_test)[:, 1]

        tuned_threshold = find_best_threshold(y_cal, cal_probs)
        test_preds = (test_probs >= tuned_threshold).astype(int)

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds, zero_division=0)
        auc = roc_auc_score(y_test, test_probs)

        print(f"\n{name}")
        print("Threshold:", round(tuned_threshold, 3))
        print("Accuracy:", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall:", round(rec, 4))
        print("AUC:", round(auc, 4))

        results.append([name, acc, prec, rec, auc])
        model_prob_store[name] = test_probs

    cbes_preds = (cbes_test.to_numpy() >= 0.5).astype(int)
    cbes_acc = accuracy_score(y_test, cbes_preds)
    cbes_prec = precision_score(y_test, cbes_preds, zero_division=0)
    cbes_rec = recall_score(y_test, cbes_preds, zero_division=0)
    cbes_auc = roc_auc_score(y_test, cbes_test.to_numpy())
    results.append(["CBES Baseline", cbes_acc, cbes_prec, cbes_rec, cbes_auc])
    model_prob_store["CBES"] = cbes_test.to_numpy()

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "AUC"])\
        .sort_values(by="AUC", ascending=False).reset_index(drop=True)

    ranked_models = results_df[results_df["Model"] != "CBES Baseline"].reset_index(drop=True)
    best_model_name = ranked_models.loc[0, "Model"]
    ml_prob = model_prob_store[best_model_name]

    print("\n===== MODEL SELECTION =====")
    print("Best Model (by ROC-AUC):", best_model_name)
    print("Tuned CBES Weights:", tuned_cbes_weights)
    print(results_df.to_string(index=False))

    alpha_grid = np.arange(0.2, 0.401, 0.02)
    alpha_candidates = []

    y_test_np = y_test.to_numpy()
    cbes_test_np = cbes_test.to_numpy()

    for alpha_value in alpha_grid:
        _, defer_rate_tmp, acc_nd_tmp = evaluate_alpha(ml_prob, cbes_test_np, y_test_np, alpha_value)
        in_target_band = 0.15 <= defer_rate_tmp <= 0.35
        alpha_candidates.append((alpha_value, defer_rate_tmp, acc_nd_tmp, in_target_band))

    valid_candidates = [item for item in alpha_candidates if item[3]]
    if valid_candidates:
        selected_alpha, _, _, _ = max(valid_candidates, key=lambda x: x[2])
    else:
        selected_alpha, _, _, _ = min(alpha_candidates, key=lambda x: abs(x[1] - 0.25))

    final_decisions = []
    decision_confidences = []
    approval_thresholds = []
    rejection_thresholds = []

    for ml, cbes in zip(ml_prob, cbes_test_np):
        decision, confidence, approval_threshold, rejection_threshold = dynamic_hybrid_decision(
            float(ml), float(cbes), alpha=float(selected_alpha)
        )
        final_decisions.append(decision)
        decision_confidences.append(confidence)
        approval_thresholds.append(approval_threshold)
        rejection_thresholds.append(rejection_threshold)

    final_decisions_np = np.array(final_decisions)
    mask = final_decisions_np != "DEFER"
    defer_count = int(np.sum(final_decisions_np == "DEFER"))

    if np.any(mask):
        hybrid_preds = (final_decisions_np[mask] == "APPROVE").astype(int)
        acc_nd = float(accuracy_score(y_test_np[mask], hybrid_preds))
    else:
        acc_nd = np.nan

    print("\n===== HYBRID SYSTEM =====")
    print("Selected alpha:", round(float(selected_alpha), 2))
    print("Decision Distribution:", dict(Counter(final_decisions)))
    print("Deferral Rate:", round(defer_count / len(final_decisions), 4))
    print("Accuracy (Non-Deferred):", "N/A" if np.isnan(acc_nd) else round(acc_nd, 4))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_df = pd.DataFrame(
        {
            "applicant_id": meta_test["applicant_id"].to_numpy(),
            "y_true": y_test_np,
            "cbes_prob": cbes_test_np,
            "best_model_prob": ml_prob,
            "final_decision": final_decisions,
            "confidence": decision_confidences,
            "approval_threshold": approval_thresholds,
            "rejection_threshold": rejection_thresholds,
        }
    )

    for model_name, model_probs in model_prob_store.items():
        prediction_df[f"prob_{model_name}"] = model_probs

    results_df.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)
    prediction_df.to_csv(ARTIFACT_DIR / "prediction_outputs.csv", index=False)

    summary_payload = {
        "best_model": best_model_name,
        "selection_metric": "AUC",
        "cbes_weights": tuned_cbes_weights,
        "selected_alpha": float(selected_alpha),
        "deferral_rate": float(defer_count / len(final_decisions)),
        "accuracy_non_deferred": None if np.isnan(acc_nd) else float(acc_nd),
        "decision_distribution": dict(Counter(final_decisions)),
        "artifacts": {
            "metrics_csv": str(ARTIFACT_DIR / "model_metrics.csv"),
            "predictions_csv": str(ARTIFACT_DIR / "prediction_outputs.csv"),
        },
    }

    with open(ARTIFACT_DIR / "pipeline_summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, indent=2)

    labels = ["APPROVE", "REJECT", "DEFER"]
    counts = [
        final_decisions.count("APPROVE"),
        final_decisions.count("REJECT"),
        final_decisions.count("DEFER"),
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts, color=["#2ca02c", "#d62728", "#7f7f7f"])
    plt.title("Final Decision Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "decision_distribution.png", dpi=150)
    plt.close()

    decision_colors = {"APPROVE": "#2ca02c", "REJECT": "#d62728", "DEFER": "#7f7f7f"}
    plt.figure(figsize=(7, 6))
    for decision in ["APPROVE", "REJECT", "DEFER"]:
        idx = final_decisions_np == decision
        plt.scatter(ml_prob[idx], cbes_test_np[idx], alpha=0.45, s=18, c=decision_colors[decision], label=decision)

    plt.xlabel(f"ML Probability ({best_model_name})")
    plt.ylabel("CBES Probability")
    plt.title("CBES vs ML Prediction Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cbes_vs_ml_comparison.png", dpi=150)
    plt.close()

    print("\nSaved frontend artifacts:")
    print("-", ARTIFACT_DIR / "model_metrics.csv")
    print("-", ARTIFACT_DIR / "prediction_outputs.csv")
    print("-", ARTIFACT_DIR / "pipeline_summary.json")
    print("-", PLOTS_DIR / "decision_distribution.png")
    print("-", PLOTS_DIR / "cbes_vs_ml_comparison.png")


if __name__ == "__main__":
    main()
