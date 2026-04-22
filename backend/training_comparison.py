import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from lightgbm import LGBMClassifier


def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    df["EMI_INCOME_RATIO"] = df["emi"] / (df["monthly_income"] + 1)
    df["LOAN_INCOME_RATIO"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["DEBT_BURDEN"] = (df["existing_emis"] + df["emi"]) / (df["monthly_income"] + 1)
    df["ASSET_COVERAGE"] = df["total_assets"] / (df["loan_amount"] + 1)

    target = "default_risk"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    }


def evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, output_dir: Path):
    models = get_models()
    results = []

    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        if name in ["Logistic Regression", "Naive Bayes"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)

        defer_mask = (probs > 0.4) & (probs < 0.6)
        non_defer_mask = ~defer_mask
        deferral_rate = float(np.mean(defer_mask))
        acc_non_defer = float(accuracy_score(y_test[non_defer_mask], preds[non_defer_mask])) if np.any(non_defer_mask) else np.nan

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "AUC": auc,
                "Deferral Rate": deferral_rate,
                "Accuracy (Non-Deferred)": acc_non_defer,
            }
        )

        print(f"\n{name} Confusion Matrix:")
        print(cm)

        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False).reset_index(drop=True)

    print("\n===== FINAL RESULTS =====")
    print(results_df.to_string(index=False))

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    roc_path = output_dir / "roc_curve_comparison.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    for _, row in results_df.iterrows():
        plt.scatter(row["Deferral Rate"], row["Accuracy"], label=row["Model"])
        plt.text(row["Deferral Rate"], row["Accuracy"], row["Model"], fontsize=8)
    plt.xlabel("Deferral Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Deferral Tradeoff")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    acc_def_path = output_dir / "accuracy_vs_deferral.png"
    plt.savefig(acc_def_path, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "AUC"]].plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    perf_path = output_dir / "model_performance_comparison.png"
    plt.savefig(perf_path, dpi=150)
    plt.close()

    results_df["Coverage"] = 1 - results_df["Deferral Rate"]
    plt.figure(figsize=(8, 6))
    for _, row in results_df.iterrows():
        if not np.isnan(row["Accuracy (Non-Deferred)"]):
            plt.scatter(row["Coverage"], row["Accuracy (Non-Deferred)"], label=row["Model"])
            plt.text(row["Coverage"], row["Accuracy (Non-Deferred)"], row["Model"], fontsize=8)
    plt.xlabel("Coverage (Decisions Made)")
    plt.ylabel("Accuracy (Non-Deferred)")
    plt.title("Coverage vs Accuracy Tradeoff")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    coverage_path = output_dir / "coverage_vs_accuracy.png"
    plt.savefig(coverage_path, dpi=150)
    plt.close()

    print("\nSaved plots:")
    print(f"- {roc_path}")
    print(f"- {acc_def_path}")
    print(f"- {perf_path}")
    print(f"- {coverage_path}")


def main():
    parser = argparse.ArgumentParser(description="Old-style model comparison training")
    parser.add_argument("--csv-path", default="synthetic_indian_loan_dataset.csv")
    parser.add_argument("--output-dir", default="comparison_outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_and_prepare_data(args.csv_path)
    evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, output_dir)


if __name__ == "__main__":
    main()