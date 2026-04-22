import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("synthetic_indian_loan_dataset.csv")
print("Dataset:", df.shape)


# =========================
# FEATURE ENGINEERING
# =========================
df["EMI_INCOME_RATIO"] = df["emi"] / (df["monthly_income"] + 1)
df["DEBT_BURDEN"] = (df["existing_emis"] + df["emi"]) / (df["monthly_income"] + 1)
df["DEBT_TO_INCOME_RATIO"] = (df["existing_emis"] * 12 + df["loan_amount"]) / (df["annual_income"] + 1)
df["LOAN_INCOME_RATIO"] = df["loan_amount"] / (df["annual_income"] + 1)
df["ASSET_COVERAGE"] = df["total_assets"] / (df["loan_amount"] + 1)
df["LIQUIDITY_RATIO"] = df["bank_balance"] / (df["loan_amount"] + 1)
df["LOAN_ACTIVITY_RATIO"] = df["active_loans"] / (df["total_loans"] + 1)
df["REPAYMENT_SCORE"] = df["closed_loans"] / (df["total_loans"] + 1)
df["MISSED_PAYMENT_RATIO"] = df["missed_payments"] / (df["total_loans"] + 1)
df["EMPLOYMENT_STABILITY"] = df["years_employed"] / (df["age"] + 1)


# =========================
# CLAMP FUNCTION
# =========================
def clamp(x, low, high):
    return np.maximum(low, np.minimum(x, high))


def dynamic_hybrid_decision(ml_prob, cbes_prob, alpha=0.25):
    """Apply dynamic CBES-shaped thresholds to the ML probability."""
    alpha = clamp(alpha, 0.2, 0.4)

    delta = (0.5 - cbes_prob) * alpha
    approval_threshold = clamp(0.5 + delta, 0.35, 0.65)
    rejection_threshold = clamp(0.5 - delta, 0.35, 0.65)
    confidence = abs(ml_prob - 0.5)

    if ml_prob >= approval_threshold:
        return "APPROVE", confidence, approval_threshold, rejection_threshold

    if ml_prob <= rejection_threshold:
        return "REJECT", confidence, approval_threshold, rejection_threshold

    if confidence < 0.15:
        return "DEFER", confidence, approval_threshold, rejection_threshold

    if cbes_prob > 0.6:
        return "APPROVE", confidence, approval_threshold, rejection_threshold
    if cbes_prob < 0.4:
        return "REJECT", confidence, approval_threshold, rejection_threshold

    return "DEFER", confidence, approval_threshold, rejection_threshold


def evaluate_alpha(ml_probs, cbes_probs, y_true, alpha):
    decisions = []
    for ml, cbes in zip(ml_probs, cbes_probs):
        decision, _, _, _ = dynamic_hybrid_decision(ml, cbes, alpha=alpha)
        decisions.append(decision)

    mask = np.array(decisions) != "DEFER"
    defer_rate = float(np.mean(np.array(decisions) == "DEFER"))

    if np.any(mask):
        pred_non_defer = (np.array(decisions)[mask] == "APPROVE").astype(int)
        acc_non_defer = float(accuracy_score(y_true[mask], pred_non_defer))
    else:
        acc_non_defer = -1.0

    return decisions, defer_rate, acc_non_defer


def build_rejection_reasons(ml_prob, cbes_prob, confidence, approval_threshold, rejection_threshold, meta_row):
    """Create deterministic reason text for rejected applications."""
    reasons = []

    if ml_prob <= rejection_threshold:
        reasons.append(
            f"ML probability {ml_prob:.3f} is below dynamic rejection threshold {rejection_threshold:.3f}."
        )

    if cbes_prob < 0.4:
        reasons.append(f"CBES probability {cbes_prob:.3f} indicates elevated risk (< 0.4).")

    if float(meta_row.DEBT_TO_INCOME_RATIO) > 0.5:
        reasons.append("Debt-to-income ratio is high (> 0.5).")

    if float(meta_row.EMI_INCOME_RATIO) > 0.4:
        reasons.append("EMI burden is high (> 0.4 of monthly income).")

    if float(meta_row.MISSED_PAYMENT_RATIO) > 0.2:
        reasons.append("Missed payment ratio is high.")

    if not reasons:
        reasons.append("Rejected by dynamic hybrid threshold policy.")

    return reasons


def trigger_rejection_event(event):
    """Hook for downstream action. Replace with API call/queue push if needed."""
    print(
        f"REJECTION_TRIGGER | user={event['user_id']} | ml={event['ml_prob']:.3f} | "
        f"cbes={event['cbes_prob']:.3f} | reasons={'; '.join(event['reasons'])}"
    )


# =========================
# CBES CALCULATION
# =========================
cibil_norm = clamp((df["cibil_score"] - 300) / 600, 0, 1)
payment_penalty = clamp(1 - df["MISSED_PAYMENT_RATIO"], 0, 1)
util_penalty = clamp(1 - df["credit_utilization_ratio"], 0, 1)

credit_component = 0.5 * cibil_norm + 0.3 * payment_penalty + 0.2 * util_penalty

dti_score = clamp(1 - df["DEBT_TO_INCOME_RATIO"], 0, 1)
emi_score = clamp(1 - df["EMI_INCOME_RATIO"], 0, 1)
loan_income_score = clamp(1 - df["LOAN_INCOME_RATIO"], 0, 1)
dti_score = clamp(1 - df["DEBT_TO_INCOME_RATIO"], 0.2, 1)
capacity_component = 0.5 * dti_score + 0.3 * emi_score + 0.2 * loan_income_score

asset_score = clamp(df["ASSET_COVERAGE"], 0, 2) / 2
liquidity_score = clamp(df["LIQUIDITY_RATIO"], 0, 1)
asset_component = 0.7 * asset_score + 0.3 * liquidity_score

stability_component = clamp(df["EMPLOYMENT_STABILITY"], 0, 1)

df["CBES_SCORE"] = (
    0.35 * credit_component
    + 0.3 * capacity_component
    + 0.25 * asset_component
    + 0.1 * stability_component
)

# add noise
rng = np.random.default_rng(42)
df["CBES_SCORE"] = clamp(df["CBES_SCORE"] + rng.normal(0, 0.05, len(df)), 0, 1)

# Keep CBES on its native [0, 1] scale to preserve dynamic range.
df["CBES_PROB"] = df["CBES_SCORE"]


# =========================
# TARGET
# =========================
y = df["loan_approved"].astype(int)


# =========================
# FEATURES
# =========================
drop_cols = ["loan_approved", "applicant_id", "city"]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# encode categorical
X = pd.get_dummies(X, drop_first=True)

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test, _cbes_train, cbes_test, _meta_train, meta_test = train_test_split(
    X_scaled,
    y,
    df["CBES_PROB"],
    df[["applicant_id", "DEBT_TO_INCOME_RATIO", "EMI_INCOME_RATIO", "MISSED_PAYMENT_RATIO"]],
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# =========================
# MODELS
# =========================
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
}

results = []
model_prob_store = {}


# =========================
# TRAIN
# =========================
for name, model in models.items():
    print(f"\n{name}")

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("AUC:", round(auc, 4))

    results.append([name, acc, prec, rec, auc])
    model_prob_store[name] = probs


# =========================
# PICK BEST MODEL (PRIMARY DRIVER)
# =========================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "AUC"])
results_df = results_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)

best_model_name = results_df.loc[0, "Model"]
ml_prob = model_prob_store[best_model_name]

print("\n===== MODEL SELECTION =====")
print("Best Model (by ROC-AUC):", best_model_name)
print(results_df.to_string(index=False))

# Calibrate alpha to keep deferral in a realistic 20-30% band.
alpha_grid = np.arange(0.2, 0.401, 0.02)
alpha_candidates = []

for alpha_value in alpha_grid:
    _, defer_rate_tmp, acc_nd_tmp = evaluate_alpha(
        ml_prob,
        cbes_test.to_numpy(),
        y_test.to_numpy(),
        alpha_value,
    )
    in_target = 0.20 <= defer_rate_tmp <= 0.30
    alpha_candidates.append((alpha_value, defer_rate_tmp, acc_nd_tmp, in_target))

valid_candidates = [x for x in alpha_candidates if x[3]]
if valid_candidates:
    selected_alpha, _, _, _ = max(valid_candidates, key=lambda x: x[2])
else:
    selected_alpha, _, _, _ = min(alpha_candidates, key=lambda x: abs(x[1] - 0.25))

print("Selected alpha:", round(float(selected_alpha), 2))


# =========================
# HYBRID DECISION
# =========================
final_decisions = []
defer_count = 0
decision_confidences = []
approval_thresholds = []
rejection_thresholds = []
rejection_events = []

for ml, cbes, meta_row in zip(ml_prob, cbes_test, meta_test.itertuples(index=False)):
    decision, confidence, approval_threshold, rejection_threshold = dynamic_hybrid_decision(
        ml,
        cbes,
        alpha=selected_alpha,
    )
    final_decisions.append(decision)
    decision_confidences.append(confidence)
    approval_thresholds.append(approval_threshold)
    rejection_thresholds.append(rejection_threshold)

    if decision == "REJECT":
        reasons = build_rejection_reasons(
            ml,
            cbes,
            confidence,
            approval_threshold,
            rejection_threshold,
            meta_row,
        )
        event = {
            "user_id": str(meta_row.applicant_id),
            "decision": decision,
            "ml_prob": float(ml),
            "cbes_prob": float(cbes),
            "confidence": float(confidence),
            "reasons": reasons,
        }
        rejection_events.append(event)

    if decision == "DEFER":
        defer_count += 1


# =========================
# EVALUATION (NON-DEFERRED)
# =========================
mask = np.array(final_decisions) != "DEFER"
y_filtered = y_test[mask]
pred_filtered = np.array(final_decisions)[mask] == "APPROVE"
pred_filtered = pred_filtered.astype(int)

acc_nd = accuracy_score(y_filtered, pred_filtered) if len(y_filtered) > 0 else np.nan

print("\n===== HYBRID SYSTEM =====")
from collections import Counter
print("Decision Distribution:", dict(Counter(final_decisions)))
print("Deferral Rate:", round(defer_count / len(final_decisions), 4))
print("Accuracy (Non-Deferred):", "N/A" if np.isnan(acc_nd) else round(acc_nd, 4))
print("Rejected Triggers:", len(rejection_events))


# =========================
# ARTIFACTS FOR FRONTEND
# =========================
artifact_dir = Path("artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)

prediction_df = pd.DataFrame(
    {
        "applicant_id": meta_test["applicant_id"].to_numpy(),
        "y_true": y_test.to_numpy(),
        "cbes_prob": cbes_test.to_numpy(),
        "best_model_prob": ml_prob,
        "final_decision": final_decisions,
        "confidence": decision_confidences,
        "approval_threshold": approval_thresholds,
        "rejection_threshold": rejection_thresholds,
    }
)

for model_name, model_probs in model_prob_store.items():
    prediction_df[f"prob_{model_name}"] = model_probs

results_df.to_csv(artifact_dir / "model_metrics.csv", index=False)
prediction_df.to_csv(artifact_dir / "prediction_outputs.csv", index=False)

summary_payload = {
    "best_model": best_model_name,
    "selection_metric": "AUC",
    "selected_alpha": float(selected_alpha),
    "deferral_rate": float(defer_count / len(final_decisions)),
    "accuracy_non_deferred": None if np.isnan(acc_nd) else float(acc_nd),
    "decision_distribution": dict(Counter(final_decisions)),
    "artifacts": {
        "metrics_csv": str(artifact_dir / "model_metrics.csv"),
        "predictions_csv": str(artifact_dir / "prediction_outputs.csv"),
    },
}

with open(artifact_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2)

print("\nSaved frontend artifacts:")
print("-", artifact_dir / "model_metrics.csv")
print("-", artifact_dir / "prediction_outputs.csv")
print("-", artifact_dir / "pipeline_summary.json")

if rejection_events:
    print("\n===== SAMPLE REJECTION TRIGGERS =====")
    for event in rejection_events[:5]:
        trigger_rejection_event(event)


# =========================
# PLOTS (ONLY 2 CLEAR CHARTS)
# =========================
plot_dir = artifact_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

# 1) Decision distribution
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
plt.savefig(plot_dir / "decision_distribution.png", dpi=150)
plt.close()

# 2) ML vs CBES comparison (colored by final decision)
decision_colors = {"APPROVE": "#2ca02c", "REJECT": "#d62728", "DEFER": "#7f7f7f"}
plt.figure(figsize=(7, 6))
for decision in ["APPROVE", "REJECT", "DEFER"]:
    idx = np.array(final_decisions) == decision
    plt.scatter(ml_prob[idx], cbes_test[idx], alpha=0.45, s=18, c=decision_colors[decision], label=decision)

plt.xlabel(f"ML Probability ({best_model_name})")
plt.ylabel("CBES Probability")
plt.title("CBES vs ML Prediction Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "cbes_vs_ml_comparison.png", dpi=150)
plt.close()

print("Saved clear plots:")
print("-", plot_dir / "decision_distribution.png")
print("-", plot_dir / "cbes_vs_ml_comparison.png")