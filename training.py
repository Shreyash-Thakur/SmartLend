
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("synthetic_indian_loan_dataset.csv")
print("Dataset:", df.shape)


for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category").cat.codes


df["EMI_INCOME_RATIO"] = df["emi"] / (df["monthly_income"] + 1)
df["LOAN_INCOME_RATIO"] = df["loan_amount"] / (df["annual_income"] + 1)
df["DEBT_BURDEN"] = (df["existing_emis"] + df["emi"]) / (df["monthly_income"] + 1)
df["ASSET_COVERAGE"] = df["total_assets"] / (df["loan_amount"] + 1)


def compute_cbes(row):
    score = 0

    # Income stability
    if row["annual_income"] > 800000:
        score += 2
    elif row["annual_income"] > 400000:
        score += 1
    else:
        score -= 1

    # CIBIL
    if row["cibil_score"] > 750:
        score += 3
    elif row["cibil_score"] > 650:
        score += 1
    else:
        score -= 2

    # Debt burden
    if row["DEBT_BURDEN"] > 0.6:
        score -= 2
    elif row["DEBT_BURDEN"] > 0.4:
        score -= 1
    else:
        score += 1

    # Missed payments
    if row["missed_payments"] > 2:
        score -= 2
    elif row["missed_payments"] > 0:
        score -= 1
    else:
        score += 1

    # Assets
    if row["ASSET_COVERAGE"] > 1:
        score += 2
    else:
        score -= 1

    return score

df["CBES"] = df.apply(compute_cbes, axis=1)

# Convert CBES to probability-like score
df["CBES_PROB"] = 1 / (1 + np.exp(-df["CBES"]))


TARGET = "default_risk"
X = df.drop(columns=[TARGET])
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

ml_probs = model.predict_proba(X_test_scaled)[:, 1]
ml_preds = model.predict(X_test_scaled)

cbes_probs = X_test["CBES_PROB"].values
cbes_preds = (cbes_probs > 0.5).astype(int)


def hybrid_decision(ml_p, cbes_p):
    combined = 0.6 * ml_p + 0.4 * cbes_p

    # disagreement logic
    if abs(ml_p - cbes_p) > 0.3:
        return "DEFER"

    if combined < 0.4:
        return "APPROVE"
    elif combined > 0.6:
        return "REJECT"
    else:
        return "DEFER"

hybrid_decisions = np.array([
    hybrid_decision(m, c) for m, c in zip(ml_probs, cbes_probs)
])


def evaluate(name, preds, probs):
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n{name}")
    print("Accuracy:", round(acc,4))
    print("Precision:", round(prec,4))
    print("Recall:", round(rec,4))
    print("AUC:", round(auc,4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

# ML
evaluate("ML MODEL", ml_preds, ml_probs)

# CBES
evaluate("CBES MODEL", cbes_preds, cbes_probs)

# HYBRID (only non-deferred)
mask = hybrid_decisions != "DEFER"

hybrid_preds = (ml_probs > 0.5).astype(int)

if mask.sum() > 0:
    acc_h = accuracy_score(y_test[mask], hybrid_preds[mask])
else:
    acc_h = 0

print("\nHYBRID SYSTEM")
print("Deferral Rate:", round((~mask).sum()/len(mask),4))
print("Accuracy (Non-Deferred):", round(acc_h,4))


labels = ["ML", "CBES", "Hybrid ND"]
accuracy_vals = [
    accuracy_score(y_test, ml_preds),
    accuracy_score(y_test, cbes_preds),
    acc_h
]

plt.figure()
plt.bar(labels, accuracy_vals)
plt.title("Model Comparison")
plt.show()

# CBES vs ML probability scatter
plt.figure()
plt.scatter(ml_probs, cbes_probs, alpha=0.3)
plt.xlabel("ML Probability")
plt.ylabel("CBES Probability")
plt.title("ML vs CBES Agreement")
plt.show()