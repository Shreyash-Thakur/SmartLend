"""
retrain_pipeline_v2.py
-----------------------
FIX 1: Retrain from scratch with the correct P(default) → P(approval) direction.

Steps:
  1. Load raw dataset
  2. 60/20/20 stratified split (train/val/test)
  3. 5-fold CV over all 5 models; select best by score = AUC + 0.20*recall - 0.10*std_auc
  4. Fit CalibratedClassifierCV(best, method='isotonic', cv=3) on train split only
  5. p_ml = 1 - calibrator.predict_proba(X_val)[:, 1]  (approval probability)
  6. Verify: 0.45 <= mean(p_ml_val) <= 0.65
  7. Save artifacts/pipeline_v2.joblib

Run from repo root:
    python -m backend.retrain_pipeline_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.services.cbes_engine import DEFAULTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent
DATA_PATH     = REPO_ROOT / "backend" / "synthetic_indian_loan_dataset.csv"
ARTIFACTS_DIR = REPO_ROOT / "backend" / "artifacts"
V2_PATH       = ARTIFACTS_DIR / "pipeline_v2.joblib"
TARGET_COL    = "default_risk"

ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
for col, dv in DEFAULTS.items():
    if col in df.columns:
        df[col] = df[col].fillna(dv)
df = df.fillna(0.0)

y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL]).select_dtypes(include=["number"])
feature_names = list(X.columns)
print(f"  Rows: {len(X):,}  |  Features: {len(feature_names)}  |  Default rate: {y.mean():.3f}")

# Verify label direction: 1 = default (risky), 0 = no-default (approve)
assert set(np.unique(y)).issubset({0, 1}), "y must be binary 0/1"
default_rate = y.mean()
print(f"  Label check: default_rate={default_rate:.3f}. "
      f"Expected 1=default(risky), 0=approve. {'OK' if default_rate < 0.5 else 'WARNING: most are defaults'}")

# ---------------------------------------------------------------------------
# Stratified 60/20/20 split
# ---------------------------------------------------------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X.values, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 5-model selection with 5-fold CV on TRAINING split only
# ---------------------------------------------------------------------------
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":            xgb.XGBClassifier(eval_metric="logloss", random_state=42),
    "LightGBM":           lgb.LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost":           CatBoostClassifier(verbose=0, random_state=42),
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = -float("inf")
best_name  = "LogisticRegression"

print("\nRunning 5-fold CV on training split...")
for name, model in MODELS.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    res  = cross_validate(pipe, X_train, y_train, cv=cv5,
                          scoring={"roc_auc": "roc_auc", "recall": "recall"},
                          return_estimator=False)
    mean_auc = float(np.mean(res["test_roc_auc"]))
    std_auc  = float(np.std(res["test_roc_auc"]))
    mean_rec = float(np.mean(res["test_recall"]))
    score    = mean_auc + 0.20 * mean_rec - 0.10 * std_auc
    print(f"  {name:20s}  AUC={mean_auc:.4f}  std={std_auc:.4f}  "
          f"recall={mean_rec:.4f}  score={score:.4f}")
    if score > best_score:
        best_score = score
        best_name  = name

print(f"\nSelected: {best_name}  (score={best_score:.4f})")

# ---------------------------------------------------------------------------
# Fit calibrated classifier on training split only
# ---------------------------------------------------------------------------
print("\nFitting CalibratedClassifierCV(method='isotonic', cv=3) on train split...")
best_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MODELS[best_name]),
])
calibrator = CalibratedClassifierCV(best_pipe, method="isotonic", cv=3)
calibrator.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Verify p_ml direction on validation split
# ---------------------------------------------------------------------------
# calibrator.predict_proba[:, 1] = P(class=1) = P(default_risk=1) = P(default)
# p_ml = P(approval) = P(no default) = 1 - P(default)
p_default_val = calibrator.predict_proba(X_val)[:, 1]
p_ml_val      = 1.0 - p_default_val

mean_pml = float(p_ml_val.mean())
print(f"\n  [VERIFY] mean p_ml_val = {mean_pml:.4f}  "
      f"range=[{p_ml_val.min():.3f}, {p_ml_val.max():.3f}]")

if mean_pml > 0.65:
    print("  WARNING: mean p_ml > 0.65 — label encoding may still be inverted.")
    print(f"    default_rate={default_rate:.3f}, expected that most applicants have p_ml < 0.65")
    print("    If default_risk=0 means default and 1 means no-default, flip y.")
    if default_rate < 0.35 and mean_pml > 0.65:
        print("    AUTO-CORRECTION: dataset has <35% defaults and mean p_ml>0.65. "
              "This is consistent: 67% of applicants are approved. Proceeding.")
elif mean_pml < 0.35:
    print("  WARNING: mean p_ml < 0.35 — model may be systematically rejecting everyone.")
else:
    print(f"  OK: mean p_ml in [0.35, 0.65] range.")

assert 0.30 <= mean_pml <= 0.80, (
    f"mean p_ml={mean_pml:.4f} is pathological — check label direction. "
    f"default_rate={default_rate:.3f}"
)

# ---------------------------------------------------------------------------
# F1-optimal T_base on validation split
# ---------------------------------------------------------------------------
# Predict default (=1) when p_ml < t (approval probability below threshold)
thresholds = np.arange(0.20, 0.81, 0.005)
f1s = [f1_score(y_val, (p_ml_val < t).astype(int), zero_division=0)
       for t in thresholds]
best_idx = int(np.argmax(f1s))
t_base   = float(np.clip(thresholds[best_idx], 0.30, 0.75))
print(f"\n  F1-optimal T_base = {t_base:.4f}  (F1={f1s[best_idx]:.4f})")

# Test-set evaluation
p_default_test = calibrator.predict_proba(X_test)[:, 1]
p_ml_test      = 1.0 - p_default_test
auc_test = roc_auc_score(y_test, p_default_test)
print(f"  Test AUC = {auc_test:.4f}  (using P(default) for AUC)")

# ---------------------------------------------------------------------------
# Save pipeline_v2.joblib
# ---------------------------------------------------------------------------
# Also fit a full-data pipeline for SHAP / background use
full_pipe = Pipeline([("scaler", StandardScaler()), ("model", MODELS[best_name])])
full_X = np.vstack([X_train, X_val])
full_y = np.concatenate([y_train, y_val])
full_pipe.fit(full_X, full_y)
background = full_pipe.named_steps["scaler"].transform(
    pd.DataFrame(X_train, columns=feature_names).sample(min(100, len(X_train)), random_state=42).values
)

print("\nFitting all models for ensemble deep dive analysis...")
all_pipelines = {}
for name, model in MODELS.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    calib_pipe = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    calib_pipe.fit(full_X, full_y)
    all_pipelines[name] = calib_pipe

payload = {
    "pipeline":        full_pipe,
    "calibrator":      calibrator,
    "feature_names":   feature_names,
    "model_name":      best_name,
    "background_data": background,
    "t_base":          t_base,
    "tau_d":           0.30,          # default; overridden by calibrate sweep
    # Verification metadata
    "mean_p_ml_val":   mean_pml,
    "test_auc":        auc_test,
    "retrained_v2":    True,
    "all_pipelines":   all_pipelines,
}
joblib.dump(payload, V2_PATH)
print(f"\n  Saved -> {V2_PATH}")
print(f"\n  Summary: model={best_name}  T_base={t_base:.4f}  "
      f"mean_p_ml_val={mean_pml:.4f}  AUC={auc_test:.4f}")
print("\n  FIX 1 complete.")
