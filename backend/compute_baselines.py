import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.app.services.cbes_engine import DEFAULTS

DATA_PATH = Path(__file__).resolve().parent / "synthetic_indian_loan_dataset.csv"
TARGET_COL = "default_risk"

def compute_matrices():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    for col, default_val in DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    df = df.fillna(0.0)
    
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])
    X = X.select_dtypes(include=['number'])
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for name, model in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        print(f"--- {name} ---")
        print(f"TN (True Approve): {tn} | FP (False Reject): {fp}")
        print(f"FN (False Approve): {fn}  | TP (True Reject): {tp}")
        print(f"Matrix:\n  [[{tn}, {fp}]\n   [{fn},  {tp}]]\n")

if __name__ == "__main__":
    compute_matrices()
