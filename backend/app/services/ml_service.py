import os
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, recall_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# We import the exact conservative risk-aware defaults defined directly in CBES Engine
from backend.app.services.cbes_engine import DEFAULTS

# Artifact Paths
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.csv"
TARGET_COL = "default_risk"

def train_pipeline(df: pd.DataFrame) -> None:
    """Train the unified pipeline using cross-validation over the top 5 model architectures.
    Performs score targeting, calibration, and joblib caching safely.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    df = df.copy()
    
    # Defensive replacement of any NaNs to conservative risk defaults across numerical distributions
    for col, default_val in DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    df = df.fillna(0.0) # Any leftover unknown fields get hard floor
    
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])
    
    # Filter identifiers mapping strings out
    X = X.select_dtypes(include=['number'])
    feature_names = list(X.columns)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    best_score = -float('inf')
    best_name = None
    best_pipeline = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_log = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        
        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring={"roc_auc": "roc_auc", "recall": "recall"}, return_estimator=False
        )
        
        mean_auc = np.mean(cv_results["test_roc_auc"])
        std_auc = np.std(cv_results["test_roc_auc"])
        mean_recall = np.mean(cv_results["test_recall"])
        
        score = mean_auc + 0.20 * mean_recall - 0.10 * std_auc
        
        metrics_log.append({
            "model": name,
            "roc_auc": mean_auc,
            "std_auc": std_auc,
            "recall": mean_recall,
            "custom_score": score
        })
        
        if score > best_score:
            best_score = score
            best_name = name
            
            best_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            best_pipeline.fit(X, y)

    # Save metrics 
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(METRICS_PATH, index=False)

    # Calibrate probability logic using strictly out-of-fold cross-validation (cv=5)
    # This avoids doing calibration on the same exact data the pipeline naturally trained on.
    unfitted_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", models[best_name])
    ])
    calib_method = "isotonic" if len(X) > 1000 else "sigmoid"
    calibrator = CalibratedClassifierCV(unfitted_pipeline, method=calib_method, cv=5)
    calibrator.fit(X, y)

    # Cache representative background for LinearExplainer if required (using best_pipeline full fit)
    background_data = best_pipeline.named_steps["scaler"].transform(X.sample(min(100, len(X)), random_state=42))

    payload = {
        "pipeline": best_pipeline,
        "calibrator": calibrator,
        "feature_names": feature_names,
        "model_name": best_name,
        "background_data": background_data
    }
    
    joblib.dump(payload, PIPELINE_PATH)


class MLPredictor:
    def __init__(self):
        """Loads and caches artifact purely read-only state. Prevents runtime retraining constraint."""
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at {PIPELINE_PATH}. Refusing to retrain at startup.")
            
        payload = joblib.load(PIPELINE_PATH)
        self.pipeline = payload["pipeline"]
        self.calibrator = payload["calibrator"]
        self.feature_names = payload["feature_names"]
        self.model_name = payload["model_name"]
        self.background_data = payload.get("background_data")
        
        # Unpack from unified pipeline explicitly for SHAP overhead caching
        self.scaler = self.pipeline.named_steps["scaler"]
        self.classifier = self.pipeline.named_steps["model"]
        
        # Cache explainer mapping
        try:
            if "Logistic" in str(self.model_name):
                self.explainer = shap.LinearExplainer(self.classifier, self.background_data)
            else:
                self.explainer = shap.TreeExplainer(self.classifier)
        except Exception:
            self.explainer = None

    def predict_application(self, input_data: Dict[str, Any]) -> Any:
        from backend.app.services.cbes_engine import compute_cbes
        from backend.app.services.decision_engine import hybrid_decision
        
        # Fill missing values aggressively toward worst risk profile prior to scaling
        sanitized = {}
        for col in self.feature_names:
            val = input_data.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = DEFAULTS.get(col, 0.0)
            sanitized[col] = val
            
        df = pd.DataFrame([sanitized], columns=self.feature_names)
        
        # Identical inference transformation matching calibrator requirements
        p_ml = float(self.calibrator.predict_proba(df)[0, 1])
        
        # Compute CBES locally
        p_cbes, cbes_breakdown = compute_cbes(input_data)
        
        # Optimize SHAP top 3 calculation via exact internal scaler transform mapped prior to wrapper
        top_3_shap = []
        if self.explainer:
            X_scaled = self.scaler.transform(df)
            try:
                shap_values = self.explainer.shap_values(X_scaled)
                
                # Dimensionality correction maps across sklearn + xgboost/lgb/cat
                if isinstance(shap_values, list):
                    vals = np.array(shap_values[-1][0])
                elif len(np.array(shap_values).shape) == 3:
                    vals = np.array(shap_values)[0, :, -1]
                else:
                    vals = np.array(shap_values)[0]
                    
                impacts = pd.Series(vals, index=self.feature_names)
                top = impacts.abs().sort_values(ascending=False).head(3)
                
                # We return exact SHAP absolute mapping keys without the giant explanation array
                top_3_shap = [{"name": k, "impact": float(v), "value": float(sanitized.get(k, 0.0))} for k, v in top.items()]
            except Exception:
                pass

        # Grab TAU_D from artifact or default to 0.3
        import joblib
        try:
            payload = joblib.load(PIPELINE_PATH)
            tau_d = float(payload.get("tau_d", 0.30))
        except Exception:
            tau_d = 0.30
            
        # Execute Decision Engine
        decision_result = hybrid_decision(
            p_ml=p_ml,
            p_cbes=p_cbes,
            tau_d=tau_d,
            shap_explanation=top_3_shap,
            cbes_breakdown=cbes_breakdown
        )
        
        # In order to satisfy the legacy API structure expecting these parameters as raw dots
        # we attach engineered_features, cbes_weights, cbes_components, selected_model
        decision_result.engineered_features = sanitized
        decision_result.cbes_weights = {} # Weight is fixed inside cbes natively now
        decision_result.cbes_components = cbes_breakdown
        decision_result.selected_model = self.model_name
        
        return decision_result

# Global lazy initializer cache
_predictor = None

def get_predictor() -> MLPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor
