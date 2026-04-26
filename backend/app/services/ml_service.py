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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, f1_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# We import the exact conservative risk-aware defaults defined directly in CBES Engine
from backend.app.services.cbes_engine import DEFAULTS
from backend.app.services.decision_engine import hybrid_decision

# Artifact Paths
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
PIPELINE_V2_PATH = ARTIFACTS_DIR / "pipeline_v2.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.csv"
TARGET_COL = "default_risk"

# Dataset path (used by training_data_service)
DATASET_PATH = Path(__file__).resolve().parents[2] / "synthetic_indian_loan_dataset.csv"

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
    unfitted_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", models[best_name])
    ])
    calib_method = "isotonic" if len(X) > 1000 else "sigmoid"
    calibrator = CalibratedClassifierCV(unfitted_pipeline, method=calib_method, cv=5)
    calibrator.fit(X, y)

    # ── T_base discovery (F1-optimal threshold on a held-out validation split) ──
    # We use 20% of the training data as the calibration/validation split.
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Fit a fresh calibrator on the sub-train to get honest val probabilities
    _calib_val = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()), ("model", models[best_name])]),
        method=calib_method, cv=5
    )
    _calib_val.fit(X_train_t, y_train_t)
    probs_val = _calib_val.predict_proba(X_val_t)[:, 1]

    # F1 sweep: p_ml is P(approval) = P(no-default), so predict default when p_ml < t
    thresholds = np.arange(0.30, 0.70, 0.01)
    f1_scores  = [f1_score(y_val_t, (probs_val < t).astype(int), zero_division=0)
                  for t in thresholds]
    t_base = float(round(thresholds[int(np.argmax(f1_scores))], 4))
    print(f"[train_pipeline] F1-optimal T_base = {t_base:.4f} "
          f"(F1 = {max(f1_scores):.4f})")

    # Cache representative background for LinearExplainer (using best_pipeline full fit)
    background_data = best_pipeline.named_steps["scaler"].transform(
        X.sample(min(100, len(X)), random_state=42)
    )

    payload = {
        "pipeline":        best_pipeline,
        "calibrator":      calibrator,
        "feature_names":   feature_names,
        "model_name":      best_name,
        "background_data": background_data,
        "t_base":          t_base,    # F1-optimal threshold — used by decision_engine
        "tau_d":           0.30,      # default; overridden by calibrate_and_save()
    }

    joblib.dump(payload, PIPELINE_PATH)


class MLPredictor:
    def __init__(self):
        """Loads and caches artifact purely read-only state. Prevents runtime retraining constraint."""
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at {PIPELINE_PATH}. Refusing to retrain at startup.")

        artifact_path = PIPELINE_V2_PATH if PIPELINE_V2_PATH.exists() else PIPELINE_PATH
        payload = joblib.load(artifact_path)
        self.pipeline = payload["pipeline"]
        self.calibrator = payload["calibrator"]
        self.feature_names = payload["feature_names"]
        self.model_name = payload["model_name"]
        self.background_data = payload.get("background_data")
        self.t_base = float(payload.get("t_base", 0.50))   # F1-optimal threshold

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

        # calibrator.predict_proba(df)[:, 1] = P(default_risk=1) = P(default)
        # the decision engine approves when p_ml >= t_approve, so p_ml must be
        # P(approval) = P(no default) = 1 - P(default)
        p_ml = float(1.0 - self.calibrator.predict_proba(df)[0, 1])
        
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

        # Grab TAU_D and T_base from artifact
        try:
            artifact_path = PIPELINE_V2_PATH if PIPELINE_V2_PATH.exists() else PIPELINE_PATH
            _payload = joblib.load(artifact_path)
            tau_d  = float(_payload.get("tau_d",  0.30))
            t_base = float(_payload.get("t_base", self.t_base))
            all_pipelines = _payload.get("all_pipelines", {})
        except Exception:
            tau_d  = 0.30
            t_base = self.t_base
            all_pipelines = {}

        all_model_predictions = {}
        if all_pipelines:
            for name, pipe in all_pipelines.items():
                all_model_predictions[name] = float(1.0 - pipe.predict_proba(df)[0, 1])

        active_model_file = ARTIFACTS_DIR / "active_model.txt"
        active_model = self.model_name
        if active_model_file.exists():
            active_model = active_model_file.read_text().strip()

        if active_model in all_model_predictions:
            p_ml = all_model_predictions[active_model]
            used_model_name = active_model
        else:
            p_ml = float(1.0 - self.calibrator.predict_proba(df)[0, 1])
            used_model_name = self.model_name
            
        # Execute Decision Engine
        decision_result = hybrid_decision(
            p_ml=p_ml,
            p_cbes=p_cbes,
            tau_d=tau_d,
            t_base=t_base,
            shap_explanation=top_3_shap,
            cbes_breakdown=cbes_breakdown,
            all_model_predictions=all_model_predictions,
        )
        
        # In order to satisfy the legacy API structure expecting these parameters as raw dots
        # we attach engineered_features, cbes_weights, cbes_components, selected_model
        decision_result.engineered_features = sanitized
        decision_result.cbes_weights = {} # Weight is fixed inside cbes natively now
        decision_result.cbes_components = cbes_breakdown
        decision_result.selected_model = used_model_name
        
        return decision_result

# Global lazy initializer cache
_predictor = None

def get_predictor() -> MLPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


# ---------------------------------------------------------------------------
# Thin wrapper used by model_analysis_service and training_data_service
# ---------------------------------------------------------------------------

_cached_tau_d: float | None = None
_cached_t_base: float | None = None

def dynamic_hybrid_decision(
    p_ml:   float,
    p_cbes: float,
    tau_d:  float | None = None,
    t_base: float | None = None,
) -> tuple[str, float, float, float]:
    """Stateless wrapper around hybrid_decision.

    Loads TAU_D and T_base from the pipeline artifact when not provided.
    Returns (decision, confidence, t_approve, t_reject).
    """
    global _cached_tau_d, _cached_t_base

    _tau_d  = 0.30
    _t_base = 0.50

    if _cached_tau_d is not None and _cached_t_base is not None:
        _tau_d = _cached_tau_d
        _t_base = _cached_t_base
    else:
        artifact_path = PIPELINE_V2_PATH if PIPELINE_V2_PATH.exists() else PIPELINE_PATH
        if artifact_path.exists():
            try:
                _pl = joblib.load(artifact_path)
                _tau_d  = float(_pl.get("tau_d",  _tau_d))
                _t_base = float(_pl.get("t_base", _t_base))
                _cached_tau_d = _tau_d
                _cached_t_base = _t_base
            except Exception:
                pass

    if tau_d  is not None: _tau_d  = tau_d
    if t_base is not None: _t_base = t_base

    result = hybrid_decision(
        p_ml=p_ml, p_cbes=p_cbes,
        tau_d=_tau_d, t_base=_t_base,
    )
    return result.decision, result.confidence, result.t_approve, result.t_reject

