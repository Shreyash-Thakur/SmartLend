import functools
import hashlib
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
from backend.app.services.decision_engine import ENGINE_VERSION, hybrid_decision
from backend.app.services.threshold_selection import select_t_base

# Artifact Paths
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
PIPELINE_V2_PATH = ARTIFACTS_DIR / "pipeline_v2.joblib"
# v3: trained on the REAL Home Credit extract with the two leaked output columns
# (`loan_approved`, `confidence_score`) removed — see backend/retrain_serving_model_v3.py
# and reports/serving_model_retrain.json. v1/v2 are kept on disk for rollback.
PIPELINE_V3_PATH = ARTIFACTS_DIR / "pipeline_v3_real.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.csv"
TARGET_COL = "default_risk"


def _active_artifact_path() -> Path:
    """Newest artifact present on disk, newest-first with fallback.

    v3 (real Home Credit, leak removed) wins when it exists; otherwise the old
    v2/v1 synthetic artifacts keep the service running exactly as before. Every
    load site in this module goes through here so they cannot disagree about
    which artifact is live.
    """
    for candidate in (PIPELINE_V3_PATH, PIPELINE_V2_PATH, PIPELINE_PATH):
        if candidate.exists():
            return candidate
    return PIPELINE_PATH

# Dataset path (used by training_data_service)
DATASET_PATH = Path(__file__).resolve().parents[2] / "synthetic_indian_loan_dataset.csv"

# NOTE (2026-08-30): train_pipeline's data source (synthetic_indian_loan_dataset.csv)
# has been deleted — see docs/superpowers/specs/2026-08-29-home-credit-swap-design.md
# section 3.4. It will not run until the deferred training work (same spec,
# section 2a) rebuilds a Home Credit-based training pipeline. Left in place,
# not fixed, so this history isn't silently lost. Additionally, cbes_engine.DEFAULTS
# changed from the old 15-key dict (conservative worst-case values like
# cibil_score: 300) to the new 7-key dict, but this file's `feature_names` still
# come from the old synthetic-trained model, so any code path in this file that
# does `DEFAULTS.get(col, 0.0)` (see MLPredictor.predict_application below,
# around line 200) and the NaN-fill in this function just below now silently
# imputes 0.0 for keys that no longer exist in DEFAULTS instead of the old
# conservative defaults — not fixed here, same deferred-schema-work reason as
# above.
def train_pipeline(df: pd.DataFrame, t_base_method: str = "cost") -> None:
    """Train the unified pipeline using cross-validation over the top 5 model architectures.
    Performs score targeting, calibration, and joblib caching safely.

    t_base_method selects how the approve/reject threshold is fitted — see
    threshold_selection.select_t_base. "cost" (default) minimises expected
    misclassification cost; "f1_legacy" reproduces the old degenerate
    fixed-range F1 sweep and exists only for comparison/rollback.
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

    # ── T_base discovery on a held-out validation split ────────────────────
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
    # predict_proba[:, 1] = P(y=1) = P(default); the engine score is
    # p_ml = P(approval) = 1 - P(default).
    p_ml_val = 1.0 - _calib_val.predict_proba(X_val_t)[:, 1]

    # WHY NOT THE OLD F1 SWEEP: the previous code took the F1-argmax over a
    # hardcoded np.arange(0.30, 0.70, 0.01). On a calibrated model with a
    # single-digit default rate, p_ml concentrates near 1 - base_rate (~0.92
    # on Home Credit), so <2% of applicants score below 0.70 and F1 for
    # catching defaulters is near zero AND monotonically increasing across
    # the whole swept window. The argmax was pinned at the edge of the range
    # — an artifact of the sweep bounds, not an optimum (v3 artifact: t_base
    # = 0.65 with F1 = 0.0024). See threshold_selection.py and
    # research/thresholds/t_base.py before changing this back.
    sel = select_t_base(y_default=y_val_t, p_ml=p_ml_val, method=t_base_method)
    t_base = float(round(sel["t_base"], 4))
    print(f"[train_pipeline] T_base = {t_base:.4f} "
          f"(method={sel['method']}, criterion={sel['criterion_value']:.4f})")
    if sel["engine_will_clip"]:
        print(f"[train_pipeline] WARNING: t_base={t_base} lies outside "
              f"decision_engine's clip range [0.30, 0.75] and will be clamped.")

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
        "t_base":          t_base,    # threshold_selection.select_t_base — used by decision_engine
        "tau_d":           0.30,      # default; overridden by calibrate_and_save()
    }

    joblib.dump(payload, PIPELINE_PATH)


@functools.lru_cache(maxsize=8)
def _threshold_artifact_hash(artifact_name: str, t_base: float, tau_d: float) -> str:
    """Short fingerprint of the thresholds a decision was made under.

    Written onto every relearning-loop capture row (spec section 3,
    `threshold_artifact_hash`). It covers the pipeline artifact identity, the
    two engine thresholds, and the CBES percentile breakpoints file — i.e.
    everything that can move a decision boundary without the code changing. The
    100MB pipeline itself is deliberately NOT hashed byte-for-byte: that would
    put a multi-second read on the request path for a provenance field.

    Never raises. A missing thresholds file degrades to a distinguishable
    "missing" marker rather than failing a lending decision.
    """
    key = f"{artifact_name}|t_base={t_base:.6f}|tau_d={tau_d:.6f}"
    try:
        cbes_bytes = (ARTIFACTS_DIR / "cbes_thresholds.json").read_bytes()
        key += "|cbes=" + hashlib.sha256(cbes_bytes).hexdigest()[:16]
    except OSError:
        key += "|cbes=missing"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class MLPredictor:
    def __init__(self):
        """Loads and caches artifact purely read-only state. Prevents runtime retraining constraint."""
        # Unchanged guard: the original v1 artifact is the "an artifact exists at
        # all" sentinel, and enforces the no-retraining-at-startup constraint.
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at {PIPELINE_PATH}. Refusing to retrain at startup.")

        # v3 (real data, leak removed) first, falling back to v2/v1 if absent.
        artifact_path = _active_artifact_path()
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
            artifact_path = _active_artifact_path()  # v3-first, v2/v1 fallback
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

        # --- relearning-loop provenance -----------------------------------
        # `hybrid_decision` returns t_approve/t_reject but not the t_base they
        # were derived from, and knows nothing about which artifact produced
        # p_ml. The capture layer needs all three to tell a fixed router's rows
        # apart from a broken router's, so they are attached here at the one
        # place that actually holds them. These are inert annotations: nothing
        # in the decision path reads them back.
        decision_result.t_base = float(t_base)
        decision_result.tau_d = float(tau_d)
        decision_result.engine_version = f"{ENGINE_VERSION}+model={used_model_name}"
        decision_result.threshold_artifact_hash = _threshold_artifact_hash(
            _active_artifact_path().name,  # v3-first, v2/v1 fallback
            float(t_base),
            float(tau_d),
        )

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
        artifact_path = _active_artifact_path()  # v3-first, v2/v1 fallback
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

