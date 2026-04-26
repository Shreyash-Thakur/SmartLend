import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Make sure we can import backend packages
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.services.ml_service import train_pipeline, get_predictor
from backend.app.services.cbes_engine import compute_cbes
from backend.app.services.calibrate import calibrate_and_save
from backend.app.services.analysis import (
    plot_roc_curves,
    plot_pml_vs_pcbes_scatter,
    plot_disagreement_histogram,
    plot_deferral_vs_accuracy,
    plot_confidence_distribution
)

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent / "synthetic_indian_loan_dataset.csv"
TARGET_COL = "default_risk"

def run_evaluation_loop():
    print("🚀 Starting End-to-End Evaluation Loop...")
    
    if not DATA_PATH.exists():
        print(f"❌ Could not find dataset at {DATA_PATH}")
        return

    # 1. Load Data
    print("\n📦 Loading Dataset...")
    df = pd.read_csv(DATA_PATH)
    total_samples = len(df)
    print(f"Loaded {total_samples} loan applications.")

    # We will use the entire dataset for training the unified pipeline
    # The pipeline internally handles CV for calibration and score evaluation
    print("\n🧠 Training ML Pipeline...")
    train_pipeline(df)

    # 2. Extract predictions across the entire set to generate proof
    print("\n🔍 Extracting full dataset evaluations for Calibration and Analysis...")
    y_true = df[TARGET_COL].values
    
    # We drop the target column to iterate through the predictor
    X_inference = df.drop(columns=[TARGET_COL])
    
    # Force initialize the predictor now that training is done
    predictor = get_predictor()
    
    p_ml_list = []
    p_cbes_list = []
    confidences = []
    decisions = []
    
    # Since doing SHAP on thousands of records sequentially will take a while,
    # we turn off SHAP explainer temporarily or just run it via predict_application.
    # Actually, we can just extract p_ml directly from calibrator to speed up for this run.
    print("Computing p_ml and p_cbes for all records...")
    
    # Vectorized ML inference
    # Fill defaults exactly like ml_service does
    from backend.app.services.cbes_engine import DEFAULTS
    X_filled = X_inference.copy()
    for col in predictor.feature_names:
        if col in X_filled.columns:
            X_filled[col] = X_filled[col].fillna(DEFAULTS.get(col, 0.0))
    X_filled = X_filled.fillna(0.0)
    
    # Reindex sequentially
    X_final = X_filled[predictor.feature_names]
    p_ml_array = predictor.calibrator.predict_proba(X_final)[:, 1]
    
    # sequential CBES evaluation (since compute_cbes uses dicts)
    p_cbes_array = []
    for idx, row in X_inference.iterrows():
        pc, _ = compute_cbes(row.to_dict())
        p_cbes_array.append(pc)
        
    p_cbes_array = np.array(p_cbes_array)
    
    # 3. Calibration Sweep
    print("\n⚖️ Sweeping for TAU_D Calibration...")
    calib_result = calibrate_and_save(
        p_ml=p_ml_array, 
        p_cbes=p_cbes_array, 
        y_true=y_true, 
        lower_target=0.20, 
        upper_target=0.30
    )
    
    print(f"🎯 Calibrated TAU_D: {calib_result.tau_d}")
    print(f"📉 Deferral Rate: {calib_result.deferral_rate * 100:.2f}%")
    print(f"✅ Non-Deferred Accuracy: {calib_result.non_deferred_accuracy * 100:.2f}%")
    print(f"Calibration successfully persisted to pipeline.joblib!")

    # 4. Finalizing Decisions for plotting
    from backend.app.services.decision_engine import hybrid_decision
    for pm, pc in zip(p_ml_array, p_cbes_array):
        res = hybrid_decision(pm, pc, calib_result.tau_d)
        confidences.append(res.confidence)
        decisions.append(res.decision)
        
    confidences = np.array(confidences)
    decisions = np.array(decisions)

    # 5. Calculate final summary metrics exactly over non-deferred volume
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    non_deferred_mask = decisions != "DEFER"
    
    deferral_rate = calib_result.deferral_rate * 100
    if np.sum(non_deferred_mask) > 0:
        y_true_eval = y_true[non_deferred_mask]
        p_ml_eval = p_ml_array[non_deferred_mask]
        y_pred_eval = (p_ml_eval >= 0.5).astype(int) # using default since thresholds varied dynamically, or evaluate actual decisions
        # Wait, the threshold shifted! We should use decisions array: "APPROVE" = 0 risk (or 1 depending on target)?
        # Let's say TARGET is default_risk (1=default, 0=clean). APPROVE = 0, REJECT = 1.
        y_decision_labels = np.where(decisions[non_deferred_mask] == "REJECT", 1, 0)
        
        acc = accuracy_score(y_true_eval, y_decision_labels) * 100
        prec = precision_score(y_true_eval, y_decision_labels, zero_division=0) * 100
        rec = recall_score(y_true_eval, y_decision_labels, zero_division=0) * 100
        auc_val = roc_auc_score(y_true, p_ml_array) * 100 # Overall AUC
    else:
        acc = prec = rec = auc_val = 0.0

    print("\n" + "="*40)
    print("🏆 SYSTEM PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Deferral Rate           : {deferral_rate:.2f}%")
    print(f"Accuracy (Non-Deferred) : {acc:.2f}%")
    print(f"Precision               : {prec:.2f}%")
    print(f"Recall                  : {rec:.2f}%")
    print(f"Overall AUC             : {auc_val:.2f}%")
    print("="*40 + "\n")

    # 6. Generate Plots
    print("📈 Rendering Analysis Visualizations...")
    
    # We mock model_probs_dict with current and baseline for ROC
    base_probs = p_cbes_array # Using CBES as a baseline plot vs ML
    model_probs_dict = {
        "Ensemble ML": p_ml_array,
        "Pure CBES Engine": base_probs
    }
    
    plot_roc_curves(y_true, model_probs_dict)
    plot_pml_vs_pcbes_scatter(p_ml_array, p_cbes_array, decisions=decisions)
    plot_disagreement_histogram(p_ml_array, p_cbes_array, calib_result.tau_d)
    plot_deferral_vs_accuracy(calib_result.curve)
    plot_confidence_distribution(confidences)
    
    print("\n✨ Evaluation Loop Complete! Your Hybrid System is locked and loaded.")

if __name__ == "__main__":
    run_evaluation_loop()
