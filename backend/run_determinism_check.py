import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.services.ml_service import get_predictor

def check_determinism():
    print("🧪 Commencing Strict Determinism & Latency Checks...")
    
    predictor = get_predictor()
    
    # Dummy mock record identical across evaluation
    dummy_input = {
        "monthly_income": 45000,
        "loan_amount": 250000,
        "loan_tenure": 24,
        "cibil_score": 750,
        "active_loans": 1,
        "credit_utilization": 0.40,
        "missed_payment_ratio": 0.0,
        "bank_balance": 150000,
        "total_assets": 800000
    }
    
    print("\nRunning initial inference to warm cache (predicting SHAP trees)...")
    start = time.perf_counter()
    res1 = predictor.predict_application(dummy_input)
    print(f"Warmup Latency: {(time.perf_counter() - start) * 1000:.2f} ms")

    # Evaluate 5 times structurally to ensure SHAP / Calibration outputs exact stability
    print("\nConducting 5 identical requests...")
    results = []
    latencies = []
    for i in range(5):
        s = time.perf_counter()
        res = predictor.predict_application(dummy_input)
        latencies.append((time.perf_counter() - s) * 1000)
        results.append({
            "p_ml": res.ml_prob,
            "p_cbes": res.cbes_prob,
            "decision": res.decision,
            "confidence": res.confidence,
            "reason": res.decision_reason
        })

    # Display Average Latency
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Inference Latency (with SHAP): {avg_latency:.2f} ms")
    
    if avg_latency > 500:
        print("⚠️ Latency Warning: SHAP evaluation might be causing spikes!")
    else:
        print("✅ Latency Check: Passed (< 500ms bounds).")
        
    # Check absolute determinism
    reference = results[0]
    is_deterministic = True
    for idx, r in enumerate(results[1:]):
        if r != reference:
            is_deterministic = False
            print(f"❌ Determinism Failure at Run {idx + 2}!")
            print(f"Expected: {reference}")
            print(f"Got: {r}")
            break
            
    if is_deterministic:
        print("\n✅ Determinism Check: PERFECT MATCH across all runs.")
        print(f"Stable Output:\n  - ML Prob: {reference['p_ml']:.4f}\n  - CBES Prob: {reference['p_cbes']:.4f}\n  - Decision: {reference['decision']}\n  - Reason: {reference['reason']}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    check_determinism()
