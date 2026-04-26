# SmartLend

**A Proper Research-LYT Project**

SmartLend is an advanced, production-grade hybrid loan decisioning system that integrates machine learning models with a novel, proprietary heuristic scoring system. Designed for rigorous financial research and operational robustness, SmartLend focuses on the critical task of accurately predicting loan defaults while intelligently deferring highly uncertain cases to human underwriters.

---

## 🔬 Research Novelty: The Hybrid ML + CBES Architecture

SmartLend introduces a significant departure from standard "black-box" ML loan approval systems by combining a calibrated Machine Learning model with the **Credit Based Evaluation System (CBES)**.

### The CBES Score Novelty
The CBES score is a novel, deterministic financial heuristic model designed to provide transparent, risk-aware credit evaluation. Unlike raw ML probabilities that can be highly sensitive to training data bias, CBES computes an interpretable "probability of approval" through a rigid mathematical framework.

It calculates an applicant's score by evaluating **5 key pillars** (weighted for optimal signal):
1. **Credit (35%)**: Evaluates CIBIL score, missed payment ratio, and credit utilization.
2. **Capacity (25%)**: Assesses Debt-to-Income (DTI) ratio and EMI-to-Net-Income ratios.
3. **Behaviour (20%)**: Looks at repayment trends over the last 12 months and active loan density.
4. **Liquidity (10%)**: Compares bank balances and total assets against the requested loan amount.
5. **Stability (10%)**: Measures employment tenure relative to age.

**Mathematical Foundation:**
Each component is normalized and passed through a softened component sigmoid curve ($k=4$). The components are then aggregated using their respective weights and passed through an aggregate sigmoid function ($k=5$). This ensures the final output gracefully spans a probability space, heavily penalizing extreme risk while preventing linear overconfidence.

### Disagreement-Driven Abstention (The Hybrid Engine)
Instead of blindly trusting the ML model or the CBES heuristic, the SmartLend system employs a **Two-Stage Blending & Deferral Architecture**:
1. **Isotonic Calibration:** The ML model (optimized via 5-fold CV) is calibrated using `CalibratedClassifierCV` (Isotonic regression) to output true default probabilities.
2. **Signal Divergence:** The system measures the structural divergence between the ML probability ($P_{ML}$) and the CBES probability ($P_{CBES}$).
3. **Abstention / Deferral:** If the ML model and CBES violently disagree (i.e., their difference exceeds a strictly defined threshold $\tau_d$), the system *abstains* and defers the decision to a human underwriter.

This mechanism ensures mathematical safety, drastically reducing false positives on risky applicants by catching anomalies the ML model might miss but the deterministic CBES catches (and vice versa).

---

## 🤖 ML Model Analysis & Selection Pipeline

To ensure the highest quality predictions, the ML pipeline dynamically evaluates a suite of 5 diverse classifier architectures before selection and calibration. 

### Evaluated Architectures:
1. **Logistic Regression (Baseline):** Offers high interpretability and stable baseline performance but lacks the capacity for deep non-linear feature interactions.
2. **Random Forest:** An ensemble method that provides strong robustness against outliers and captures non-linear relationships, though it can sometimes suffer from overconfidence.
3. **XGBoost:** A highly optimized gradient boosting framework known for its execution speed and model performance on complex tabular datasets.
4. **LightGBM:** A fast, distributed gradient boosting framework that uses tree-based learning algorithms, offering excellent efficiency and accuracy.
5. **CatBoost:** A powerful gradient boosting algorithm that excels with categorical data and is highly resistant to overfitting.

### Selection & Calibration Process:
Each model undergoes a rigorous **5-fold Stratified Cross-Validation** on the training set. We optimize not just for raw accuracy, but for a custom, risk-aware composite score:
`Score = AUC + 0.20 * Recall - 0.10 * Std(AUC)`

This custom metric ensures the selected model:
- Discriminates well between classes (High AUC).
- Is highly sensitive to actual default cases (Boosted Recall).
- Maintains stable performance across different data slices (Low Standard Deviation of AUC).

Once the best model is selected, it undergoes **Isotonic Calibration** (via 3-fold CV) to ensure the output probabilities represent true real-world likelihoods, rather than distorted, uncalibrated scores. This is critical for our hybrid engine to accurately measure disagreement against the CBES heuristic.

---

## 📊 System Performance

| Metric | Value |
|--------|-------|
| **Test AUC** | 71.0% |
| **Deferral Rate** | 24.9% |
| **Non-deferred Accuracy** | 60.6% |
| **Automated Tests** | 12/12 passing |

---

## 🚀 Quick Start

Run the application stack using the provided helper scripts:

- **Windows:** `start.bat`
- **Linux/Mac:** `start.sh`

### Services
- **Backend (FastAPI):** http://localhost:8000
- **Frontend (React/Vite):** http://localhost:5173
- **API Documentation:** http://localhost:8000/docs