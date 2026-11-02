# SmartLend — ML Code Walkthrough

**Updated:** 2 November 2026, from `main` at commit `eba4264`.
**Focus:** the ML. The frontend and deployment are left out, and the backend
gets one short section. Every claim names the file it comes from; the links
open the file.

**Order of this file:**
- **Part 0:** slide 6, explained in short.
- **Part 1:** the architecture diagram, plus the backend in short.
- **Part 2:** the questions she will ask, with answers taken from the code.
  **Read this first.**
- **Part 3:** one application, followed through the code.
- **Part 4:** the full ML flow, step by step:
  data → split → features → training → early stopping → calibration →
  scoring → CBES → decision.
- **Part 5:** the research ML (where the paper numbers come from).
- **Part 6:** every score: what it is, and which file computes it.
- **Part 7:** weak spots, with an honest answer for each.

---

# Part 0 — Slide 6 (convergence): what went wrong

**In one line:** slide 6 contradicts itself. The verdict line says *"not yet
converged — more data / rounds can still help"*, but the footer says
*"boosting iterations have converged"*. The truth is: **the training rounds
have converged; only the amount of data has not.**

1. **Boosting curve (more trees, same data).**
   - The best score came at tree **439** (AUC 0.7607).
   - Training stopped itself at tree **540**, because 100 trees in a row
     brought no gain.
   - So *"rounds can still help"* is **wrong**.
2. **Learning curve (same model, more rows).**
   - 100k → 246k rows still added **+0.010 AUC** (0.7486 → 0.7586).
   - That is about 3× the ±0.0036 noise, so **more data would still help**.
   - "Doubling" on the slide is loose wording: 100k → 246k is about 2.5×.
3. **Its numbers come from a different setup than the leaderboard.**
   - The setup: XGBoost on all numeric columns, with a single 80/20 split.
   - That is why 0.7586 / 0.7607 do not match 0.7651. Both are correct for
     their own setup.
4. **"Uncertainty deferral — Granted"** refers to the SGLB *research* study.
   The live engine still runs the old rule by default (§4.8).

**Code:** [`research/analysis/convergence.py`](../research/analysis/convergence.py)
→ [`reports/convergence.json`](../reports/convergence.json)

**If she asks:** *"The slide merged two curves. Iterations converged; we
early-stop at about 440 trees. The data curve was still rising, so the
headroom was in information. We then confirmed that with features: +0.0155
AUC from FE-v2 and +0.0111 more from FE-v3."*

---

# Part 1 — Architecture + the backend in short

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ [A] FRONTEND  (React, frontend/) – only shows results, no ML         │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 │ HTTP /api/...
 ┌───────────────────────────────▼──────────────────────────────────────┐
 │ [B] API  FastAPI routers  backend/app/routers/applications.py        │
 └───────────────────────────────┬──────────────────────────────────────┘
 ┌───────────────────────────────▼──────────────────────────────────────┐
 │ [C] FEATURE BUILDER  form + bank profile → 15 features               │
 │     backend/app/services/customer_profile_service.py                 │
 └───────────────┬───────────────────────────────────┬──────────────────┘
 ┌───────────────▼─────────────────┐   ┌─────────────▼──────────────────┐
 │ [D] ML MODEL                    │   │ [E] CBES RULE SCORE            │
 │ calibrated Logistic Regression  │   │ 5 weighted pillars, no training│
 │ → p_ml (chance of repaying)     │   │ → p_cbes + pillar breakdown    │
 │ → top-3 SHAP reasons            │   │ services/cbes_engine.py        │
 │ services/ml_service.py          │   └─────────────┬──────────────────┘
 │ artifacts/pipeline_v3_real.joblib                 │
 └───────────────┬─────────────────┘                 │
                 └──────────────┬────────────────────┘
 ┌──────────────────────────────▼───────────────────────────────────────┐
 │ [F] DECISION ENGINE  services/decision_engine.py → hybrid_decision   │
 │   blend = 0.90·p_ml + 0.10·p_cbes → gates → APPROVE / REJECT / DEFER │
 │   deferral mode (SMARTLEND_DEFERRAL_MODE):                           │
 │     1 disagreement  (default, old rule – measured upside down)       │
 │     2 uncertainty   (fix #1: model least sure)                       │
 │     3 expected_loss (fix #2: most money at risk)          ← NEW      │
 └───────────┬──────────────────────────────────────┬───────────────────┘
      APPROVE / REJECT                    DEFER (+3% random exploration)
 ┌───────────▼──────────────────────────────────────▼───────────────────┐
 │ [G] EXPLANATIONS  SHAP → sentences   services/explainability_service │
 │ [H] DATABASE + HUMAN REVIEW QUEUE    services/deferred_review_service│
 │ [I] RELEARNING GATE  0/4 pass → never retrain                        │
 │     research/relearning/gate.py                                      │
 └──────────────────────────────────────────────────────────────────────┘
 ═══════════════════════════ offline, separate ══════════════════════════
 ┌──────────────────────────────────────────────────────────────────────┐
 │ [J] RESEARCH LAB  research/ → reports/*.json                         │
 │   trains XGBoost / LightGBM / CatBoost / RF / LR / TabPFN …          │
 │   runs every deferral study; makes every number on the slides        │
 └──────────────────────────────────────────────────────────────────────┘
```

**The backend in short.**
- **What it is:** `backend/` is a FastAPI server.
- **`routers/`** holds the URLs. The ML endpoint is
  [`applications.py`](../backend/app/routers/applications.py).
- **`services/`** does the work. The ML lives in four files:
  - [`customer_profile_service.py`](../backend/app/services/customer_profile_service.py)
    builds the features **[C]**;
  - [`ml_service.py`](../backend/app/services/ml_service.py) runs the model
    and SHAP **[D]**;
  - [`cbes_engine.py`](../backend/app/services/cbes_engine.py) computes the
    rule score **[E]**;
  - [`decision_engine.py`](../backend/app/services/decision_engine.py) makes
    the decision **[F]**.
- **`artifacts/`** holds the trained model file, the CBES cutoffs and the
  307k-row prediction table.
- **It never trains.** It only loads the saved model.
- **The other services** (database, AI briefing, voice, cache) are plumbing.
  They are not ML.

---

# Part 2 — The questions she will ask (answers taken from the code)

## Q1. "How many batches?"

**Short answer: training uses no mini-batches. Every model trains on its
full training set at once. Batches only appear when a foundation model
*predicts*, because of GPU memory.**

| Model | Batches? | Where in the code |
|---|---|---|
| **Logistic Regression** (live model) | none. The solver (sklearn's default, lbfgs) uses all training rows at each step, up to `max_iter=1000` steps | [`backend/retrain_serving_model_v3.py`](../backend/retrain_serving_model_v3.py) line ~171 |
| **XGBoost / LightGBM / CatBoost** | none. Each new tree is built on all the training rows. They train in **rounds** (trees), not batches: up to 2,000 rounds | [`research/analysis/features_vs_fusion.py`](../research/analysis/features_vs_fusion.py) line 71, `GBM_PARAMS` |
| **TabPFN-2.5** | **16 batches of 500 rows**, for *prediction* only (8,000 rows scored on an 8 GB laptop GPU). It does no training at all; it reads 5,000 example rows as "context" | [`reports/tabpfn_quick_run.json`](../reports/tabpfn_quick_run.json): `"batch": 500` |
| **TabICLv2** (challenger) | `batch_size=2`, plus disk offload, just to fit on the GPU. It still ran out of disk | [`research/analysis/challenger_screen.py`](../research/analysis/challenger_screen.py) line 83 |
| Remote TabPFN check | 200 rows per request (default) | [`backend/run_tabpfn_connection_check.py`](../backend/run_tabpfn_connection_check.py) |

> **Say it like this:** *"Tree boosting doesn't use mini-batches. It
> trains in rounds, with each tree fit on the whole training fold. Our only
> batching is at inference for TabPFN: 16 batches of 500 rows, because of
> 8 GB of GPU memory."*

## Q2. "What was the data split?"

**Short answer: 307,511 rows, 8.07% defaulters. Every split is
*stratified* (keeps 8.07% in each part) and uses a fixed seed, so it can be
repeated exactly.**

| Used for | Split | Seed | Code |
|---|---|---|---|
| **Live model (LR)** | **80% train / 20% test**: 246,008 / 61,503 rows | `random_state=42`, stratified | [`retrain_serving_model_v3.py`](../backend/retrain_serving_model_v3.py) line 160 |
| ↳ calibration inside the 80% | 5-fold (`CalibratedClassifierCV(cv=5)`) | — | same file, line 175 |
| ↳ choosing the approve threshold | the 20% test part is cut **50/50**: pick the threshold on one half, report on the other | 42 | same file, line 210 |
| **All research models** (leaderboard, FE-v2, FE-v3) | **5-fold stratified cross-validation**. Each row is predicted by a model that never saw it ("out-of-fold") | `StratifiedKFold(5, shuffle, 42)` | [`features_vs_fusion.py`](../research/analysis/features_vs_fusion.py) line 170 |
| ↳ early-stopping check inside each fold | **10% of the training folds** is held back to decide when to stop. The test fold is never used for this | `seed + fold` | same file, line 108 |
| **Deferral studies** (thresholds for routers) | **50% tune / 50% test** of the 307k table: the threshold is learned on tune and measured on test (153,756 rows) | **20260831** | [`research/deferral/evaluate.py`](../research/deferral/evaluate.py) `split_tune_test` |
| **Final, strictest audit** | **50% tune / 25% selection / 25% final**. The final slice is touched once | 20260831 | [`research/deferral/audit_final_test.py`](../research/deferral/audit_final_test.py) |
| **Convergence study** (slide 6) | one fixed **80/20** holdout. Smaller training sizes drawn 3 times each, plus a 10% early-stopping slice | 42 | [`convergence.py`](../research/analysis/convergence.py) lines 153, 275 |
| **TabPFN / TabFM** | 5,000 context rows + **8,000 separate rows scored** | seeded, stratified | [`score_tabfm.py`](../research/analysis/score_tabfm.py) line 131 |
| Challenger screen | a 30,000-row sample, 5-fold | 42 | [`challenger_screen.py`](../research/analysis/challenger_screen.py) |

> **Say it like this:** *"Live model: stratified 80/20, seed 42.
> Research: stratified 5-fold out-of-fold, with early stopping on an inner
> 10% of the training folds. Any threshold is chosen on a tune half and
> measured on a test half."*

## Q3. "Early stopping: after how many rounds?"

**Short answer: up to 2,000 trees at learning rate 0.05. Stop if the
validation AUC has not improved for 100 rounds in a row. In practice the best
tree was 439, and training stopped at 540.**

- **The setting:**
  `GBM_PARAMS = {"n_estimators": 2000, "learning_rate": 0.05, "early_stopping": 100}`
  in [`features_vs_fusion.py`](../research/analysis/features_vs_fusion.py)
  line 71. It is reused unchanged by `features_v3_side_tables.py`,
  `challenger_screen.py` and `cost_sensitive_catboost.py`.
- **The same rule, written for each library:**
  - LightGBM: `lgb.early_stopping(100)` (line 115);
  - XGBoost: `early_stopping_rounds=100` (line 122);
  - CatBoost: `od_type="Iter", od_wait=100` (line 129).
  - All three watch **AUC** on the 10% early-stopping slice.
- **The measured result** (`boosting_convergence` in
  [`reports/convergence.json`](../reports/convergence.json)):
  - best round **439**, AUC **0.7607**;
  - stopped at round **540** (439 + about 100 patience);
  - AUC at round 50 was 0.7453.
- **Two exceptions, so you are not caught out:**
  - In the *learning-curve* half of `convergence.py`, the per-size models use
    lr 0.1 with patience 50 (line 155). They are re-fit at each data size, so a
    faster setting is enough.
  - The **SGLB** uncertainty model trains a fixed **1,000** rounds with **no
    early stopping** (`SGLB_ITERATIONS = 1000` in
    [`uncertainty_deferral_catboost.py`](../research/analysis/uncertainty_deferral_catboost.py)
    line 65). Its "virtual ensemble" needs a fixed training path.

## Q4. "Why did you stop there?"

**Short answer: after tree 439, new trees stopped improving the score on
data the model had not trained on. More trees would only memorise noise in the
training rows (overfitting). Early stopping finds that point automatically.**

- **How the code decides:**
  1. After each tree, it scores the 10% slice the model does *not* train on.
  2. If 100 trees in a row bring no new best, it stops and keeps the best
     tree count.
- **Why 100 patience?** AUC moves up and down a little from tree to tree.
  Waiting 100 rounds makes sure a small dip is not mistaken for the end.
- **Why not pick the count by hand?**
  - It would be a guess.
  - If you chose it by looking at the test fold, the test score would be
    biased.
  - So the slice comes **only from the training folds**.
- **Evidence it really converged:**
  - the curve in `boosting_auc_curve_every_10` flattens near 0.760 from about
    tree 400;
  - the 14-configuration tuning study also found only +0.001, which is inside
    the noise ([`reports/tuning_cpu.json`](../reports/tuning_cpu.json)).
- **Model rounds converged; data did not.** The learning curve was still
  rising. So the next gains came from **features** (FE-v2 and FE-v3), not
  from more trees.

## Q5. "Why do the other models fail?"

**Short answer: "fail" here means "lose to the boosted trees". Each loss
has a measured reason.**

| Model | Score | Why it loses | Source |
|---|---|---|---|
| **XGBoost / CatBoost / LightGBM** | 0.7651 / 0.7643 / 0.7631 | They **don't fail**: all three are within the ±0.0036 noise of each other. Boosted trees handle missing values, odd distributions and feature interactions natively | leaderboard, [`docs/STATUS.md`](STATUS.md) |
| **Logistic Regression** (research) | 0.7378 | It draws one straight line through the features, so it **cannot learn interactions** (for example, low income is risky *only if* the loan is big). Its learning curve is flat (+0.0007): more data does not help a linear model | [`convergence.json`](../reports/convergence.json) |
| **Logistic Regression** (live) | 0.6919 | It is the same kind of model, and also limited to **15 form fields** instead of the full bureau history | [`serving_model_retrain.json`](../reports/serving_model_retrain.json) |
| **Random Forest** | 0.7376 | It builds trees independently and averages them. Boosting builds each tree to **fix the previous trees' errors**, which suits a rare (8%) class better | leaderboard |
| **TabPFN-2.5** | 0.7284 ± 0.0127 | It does no training: it only "reads" **5,000 example rows**, because an 8 GB GPU can't hold more. The others learn from about 246k rows. It also lost in all 16 segments | [`tabpfn_quick_run.json`](../reports/tabpfn_quick_run.json), [`complementarity.json`](../reports/complementarity.json) |
| **Google TabFM** | 0.6911 | Same limit (5,000 context rows), on just the 15 live features | [`tabfm_quick_run.json`](../reports/tabfm_quick_run.json) |
| **RealMLP** (neural net) | 0.736 vs CatBoost 0.748 on 30k rows | It lost on **every one of the 5 folds** (−0.0121 on average). Neural nets need more data and tuning to match trees on tabular data | [`challenger_screen.json`](../reports/challenger_screen.json) |
| **TabICLv2** | no score | It crashed with "no space left on device". Even at batch size 2 it needed about 18 GB of disk offload per fold | same file |
| **CBES** (rule score) | 0.565 | It uses 8 fields and fixed percentile rules, with no learning. The delinquency cutoffs are all 0, so that input is just a yes/no switch | [`cbes_thresholds.json`](../backend/artifacts/cbes_thresholds.json) |
| **Model fusion** (averaging, stacking, TabPFN + XGBoost) | at best +0.0016 | **Below the noise floor.** The models make the **same mistakes** (correlation 0.95–0.96), so combining them adds nothing | [`features_vs_fusion.json`](../reports/features_vs_fusion.json) |
| **Cost-weighted CatBoost** (defaulters ×7.5) | no gain | Weighting the training never beats plain training plus a cost-based threshold (Elkan 2001) | [`cost_sensitive_catboost.json`](../reports/cost_sensitive_catboost.json) |

> **The pattern to say:** *"Every model on the same features lands
> around 0.76, and they are 95% correlated. So the limit was the
> information in the features, not the model. Better features moved
> CatBoost from 0.7603 to 0.7757 (FE-v2), then to 0.7868 (FE-v3)."*

**If she means "why does the deferral rule fail":** the old disagreement rule
compares two scores that sit **0.31 apart on average**. So it measures that
gap, not real disagreement. That gap is widest when the model is most
confident, which means it sends humans the *easiest* cases (§5.2).

---

# Part 3 — One loan application, step by step

**Open:** [`backend/app/routers/applications.py`](../backend/app/routers/applications.py),
function `_create_application_record`.

```
form ── POST /api/applications ──►
[B] _create_application_record()
     1. _validate_payload(form)
          └─[C] resolve_application_payload()      form + bank profile → 15 features
     2. predictor = get_predictor()                model loaded ONCE, kept in memory
     3. predictor.predict_application(payload)
          ├─[D] p_ml = 1 − P(default)              calibrated Logistic Regression
          ├─[E] p_cbes, pillars = compute_cbes()
          ├─[G] top-3 SHAP reasons
          └─[F] hybrid_decision(...) → APPROVE / REJECT / DEFER
     4. stamp: engine version, model file, threshold fingerprint
     5. not deferred? 3% random coin flip may still send it to a human
     6. save to loan_applications                   [H]
     7. DEFER or coin flip → deferred_reviews       [H]
◄── reply: decision, confidence, SHAP reasons, CBES pillars
```

The analyst's later verdict is **saved, never trained on** (§4.10).

---

# Part 4 — The full ML flow, step by step

## 4.1 The data and the label direction

**Block [J] / [D].**

- **The dataset:** Home Credit Default Risk, **307,511 real loan
  applications**, joined with `bureau.csv` (their loans at other lenders).
  The raw files are in `data/raw/home_credit/` (not in git).
- **The label:** `TARGET` has **1 = defaulted**, 0 = repaid. **8.07%**
  defaulted.
- **Why we judge by AUC, not accuracy:** "approve everyone" already scores
  92% accuracy while catching zero defaulters.
- **Direction trap.** The live engine uses
  `p_ml = chance of repaying = 1 − P(default)`.
  - In [`prediction_outputs.csv`](../backend/artifacts/prediction_outputs.csv),
    `y_true == 1` means **GOOD**.
  - In training labels, 1 = **DEFAULT**.
  - SHAP is on the P(default) scale.

## 4.2 Features

**Block [C] for the live model, [J] for research.**

**Live model: 15 features**, built by `_build_profile` in
[`customer_profile_service.py`](../backend/app/services/customer_profile_service.py):

| Feature | From Home Credit | How |
|---|---|---|
| `age` | `DAYS_BIRTH` | `−days / 365.25` (Home Credit stores days as negative numbers) |
| `years_employed` | `DAYS_EMPLOYED` | the same; the special value **365243** ("not employed") becomes 0 |
| `existing_emis` | `AMT_ANNUITY` | the yearly payment |
| `debt_to_income_ratio` | annuity / income | there is no DTI column, so this stands in |
| `cibil_score` | `EXT_SOURCE_2` | rescaled to 300–900 **for display only**; not a real CIBIL score |
| `credit_utilization_ratio` | bureau debt / credit | clipped to 0–1 |
| `active_loans`, `closed_loans`, `total_loans`, `missed_payments` | `bureau.csv` | counted per person |
| `annual_income`, `monthly_income`, `dependents`, `loan_amount`, `loan_income_ratio` | application columns | |

**Why this matters:** the **same function** builds features for training and
for live scoring, so there is no train/serve mismatch.

**Research FE-v2: 319 features**, in [`research/features/engineer.py`](../research/features/engineer.py):
- **`application_features()`:**
  - ratios: annuity/income, credit/income, credit/annuity (about the loan
    length), years employed/age;
  - **EXT_SOURCE combinations** (mean, min, max, product). These are the
    strongest signal.
  - **a count of missing values per row.**
- **`bureau_features()`:** per-person counts of active, closed and recent
  loans, total debt, overdue amounts, and the same for active loans only.
- **`build_feature_matrix()`:**
  - people with no bureau record stay *missing* (with a flag), never
    zero-filled;
  - `TARGET` is never used to build a feature, so there is no leakage.

**Research FE-v3: +26 features** from four more tables (previous
applications, instalments, POS/cash, credit cards), in
[`research/analysis/features_v3_side_tables.py`](../research/analysis/features_v3_side_tables.py).

## 4.3 Training the live model

**Block [D].**
**Open:** [`backend/retrain_serving_model_v3.py`](../backend/retrain_serving_model_v3.py),
function `main()`.

1. **`build_training_frame`:** builds the 15 features for every row and fills
   gaps with `cbes_engine.DEFAULTS` (or 0).
2. **Split:** `train_test_split(test_size=0.2, random_state=42, stratify=y)`
   gives **246,008 train / 61,503 test** rows.
3. **Model:** `StandardScaler` → `LogisticRegression(max_iter=1000)`. The
   scaler puts all features on one scale so the solver converges.
4. **Calibrate:** `CalibratedClassifierCV(method="isotonic", cv=5)`.
   - **In plain words:** it makes the score *mean* what it says. Of people
     scored 0.90, about 90% should repay.
   - **How:** 5 rounds inside the training part. Each round trains on 4/5 and
     learns a step-shaped correction on the remaining 1/5.
   - **Why:** the engine compares `p_ml` with fixed thresholds, so it must be
     a real probability.
5. **Test on the 20% it never saw:** **ROC-AUC 0.6919**, PR-AUC 0.1706,
   Brier 0.0711 ([`serving_model_retrain.json`](../reports/serving_model_retrain.json)).
6. **Threshold:** the 20% test part is split 50/50. `select_t_base` chooses
   the threshold on one half, and its metrics are reported on the other half.
7. **Save** to `backend/artifacts/pipeline_v3_real.joblib`:
   - the model and calibrator;
   - the feature names;
   - 100 rows for SHAP;
   - `t_base` and `tau_d`.

**What it removed:**
- **`loan_approved`, `confidence_score`:** the old model's own outputs had
  leaked in as inputs (**target leakage**). Removing them lowered the AUC
  honestly.
- **Loan term, interest rate, assets:** they have no Home Credit match.

**Why LR is the live model and CatBoost (0.7868) is not:**
- The form collects about 15 fields. CatBoost needs 345 features, mostly from
  the full bureau history.
- LR is calibrated, fast, and SHAP is exact for it.

## 4.4 Training the research models

**Block [J].**
**Open:** [`research/analysis/features_vs_fusion.py`](../research/analysis/features_vs_fusion.py),
functions `_fit_predict_fold` and `oof_run`.

For each of the 5 folds:
1. **Hold out** 1 fold as the test fold.
2. **Cut 10%** from the other 4 folds as the early-stopping slice (line 108).
3. **Train on the rest:**

   | Model | Settings |
   |---|---|
   | LightGBM | `num_leaves=63` |
   | XGBoost | `max_depth=6`, `tree_method="hist"` |
   | CatBoost | default depth |
   | Logistic Regression | median impute → scaler → LR |

   - The three boosted models use up to 2,000 rounds, lr 0.05 and 100-round
     patience on AUC.
4. **Predict** the test fold and save those predictions (the "out-of-fold"
   predictions).

After all 5 folds:
- Compute the AUC on each fold and report the mean ± sd.
- To compare v1 with v2, subtract **fold by fold** on the same folds (the
  "fold-paired delta").
- **The rule, written before running:** a gain counts only if it is
  **> 0.0036** (the fold-to-fold noise) **and** every fold improves.

## 4.5 Scoring a live applicant

**Block [D] + SHAP [G].**
**Open:** [`backend/app/services/ml_service.py`](../backend/app/services/ml_service.py).

- **`_active_artifact_path()`** picks the model file: v3, else v2, else v1.
- **`MLPredictor.__init__()`:**
  - **refuses to start if there is no model file**; it never trains;
  - builds `shap.LinearExplainer` (exact for LR).
  - If SHAP fails, the reasons are shown labelled **"heuristic"**, never
    passed off as SHAP.
- **`get_predictor()`** loads the model once and keeps it in memory.
- **`predict_application()`:**
  1. Fill gaps with the same defaults as training.
  2. `p_ml = 1 − calibrator.predict_proba(df)[0, 1]`.
  3. `compute_cbes()` gives `p_cbes` and the pillars (§4.7).
  4. **SHAP top 3:** how much each feature pushed this person up or down
     compared with an average applicant. Keep the 3 biggest.
     - **Caveat:** SHAP explains the plain LR, but the served score goes
       through the calibrator. So the *order* of reasons is right, but the
       values don't add up exactly to the served score.
  5. Call `hybrid_decision()` (§4.8).
  6. Stamp the engine version, the model file and the threshold fingerprint.

The sentences come from `_build_top_factors()` in
[`explainability_service.py`](../backend/app/services/explainability_service.py).

## 4.6 The approve threshold

**Block [F].**
**Open:** [`backend/app/services/threshold_selection.py`](../backend/app/services/threshold_selection.py),
function `select_t_base`.

The search tries 199 **percentiles of the real scores**.

| Method | Idea |
|---|---|
| `cost` (default) | lowest cost of mistakes: 0.45 per bad approval (Basel II LGD) + 0.2644 per good rejection (Verbraken 2014). Theory gives ≈ 0.63 |
| `youden` | catch the most defaulters for the fewest false alarms. The research blend uses this (t* = 0.885) |
| `f1`, `approval_rate` | for comparison |
| `f1_legacy` | the old 0.30–0.70 search, **kept only to show the bug**: most people score about 0.92, so that range could reject almost nobody |

## 4.7 CBES, the rule score

**Block [E].**
**Open:** [`cbes_engine.py`](../backend/app/services/cbes_engine.py)
(`compute_cbes`), [`cbes_calibration.py`](../backend/app/services/cbes_calibration.py),
and [`cbes_thresholds.json`](../backend/artifacts/cbes_thresholds.json).

1. **Cutoffs:** for each input, store the 10th, 30th, 50th, 70th and 90th
   percentile of the real data. These map to 0, 0.25, 0.5, 0.75 and 1, with a
   straight line in between. If lower is better, the score is flipped.
2. **Five pillars:**

   | Pillar | Weight | What goes in |
   |---|---|---|
   | Credit | 0.35 | 70% credit score + 30% no missed payments |
   | Capacity | 0.30 | 60% low DTI + 40% low loan-to-income |
   | Behaviour | 0.20 | fewer active loans |
   | Stability | 0.10 | years employed |
   | Region | 0.05 | region rating |

   - Each pillar is softened by `1/(1+e^(−4(x−0.5)))`, which keeps it within
     0.12–0.88.
   - ⚠ The old slide says "Liquidity". The code has Stability and Region.
3. **Combine:**
   `p_cbes = sigmoid(5·(0.35·credit + 0.30·capacity + 0.20·behaviour + 0.10·stability + 0.05·region − 0.5))`.
4. **Missing input:** gets the *worst* realistic value.

**Worked example:**
- **The applicant:** EXT_SOURCE_2 0.60, 0 missed payments, 1 active loan,
  DTI 0.15, 5 years employed, loan-to-income 2.22, region 2.
- **Result:** raw 0.669 → **p_cbes = 0.699**.

**Its value:**
- **AUC 0.565 alone**, against about 0.765 for the ML model.
- Its average is 0.61, while the ML model's is 0.92. **That 0.31 gap caused
  the deferral bug.**
- It is kept as an explanation layer and as a fallback for the 14.3% of
  applicants with no bureau record.

## 4.8 The decision engine

**Block [F].**
**Open:** [`backend/app/services/decision_engine.py`](../backend/app/services/decision_engine.py),
function `hybrid_decision`.

**Stage 0, safety limits:**
- probabilities within 0–1;
- `tau_d` within 0.10–0.90;
- `t_base` within 0.30–0.98.

The upper limit used to be 0.75, which made the system approve 99.5%.

**Stage A, blend:**
- `p_blend = 0.90·p_ml + 0.10·p_cbes` (`_BLEND_ALPHA = 0.10`).
- **The measured AUC at each weight** (comment above `_BLEND_ALPHA`):
  - α=0: 0.7669
  - α=0.10: 0.7613
  - α=0.25: 0.7393
- More CBES weight means worse ranking.
- ⚠ The docstring at lines 10–13 still says 0.75/0.25.

**Stage B, the values the gates use:**
```
D          = |p_ml − p_cbes|
confidence = 0.6·|p_ml−0.5|·2 + 0.2·|p_cbes−0.5|·2 + 0.2·(1 − D)
t_approve  = t_base − 0.05·clip(p_cbes − 0.5, −0.15, 0.15)     moves at most ±0.0075
```

**Stage C, the gates.** One of three modes runs, chosen by
`SMARTLEND_DEFERRAL_MODE`:

| Mode | When it defers | Otherwise |
|---|---|---|
| **1 `disagreement`** (**default**, old rule) | `D > tau_d`, or confidence < 0.18, or the grey zone | approve if `p_blend ≥ t_approve`; reject if `p_ml ≤ t_reject`; then CBES ≥ 0.60 / ≤ 0.40 settles the grey zone |
| **2 `uncertainty`** (fix #1, Chow's rule) | `|p_ml − t_approve| < 0.2458`: the model is near its own line | approve if `p_ml ≥ t_approve`, else reject |
| **3 `expected_loss`** (fix #2, NEW) | the cost of being wrong > **0.198976**: approving costs `0.45·(1−p_ml)`, rejecting costs `0.2644·p_ml` | same hard approve/reject |

- `_resolve_deferral_mode()` reads an explicit argument first, then the
  environment variable, and otherwise uses **mode 1**.
- So **both fixes are off unless switched on**. Nothing in `terraform/` or
  `docker-compose.yml` sets the variable.

**Worked example** (p_ml 0.93, p_cbes 0.699, t_base 0.65, tau_d 0.30):

| Value | Result | Gate check |
|---|---|---|
| p_blend | **0.907** | |
| D | 0.231 | not > 0.30 |
| confidence | 0.750 | |
| t_approve | 0.6425 | |

| Mode | Check | Decision |
|---|---|---|
| 1 | 0.907 ≥ 0.6425 | **APPROVE** |
| 2 | distance 0.2875 > 0.2458 | **APPROVE** |
| 3 | loss 0.45 × 0.07 = 0.0315 < 0.199 | **APPROVE** |

## 4.9 The 307k prediction table

**Block [J], used by [I].**
**Open:** [`backend/artifacts/prediction_outputs.csv`](../backend/artifacts/prediction_outputs.csv).

- **What it is:** one row per applicant (307,511).
- **What is in it:** out-of-fold scores from LR, LightGBM, XGBoost, CatBoost,
  Random Forest and CBES.
- `best_model_prob` is XGBoost v1 (**0.7651**).
- **Why it matters:** every deferral and gate study reads this file, so none
  of them needs retraining.
- **Its decision columns** were rewritten on the 90/10 blend by
  [`research/blend/regenerate.py`](../research/blend/regenerate.py):
  - Youden line t* = 0.8853, chosen on the tune half;
  - **22.49% deferred**, so **77.51% of cases are automatic**;
  - **74.13% accuracy** on the automatic cases
    ([`blend_decision.json`](../reports/blend_decision.json)).
- ⚠ The script that produced the original OOF scores is not committed. If
  asked: *"the 31 Aug OOF run; the file and its commit are the record."*

## 4.10 Why we never retrain

**Blocks [H] and [I].**
**Open:**
- [`research/relearning/gate.py`](../research/relearning/gate.py);
- [`relearning_service.py`](../backend/app/services/relearning_service.py).

**What is recorded:**
- Every DEFER, plus a random **3%** of automatic cases (`_should_explore`),
  goes to a human.
- The human verdict is **saved**.
- The random 3% is the only *unbiased* sample, because deferred cases were
  picked by the router.

**Why retraining stays off:** the gate allows it only if all 4 conditions pass.
1. The router picks harder cases than **random piles of the same size**
   (z ≤ −2).
2. The deferral rate is inside the band the AUC allows (**8–16%**, from the
   overlap of the good and bad score curves).
3. There are ≥ 1,000 random-exploration labels with real outcomes. There
   are 0, because outcomes take 12–24 months.
4. There is a written retraining design.

**Today: 0/4 → DO NOT OPEN THE LOOP**
([`relearning_gate.json`](../reports/relearning_gate.json)). The service file
has no training code, and a test checks that.

---

# Part 5 — The research ML (where the paper numbers come from)

## 5.1 The six candidate deferral signals

**Open:** [`research/deferral/signals.py`](../research/deferral/signals.py),
[`research/deferral/evaluate.py`](../research/deferral/evaluate.py).

| Class | Signal | Idea |
|---|---|---|
| `CurrentDisagreement` | \|p_ml − p_cbes\| | the old rule |
| `RankDisagreement` | \|rank − rank\| | ranks remove the 0.31 gap |
| `ZScoreDisagreement` | \|z − z\| | removes the gap and the scale difference |
| `IsotonicDisagreement` | \|iso − iso\| | puts both onto real outcomes first |
| **`MLUncertainty`** | −\|p_ml − t\| | ignore CBES; defer what is nearest the model's line |
| `MLUncertaintyHalf` | −\|p_ml − 0.5\| | the same, with the line fixed at 0.5 |

**The fair test:**
1. 50/50 tune/test split, seed 20260831.
2. Every signal defers **the same 22.5%** (the middle of the 20–25% capacity
   band), so the only question is *which* cases it picks.
3. Each is scored by the unchanged gate and by risk-coverage:
   - **selective risk** = the error rate on the automatic cases;
   - **position**: 0 = random, 1 = perfect.

## 5.2 What was wrong with the old rule

- p_ml is **0.31 higher** than p_cbes on average, and higher on **98.4%** of
  rows.
- So `D` measured that fixed gap, and the gap is widest when the model is
  most confident.
- So the rule sent humans the **easiest** cases:
  - z **+38 / +43** (the wrong direction);
  - 52% of cases deferred.
- **Source:** [`deferral_fix.json`](../reports/deferral_fix.json),
  [`docs/DEFERRAL-FIX.md`](DEFERRAL-FIX.md).

## 5.3 The audits: 10,000 shuffles + a final untouched slice

**Open:** [`audit_10k.py`](../research/deferral/audit_10k.py),
[`audit_final_test.py`](../research/deferral/audit_final_test.py).

**10,000-shuffle audit** on 153,756 test rows, about 22.5% deferred
([`deferral_audit_10k.json`](../reports/deferral_audit_10k.json)):

| Router | balance z | accuracy z | verdict |
|---|---|---|---|
| deployed blend band | +1.3 | −0.9 | FAIL (about random) |
| old disagreement | +16.8 | +36.0 | FAIL (upside down) |
| rank / z-score / isotonic | −12 / −43 / −61 | −35 / −59 / −99 | PASS |
| **ML uncertainty (wired in the engine)** | −6.2 | **−59.7** | PASS |
| ML uncertainty, line at 0.5 | −96.0 | −263.1 | PASS |

**Final untouched slice** (50/25/25 split; winner picked on the 25%
selection slice by a rule fixed beforehand):
- winner `ml_uncertainty_0.5`: **accuracy z −186.3, PASS**;
- the wired version: −42.7, PASS;
- the deployed band: FAIL again.

## 5.4 Cost-aware routing (mode 3)

**Open:**
- [`cost_aware_routing.py`](../research/deferral/cost_aware_routing.py);
- [`calibrate_expected_loss.py`](../research/deferral/calibrate_expected_loss.py),
  which works out `DEFAULT_THETA_EL`.

| At the same 22.5% budget | no deferral | uncertainty | **expected loss** |
|---|---|---|---|
| cost per automatic decision | 0.0731 | 0.0625 | **0.0291** |
| share of defaulters reaching a human | 0% | 24.7% | **42.0%** |

([`expected_loss_calibration.json`](../reports/expected_loss_calibration.json))

**What it means:**
- **uncertainty** catches the model's *mistakes*;
- **expected loss** protects the *money*.

Choosing between them is a business decision, so the engine offers both.

## 5.5 Features beat fusion (FE-v2, FE-v3)

| Model | v1 | v2 (319) | v3 (+26) |
|---|---|---|---|
| CatBoost | 0.7603 | 0.7757 | **0.7868** |
| XGBoost | 0.7584 | 0.7749 | 0.7853 |
| LightGBM | 0.7560 | 0.7721 | — |
| LR | 0.7377 | 0.7560 | — |

- Every fold improved in both steps.
- The best model combination added +0.0016, **below the noise**.
- **Sources:**
  - [`features_vs_fusion.json`](../reports/features_vs_fusion.json)
  - [`features_v3_side_tables.json`](../reports/features_v3_side_tables.json)

## 5.6 The other studies, one line each (`research/analysis/`)

- **`uncertainty_deferral_catboost.py`:**
  - **What it is:** SGLB, a CatBoost setting with 10 "virtual" models inside
    one model.
  - **How it works:** their spread measures what the model doesn't *know*.
  - **Result:** position **0.40**, against 0.14 for Chow's rule. This is the
    "Granted" on slide 6.
- **`monotone_catboost.py`:**
  - 21 features forced to push the sensible way;
  - −0.0008 AUC (noise), 0 violations in 14,400 checks.
- **`cost_sensitive_catboost.py`:** 7.5× weights give no gain, so costs
  belong in the threshold.
- **`taiwan_replication.py`:**
  - the same audit code on 30,000 Taiwan credit-card rows (AUC 0.781);
  - every router beat random again.
- **`shift_stress_tests.py`:** when the input data is damaged, the deferral
  rate jumps from 22.4% to 56.6%, so it works as a drift alarm.
- **`fairness_subgroups.py`, `fairness_routing_comparison.py`:** how each
  router treats subgroups.
- **`complementarity.py`, `score_tabfm.py`:** TabPFN and TabFM against
  XGBoost; no combination helps.
- **`challenger_screen.py`:** RealMLP loses every fold; TabICLv2 ran out of
  disk.

---

# Part 6 — Every score: what it is, and where it comes from

| Number | What it is | Model + data | Code | Report |
|---|---|---|---|---|
| **0.6919** | **live model** AUC | calibrated LR, 15 features, 61,503 held-out rows | [`retrain_serving_model_v3.py`](../backend/retrain_serving_model_v3.py) | `serving_model_retrain.json` |
| 0.7651 ± 0.0036 | XGBoost in the 307k table | XGBoost v1, 5-fold OOF | 31 Aug OOF run (not committed) | `prediction_outputs.csv` |
| 0.7378 / 0.7376 | research LR / Random Forest | OOF, 307k | same run | leaderboard |
| 0.7757 | best at report freeze | CatBoost FE-v2, 5-fold OOF | [`features_vs_fusion.py`](../research/analysis/features_vs_fusion.py) | `features_vs_fusion.json` |
| **0.7868** | **best now** | CatBoost FE-v3 | [`features_v3_side_tables.py`](../research/analysis/features_v3_side_tables.py) | `features_v3_side_tables.json` |
| 0.5650 | CBES alone | rule score, 307k | [`cbes_engine.py`](../backend/app/services/cbes_engine.py) | leaderboard |
| 0.7284 / 0.6911 | TabPFN / TabFM | 5,000 context rows, 8,000 scored | `complementarity.py`, `score_tabfm.py` | `tabpfn_quick_run.json`, `tabfm_quick_run.json` |
| 77.51% / 74.13% | automatic share / accuracy on those | 90/10 blend, Youden line | [`regenerate.py`](../research/blend/regenerate.py) | `blend_decision.json` |
| 0.7586 / 0.7607 | slide-6 numbers | XGBoost, all numeric columns, 80/20 | [`convergence.py`](../research/analysis/convergence.py) | `convergence.json` |
| 439 / 540 | best tree / stopping tree | same | same | same |
| −59.7 / −186.3 | audit z (wired uncertainty / final winner) | 10,000 shuffles | `audit_10k.py`, `audit_final_test.py` | `deferral_audit_*.json` |
| 0.0731 → 0.0291 | cost per automatic decision, with mode 3 | test half | `calibrate_expected_loss.py` | `expected_loss_calibration.json` |

**AUC in one line:** pick one defaulter and one good customer at random; AUC
is the chance the model ranks the defaulter as riskier. 0.5 is a coin flip.
**± 0.0036:** how much AUC moves fold to fold. Smaller gains count as noise.

---

# Part 7 — Weak spots (say them before she finds them)

1. **Different models give different headline numbers.**
   - live LR 0.6919, the table's XGBoost 0.7651, the best CatBoost 0.7868;
   - the leaderboard "LR 0.7378" is the research LR.
   - **Answer:** *"live is limited to the 15 fields a form collects."*
2. **Both deferral fixes are off by default.** The engine runs the old rule
   unless `SMARTLEND_DEFERRAL_MODE` is set.
3. **The constants were measured on the research table, not on the live LR's
   scores.**
   - `0.2458` is from the older operating point;
   - `0.198976` is from the current table.
4. **`ENGINE_VERSION`** still says 2026-04-27 after the α change and the
   third mode was added.
5. **Old comments:**
   - the docstring says 0.75/0.25;
   - `ENGINE_CLIP_HI = 0.75` in `threshold_selection.py`.
6. **The live model file stores `t_base = 0.65`.** That is the edge of the
   old F1 search: the file was made before the default switched to `cost`.
7. **The 0.5-line uncertainty beat the wired one** (−263 against −60). Both
   pass.
8. **The deployed blend band in the table is about random** (z +1.3 / −0.9),
   which is why gate condition 1 fails.
9. **Slide 6** contradicts itself (Part 0), and the **CBES slide** says
   "Liquidity".
10. **The script that made the original OOF table is not committed.**
11. **Home Credit is not Indian data**, and `cibil_score` is EXT_SOURCE_2
    rescaled.
