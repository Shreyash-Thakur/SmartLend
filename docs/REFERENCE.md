# SmartLend — Reference Sheet

**Updated:** 22 September 2026 · *dashboard live on real data*
**Purpose:** everything you may be asked to justify, in one place.

> **Canonical CBES AUC statement** (four similar numbers exist; know which is which):
> **0.5650** on all 307,511 out-of-fold rows (`reports/complementarity.json`) — quote this one.
> 0.5636 = single 20% holdout (`real_data_baselines.json`); 0.5621 = blend-study test half;
> 0.5638 = an earlier artifact era. All the same engine, different evaluation slices.

---

## 1. The dataset

| | |
|---|---|
| **Name** | Home Credit Default Risk (`application_train` + credit-bureau aggregates) |
| **Source** | https://www.kaggle.com/competitions/home-credit-default-risk |
| **File** | `creddefer_full_merged.csv` |
| **Rows** | 307,511 real loan applications |
| **Columns** | 131 (122 native + 9 merged bureau aggregates) |
| **Target** | `TARGET` — 1 = client had payment difficulties |
| **Default rate** | 8.07% |
| **Join key** | `SK_ID_CURR` |

### Verification performed

| Check | Result |
|---|---|
| Row count vs published Home Credit | 307,511 — exact match |
| Default rate vs published figure | 0.0807 — exact match |
| Duplicate applicants after merge | none — 307,511 unique `SK_ID_CURR` |
| Bureau signals directionally sane | **yes, all nine** |

**Why that last check matters.** A misaligned join produces columns that look fine but predict nothing. Ours discriminate correctly:

| Merged column | Defaulters vs repayers | Correct direction? |
|---|---|---|
| `overdue_credits` | **2.66× higher** | ✅ |
| `max_credit_overdue` | 1.38× higher | ✅ |
| `active_credits` | 1.22× higher | ✅ |
| `closed_credits` | 0.89× (lower) | ✅ closed loans = good history |
| `avg_days_credit` | −908 vs −1098 days | ✅ defaulters sought credit more recently |

### Known data issues (disclose if asked)

1. **14.3% (44,020 applicants) have no credit-bureau record at all.** Kept as blank + indicator column, deliberately **not** filled with 0 — "no overdue loans" and "no information" are different facts. These are thin-file applicants: young, first-time, or informally employed.
2. `total_credit_debt` contains negative values (min −6,981,558). This exists in Home Credit's original `AMT_CREDIT_SUM_DEBT`; it is a source-data quirk, not our merge. Needs clipping before computing utilisation.
3. **The file contains approved applicants only.** Rejected applicants never appear, so we never observe whether they would have repaid. This is the *reject inference* problem and it is central to our research claim (§4).

---

## 2. CBES — the scoring logic

**CBES = Credit Behaviour Evaluation Score.** A hand-designed, transparent rule-based score that runs alongside the ML model.

**Code:** `backend/app/services/cbes_engine.py`
**Calibration:** `backend/app/services/cbes_calibration.py` → `backend/artifacts/cbes_thresholds.json`

### Inputs — exactly 8 fields

| Field | Home Credit source |
|---|---|
| `credit_score` | `EXT_SOURCE_2` (proxy) |
| `delinquencies` | `overdue_credits` (bureau) |
| `active_loans` | `active_credits` (bureau) |
| `dti` | `AMT_ANNUITY / AMT_INCOME_TOTAL` |
| `employment_tenure_years` | `−DAYS_EMPLOYED / 365.25` |
| `annual_income` | `AMT_INCOME_TOTAL` |
| `loan_amount` | `AMT_CREDIT` |
| `region` | `REGION_RATING_CLIENT` |

✅ **All 8 are available in the merged dataset.** CBES and the data match.

### The five pillars and their weights

| # | Pillar | Weight | Built from |
|---|---|---|---|
| 1 | **Credit** | **0.35** | 0.70 × external score + 0.30 × delinquency history |
| 2 | **Capacity** | **0.30** | 0.60 × debt-to-income + 0.40 × loan-to-income |
| 3 | **Behaviour** | **0.20** | concurrent active credit lines |
| 4 | **Stability** | **0.10** | employment tenure |
| 5 | **Region** | **0.05** | urbanicity proxy (ordinal 1–3) |

```
CBES_raw = 0.35·credit + 0.30·capacity + 0.20·behaviour
         + 0.10·stability + 0.05·region

p_cbes   = sigmoid( 5 · (CBES_raw − 0.5) )
```

Each pillar passes through `component_sigmoid(x) = 1/(1+e^(−4(x−0.5)))`, which spans **[0.27, 0.73]** rather than [0.02, 0.98] — deliberately softened so no single pillar can dominate.

### Two design decisions to be ready to defend

**1. Thresholds are percentiles from the real data, not bank conventions.**
`EXT_SOURCE_2` is a normalised score in [0,1] with no published prime/subprime cutoff the way CIBIL or FICO has. Inventing one would be dishonest. Instead each pillar is scored against percentile breakpoints computed from the actual training distribution, so a rule reads as *"bottom 20% of applicants by this dataset's own score distribution."*

**2. Missing fields default to the worst observed value, not the average.**
`DEFAULTS` sets a missing credit score to 0, missing delinquencies to 10, and so on. A missing field therefore never *masks* risk as neutral. Conservative by construction.

### Direction convention (easy to get asked, easy to trip on)

- `p_ml` = probability of **approval** (`risk_score = 1 − p_ml`)
- `p_cbes` = **creditworthiness** — higher is a better applicant
- Both point the **same way**. There is no sign error in the blend.

---

## 3. The hybrid system: the flaw, the diagnosis, and the fix

```
Stage A — blend:     p_blend = 0.90·p_ml + 0.10·p_cbes     (α = 0.10, measured choice)
Stage B — routing:   legacy default:  D = |p_ml − p_cbes| > TAU_D  → defer
                     measured fix (opt-in): |p_ml − t_approve| < TAU_U (0.2458) → defer
```

### The finding, told in order

1. **Measured on real data, the disagreement router was inverted**: 52% deferral
   (AUC-implied ceiling 16%), and it deferred the cases the model was MOST
   confident about — gate z = +38.1σ balance / +43.0σ accuracy against a
   random-router null (`reports/relearning_gate_before_deferral_fix.json`).
2. **Root cause, confirmed**: CBES scores 0.5650 AUC and sits ~0.31 below p_ml
   population-wide (mean p_ml 0.9207 vs p_cbes 0.6133), so D mostly measures a
   fixed scale offset that WIDENS with confidence (corr +0.38). The decisive
   control: at a matched 22.6% rate the incumbent signal is still inverted
   (z = +18.3) — the defect is the signal, not the threshold
   (`reports/deferral_fix.json`).
3. **The fix, measured at the required 20–25% rate**: plain model uncertainty
   (Chow's rule) scores z = **−99.7** — humans finally get the hard cases.
   Every *repaired* disagreement variant (rank, z-score, isotonic) also fixes
   the inversion but **loses to plain uncertainty** — the publishable negative
   result. (An earlier draft of this sheet recommended "rank-normalise and
   re-tune"; that idea was implemented, measured at z = −13.4, and rejected.)
4. **Wired in**: `SMARTLEND_DEFERRAL_MODE=uncertainty` with TAU_U = 0.2458.
   The demo default remains the legacy router (deliberate stability choice);
   the regenerated artifact runs at **22.5% deferral** on the α = 0.10 blend.

### Say this if challenged

> "We built a deferral rule on a plausible premise, measured it on real data,
> found it inverted, confirmed the mechanism — a fixed calibration offset
> between two differently-scaled scores — fixed it, and re-measured. The fix
> is standard model uncertainty; the interesting part is that even perfectly
> calibrated ML-vs-rule disagreement loses to it, because CBES at 0.565 AUC
> carries too little signal to disagree *informatively*."

## 4. Research positioning — what is ours vs cited

### Established work we build on (cite, never claim)

| Topic | Reference | Link |
|---|---|---|
| Conformal prediction (foundational) | Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*, Springer 2005 | https://link.springer.com/book/10.1007/b106715 |
| Conformal under covariate shift | Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019 | https://arxiv.org/abs/1904.06019 |
| Practical CP tutorial | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* | https://arxiv.org/abs/2107.07511 |
| Selective classification magnifies group disparities | Jones, Sagawa, Koh, Kumar & Liang, ICLR 2021 | https://arxiv.org/abs/2010.14134 |
| Profit-based credit evaluation (EMP) | Verbraken, Bravo, Weber & Baesens, *EJOR* 2014 | https://doi.org/10.1016/j.ejor.2014.04.001 |
| Selective labels problem | Lakkaraju, Kleinberg, Leskovec, Ludwig & Mullainathan, KDD 2017 | https://doi.org/10.1145/3097983.3098066 |
| Sample selection bias | Heckman, *Econometrica* 1979 | https://doi.org/10.2307/1912352 |
| Regression discontinuity (robust) | Calonico, Cattaneo & Titiunik, *Econometrica* 2014 | https://doi.org/10.3982/ECTA11757 |
| Conformal library | MAPIE | https://mapie.readthedocs.io |

> ⚠️ **Verify before citing aloud.** In earlier sessions I referenced several very recent arXiv preprints (on profit-aware conformal abstention, reject-inference critique, and tabular foundation models in credit). I cannot re-verify those IDs offline. **Do not quote a specific arXiv number tomorrow unless you have personally opened the page.** The nine references above are long-established and safe.

### Our claim

> Conformal prediction gives a coverage guarantee only if calibration and test data are exchangeable. Credit data breaks this **by construction**: repayment is observed only for approved applicants. A deferral system calibrated on approved-only data therefore produces a guarantee that **looks valid on the sample it can measure and is void on the population it serves**.
>
> The textbook fix — weighted conformal prediction with reject-inference weights — **also fails**, because it requires *positivity* (approval probability bounded away from zero). Real lending uses hard cutoffs, so `P(approved | x) = 0` exactly on the rejected region and the weight `1/P̂` diverges. The correction is silently invalid **precisely on the applicants it was meant to protect**.

**What is genuinely ours:** measuring the size of that degradation on real data, and characterising exactly where the standard correction breaks.
**What is not ours:** conformal prediction, reject inference, weighted CP, fairness-of-abstention — all cited above.

---

## 5. Model results on real data ⭐

**Headline: 0.71 (synthetic, meaningless) → 0.7651 ± 0.0036 (real data, XGBoost, out-of-fold CV).**

Current dashboard numbers (`backend/artifacts/model_metrics.csv`, 5-fold OOF CV
over all 307,511 rows; foundation models on 8,000 id-indexed rows):

| Model | ROC-AUC | Sample |
|---|---|---|
| **XGBoost** | **0.7651 ± 0.0036** | 307,511 OOF |
| CatBoost | 0.7643 | 307,511 OOF |
| LightGBM | 0.7631 | 307,511 OOF |
| LogisticRegression | 0.7378 | 307,511 OOF |
| RandomForest | 0.7376 | 307,511 OOF |
| TabPFN-2.5 | 0.7284 ± 0.0127 | 8,000 rows, 5,000-row context |
| TabFM 1.0 (Google) | 0.6911 ± 0.0113 | same 8,000 rows, 15-feature serving frame |
| **CBES standalone** | **0.5650** | 307,511 OOF — rule-based, 8 fields, untrained |

(The earlier single-holdout table — XGB 0.7670 on 61,503 rows, seed 42 — is in
`reports/real_data_baselines.json`; real numbers, superseded by CV. Quote the
interval, not the point.)

### ⭐ The most important number here: CBES = 0.5650

Random guessing scores 0.5. **CBES scores 0.5650 — it is only marginally better than a coin flip**, while the ML models reach ~0.765.

This explains the hybrid's failure completely, and it is a much stronger answer than "we don't know":

1. CBES carries very little signal (0.5636 ≈ near-noise).
2. The blend mixes **25% of that near-noise** into a 0.767 model → drags it down.
3. `D = |p_ml − p_cbes|` is therefore mostly *"how far is the ML score from a nearly-uninformative number"* → deferring on it is close to deferring at random, or worse.

**So the old hybrid's underperformance was never a mystery. It was the predictable consequence of blending and deferring on a weak signal — and both have since been fixed and measured (α cut to 0.10, deferral switched to model uncertainty).**

Be ready to say: *"We have the number that explains it: CBES is 0.5650 AUC — too weak to blend at 25% and too weak to defer on. We measured both consequences, cut the blend weight to 0.10, and replaced disagreement-routing with model uncertainty."*

**In fairness to CBES:** it uses **8 fields**, the ML models use **129**. It is untrained — pure domain rules. It exists for *interpretability*, so a human reviewer can see why a decision was made. Judged as an explanation tool it is reasonable; judged as a predictor it is weak, and the system currently treats it as a predictor.

### Two framing points

**Why PR-AUC is also reported.** Only 8.07% of applicants default. A model predicting "never defaults" scores 92% accuracy and is useless. ROC-AUC alone flatters imbalanced problems, so PR-AUC is the honest companion metric.

**Why the old 0.71 didn't count.** It was measured on data generated by a hand-written formula in `generate_indian_loan_dataset.py`. The model was recovering the formula we ourselves wrote. **0.7670 on 307,511 real applications is a real number; 0.71 was not.** Say this plainly — it is a strength, not a weakness.

---

## 5b. Calibration — and why it changes the model choice

Same five models, shared 10,000-row holdout, four metrics:

| Model | ROC-AUC | PR-AUC | ECE ↓ | Brier ↓ |
|---|---|---|---|---|
| **XGBoost** | **0.7725** | 0.2828 | 0.0038 | 0.0662 |
| LightGBM | 0.7689 | 0.2823 | **0.0023** | 0.0663 |
| CatBoost | 0.7704 | 0.2821 | 0.0067 | 0.0663 |
| Logistic Regression | 0.7483 | 0.2338 | 0.0043 | 0.0683 |
| Random Forest | 0.7369 | 0.2224 | 0.0203 | 0.0696 |

**The two rankings disagree.** XGBoost ranks applicants best; LightGBM's probabilities are the most truthful (ECE = Expected Calibration Error, lower is better).

**Why this matters here specifically:** the deferral rule consumes `p_ml` as a *probability*, not as a ranking. A better-calibrated score makes `|p_ml − p_cbes|` mean closer to what we intend. So there is a defensible argument for deploying **LightGBM** despite XGBoost's higher AUC — and that is a more interesting model-selection answer than "pick the top row."

Random Forest being ~9× worse calibrated matches theory for vote-averaging ensembles — a useful check that the metric behaves.

## 5c. Foundation models — TabPFN-2.5 and Google TabFM, both honestly scored

**TabPFN-2.5 (Prior Labs).** Initially hardware-blocked (8 GB VRAM caps the
in-context training set), then scored twice: 0.7446 on the full 61,503-row
holdout from a 5,000-row context (77 GPU-minutes), and re-scored at 8,000 rows
**with `SK_ID_CURR` saved per probability** (0.7284 ± 0.0127 — same model,
smaller sample, overlapping intervals; quote the interval). The row IDs are
what unblocked the fusion analysis (§"hybrid" in `docs/FUTURE-SCOPE.md`:
verdict KILL). To lift the context cap, `colab/tabpfn_colab_server.ipynb`
hosts it on a Colab GPU as a live endpoint the backend can call.

**Google TabFM 1.0** (github.com/google-research/tabfm, June 2026) — added on
reviewer request. Zero-shot in-context model, same class as TabPFN, different
lineage. Scored on the exact same 8,000 rows (committed script:
`research/analysis/score_tabfm.py`): **0.6911 ± 0.0113** — matching the
*trained* serving LogisticRegression on the same 15-field vocabulary with no
training at all.

Verified corrections worth carrying:

| Claim often repeated | Reality |
|---|---|
| "TabPFN published in *Nature*" | The *Nature* paper describes **TabPFN v2**, **not v2.5**. Cite **arXiv:2511.08667** for v2.5 — a **preprint**. |
| "Calibration is baked in" | Prior Labs' own report: results are *"computed using uncalibrated, default scores."* |
| "Fine to use, it's open" | TabPFN-2.5's licence is **non-commercial and covers outputs**. TabFM's **code** is Apache-2.0 but its **weights** are `tabfm-non-commercial-v1.0`. Both are research baselines, neither is deployable commercially. |
| "`pip install tabpfn` gives v2.5" | It installs **TabPFN-3**; v2.5 must be pinned via the checkpoint path (`models/README.md`). |

## 5d. Running the dashboard

```bash
# terminal 1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# terminal 2
cd frontend && npm run dev
```

Open **http://localhost:5173** → Model Analysis. Vite proxies `/api` to port 8000.

The page shows: the active/serving model card (with artifact + engine version — the version is also a permanent badge in the org header), sortable leaderboard (each model at its own operating threshold), AUC bar chart, error-profile table, ML-vs-CBES decision-landscape scatter, decision mix by probability band, and a paginated case drill-down. (The old 2×2 confusion-matrix card was removed in commit c3b6628 after the matrices were re-scored at real thresholds.)

## 6. Quick answers to likely questions

**"Why did you move off your own dataset?"**
It was synthetic — labels came from a formula we wrote, so any accuracy measured on it was circular. We now use 307,511 real applications, and accuracy went *up* to 0.767.

**"Why was your hybrid worse than a simple model?"**
It was, and we measured why: the disagreement signal is dominated by a scale offset between the two scores rather than real disagreement. That diagnosis is our contribution — and the measured fix is plain model uncertainty, which beats every repaired version of the disagreement signal (z = −99.7 vs −13 to −71; `reports/deferral_fix.json`).

**"Is CBES just made-up weights?"**
The pillar weights are a domain prior. The *thresholds* inside each pillar are calibrated from the real training distribution, not invented. CBES is deliberately restricted to 8 portable fields so it stays interpretable and comparable across datasets.

**"What is actually novel?"**
Not conformal prediction, not reject inference. What is ours: measuring how far the coverage guarantee degrades under real credit selection bias, and showing that the standard correction breaks down exactly on the applicants it was meant to protect.

**"What about fairness?"**
Home Credit carries `CODE_GENDER`, age and region. Missing data concentrates in thin-file applicants (the 44,020 with no bureau record), so imputation defaults would systematically hit them. We keep missingness explicit and measurable rather than filling it in.

---


**"Which model are you deploying?"**
XGBoost has the best AUC at 0.7670, but LightGBM is better calibrated (ECE 0.0023 vs 0.0038) and our deferral layer consumes probabilities rather than rankings — so LightGBM is arguably the better production choice. The three boosters are within 0.0004 AUC of each other, so the decision rests on calibration, not accuracy.

**"Did you try a foundation model?"**
Two. TabPFN-2.5 (0.7284 ± 0.0127 on 8,000 id-indexed rows from a 5,000-row context; 0.7446 on the earlier full holdout) and Google TabFM (0.6911 ± 0.0113 zero-shot on the same rows). Both beat or match trained linear baselines with a fraction of the data, both lose to the tree ensembles, both carry non-commercial licences, and the pre-registered fusion experiment returned KILL — neither adds anything an honest evaluation can claim on top of XGBoost. We also host TabPFN on a Colab GPU as a live endpoint to lift the local 8 GB context cap.

**"Why was your deferral rate so high?"**
It was 52% on real data under the original rule — and that was the finding, not an embarrassment: the signal was inverted (it deferred the model's most confident cases), we confirmed the mechanism, and we fixed it. The regenerated artifact defers **22.5%**, inside the 20–25% underwriter-capacity band; the remaining gap to the AUC-implied 16% ceiling is a capacity decision we state rather than hide.

## 7. File map

| Item | Path |
|---|---|
| CBES engine | `backend/app/services/cbes_engine.py` |
| CBES thresholds | `backend/artifacts/cbes_thresholds.json` |
| Hybrid decision logic | `backend/app/services/decision_engine.py` |
| Calibration report | `backend/artifacts/calibration_report.txt` |
| Canonical schema | `research/data/canonical.py` |
| Missingness profiler | `research/data/profile.py` |
| Real-data results | `reports/real_data_baselines.json` |
| Research design | `docs/superpowers/specs/2026-08-18-conformal-credit-deferral-design.md` |
| Project status | `docs/STATUS.md` |
