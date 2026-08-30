# SmartLend — Reference Sheet

**For:** mid-evaluation, 31 August 2026 · *dashboard live on real data*
**Purpose:** everything you may be asked to justify, in one place.

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

## 3. The hybrid system and its known flaw

```
Stage A — blend:     p_blend = 0.75·p_ml + 0.25·p_cbes
Stage B — disagree:  D = |p_ml − p_cbes|
                     if D > TAU_D (0.43) → defer to a human
```

### The honest result

| System | Decides | Accuracy |
|---|---|---|
| Plain LogisticRegression | 100% | **70.5%** |
| Hybrid (after 3 fixes) | 74.5% | **62.7%** |

Abstaining on 25.5% of cases made accuracy **7.8 points worse**. A selective classifier should be *more* accurate on what it keeps.

### The diagnosis — now measured, not inferred

**On real data the hybrid defers 52.38% of applications** (up from 25.5% on synthetic) and is 60.61% accurate on what it keeps, against 70.5% for plain logistic regression deciding everything.

**Direct evidence of the inversion.** Group applicants by model confidence, then ask who gets deferred:

| Model confidence | Applicants | Deferred | % deferred |
|---|---|---|---|
| 0.4–0.6 — **genuinely uncertain** | 280 | 12 | **4.3%** |
| 0.8–1.0 — **highly confident** | 22,942 | 12,580 | **54.8%** |

**The rule defers 55% of the cases the model is sure about, and 4% of the cases it is unsure about.** This is visible live on the dashboard's probability-band chart.

**The mechanism.** Deferral triggers on `|p_ml − p_cbes| > 0.43`. But CBES scores **0.5638 AUC** (random = 0.5) and sits systematically **~0.31 below** the ML score population-wide:

| | Mean |
|---|---|
| `p_ml` | 0.6722 |
| `p_cbes` | 0.3646 |
| `D` | 0.3105 |

So `D` is dominated by a **fixed offset between two differently-scaled scores**, not by genuine case-by-case disagreement. It fires where the offset is widest, which tracks model confidence rather than difficulty.

**This is why three previous fixes failed** — they tuned a threshold on an inverted signal.

**The fix:** rank-normalise or z-score both signals before differencing, then re-tune `TAU_D`.

### Say this if challenged

> "The hybrid underperforms, and we can now show exactly why. It defers 55% of the cases the model is most confident about and only 4% of the genuinely uncertain ones — the deferral signal is inverted. The cause is that CBES is nearly uninformative at 0.5638 AUC and sits on a different scale to the ML score, so their difference measures scale mismatch rather than disagreement. That diagnosis is our contribution; the fix is to normalise both signals before comparing them."

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

**Headline: 0.71 (synthetic, meaningless) → 0.7670 (real data, XGBoost).**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **XGBoost** | **0.7670** | 0.2624 |
| **LightGBM** | **0.7667** | 0.2632 |
| **CatBoost** | **0.7666** | 0.2638 |
| LogisticRegression | 0.7405 | 0.2226 |
| RandomForest | 0.7388 | 0.2204 |
| **CBES standalone** | **0.5636** | rule-based, 8 fields, untrained |

Held-out test set: 61,503 rows (20%), stratified, seed 42.
Full output: `reports/real_data_baselines.json`.

### ⭐ The most important number here: CBES = 0.5636

Random guessing scores 0.5. **CBES scores 0.5636 — it is only marginally better than a coin flip**, while the ML models reach 0.767.

This explains the hybrid's failure completely, and it is a much stronger answer than "we don't know":

1. CBES carries very little signal (0.5636 ≈ near-noise).
2. The blend mixes **25% of that near-noise** into a 0.767 model → drags it down.
3. `D = |p_ml − p_cbes|` is therefore mostly *"how far is the ML score from a nearly-uninformative number"* → deferring on it is close to deferring at random, or worse.

**So the 62.7% is not a mystery. It is the predictable consequence of blending and deferring on a weak signal.**

Be ready to say: *"We now have the number that explains it. CBES is 0.5636 AUC — too weak to blend at 25% and too weak to defer on. Our next step is to either strengthen CBES or change how the two signals are combined."*

**In fairness to CBES:** it uses **8 fields**, the ML models use **129**. It is untrained — pure domain rules. It exists for *interpretability*, so a human reviewer can see why a decision was made. Judged as an explanation tool it is reasonable; judged as a predictor it is weak, and the system currently treats it as a predictor.

### Two framing points### Two framing points

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

## 5c. TabPFN-2.5 — evaluated, hardware-blocked

Assessed as a candidate. A tabular foundation model from Prior Labs.

**Outcome: could not be fairly scored.** It exhausted 8 GB of VRAM at every configuration tried, including a 500-row prediction batch. The binding constraint is the **training context** — TabPFN holds it in memory for each forward pass at cost quadratic in rows — not the batch size. Prior Labs recommend A100/H100-class hardware. Fitting our data on this GPU would mean subsampling to ~5%, which is not a comparison worth reporting.

Three verified corrections worth carrying:

| Claim often repeated | Reality |
|---|---|
| "Published in *Nature*" | The *Nature* paper describes **TabPFN v2** (10k×500), **not v2.5** (50k×2000). Cite **arXiv:2511.08667** for v2.5 — a **preprint**. |
| "Calibration is baked in" | Prior Labs' own report: results are *"computed using uncalibrated, default scores."* **No published ECE/Brier exists.** |
| "Fine to use, it's open" | Licence is **non-commercial and covers outputs**, not just weights. Academic use is explicitly allowed; commercial deployment is not. |

Also: `pip install tabpfn` installs **TabPFN-3**, not v2.5. The version must be pinned or you benchmark a different model than you report.

## 5d. Running the dashboard

```bash
# terminal 1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# terminal 2
cd frontend && npm run dev
```

Open **http://localhost:5173** → Model Analysis. Vite proxies `/api` to port 8000.

The page shows: sortable leaderboard (best value per metric highlighted), AUC bar chart, 2×2 confusion matrix with derived rates, false-positive vs false-negative error profile, decision mix by probability band (**this is where the inversion is visible**), and a paginated 25,000-case drill-down.

## 6. Quick answers to likely questions

**"Why did you move off your own dataset?"**
It was synthetic — labels came from a formula we wrote, so any accuracy measured on it was circular. We now use 307,511 real applications, and accuracy went *up* to 0.767.

**"Why is your hybrid worse than a simple model?"**
It is, and we measured why: the disagreement signal is dominated by a scale offset between the two scores rather than real disagreement. That diagnosis is our contribution — the fix is to rank-normalise both signals before comparing.

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
Yes — TabPFN-2.5. It needs A100/H100-class memory for our data volume; our 8 GB GPU fits about 5% of the dataset, which would not be a fair comparison. We also found its licence is non-commercial and restricts outputs, so it could not be deployed even if it won.

**"Why is your deferral rate so high?"**
52.38% on real data, up from 25.5% on synthetic — and that is the finding, not an embarrassment. The rule defers 55% of the cases the model is most confident about and 4% of the uncertain ones. The signal is inverted, we know the mechanism, and we know the fix.

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
