# Defence brief — SHAP explainability and the CBES score

**Scope:** exactly how these two mechanisms work *in this repository*, on the code as it
stands. Every claim carries a `file:line` so it can be opened live in front of an examiner.
Nothing here is a description of how SHAP or scorecards "usually" work.

**Verified against:** `backend/artifacts/pipeline_v3_real.joblib` (15 features,
`model_name = "LogisticRegression"`, `StandardScaler + LogisticRegression(max_iter=1000)`
wrapped in `CalibratedClassifierCV(isotonic, cv=5)`), and
`backend/artifacts/cbes_thresholds.json` (regenerated 2026-08-30).

> **Read the "Known defects" section (§3) before the viva.** There are four live issues in
> the explainability path that an examiner can trigger by clicking one application.

---

# PART 1 — SHAP

## 1.1 The thirty-second answer

> We use **`shap.LinearExplainer`**, because the served model is a logistic regression.
> It explains **P(default)**, so a **positive SHAP value pushes the applicant toward
> default, i.e. toward REJECT**. The baseline is the mean prediction over a fixed
> 100-row sample of the training set stored inside the model artifact. We take the
> three largest-magnitude attributions, re-sign them so "positive" means "supported
> the decision we actually made", and render them as a horizontal bar chart.

## 1.2 Which explainer, and where

`backend/app/services/explainability_service.py` **does not compute SHAP.** It is a
*presentation* layer that consumes an already-computed list. SHAP is computed in
`backend/app/services/ml_service.py`.

| Fact | Location |
|---|---|
| `import shap` | `backend/app/services/ml_service.py:11` |
| Explainer chosen | `ml_service.py:254-260` |
| Model handed to the explainer | `ml_service.py:250-251` |
| SHAP values computed per request | `ml_service.py:284-305` |

```python
# ml_service.py:249-260
self.scaler     = self.pipeline.named_steps["scaler"]
self.classifier = self.pipeline.named_steps["model"]
try:
    if "Logistic" in str(self.model_name):
        self.explainer = shap.LinearExplainer(self.classifier, self.background_data)
    else:
        self.explainer = shap.TreeExplainer(self.classifier)
except Exception:
    self.explainer = None
```

The branch is selected by a **substring test on the artifact's `model_name` string**.
The live artifact stores `model_name = "LogisticRegression"`, so the `LinearExplainer`
branch is taken. `TreeExplainer` is dead code on the current artifact — it exists only
because `train_pipeline()` (`ml_service.py:96-102`) can select XGBoost / LightGBM /
CatBoost / RandomForest, and that path is currently unrunnable (its dataset was deleted;
see the NOTE at `ml_service.py:57-69`).

**Verified live.** Instantiating `MLPredictor()` on the current artifact reports
`explainer = LinearExplainer`, 15 features, and returns three attributions per call.

## 1.3 What is being explained — model, features, direction

### The model
`shap.LinearExplainer` is given `self.pipeline.named_steps["model"]` — the **plain
`LogisticRegression`** fitted on `X_train` at `retrain_serving_model_v3.py:224-225`.

⚠️ **This is not the model that produces the served score.** `p_ml` comes from
`self.calibrator` — a `CalibratedClassifierCV(isotonic, cv=5)` (`ml_service.py:279`,
`retrain_serving_model_v3.py:176-177`). The calibrator internally holds five *other*
logistic regressions plus an isotonic step. The plain pipeline is a sibling model fitted
on the same data, deliberately saved so SHAP has something linear to read
(`retrain_serving_model_v3.py:222-225`). See §3.1 — say this before you are asked.

### The features
The 15 columns in the artifact's `feature_names`
(`retrain_serving_model_v3.py:104`, `FEATURES = PROFILE_FEATURES + ["loan_amount", "loan_income_ratio"]`):

`age`, `dependents`, `years_employed`, `annual_income`, `monthly_income`, `existing_emis`,
`cibil_score`, `total_loans`, `active_loans`, `closed_loans`, `missed_payments`,
`credit_utilization_ratio`, `debt_to_income_ratio`, `loan_amount`, `loan_income_ratio`.

These are the *serving vocabulary*, built for each Home Credit row by the same function
the live API uses — `customer_profile_service._build_profile`
(`retrain_serving_model_v3.py:127`). So a feature is constructed identically at fit time
and at score time; there is no second, divergent formula.

Eight of the original 25 serving features were **dropped rather than invented** because
Home Credit does not carry them, and two were dropped as **target leaks**
(`loan_approved`, `confidence_score` — the model's own outputs had been training inputs).
The full list with per-feature reasons is `retrain_serving_model_v3.py:74-96` and
`reports/serving_model_retrain.json`.

### The direction — get this right
The training label is `TARGET`, where **1 = defaulted**
(`retrain_serving_model_v3.py:63`). The logistic regression's positive class is therefore
"defaulted". `LinearExplainer` on a binary sklearn `LogisticRegression` attributes the
**log-odds of class 1**. Therefore:

> **Positive SHAP value → pushes P(default) up → pushes toward REJECT.**
> **Negative SHAP value → pushes P(default) down → pushes toward APPROVE.**

Sanity check you can run live: `classifier.coef_` for `missed_payments` is **+0.061**
(more missed payments ⇒ more default) and for `cibil_score` is **−0.451** (higher score
⇒ less default). Signs are as expected.

**The served score is the complement.** `p_ml = 1 − P(default)` = P(approval), stated at
`ml_service.py:276-279` and again at `retrain_serving_model_v3.py:191`. So SHAP is in the
**opposite** orientation to `p_ml`. That flip is the single most likely thing to be asked
about and the easiest to state backwards.

### The re-signing step (why the UI shows the opposite sign sometimes)
`explainability_service.py:30-32`:

```python
def _impact_sign_for_decision(decision: str, raw_impact: float) -> float:
    return raw_impact if decision == "REJECT" else -raw_impact
```

The displayed `impact` is **not** the raw SHAP value. It is re-expressed as
*"did this feature support the decision we actually made?"*:

| Decision | Raw SHAP (toward default) | Displayed impact | Meaning |
|---|---|---|---|
| REJECT | +0.55 | **+0.55** | supported the rejection |
| REJECT | −0.20 | −0.20 | argued against the rejection |
| APPROVE | −0.55 | **+0.55** | supported the approval |
| APPROVE | +0.20 | −0.20 | was a risk factor despite approval |

`direction` is set to `"supports_decision"` / `"opposes_decision"` on that re-signed value
(`explainability_service.py:82`). The frontend paints `impact >= 0` **green**
(`FeatureContributionChart.tsx:45,58`), so **green always means "supported the decision on
screen"**, never "good for the applicant". Be precise about this if asked.

## 1.4 The baseline / background dataset

SHAP values are differences from a reference expectation, so the reference must be named.

| Question | Answer | Location |
|---|---|---|
| What is the background? | 100 rows sampled from the **training split**, already `StandardScaler`-transformed | `retrain_serving_model_v3.py:226-228` |
| Sampling | `X_train.sample(min(100, len(X_train)), random_state=42)` — fixed seed, so it is deterministic and reproducible | same |
| Where does it live? | Serialised into the artifact under `background_data`; shape `(100, 15)` | `retrain_serving_model_v3.py:235`; `ml_service.py:246` |
| What is the baseline value? | `E[f(x)]` over those 100 rows, in **log-odds of default** | implicit in `shap.LinearExplainer` |

**No double-scaling bug.** The background was scaled by the pipeline's own
`StandardScaler`, and at inference `ml_service.py:287` applies **the same fitted scaler**
(`self.scaler`, unpacked from the same pipeline) before calling the explainer. Explanation
space and background space match.

**Honest caveat:** 100 rows is a small reference set. For a *linear* model this matters far
less than it would for a tree model — `LinearExplainer`'s output for feature *i* is
`coef_[i] · (x_i − E[x_i])`, and the only thing the background supplies is the mean
`E[x_i]`, which is stable at n=100. This is a genuine reason the linear choice is
defensible here.

## 1.5 Top-3 selection and rendering — the full trace

| # | Step | Location |
|---|---|---|
| 1 | Missing features imputed toward worst-case, row built in `feature_names` order | `ml_service.py:266-274` |
| 2 | Row scaled with the pipeline's scaler | `ml_service.py:287` |
| 3 | `shap_values(X_scaled)`; shape normalised across sklearn / xgboost / lgb layouts | `ml_service.py:289-297` |
| 4 | **Top 3 by absolute value** — `impacts.abs().sort_values(ascending=False).head(3)` | `ml_service.py:299-300` |
| 5 | Emitted as `[{name, impact, value}]`; `value` is the **raw, unscaled** feature value | `ml_service.py:303` |
| 6 | Carried through the decision engine untouched | `decision_engine.py:146, 289` |
| 7 | Persisted onto the application row as `_decision_meta["shap_explanation"]` | `routers/applications.py:432, 455-459` |
| 8 | Re-signed, labelled, given a reason string, re-sorted by `abs(impact)`, sliced `[:5]` | `explainability_service.py:59-91` |
| 9 | Mapped to `decision.featureImportance` (`baseValue` ← `targetValue`) | `decision_service.py:56-64, 107` |
| 10 | Rendered as a horizontal bar chart, top 5, green/red by sign | `pages/ApplicationReview.tsx:243` → `components/sections/FeatureContributionChart.tsx:20-67` |

Note the `[:5]` at step 8 and `maxFeatures = 5` at `FeatureContributionChart.tsx:18` are
both **larger than the 3 values that actually exist**, because step 4 already truncated to
three. The UI therefore shows **three** bars. There is no bug here, but do not claim "top 5
SHAP features" — the honest statement is *"top 3 by |SHAP|, in a component that would
display up to 5"*.

Ranking is by **absolute** SHAP value, so the three shown are the three most *influential*
features, not the three most *negative* ones. A strongly favourable feature can occupy a
slot on a rejected application.

**Only reached when SHAP succeeded.** `explainability_service.py:61` guards on
`if shap_explanation:`. If SHAP failed (`ml_service.py:304-305` swallows any exception)
the code silently falls through to a **hand-written heuristic** at
`explainability_service.py:93-188` — nine fixed rules like `impact = dti − 0.45`. That
fallback is *not* SHAP, is not labelled as such anywhere in the UI, and is built on the
old India-specific vocabulary. See §3.4.

## 1.6 Limitations of this SHAP setup — the honest list

1. **We explain a sibling model, not the served model.** SHAP reads the plain pipeline;
   the score comes from the isotonic-calibrated ensemble. Rankings are close (same data,
   same estimator spec) but the magnitudes are in the *uncalibrated* log-odds space and do
   not decompose the served probability. `ml_service.py:250-251` vs `ml_service.py:279`.
2. **Additivity does not survive calibration.** Isotonic regression is a non-linear
   monotone map. SHAP's guarantee is that attributions sum to
   `f(x) − E[f(x)]` in log-odds; nothing in this code claims — and nothing can claim —
   that they sum to the displayed `riskScore`.
3. **Correlated features split credit arbitrarily.** `annual_income` and `monthly_income`
   are near-perfectly collinear by construction; `total_loans`, `active_loans` and
   `closed_loans` satisfy an identity; `debt_to_income_ratio` and `loan_income_ratio`
   share terms. `LinearExplainer` with `feature_perturbation="interventional"` (the
   default when a background matrix is supplied) attributes to each variable
   independently, so the split between two collinear columns is determined by which one
   the fitting procedure happened to load the coefficient onto. Whether `monthly_income`
   or `annual_income` appears in the top-3 is not a substantive finding.
4. **SHAP explains the model, never the ground truth.** It answers *"why did this model
   output this number"*, not *"why will this person default"*. If the model is wrong, SHAP
   faithfully explains a wrong answer. On this artifact, held-out ROC-AUC is **0.6919**
   with PR-AUC **0.1706** at an 8.07% default rate (`retrain_serving_model_v3.py:180-187`,
   values read back from the artifact's `test_metrics`) — so it explains a modest model.
5. **The reason strings are templates, not derived text.** Every sentence a customer sees
   is one of six f-strings at `explainability_service.py:70-75`, parameterised only by the
   label and the value. No LLM, no causal claim.
6. **Counterfactuals are not counterfactuals.** `_counterfactual_target`
   (`explainability_service.py:39-51`) is a hardcoded lookup table. It is not derived from
   the model, no re-scoring is done, and for a feature absent from the table it returns
   the current value unchanged (§3.2).
7. **Silent failure mode.** `except Exception: pass` at `ml_service.py:304-305`. A broken
   explainer produces an empty list and the UI degrades to heuristics without warning.
8. **The artifact-selection substring test is fragile.** `"Logistic" in str(model_name)`
   (`ml_service.py:255`). Any future artifact named e.g. `"LogitBoost"` picks the wrong
   explainer; a model that is neither linear nor tree-based falls into `TreeExplainer` and
   throws, silently disabling explanations.

---

# PART 2 — CBES

## 2.1 The thirty-second answer

> CBES is a hand-designed, five-pillar heuristic score over **8 input fields**. Each pillar
> maps raw values onto [0,1] via percentile breakpoints computed from the real Home Credit
> distribution, passes them through a k=4 sigmoid, and the five are combined with fixed
> weights (0.35/0.30/0.20/0.10/0.05) before a final k=5 sigmoid. **A high CBES means a
> good applicant** — it is an approval probability, on the same orientation as `p_ml`.
> Standalone it scores **0.5650 AUC** against 0.5 for random. We keep it for
> interpretability and portability, not for discrimination, and we say so.

## 2.2 The five pillars

Source: `backend/app/services/cbes_engine.py:111-150`.

| # | Pillar | Weight | Computed from | Sub-weights | Line |
|---|---|---|---|---|---|
| 1 | **Credit** | **0.35** | `credit_score` (EXT_SOURCE_2, higher better) and `delinquencies` (lower better) | 0.70 / 0.30 | 112-117 |
| 2 | **Capacity** | **0.30** | `dti` (lower better) and `loan_to_income = loan_amount / annual_income` (lower better) | 0.60 / 0.40 | 120-125 |
| 3 | **Behaviour** | **0.20** | `active_loans` — concurrent open credit lines (lower better) | — | 128-131 |
| 4 | **Stability** | **0.10** | `employment_tenure_years` (higher better) | — | 134-137 |
| 5 | **Region** | **0.05** | `region` = `REGION_RATING_CLIENT`, ordinal 1 = best … 3 = worst | — | 139-142 |

Weights sum to exactly 1.00. `loan_to_income` is derived inside the engine at
`cbes_engine.py:109`; the eight raw inputs are read at `cbes_engine.py:100-107`.

The region pillar is deliberately weighted at 0.05 and the comment at
`cbes_engine.py:139-140` states it is a **urbanicity proxy, explicitly not a geography
claim**. Note it is also the **only pillar whose breakpoints are hardcoded**
(`[1, 1, 2, 3, 3]`, `cbes_engine.py:141`) rather than loaded from the calibration artifact —
because an ordinal 1–3 rating has no meaningful percentiles.

## 2.3 The exact formula

**Step 1 — percentile mapping** (`cbes_engine.py:66-90`). Each raw value is linearly
interpolated across five breakpoints (p10, p30, p50, p70, p90) onto the positions
`[0.0, 0.25, 0.5, 0.75, 1.0]`, clipped at both ends. `higher_is_better=False` returns
`1 − score`.

**Step 2 — pillar sigmoid, k = 4** (`cbes_engine.py:93-95`):

```
component_sigmoid(x) = 1 / (1 + exp(−4 · (x − 0.5)))
```

**Step 3 — weighted sum** (`cbes_engine.py:144-150`):

```
CBES_raw = 0.35·credit + 0.30·capacity + 0.20·behaviour + 0.10·stability + 0.05·region
```

**Step 4 — aggregate sigmoid, k = 5** (`cbes_engine.py:151`):

```
p_cbes = 1 / (1 + exp(−5 · (CBES_raw − 0.5)))
```

### What each sigmoid actually does

| | k = 4 (per pillar) | k = 5 (aggregate) |
|---|---|---|
| Input range | [0, 1] | [0.1192, 0.8808] |
| Output range | **[0.1192, 0.8808]** | **[0.1296, 0.8704]** |
| Effect | Compresses the extremes. A percentile-0 applicant scores 0.119, not 0.0 — no single pillar can zero out the score. | Steepens the mid-range so the blended score separates around the 0.5 axis, while still refusing to reach 0 or 1. |

**Verified empirically**: an all-best synthetic applicant returns `p_cbes = 0.87034`; an
all-worst applicant and an *empty dict* both return `p_cbes = 0.12966`. These are the exact
theoretical bounds.

> ⚠️ **The docstring at `cbes_engine.py:94` used to be wrong** — it claimed k=4 gives a
> `[0.27, 0.73]` span, which is the **k=2** span. The true k=4 span is `[0.1192, 0.8808]`.
> **Fixed (2026-08-31):** the docstring now states `[0.1192, 0.8808]` and records that the
> old figure was the k=2 value. It was a comment error, not a maths error; the code was
> always k=4 and the measured outputs always agreed with it.

**Consequence you must be ready to state:** `p_cbes` is structurally confined to
`[0.130, 0.870]` and its observed mean over all 307,511 rows is **0.6133**, while the
calibrated `p_ml` concentrates near **0.92** (8% base rate). The ~0.31 systematic offset is
a *scale* artifact of these two sigmoids, and it is the documented root cause of the
inverted deferral behaviour (`docs/DEFERRAL-FIX.md`, `docs/REFERENCE.md:135-151`,
`README.md:58`). If an examiner asks "why does the system defer the easy cases?", the
answer is these two constants, not a logic bug in the gates.

### Provenance of k=4 and k=5 — say this before you are asked
`docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md:22`, after a
cited literature review, records the finding verbatim: mapping a rank-based aggregate
through a monotonic squashing function is the standard FICO base-score/PDO pattern and the
*shape* is defensible and citable, but **`k=4` and `k=5` are hand-picked engineering
choices with no source behind them**. Do not present them as literature-derived.

## 2.4 How the thresholds were derived

Script: **`backend/app/services/cbes_calibration.py`**, run as
`python -m backend.app.services.cbes_calibration` (`main()`, lines 95-103).

| Question | Answer | Location |
|---|---|---|
| Which percentiles? | **p10, p30, p50, p70, p90** | `cbes_calibration.py:30` |
| Which distribution? | The full real Home Credit training distribution — `application_train.csv` left-joined with `bureau.csv` aggregates, mapped through the canonical data layer | `cbes_calibration.py:76-92` |
| Which columns? | `credit_score`, `delinquencies`, `active_loans`, `dti`, `employment_tenure_years`, `loan_to_income` | `cbes_calibration.py:32-39` |
| NaN handling | Dropped before `np.percentile` | `cbes_calibration.py:46-48` |
| Monotonicity | Edges forced non-decreasing | `cbes_calibration.py:49-52` |
| Output | `backend/artifacts/cbes_thresholds.json` | `cbes_calibration.py:27-28, 99` |
| Loading | **Lazily, on first `compute_cbes` call**, not at import — so the module cannot crash on import if the artifact is absent | `cbes_engine.py:39-51` |

Current artifact values:

| Column | p10 | p30 | p50 | p70 | p90 |
|---|---|---|---|---|---|
| `credit_score` | 0.2157 | 0.4405 | 0.5660 | 0.6459 | 0.7220 |
| `delinquencies` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `active_loans` | 0.0 | 1.0 | 1.0 | 2.0 | 4.0 |
| `dti` | 0.0800 | 0.1248 | 0.1628 | 0.2125 | 0.3016 |
| `employment_tenure_years` | 0.9117 | 2.5161 | 4.5120 | 7.6441 | 14.6010 |
| `loan_to_income` | 1.3318 | 2.2663 | 3.2651 | 4.7250 | 7.4875 |

**Why percentiles at all.** `credit_score` is EXT_SOURCE_2 — a normalised external score
with **no established prime/subprime cutoff**. Inventing a bank-style cutoff for it would
be, in the module docstring's own words, "scientifically dishonest"
(`cbes_engine.py:1-17`). Percentile bands make each rule legible as *"bottom 20% of
applicants by this dataset's own distribution"*.

### Two degeneracies you must volunteer

- **`delinquencies` collapses to a binary step.** All five breakpoints are 0.0, because
  most Home Credit applicants have zero delinquencies. `_interp` (`cbes_engine.py:78-90`)
  then returns 0.0 for `value <= 0` and 1.0 for anything above, so after inversion the
  sub-score is **1.0 for zero delinquencies and 0.0 for any**. That is **30% of the credit
  pillar = 10.5% of the total score** behaving as an on/off switch, not a graded measure.
  The script detects this and prints a warning rather than raising
  (`cbes_calibration.py:53-60`) — a column collapsing is a property of the data, not a bug.
- **`active_loans` is nearly as coarse** — p30 = p50 = 1.0, so a third of the applicant
  population sits on a flat segment of the behaviour pillar.

## 2.5 The DEFAULTS behaviour

`cbes_engine.py:26-37`:

```python
DEFAULTS = {
    "credit_score": 0.0,            "delinquencies": 10.0,
    "active_loans": 10.0,           "dti": 1.0,
    "employment_tenure_years": 0.0, "annual_income": 1.0,
    "loan_amount": 10_000_000.0,    "region": 3.0,
}
```

**What happens on a missing field.** `_safe_float` (`cbes_engine.py:54-63`) substitutes the
default when the value is `None`, non-numeric, `NaN`, or `inf`.

**Why the worst observed value and not the mean.** The comment at `cbes_engine.py:26-27` is
the whole argument: *"as bad as observed data in this dataset gets, so a missing field
never masks risk as neutral."* Mean-imputation would hand an applicant with no bureau file
an average credit pillar — it would convert *absence of evidence* into *evidence of
average creditworthiness*, which is precisely the failure mode a lender cannot accept.
Worst-case imputation makes missingness costly and therefore visible.

**Verified:** `compute_cbes({})` returns exactly `0.12966`, the theoretical floor. A payload
with no CBES fields is scored as the worst possible applicant, not a median one.

**Train/serve consistency (important, and defensible).** The same `DEFAULTS` dict is
imported by the ML retraining script (`retrain_serving_model_v3.py:52`) and applied to the
training frame at line 146 — *and* by `MLPredictor.predict_application` at
`ml_service.py:266-272`. A missing value therefore means the identical number at fit time
and at score time. The caveat: only three of the 15 ML features appear in this 8-key dict
(`active_loans`, `annual_income`, `loan_amount`), so the other twelve fall through
`DEFAULTS.get(col, 0.0)` to **0.0** — including `cibil_score`. That is documented at
`ml_service.py:57-69` and is consistent between train and serve, but "0.0" is a *low*
credit score, so it happens to remain conservative.

## 2.6 Direction convention — unambiguous

> **High CBES = good applicant = high probability of approval.**
> `p_cbes` is an approval probability, oriented the same way as `p_ml`, and the
> **opposite** way to a raw SHAP value.

Four independent places in the code establish this:

1. **By construction.** `credit_score` and `employment_tenure_years` use
   `higher_is_better=True`; `delinquencies`, `dti`, `loan_to_income` and `active_loans` use
   `higher_is_better=False` (`cbes_engine.py:112-136`). Every pillar rises as the applicant
   improves, and both sigmoids are monotone increasing.
2. **The consuming contract.** `hybrid_decision`'s docstring: *"`p_cbes` : CBES probability
   (approval likelihood)"* — `decision_engine.py:158-159`.
3. **The tiebreak gates.** `elif p_cbes >= 0.60: decision = "APPROVE"` and
   `elif p_cbes <= 0.40: decision = "REJECT"` (`decision_engine.py:261-268`).
   High ⇒ approve. This is unambiguous.
4. **The blend and the tilt.** `p_blend = 0.75·p_ml + 0.25·p_cbes` with
   `_BLEND_ALPHA = 0.25` (`decision_engine.py:81, 205`), and `tilt = p_cbes − 0.5` lowers
   the approval bar when CBES is high (`decision_engine.py:211-213`). Both only make sense
   if higher is better.

**Empirically:** all-best applicant → 0.87034; all-worst → 0.12966.

**Where it is inverted, correctly.** Every research artifact works in *default* space, so
`research/analysis/complementarity.py:241` writes `pd_CBES = 1.0 − prob_CBES`. That is the
flip, and it is asserted consistent by a label-convention gate at
`complementarity.py:248-250`.

## 2.7 Measured performance: 0.5650 AUC

| | Value |
|---|---|
| CBES standalone AUC | **0.5650** |
| Random baseline | 0.5000 |
| XGBoost on the same rows | 0.7651 |
| Rows | **307,511**, all out-of-fold |
| Defaults | 24,825 (8.07% base rate) |
| Correlation with XGBoost's P(default) | Pearson **0.193**, Spearman **0.240** |
| Source | `reports/complementarity.json` → `hybrid_xgb_cbes_full_oof.auc_cbes_alone` |
| Script | `research/analysis/complementarity.py:426-427` |

**Independently reverified for this brief.** Recomputing
`roc_auc_score(1 − y_true, 1 − prob_CBES)` directly from
`backend/artifacts/prediction_outputs.csv` gives **0.5650424**. The column's min/max are
0.13056 / 0.87034 — matching the current engine's theoretical bounds exactly — confirming
the figure is measured on **this** five-pillar engine, not an earlier version.

### What 0.5650 means in plain words
Take one applicant who defaulted and one who did not, at random. Rank them by CBES.
CBES puts the defaulter at higher risk **56.5% of the time**. Coin-flipping gets 50%.
So CBES carries **real but weak** signal — about **13% of the way** from random to
XGBoost's 0.7651. It is not random, and it is not competitive.

Per-segment, `docs/FUTURE-SCOPE.md:91-100` records that XGBoost beats CBES in **all 16
segments**, by 0.14–0.26 AUC. There is no niche where CBES wins.

### Why the project keeps it — four reasons, in order of strength

1. **Interpretability that survives an adverse-action letter.** Every CBES output
   decomposes into five named pillars with fixed published weights and a percentile band
   per field. It is a reason code, not a post-hoc approximation of one. SHAP's attributions
   are estimates of a model's behaviour; CBES's pillars *are* the computation.
2. **8 fields vs 129.** CBES needs `credit_score`, `delinquencies`, `active_loans`, `dti`,
   `employment_tenure_years`, `annual_income`, `loan_amount`, `region`. The full Home
   Credit frame is 122 columns in `application_train` plus bureau derivations. A score
   requiring 8 collectible fields is deployable where a 129-column model is not.
3. **Portable across datasets.** Every threshold is a percentile of whatever distribution
   it is calibrated on. Point `cbes_calibration.py` at a new lender's book, re-run it, and
   the rule set transfers without retraining a model or re-deriving 129 features.
4. **Low error correlation makes it a genuine second opinion.** Pearson correlation with
   XGBoost's probabilities is only 0.193. That is *why* it was chosen as the disagreement
   signal — and also, honestly, why that architecture underperformed: at 0.5650 AUC the
   disagreement is mostly CBES being wrong.

### The thing you must not claim
`reports/complementarity.json` records that the best XGBoost+CBES blend weight is
`w_xgb = 1.0`, i.e. **the optimal amount of CBES in the ensemble is zero**, and the report
itself annotates that weight as *"chosen on the evaluation data itself — an optimistic
upper bound, not an honest estimate."* Blending CBES at 25% (`_BLEND_ALPHA`) **costs**
predictive performance. Present CBES as an interpretability and deployability artifact.
Do not present it as an accuracy contribution.

---

# 3. Known defects in the current code — disclose these first

**Status (2026-08-31):** §3.2, §3.3, §3.4 and the `active_model.txt` half of §3.5 have since
been **fixed in code**; the text below is retained as the original finding, each with a
"Fixed" line stating what changed. §3.1 is *not* a bug and remains open by design.

### 3.1 SHAP explains a different model than the one that scores
`shap.LinearExplainer(self.classifier, ...)` reads the plain pipeline
(`ml_service.py:251, 256`) while `p_ml` comes from the isotonic `CalibratedClassifierCV`
(`ml_service.py:279`). Both were fitted on the same training split with the same estimator
spec, so the *rankings* are close; but the attributions do not decompose the served
probability, and isotonic calibration is non-monotone in the *gaps* between features.
This is not accidental — `retrain_serving_model_v3.py:222-225` saves the plain pipeline
specifically so the existing SHAP code has a linear object to read. **Severity: honesty
issue, not a crash.** State it as a limitation, do not let it be discovered.

**Not fixed — inherent, not a defect.** Attributions cannot decompose an isotonically
calibrated probability; that is a property of calibration, not of this code. The point is
now stated in the code itself, at the explainer construction site
(`ml_service.py`, the comment above `shap.LinearExplainer`) and in the module header of
`explainability_service.py`, so a reader of either file meets it without needing this brief.

### 3.2 `explainability_service.py` still speaks the OLD 15-key India vocabulary
`FEATURE_LABELS` (lines 17-27) and `_counterfactual_target` (lines 39-51) key on
`emi_income_ratio`, `missed_payment_ratio`, `credit_component`, `asset_component`,
`stability_component` — names that **do not exist** in the retrained artifact's
`feature_names`. Of the live 15 features, only **three** (`cibil_score`,
`debt_to_income_ratio`, `loan_income_ratio`) are covered.

Live consequence, reproduced against the current artifact: a sample applicant's top-3 SHAP
features come back as `cibil_score`, `age`, `active_loans`. `age` and `active_loans` are
absent from the target table, so `_counterfactual_target` returns `value` itself
(line 51) → `delta = 0.0` and the UI can render *"improve Age from 35 to 35"*. Labels for
uncovered features fall back to title-case (`_to_label`, line 36), which is cosmetically
fine but confirms the vocabulary drift. **Severity: user-visible nonsense on the
recommendations panel.**

**Fixed (2026-08-31).** `FEATURE_LABELS` now covers all 15 artifact features (plus the
legacy CBES-component keys the heuristic fallback still emits), and
`_counterfactual_target` returns `float | None` off three tables: `IMMUTABLE_FEATURES`
(`age`, `dependents`, `total_loans`, `closed_loans`), `_ABSOLUTE_TARGETS` and
`_RELATIVE_TARGETS`. `None` is returned for an immutable feature, an unknown feature, a
zero base value under a relative target, or an applicant already at/past the target;
`_build_counterfactuals` skips those entries and any residual zero delta. **Coverage went
from 3/15 to 15/15, and a zero-delta suggestion can no longer be emitted.**

### 3.3 The frontend chart's x-axis is hardcoded to [−1, 1]
`FeatureContributionChart.tsx:35` sets `<XAxis type="number" domain={[-1, 1]} />`. SHAP
values here are **log-odds**, which are unbounded. The verified sample already produced
`cibil_score` at **0.5466**; a more extreme applicant will exceed 1.0 and the bar will
clip. **Severity: silent visual truncation of the largest attributions** — the exact ones
that matter most.

**Fixed (2026-08-31).** The domain is now derived from the plotted data via
`symmetricDomain()` in the same file: symmetric about zero at
`max(0.5, max|impact| * 1.15)`. Nothing clips, and the floor stops an all-small chart from
being magnified.

### 3.4 A silent, unlabelled heuristic fallback impersonates SHAP
If the explainer throws, `ml_service.py:304-305` swallows it and emits an empty list;
`explainability_service.py:61` then falls through to nine hand-written rules
(lines 93-188) built on the old vocabulary — with the same `topFactors` shape, the same
chart, and no indication to the user that these are not SHAP values. **Severity: an
examiner clicking an application cannot tell which mechanism produced the bars.**

**Fixed (2026-08-31).** The fallback is retained — it is correct defensive behaviour — but
it no longer impersonates SHAP. `_build_top_factors` returns `(factors, source)` with
`source` in `{"shap", "heuristic"}`; the flag is stamped on every factor, surfaced as
`explainability_payload["explanationSource"]` (added to `ApplicationExplainResponse`) and as
`decision.explanationSource` / `featureImportance[].source` in the application response; and
`FeatureContributionChart.tsx` renders an amber "Not SHAP" banner when it resolves to
`heuristic`. The `explanation` string also differs between the two. Separately,
`ml_service.py` now logs a warning instead of `except Exception: pass` at all three points
where SHAP can go missing (explainer construction, `shap_values`, no explainer at all).

### 3.5 Two stale artifacts that will be spotted
- **`backend/app/services/explainability_service.py:7-15`** — the NOTE claims live CBES
  scores are "silently near-constant (~0.13-0.22)". That was true before
  `customer_profile_service.resolve_application_payload` began populating CBES's
  snake_case vocabulary; it now supplies all seven scoring keys at
  `customer_profile_service.py:578-587`, and `LoanApplicationInput` uses
  `model_config = ConfigDict(extra="allow")` (`schemas.py:67`) so they survive validation.
  **The note is out of date on the `customer_id` path.** It remains accurate for a legacy
  full-form payload with no `customer_id`, which does bypass the profile merge
  (`routers/applications.py:222-232`) and does score `p_cbes = 0.12966` for everyone.
- **`backend/artifacts/active_model.txt` contained `TabPFN-2.5`**, but the v3 artifact
  carries no `all_pipelines`, so `all_model_predictions` is empty and the lookup always
  missed — the served score silently fell back to the calibrator. Behaviour was correct;
  the file was misleading. **Fixed (2026-08-31):** the file now reads `LogisticRegression`,
  which is the model actually serving, and the lookup logs a warning naming the requested
  model, the artifact, and the available models whenever a requested model is unavailable,
  instead of falling through silently.
- **`backend/training.py:30`** imports `COMPONENT_WEIGHTS, compute_cbes_probability` from
  `cbes_engine`, and **neither name exists** in the current module. That script cannot run.
  It does not affect serving, and — verified in §2.7 — the committed
  `prediction_outputs.csv` was nonetheless produced with the *current* engine, so the
  0.5650 figure stands.

---

# 4. Questions you will be asked

### "Isn't CBES basically random?"
No, but it is close, and I will give you the number rather than a characterisation. AUC
**0.5650** on all 307,511 out-of-fold rows against 0.5 for random — about 13% of the way
from a coin flip to our XGBoost baseline at 0.7651. It carries real signal and I would not
underwrite on it. It is in the system for interpretability, for its 8-field data
requirement, and because it re-calibrates onto a new lender's book by re-running one
script. We also measured what it costs: the optimal blend weight for CBES in an
XGBoost ensemble is zero, and we report that rather than hiding it.

### "Does SHAP explain the model or the applicant?"
The model — strictly. SHAP answers *"why did this model output this number for this row"*,
decomposing the prediction against a baseline of 100 training rows. It makes no causal
claim about the applicant. If the model is wrong, SHAP faithfully explains a wrong answer.
Ours has held-out AUC 0.6919, so it explains a modest model honestly. There is a second
layer I should flag: SHAP reads the plain logistic regression inside the artifact, while
the served probability comes from the isotonic-calibrated version, so the attributions
rank the drivers correctly but do not sum to the served probability.

### "How do you know the CBES weights are right?"
**We don't, and nothing in this repository validates them.** 0.35/0.30/0.20/0.10/0.05 are
hand-designed to reflect standard credit-underwriting priority — credit history first,
affordability second, existing exposure third — and the sub-weights within each pillar are
the same kind of judgement. There is no ablation, no per-pillar AUC, and no weight
sensitivity analysis in `reports/`. A diagnostic pass to test them against the real data is
specified in
`docs/superpowers/plans/2026-08-30-cbes-diagnostics-explanation-relearning-capture.md` and
has **not been built**. What *is* data-derived is the calibration layer: the percentile
breakpoints, from the real Home Credit distribution. The weights on top of them are prior
belief, and given the 0.5650 AUC I would not defend them as optimal — only as legible.

### "Why a sigmoid at all? Why k=4 and k=5?"
The shape is standard: mapping a rank-based aggregate through a monotone squashing function
is the FICO base-score/PDO pattern, and that is citable. The specific constants are not.
Our own literature review recorded that finding explicitly — no source prescribes a
squashing steepness, so `k=4` and `k=5` are engineering choices. Their measurable effect is
that `p_cbes` is confined to [0.130, 0.870] with a population mean of 0.613, while
calibrated `p_ml` sits near 0.92. That ~0.31 offset is what broke our disagreement-based
deferral router, and diagnosing it is one of our results.

### "Why default missing fields to the worst value instead of the mean?"
Because mean-imputation converts absence of evidence into evidence of average
creditworthiness. An applicant with no bureau file would receive an average credit pillar.
Worst-case imputation makes missingness costly and therefore visible. The same `DEFAULTS`
dict is used at training time and at inference, so a missing value means the same number in
both places — an empty payload scores exactly 0.1297, the theoretical floor.

### "Does a positive SHAP value mean the applicant is good?"
No — and the sign flips once between the model and the screen, so let me be exact. Raw SHAP
is in P(default) space: **positive pushes toward default, toward reject**. Before display,
`_impact_sign_for_decision` re-signs it to mean *"supported the decision we made"*, negating
it for APPROVE and DEFER. So a green bar on screen means *"this feature supported the
decision shown"*, not *"this is good for the applicant"* — on a rejected application, a
green bar is a reason for the rejection.

### "Why TreeExplainer versus LinearExplainer — did you choose, or did it choose you?"
It is selected automatically by a substring test on the artifact's model name
(`ml_service.py:255`), and it resolves to `LinearExplainer` because the served model is a
logistic regression. That is the correct explainer for it: exact, closed-form, no sampling
noise, and cheap enough to run on the request path. The `TreeExplainer` branch is currently
dead code, retained because the multi-model training path can select XGBoost, LightGBM or
CatBoost. I would call the substring test the weak point — it is a string match, not a type
check.

### "SHAP is exact for linear models. So what's left to go wrong?"
Correlated features. `annual_income` and `monthly_income` are collinear by construction,
`total_loans = active_loans + closed_loans`, and `debt_to_income_ratio` shares terms with
`loan_income_ratio`. `LinearExplainer` attributes to each independently, so how credit
splits between two collinear columns depends on where the fitting procedure put the
coefficient. Which of the income columns appears in the top-3 is not a substantive finding,
and I would not build an adverse-action reason code on that distinction without grouping
them first.

### "Show me a real explanation."
Live, on the current artifact, for a 35-year-old with `cibil_score` 750,
`debt_to_income_ratio` 0.15 and a ₹800k request: `p_ml = 0.9587`, `p_cbes = 0.2870`,
decision **DEFER**. Top-3 SHAP: `cibil_score` +0.547, `age` +0.155, `active_loans` +0.108 —
all positive, i.e. all pushing toward default relative to the training baseline. And note
the decision: `|0.9587 − 0.2870| = 0.67 > τ_d = 0.30`, so the legacy disagreement gate
fires. That single case is the scale-mismatch pathology in §2.3, reproducible on demand.

### "Only three features? Why not all fifteen?"
Three is a truncation at `ml_service.py:300` (`.head(3)`), taken by absolute SHAP value, to
keep the payload small and the request path fast. The presentation layer and the chart
component are both sized for five, so the honest description is "top 3 by |SHAP|, in a
component that would display up to 5". Ranking by absolute value means the three shown are
the most *influential*, not the most *negative* — a favourable feature can occupy a slot on
a rejected application.

### "What is the baseline, exactly?"
The mean model output over 100 rows sampled from the training split with
`random_state=42`, stored in the artifact as `background_data` with shape (100, 15), and
already `StandardScaler`-transformed by the same scaler used at inference — so explanation
space and background space match. For a linear model the background only supplies
`E[x_i]`, which is stable at n=100; that is one of the reasons the linear choice is
comfortable here.

### "Your CBES has a hardcoded `[1,1,2,3,3]` in it. Explain."
That is the region pillar, weight 0.05. `REGION_RATING_CLIENT` is an ordinal 1–3 rating, so
percentiles of it are meaningless; the breakpoints are written directly instead of loaded
from the calibration artifact. It is the only pillar not data-calibrated, it carries 5% of
the weight, and the code comment states it is a urbanicity proxy and explicitly not a
geography claim.

### "Thirty percent of your credit pillar is a step function. Did you know?"
Yes, and the calibration script prints a warning when it happens
(`cbes_calibration.py:53-60`). All five `delinquencies` percentiles collapse to 0.0 because
most Home Credit applicants have none, so the sub-score is 1.0 for zero delinquencies and
0.0 for any — 10.5% of the total score behaving as an on/off switch. `active_loans` is
nearly as coarse, with p30 = p50 = 1.0. That is a property of the data, and we report it
rather than smoothing it away.

---

## Appendix — reproduce every number in this brief

```bash
# CBES bounds and the empty-payload floor
python -c "from backend.app.services.cbes_engine import compute_cbes; print(compute_cbes({}))"
# -> (0.12965807146502517, {...})

# The served artifact: explainer type, features, model
python -c "import joblib; p=joblib.load('backend/artifacts/pipeline_v3_real.joblib'); \
print(p['model_name'], len(p['feature_names']), p['background_data'].shape, p['test_metrics'])"

# CBES standalone AUC, recomputed from the OOF predictions
python -c "import pandas as pd; from sklearn.metrics import roc_auc_score; \
d=pd.read_csv('backend/artifacts/prediction_outputs.csv',usecols=['prob_CBES','y_true']); \
print(roc_auc_score(1-d['y_true'], 1-d['prob_CBES']), d['prob_CBES'].min(), d['prob_CBES'].max())"
# -> 0.5650424316515033 0.130561... 0.870342...

# Regenerate the CBES percentile thresholds from the real data
python -m backend.app.services.cbes_calibration
```

### One-page file map

| Concern | File |
|---|---|
| SHAP computation | `backend/app/services/ml_service.py:250-305` |
| SHAP presentation / re-signing | `backend/app/services/explainability_service.py:30-91` |
| SHAP → API response | `backend/app/services/decision_service.py:56-64` |
| SHAP → UI | `frontend/src/pages/ApplicationReview.tsx:243`, `frontend/src/components/sections/FeatureContributionChart.tsx` |
| CBES formula | `backend/app/services/cbes_engine.py:93-160` |
| CBES calibration | `backend/app/services/cbes_calibration.py` |
| CBES thresholds artifact | `backend/artifacts/cbes_thresholds.json` |
| CBES direction contract | `backend/app/services/decision_engine.py:158-159, 261-268` |
| CBES measured AUC | `reports/complementarity.json`, `research/analysis/complementarity.py:426-427` |
| Serving model provenance | `backend/retrain_serving_model_v3.py`, `reports/serving_model_retrain.json` |
