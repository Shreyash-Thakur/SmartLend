# SmartLend — Valid Deferral Guarantees Under Credit Selection Bias

**Design specification**
**Date:** 2026-08-18
**Status:** Approved for implementation planning
**Horizon:** ~32 weeks (full academic year)

---

## 1. Context

SmartLend currently implements a hybrid loan decisioning system: a calibrated ML
model, a hand-designed five-pillar heuristic (CBES), and an abstention rule that
defers to a human underwriter when the two disagree by more than a threshold
`TAU_D = 0.43`.

Three problems block the project from being research-grade.

**The data is self-generated.** `backend/generate_indian_loan_dataset.py` draws
`default_risk` from a logistic model written by hand (`RISK_INTERCEPT = -2.85`)
and injects random label flips. The reported 71% AUC is the noise floor built
into the generator, not a finding. No result computed on it is defensible.

**The system underperforms its own baseline.** `artifacts/calibration_report.txt`
records baseline LogisticRegression at 70.5% balanced accuracy and the hybrid at
62.7% accuracy on the 74.5% of cases it does not defer. A selective classifier
that abstains on a quarter of cases should be *more* accurate on the remainder,
not less. The deferral rule is currently anti-correlated with difficulty — it
preferentially defers easy cases. This is worse than random abstention.

**The abstention rule carries no guarantee.** `TAU_D` was tuned to hit a target
deferral rate. It is a fitted constant, not a method.

This specification replaces all three.

## 2. Thesis

> Conformal deferral guarantees in credit scoring are attainable only on the
> policy-overlap region. Outside it — the applicants historical policy
> deterministically rejected — no observational method can deliver coverage, and
> the standard correction silently claims otherwise. We partition the applicant
> space by identifiability, quantify how far the guarantee degrades across four
> datasets, and give valid guarantees where they hold.

### The argument

1. Conformal prediction's coverage guarantee requires **exchangeability** between
   calibration and test data.
2. Credit data violates this **by construction**. Repayment is observed only for
   approved applicants; rejected applicants never acquire a label. This is the
   *reject inference* problem — an established credit-scoring literature entirely
   separate from the "reject option / abstention" literature despite the name
   collision.
3. Therefore a conformal deferral system calibrated on approved-only data
   produces a guarantee that **appears valid on the sample it can measure and is
   void on the population it serves**. The failure is silent.
4. The apparent fix — weighted conformal prediction (Tibshirani et al., 2019)
   with reject-inference propensity weights — **is not itself novel**. Weighting
   by acceptance probability is textbook reject inference ("augmentation"), and
   conformal prediction for MAR missing outcomes is an established reduction to
   covariate shift (arXiv 2403.04613). The composition of two known results is
   not a contribution.
5. **The fix also does not work.** Weighted conformal requires **positivity**:
   propensity bounded away from zero. Real lending policies use *deterministic
   cutoffs* (e.g. CIBIL < 600 → auto-reject), so `P(approved | x) = 0` exactly on
   that region and the weight `1/P̂` diverges. This is a **structural** positivity
   violation, which renders the target *unidentifiable* — not merely noisy.
6. So the correction is silently invalid **precisely on the applicants it was
   meant to protect**: thin-file, low-score, historically excluded. That is the
   finding.

### The genuine gap

Four tools exist. The assembly does not.

| Region | Tool | Status |
|---|---|---|
| Full overlap | Standard weighted conformal | Established |
| Limited overlap | Overlap weights / trimmed estimands | Established in causal inference |
| **Zero overlap** | Manski-style partial identification bounds (valid without positivity, since outcomes are bounded) | Established; never applied here |
| **At the cutoff** | **Regression discontinuity** — the canonical tool for deterministic thresholds | Heavily used on FICO cutoffs; never combined with conformal |

RD is the leverage point. Credit cutoffs are the textbook RD setting, manual
overrides make the boundary fuzzy (yielding genuine *local* overlap), and RD is
**native to finance and OR journals** — it reads as domain sophistication rather
than imported ML machinery.

## 3. Contributions

**Paper title (working):** *Where Deferral Guarantees Hold: Conformal Prediction
for Credit Scoring Under Structural Positivity Violations*

**Strategy:** hybrid — **empirical lead, bounded theory.** The measurement study
is publishable on its own; the theory is scoped to what can be verified by
experiment plus supervisor review, not to what would require original
partial-identification results.

| # | Contribution | Status |
|---|---|---|
| **C1** | Empirical demonstration that conformal deferral guarantees degrade under credit selection bias, quantified across four datasets | **Novel — the core, and empirical** |
| **C2** | Characterisation of where positivity *structurally* fails in credit, and demonstration that the standard weighted-CP correction diverges exactly there | **Novel — the theoretical core, bounded** |
| **C3** | RD-based local identification recovering valid guarantees near policy cutoffs | **Novel — theory, leverages supervisor strength** |
| C4 | Validation on synthetic ground-truth propensity + MNAR sensitivity analysis | Supporting, necessary |
| C5 | Profit-based (EMP) evaluation | Table stakes, not a claim |
| C6 | Group-conditional coverage and deferral-burden audit | Table stakes, not a claim |
| C7 | ML↔heuristic disagreement as a nonconformity score | Secondary, falsifiable |
| C8 | Document/ASR extraction uncertainty propagation | Out of paper scope; system/demo only |
| — | Manski-style partial-identification bounds on the zero-overlap region | **Stretch / future work — do not commit** |

Two scoping decisions that protect the timeline:

- **C5 and C6 are not claims.** Both were framed as novel in earlier drafts. They
  are prerequisites for credibility in this literature; presenting them as
  contributions invites rejection.
- **Partial identification stays out of the committed scope.** It is the natural
  completion of the argument and should be named as future work, but attempting
  original bounds inside the project timeline is the single largest risk to
  finishing. If C1–C4 land early, revisit.

### Why the paper still works if the theory stalls

C1 alone — a rigorous multi-dataset measurement of how badly conformal deferral
guarantees degrade under realistic credit selection bias — has not been done and
is publishable at ESWA tier. C2 is a characterisation, provable by construction
and verifiable by simulation. Only C3 requires genuine econometric machinery, and
it is the component the supervisor is strongest on. The dependency ordering is
deliberate: **each contribution stands without the ones after it.**

### Honest novelty accounting

Overclaiming is the fastest route to a bad viva. What is established, and cited
rather than claimed:

- Conformal prediction (Vovk, Gammerman, Shafer)
- Weighted conformal prediction under covariate shift (Tibshirani et al., 2019)
- Mondrian / group-conditional conformal prediction (implemented in MAPIE)
- Learning-to-defer and selective classification
- Reject inference in credit scoring
- Profit-based evaluation of credit scorecards — Expected Maximum Profit
  (Verbraken, Bravo, Weber & Baesens, EJOR 2014)
- **Selective classification magnifies group disparities** (Jones, Sagawa, Koh,
  Kumar & Liang, ICLR 2021). This paper already establishes the core
  fairness-of-abstention phenomenon. C6 extends it into credit scoring, adds
  distribution-free guarantees, and proposes a remedy — it does not claim the
  phenomenon.
- **Cost-sensitive conformal abstention with human-review budgets**
  (arXiv 2607.27143). Combines Mondrian CP, cost-controlled abstention and
  human-in-the-loop deferral, benchmarked across domains *including credit
  scoring*. Profit-aware conformal deferral is therefore **occupied**; it is
  table stakes for this paper, not a contribution.
- **Reject-inference reweighting by acceptance probability** — textbook, known as
  *augmentation*, decades old. The "propensity model" is not novel machinery.
- **Conformal prediction for MAR missing outcomes reduces to conformal under
  covariate shift** (arXiv 2403.04613). The reduction is established.
- Positivity violations, overlap weights, trimmed estimands and Manski bounds —
  all established in the causal inference literature.
- Regression discontinuity at credit score cutoffs — established in empirical
  finance.

### Why the remaining claim survives a crowded literature

Cost-sensitivity, Mondrian fairness and human-review budgets are
**domain-general** — they transfer across healthcare, fraud and credit, which is
why several groups have already published them.

**Two things do not transfer, and both are structural to lending:**

1. **Reject inference.** Repayment is observed only for applicants a prior policy
   approved. Radiology and fraud detection have no equivalent.
2. **Deterministic approval policy.** Credit decisions are made by hard score
   cutoffs, producing *structural* rather than random positivity violations. Most
   domains where weighted conformal is applied have stochastic or at least
   overlapping treatment assignment.

The combination means the standard toolkit fails here in a way it does not fail
elsewhere. That is the moat — not the tools, but the specific way they break.

CBES itself is an engineering contribution, not a research one, and must be
described that way.

### Positioning against adjacent work

*The Illusion of Improvement: Reject Inference Strategies in Credit Scoring*
(arXiv 2606.18479) shows reject-inference methods producing accuracy gains while
recall collapses. Read in Phase 0; the paper is adjacent to C1 and the
positioning must be explicit.

## 4. Data foundation

| Dataset | Size | Role | Key property |
|---|---|---|---|
| Home Credit Default Risk | 307K + 6 relational tables | **Primary** | Explicit `CODE_GENDER`; rich relational features |
| Lending Club | ~533K resolved | External validity | `grade`/`sub_grade` is an *observed lender decision* — directly models the selection mechanism |
| Indian CIBIL (Kaggle) | ~51K | India framing | Real CIBIL, DPD, utilisation; maps onto CBES pillars |
| Existing synthetic generator | 25K | **Controlled testbed** | Ground-truth propensity is known |

Phase 0 begins with Home Credit.

### The synthetic generator is repurposed, not discarded

It becomes the only dataset where the selection mechanism is **known**, and
therefore the only place the weighted correction can be *verified* rather than merely
applied. Required modification: simulate an explicit approval policy, discard
labels for rejected applicants, retain ground-truth propensity for validation.

### Data discipline (non-negotiable)

- A locked test split per dataset, opened once, at the end.
- Fixed seeds; every reported number traceable to a logged MLflow run.
- No threshold may be tuned on test data. `TAU_D` was tuned to hit a target
  deferral rate; that failure mode is what this rule exists to prevent.

### Expected effect on headline numbers

AUC will likely *fall* initially. 71% on self-generated data is meaningless;
~75–78% on Home Credit is real. This is correct and must be stated plainly in
the report.

## 5. Canonical schema and field matching

The four datasets have incompatible schemas. The resolution is a deliberate
asymmetry rather than full harmonisation.

- **CBES consumes only the canonical core** (~12 fields available across all
  datasets) — remaining portable, interpretable and cross-dataset comparable.
- **ML models consume full native features** (Home Credit's 120+ columns and
  relational aggregates) — no performance handicap.

### Why the asymmetry is principled

If both components saw identical features, their disagreement would measure only
model-class differences (linear vs. trees) — uninteresting. Because CBES encodes
a *portable domain prior* and the ML model exploits *dataset-specific patterns*,
their disagreement measures something meaningful: **learned patterns
contradicting domain knowledge**. This is what makes C7 defensible.

### Canonical core mapping

| Canonical field | Home Credit | Lending Club | CIBIL | Synthetic |
|---|---|---|---|---|
| `age_years` | `-DAYS_BIRTH/365` | absent | `age` | `age` |
| `annual_income` | `AMT_INCOME_TOTAL` | `annual_inc` | native | `annual_income` |
| `loan_amount` | `AMT_CREDIT` | `loan_amnt` | native | `loan_amount` |
| `installment` | `AMT_ANNUITY` | `installment` | native | `emi` |
| `credit_score` | `EXT_SOURCE_1/2/3` (proxy) | `fico_range_low` | `cibil_score` | `cibil_score` |
| `dti` | derived | `dti` | native | `debt_to_income_ratio` |
| `employment_tenure` | `-DAYS_EMPLOYED/365` | `emp_length` | native | `years_employed` |
| `credit_utilization` | `credit_card_balance` agg | `revol_util` | native | `credit_utilization_ratio` |
| `delinquencies` | `bureau` agg | `delinq_2yrs` | `DPD` | `missed_payments` |
| `active_loans` | `bureau` active count | `open_acc` | native | `active_loans` |
| `target` | `TARGET` | `loan_status` → binary | native | `default_risk` |

### Implementation

A declarative registry, not scattered conditionals:

```python
@dataclass(frozen=True)
class FieldSpec:
    canonical: str
    source: str | Callable          # column name or derivation
    unit: str                       # currency / years / ratio / score
    scale: Literal["raw", "percentile"]
    availability: Literal["native", "derived", "proxy", "absent"]
```

Each dataset ships a coverage report stating which canonical fields it can
populate and by what means.

### Missingness rule

Missing values become an explicit `NULL` **plus a missingness indicator column**.
Never a magic default.

`map_to_application_schema` in `backend/app/services/parser_service.py` currently
substitutes `cibil_score = 650`, `monthly_income = 50000` and similar when
extraction fails. The model then consumes fabricated values as fact. This is also
a **fairness defect**: thin-file applicants (young, rural, first-time borrowers)
have systematically more missing fields, so imputation defaults concentrate on
them. Missingness indicators convert a hidden bias into a measurable one. This
defect is the bridge into C8.

### Currency

Monetary fields are not comparable across ₹ and $. Cross-dataset comparisons use
**within-dataset percentiles**; raw units are retained for within-dataset
modelling.

## 6. Method

### C1 — Measuring the degradation (empirical core)

Calibrate a standard conformal deferral system on approved-only data, then
measure empirical coverage on (a) the approved-only sample and (b) the full
applicant population. **The gap between them is the paper's opening figure.**

Repeat across all four datasets and across selection severities. Also benchmark
the standard reject-inference schemes (augmentation/reweighting, parcelling,
fuzzy augmentation, Heckman two-stage) to establish how much each recovers — a
comparison the literature has not run against a *coverage* criterion.

**Propensity model.** Train `P(approved | x)` on the approve/reject decision
rather than on default. Lending Club's `grade`/`sub_grade` provides an observed
selection signal; Home Credit requires a modelled policy. Not novel machinery —
it is the textbook augmentation weight — but required as the comparison arm.

**Weighted calibration** (the arm shown to fail):

```
w(x)      = 1 / P̂(approved | x)
threshold = weighted_quantile(scores, weights=w, level=1-α)
```

### C2 — Characterising the structural positivity violation (theoretical core)

Partition the covariate space by identifiability:

| Region | Condition | What is attainable |
|---|---|---|
| Overlap | `P̂(approved\|x) > ε` | Valid weighted conformal coverage |
| Limited overlap | `0 < P̂ ≤ ε` | Coverage with inflated variance; overlap-weighted estimand |
| **Zero overlap** | `P̂ = 0` (policy cutoff region) | **No guarantee is attainable from observational data** |

Deliverables: an operational test for which region an applicant falls in, the
divergence rate of `1/P̂` as the cutoff is approached, and simulation confirming
that reported coverage in the zero-overlap region is spurious regardless of
weighting scheme. This is provable by construction and checkable by simulation —
which is why it is safe to commit to.

### C3 — RD-based local identification (theory, supervisor-supported)

Credit cutoffs are the canonical regression-discontinuity setting. Manual
overrides make the boundary **fuzzy**, producing genuine *local* overlap in a
neighbourhood of the cutoff where the propensity is strictly interior.

Approach: estimate the local outcome distribution just below the cutoff via fuzzy
RD, use it to extend valid conformal calibration into a bandwidth around the
threshold, and report how much of the previously-unguaranteed population is
recovered. Bandwidth selection and the usual RD diagnostics (density continuity,
covariate smoothness, placebo cutoffs) follow standard practice.

This is the component to develop with the supervisor, and the one to descope
first if it stalls — C1 and C2 do not depend on it.

**Prediction sets.** Build `C_α(x) ⊆ {approve, reject}`. A singleton means
decide; an ambiguous `{0,1}` or empty set means defer. Abstention now follows
from the geometry rather than a tuned constant.

### C4 — Validation and MNAR sensitivity

Verify the correction on the synthetic generator, where the
propensity function is known by construction, before trusting it where it cannot
be checked. Includes the MNAR sensitivity analysis described in §14.

### C5 — Profit-based evaluation

Specified in §8 (Evaluation protocol) rather than here, since it is a metric
rather than a method. Mandatory for journal fit.

### C6 — Fairness of abstention (table stakes, not a claim)

Marginal coverage does not imply group-conditional coverage. A system can meet
90% coverage overall while systematically routing rural, thin-file or female
applicants into the defer queue — and a human queue is not a neutral outcome; it
means slower decisions and more discretionary judgment for those least able to
absorb either.

Measured:
- Deferral burden disparity — `|P(defer | G=g) − P(defer | G=g′)|`
- Selective accuracy parity — per-group accuracy on non-deferred cases
- Group-conditional coverage gap — per-group empirical coverage vs. nominal `1−α`

Remedy: Mondrian CP (per-group calibration), reporting the efficiency cost of
equalising deferral burden.

Note: the paper's central figure is the **coverage gap between approved-only and
full-population calibration** (C1), not the fairness trade-off. Fairness results
support the main claim by showing who the invalid guarantee was failing.

Available protected attributes: gender (Home Credit `CODE_GENDER`), age band,
region/urbanicity. Regulatory framing is the **EU AI Act** (credit scoring =
high-risk) and **RBI digital lending guidelines** — not US disparate impact,
which was rolled back in 2025.

### C7 — Disagreement-augmented nonconformity

```
s'(x, y) = s(x, y) + λ · D(p_ml, p_cbes)
```

Test whether `λ > 0` improves the risk–coverage curve over plain conformal.
**This is falsifiable**: `λ` may not help, and reporting that is a legitimate
result. Falsifiability is what distinguishes this from advocacy.

### C8 — Extraction uncertainty propagation (system only, not in the paper)

```
Document → render → degrade → extract (LayoutLMv3 / Donut / VLM)
         → per-field confidence distribution (not point estimates)
         → Monte-Carlo propagation through the risk model
         → extraction uncertainty enters the conformal score
```

Hypothesis: decisions made on poorly-extracted documents are systematically
overconfident; incorporating extraction uncertainty restores calibration and
redirects deferrals toward genuinely illegible cases.

Synthetic *documents* rendered from *real* records is standard practice and
defensible. Extractors are pretrained/benchmarked on public DocILE, CORD, SROIE
and FUNSD before domain adaptation.

**Modality-agnostic interface.** The uncertainty interface is designed from day
one to accept any modality, so ASR can slot in without redesign.

### Voice (Phase 5 stretch, cut-able)

ASR supplies word-level confidence exactly as OCR does, making voice a third
modality feeding one framework. `ai4bharat/indic-conformer-600m-multilingual`
(22 Indian languages, MIT) for intake; ElevenLabs for spoken decision output.

Sharp secondary result available here: voice intake exists to serve low-literacy
and rural applicants, but ASR error rates are *higher* for exactly those speakers
(accent, dialect, noise, code-switching) — so the modality intended to include
them may systematically raise their deferral rate.

## 7. Models

**Credit baselines**
- **WOE + logistic scorecard** (`optbinning` / `scorecardpy`) — the credit
  industry standard. Omitting it is the most common reason credit-scoring papers
  are rejected. Mandatory.
- XGBoost, LightGBM, CatBoost — the accuracy bar
- `Prior-Labs/TabPFN-v2-clf` — posterior may give a better abstention signal
- FT-Transformer (`rtdl` / `pytorch-tabular`)

TabPFN caps at ~10K rows (50K for v2.5) against Home Credit's 307K. Mitigation is
retrieval-based context selection via `sentence-transformers`. Positioned as
engineering, not a claim — resampling for tabular foundation models in credit
risk is already explored (arXiv 2605.18635).

**Selective baselines — must be beaten at matched coverage**
softmax threshold, predictive entropy, MC-dropout, deep ensembles, and the
existing `TAU_D` rule. Beating a full-coverage model while abstaining on 25% of
cases proves nothing.

**Libraries**

| Purpose | Tooling |
|---|---|
| Conformal | MAPIE (Mondrian built in), crepes, TorchCP |
| Fairness | Fairlearn, AIF360 |
| **Regression discontinuity** | **`rdrobust`** (Calonico–Cattaneo–Titiunik; canonical, Python port available) — bandwidth selection, bias correction, robust CIs |
| Selection models | `statsmodels` for Heckman two-stage; `linearmodels` for panel/IV variants |
| Profit metrics | `EMP` (R; call via `rpy2` or reimplement — verify against the R output either way) |

Note: the `EMP` reference implementation is R-only. Reimplementing it in Python is
acceptable but the values **must** be checked against the R package on a shared
sample, or the profit numbers are unverifiable.

**Documents:** `microsoft/layoutlmv3-base`, `naver-clova-ix/donut-base`,
Qwen2.5-VL-class.
**Speech:** `ai4bharat/indic-conformer-600m-multilingual`.
**Text:** `ProsusAI/finbert` for free-text fields.

## 8. Evaluation protocol

**Metrics.** PR-AUC (defaults are imbalanced — ROC-AUC alone is insufficient),
Brier, ECE, AURC / risk–coverage curves, selective accuracy at matched coverage,
deferral disparity, group-conditional coverage.

**Profit-based evaluation (mandatory for journal fit).** Report **EMP — Expected
Maximum Profit** (Verbraken et al., EJOR 2014; `EMP` package on CRAN) alongside
AUC throughout. Credit-scoring reviewers in OR/DSS venues read AUC-only
evaluation as naïve.

The loss structure has three terms, and the third is what makes this a lending
paper rather than a generic ML paper:

| Outcome | Cost |
|---|---|
| False approval | LGD × EAD |
| False rejection | Foregone interest income |
| **Deferral** | **Underwriter time** |

**Headline diagnostic.** Empirical coverage on approved-only data vs. full
population. **The gap is the problem C1 solves**, and it is the paper's opening
figure.

**Statistics.** Multiple seeds with confidence intervals, DeLong's test for AUC
comparisons, corrected paired t-tests.

## 9. System architecture

A hard split that does not currently exist:

```
research/                 # experiments — never imported by the API
  data/         loaders, canonical schema, locked splits, protected attributes
  models/       baselines, TabPFN, tabular transformers
  conformal/    scores, weighted calibration, Mondrian
  fairness/     disparity metrics, audits
  docs/         generation, extraction, uncertainty
  experiments/  Hydra configs + MLflow tracking

backend/app/              # serving — loads artifacts, trains nothing
```

Targeted cleanups, only where they serve the work:
- `backend/app/routers/applications.py` is 907 lines — split by concern
- `decision_engine.py` / `decision_service.py` / `ml_service.py` / `calibrate.py`
  have overlapping responsibilities — establish clear boundaries
- `backend/artifacts/pipeline_v2.joblib` is 101 MB committed to git — move to
  DVC or an artifact store
- Rewrite `README.md` early. Its current claims ("novel proprietary
  architecture", "production-grade") are contradicted by the project's own
  calibration report.

## 10. Phasing

| Phase | Weeks | Output |
|---|---|---|
| 0 · Foundation | 1–3 | Home Credit ingested, canonical schema, locked splits, MLflow, reproduced baselines, **anti-correlated deferral diagnosed**, weighted-CP reading |
| 1 · Conformal core + degradation study | 4–9 | Standard conformal deferral beating naive selective baselines; propensity model; **coverage-gap measured across four datasets and reject-inference schemes (C1)** → *publishable on its own* |
| 2 · Positivity + RD | 10–14 | Identifiability partition and divergence characterisation (C2); fuzzy-RD local identification with supervisor (C3); synthetic-ground-truth validation and MNAR sensitivity (C4) |
| 3 · Profit & fairness | 15–19 | **EMP profit evaluation (C5)**, disparity measurements, Mondrian, efficiency-cost trade-off curve (C6) |
| 4 · Paper draft | 20–23 | Ablations, figures, full draft, arXiv preprint, **journal submission** |
| 5 · Documents | 24–29 | Doc generation, extractor fine-tuning, uncertainty propagation (C8); voice if ahead of schedule |
| 6 · Integration & viva | 30–32 | Full stack wired, demo polish, viva prep; respond to reviewer comments as they arrive |

**Two independent safety lines, by design:**

1. **End of Phase 1 (week 9): a publishable paper already exists.** The
   multi-dataset degradation study stands alone. If the Phase 2 theory never
   closes, the paper becomes a pure measurement study and still submits.
2. **End of Phase 4 (week 23): the paper is submitted** — *before* the document
   pipeline is built. A Phase 5 overrun cannot threaten the publication
   milestone.

Note the ordering: the paper is drafted and submitted *before* the document and
voice work, not after. Those components serve the viva and the demonstrator, not
the paper, so scheduling them earlier would put the only hard deadline at the
mercy of the most open-ended engineering.

**Descope order under time pressure:** C8/voice first, then C7 (disagreement
score), then C3 (RD). Never C1 or C2.

## 11. Publication strategy

**Institutional requirement:** "under review" is sufficient. This removes the
need for a conference-first detour and permits a direct Q1 journal submission.

### Venue targets

| Tier | Venue | Notes |
|---|---|---|
| Stretch | European Journal of Operational Research (Q1) | *The* credit-scoring venue; 6–12 month review |
| Stretch | Decision Support Systems (Q1) | Ambitious |
| **Primary** | **Expert Systems with Applications** (Q1, IF 7.5, CiteScore 15.0) | **Realistic for solid work** |
| **Primary** | **Engineering Applications of Artificial Intelligence** (Q1) | **Realistic** |
| Secondary | Knowledge-Based Systems (Q1, IF 7.6) | Realistic |
| Fallback | IEEE Access (Q1, 4–8 week review) | Safe; ~$1,950 APC |

### Venue hygiene

"Scopus-indexed" spans EJOR down to journals delisted for fake peer review. 852
journals have been discontinued from Scopus, 56 in 2025, most commonly for
"publication concerns" — absent peer review and citation manipulation. A
publication in a delisted journal stays on a CV but stops counting.

Three rules:

1. Verify indexing on **Scopus Sources** in the week of submission — not a blog
   list, not a badge on the journal's own site.
2. **Any journal that solicits a submission by email is disqualified.** Q1
   journals do not recruit students.
3. Post an **arXiv preprint the day the draft is finished** — establishes
   priority, costs nothing, protects against slow review.

### Schedule

Phase 3 completes ~week 19 (month 4.5). Phase 4 drafts and submits by **week 23
(~month 5.5)**, with the arXiv preprint going up the same week. ESWA review runs
3–6 months, so a decision arrives around month 9–11 — well past the point where
"under review" has been satisfied, and with Phases 5–6 running in parallel rather
than blocking it.

The critical property: **submission does not depend on the document pipeline.**

## 12. Risks

| Risk | Mitigation |
|---|---|
| **RD identification (C3) does not close** | C1 + C2 already form a paper; descope C3 to "future work" and submit the measurement study plus the positivity characterisation |
| **Reviewer says the weighted-CP composition is trivial** | Agreed — and the paper says so first. The contribution is that the composition *fails*, not that it works. Frame C2 as the finding, never C1's propensity model |
| RD assumptions fail (no density continuity at cutoff, no fuzzy boundary) | Test with standard RD diagnostics in Phase 2 *before* committing; Lending Club `grade` boundaries give multiple candidate cutoffs |
| Home Credit has no observable approval cutoff | Home Credit is the outcome dataset; RD runs on Lending Club `grade`/`sub_grade` and the synthetic policy, which have explicit thresholds |
| Propensity model is unidentifiable on Home Credit | Lending Club's `grade` gives an observed decision; synthetic gives ground truth |
| `λ > 0` does not improve C7 | Negative result is reportable; C1–C4 do not depend on C7 |
| Phase 5 overruns | Explicitly cut-able; paper already submitted at week 23 |
| MAR vs. MNAR is untestable | State as an assumption, follow the credit-scoring literature's justification, test sensitivity to it (C4) |
| Scope creep back toward partial identification | Explicitly listed as future work in §3; requires a written decision to re-enter scope |

## 13. Out of scope

**Out of scope for the project entirely:**
- LLMs as risk predictors (consistently mediocre at tabular prediction)
- Deployment, authentication, production hardening

**Out of scope for the *paper*, but in scope for the system and viva:**
- Document and ASR extraction uncertainty (C8 / Phase 5). Built for the
  demonstrator; excluded from the paper to keep the contribution sharp.

**Must never be claimed as research contributions:**
- CBES itself (engineering)
- Profit-aware conformal abstention (occupied — arXiv 2607.27143)
- Fairness-of-abstention as a phenomenon (occupied — Jones et al., ICLR 2021)

---

## 14. Open question for implementation planning

MAR vs. MNAR cannot be tested, since labels for rejected applicants are never
observed. The spec assumes MAR, consistent with the credit-scoring literature's
standard justification (standardised application procedures, third-party data,
automated policy). Phase 2 must include a **sensitivity analysis** quantifying
how far results degrade as the MNAR violation grows. This is the most likely
point of reviewer attack and must be addressed pre-emptively rather than
defensively.
