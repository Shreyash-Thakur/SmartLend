# CBES Research Synthesis — Tuning Justification, Customer Explanation, Relearning Loop

**Design specification**
**Date:** 2026-08-30
**Status:** Research complete; explanation module + relearning-loop capture schema pending implementation approval
**Relationship to prior specs:** Extends `2026-08-29-home-credit-swap-design.md` (which redesigned CBES around 7 real Home Credit fields, 5-weighted-component structure, percentile thresholds). This spec is the literature backing for that design, plus two new pieces of scope requested afterward: a customer-facing explanation module, and a relearning-loop data-capture design (not a live retraining loop — deliberately not that, see §3).

## 0. Method

Five parallel literature/regulatory searches (scorecard weight calibration, deferral/reject-inference failure modes, cross-jurisdictional explainability law, SHAP-for-credit critique, human-feedback-loop bias), then one synthesis pass. 47 findings collected, cited by source title and URL throughout. Full source list in §4.

## 1. CBES tuning recommendations

Grounded against the actual code: `backend/app/services/cbes_engine.py` (5 components — Credit 35%, Capacity 30%, Behaviour 20%, Stability 10%, Region 5% — `component_sigmoid` k=4, final `p_cbes` sigmoid k=5, percentile interpolation over p10/p30/p50/p70/p90 breakpoints) and `backend/app/services/decision_engine.py`.

### Keep as-is (change would not be evidence-backed)

**1.1 Keep percentile normalization of all seven inputs — WELL-SUPPORTED.**
Rank/percentile transforms are the industry-standard input representation, not a shortcut. WOE/IV scorecard methodology operates on binned/ranked distributions rather than raw values ([An Information-Theoretic Framework for Credit Risk Modeling](https://arxiv.org/pdf/2509.09855)), classic FICO-style scaling builds on ordinal/rank-transformed variables before a monotonic points mapping ([Credit Scoring — Scorecard Development Process](https://medium.com/@yanhuiliu104/credit-scoring-scorecard-development-process-8554c3492b2b)), and percentile ranking is specifically robust to outliers because successive ranks are independent of the magnitude of the gap between them ([US Patent 10,235,344](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10235344)). This directly defends the choice already documented in `cbes_engine.py`'s docstring — EXT_SOURCE_2 has no prime/subprime cutoff, so percentile bands are the honest scale. **No change.**

**1.2 Keep the sigmoid combination — WELL-SUPPORTED as a form, NOT SUPPORTED as to the specific k values.**
Mapping a rank-based aggregate through a monotonic scaling function is exactly the FICO base-score/PDO pattern. The *shape* is defensible and citable. But `k=4` and `k=5` are hand-picked engineering choices with **no finding behind them** — no source prescribes a squashing steepness. State this plainly in the paper; do not present the constants as literature-derived.

**1.3 Do NOT switch to equal weights (20/20/20/20/20) — WELL-SUPPORTED to reject.**
Composite-indicator literature finds equal weighting "methodologically less robust" than statistically-derived or elicited expert weights, precisely because it hides the absence of any justification ([Robustness and Sensitivity of Weighting and Aggregation in Constructing Composite Indices](https://www.sciencedirect.com/science/article/abs/pii/S1470160X13000034)). Equal weighting would be a downgrade, not a neutral fallback.

**1.4 Do NOT abandon the expert-weighted design for a pure statistical model — WELL-SUPPORTED to keep.**
Basel II's IRB framework explicitly permits and sometimes favors constrained-expert-judgment rating systems over purely statistical ones where data is thin, *provided qualitative validation accompanies them* ([Approaches to the Validation of Internal Rating Systems](https://www.bundesbank.de/resource/blob/623114/536b20aaf00fe593e6dea5faee28fbfe/mL/2003-09-approaches-data.pdf), Bundesbank). This is the strongest single defense line for the project's academic defense: CBES's hand-weighted design is a recognized regulatory-grade approach, not an ad hoc shortcut.

### Diagnostics to run (evidence-backed, no code weight changes implied)

**1.5 Multicollinearity (VIF) check across the seven inputs — highest priority.**
Double counting is the documented failure mode of additive hand-built scores: correlated component inputs each carry weight for the same underlying risk factor, inflating the total ([US Patent 11,792,197](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11792197); corroborated across scorecard literature). CBES has two concrete, visible exposures:
- Capacity mixes `dti` (weight 0.60 within Capacity) and `loan_to_income` (0.40) — both debt-burden ratios, near-certainly correlated.
- Credit's delinquency sub-score (weight 0.30 within Credit) and Behaviour's `active_loans` are both bureau-derived credit-line variables.

If VIF is high, the honest response is not necessarily reweighting — it's *documenting* the overlap as a known limitation, since VIF is a diagnostic in this literature, not a prescribed correction.

**1.6 Weight sensitivity sweep, not re-tuning.**
The weight-allocation space is combinatorially astronomical (a 10-variable scorecard has billions of allocations summing to 100%), so expert weights are "almost never provably optimal" and should be treated as one defensible point in a vast space, motivating sensitivity analysis ([Using a Genetic Algorithm to Optimize an Expert Credit Rating Model](https://www.sciencedirect.com/science/article/abs/pii/S095741742200834X)). Concrete: perturb each weight ±5 and ±10 points (renormalizing), re-run against the Home Credit sample, report how much decision-mix/AUC moves. A stable result is a *strong* defense of 35/30/20/10/5; an unstable one is a finding worth reporting either way.

**1.7 Validate weights against held-out default outcomes; compare to WOE/IV ordering.**
The classical (Cooke) model of structured expert judgment shows expert-elicited weights can be empirically scored against calibration items with known outcomes ([Expert Elicitation](https://www.journals.uchicago.edu/doi/10.1093/reep/rex022)). Home Credit's `TARGET` gives exactly such calibration items. Also fit a WOE/IV pass over the same seven inputs and compare the IV-implied ordering against 35/30/20/10/5 — if Credit and Capacity dominate IV as well, the hand weights are corroborated; if Region's 5% carries more IV than Stability's 10%, that's a documented gap worth reporting.

**1.8 Write a qualitative + outcome-based validation protocol.**
Regulatory guidance holds that expert-assigned (not data-derived) weights require process-oriented qualitative review in addition to statistical validation ([Supervisory Handbook on the Validation of IRB Rating Systems](https://www.eba.europa.eu/sites/default/files/document_library/Publications/Reports/2023/1061495/Supervisory%20handbook%20on%20the%20validation%20of%20IRB%20rating%20systems%20revised.pdf), EBA). CBES's weights should ship with a written validation protocol document, not stand as self-justifying.

### Explicitly not recommended

**1.9 Replacing the weighted sum with a learned/regression-fit weighting — NOT SUPPORTED, and self-defeating.** No finding calls for it; it would collapse CBES into a second ML model and destroy the rule-engine leg of the hybrid design.

**1.10 Changing 35/30/20/10/5 to any other specific numbers — NOT SUPPORTED.** No finding endorses any particular alternative allocation. Any weight change should follow from 1.6/1.7 results, not intuition.

**1.11 Reject-inference / deferral-driven reweighting of CBES — NOT SUPPORTED, actively hazardous right now.** Reject inference can produce an "illusion of improvement" — accuracy rising while recall on defaulters collapses ([The Illusion of Improvement: Reject Inference Strategies in Credit Scoring](https://arxiv.org/abs/2606.18479)) — and this project's deferral rule is already documented as worse than random. Do not let deferral outcomes touch CBES weights until §3's gate is met.

**1.12 Deferral-rule diagnostics (context, not CBES tuning, but cheap and citable):**
(a) Evaluate accuracy on the auto-decided subset and the deferred subset *separately* against no-deferral baselines, rather than aggregate accuracy — the CoDoC validation pattern ([Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians](https://www.nature.com/articles/s41591-023-02437-x), *Nature Medicine*).
(b) Benchmark the observed defer rate against the "natural error rate" implied by the model's own AUC — a rate far above or below it signals the referral mechanism isn't isolating genuinely hard cases ([Bounds for rating override rates](https://arxiv.org/abs/1203.2287), Tasche).
(c) The general theory of why a naive assignment squanders capacity on easy cases instead of reserving the better decision-maker for hard ones: [Who Should Predict? Exact Algorithms For Learning to Defer to Humans](https://arxiv.org/pdf/2301.06197).

## 2. Customer-facing explanation module (buildable now, no SHAP, no trained model)

**Format: one outcome line + up to 4 ranked principal-reason statements + one procedural-rights line.**

**Why up to 4, not 5:** Reg B commentary states "disclosure of more than four reasons is not likely to be helpful to the applicant" — the origin of the industry's 4-reason-code convention ([Comment for 1002.9 — Notifications](https://www.consumerfinance.gov/rules-policy/regulations/1002/interp-9/), CFPB). CBES has 5 components; **rank all five by weighted shortfall and disclose the top 4** — Region (5% weight) is the natural one dropped by this rule, not by hand-picking.

**Why they must be specific and per-applicant, not templated:** Reg B (12 CFR 1002.9) requires *specific, principal* reasons — "did not meet internal standards" is explicitly insufficient. Model complexity is no excuse; [CFPB Circular 2022-03](https://www.federalregister.gov/documents/2022/06/14/2022-12729/consumer-financial-protection-circular-2022-03-adverse-action-notification-requirements-in) confirms the obligation applies fully to algorithmic decisions, and [Circular 2023-03](https://www.federalregister.gov/documents/2024/04/17/2024-08003/consumer-financial-protection-circular-2023-03-adverse-action-notification-requirements-and-proper) holds that checking boxes on the CFPB's own sample forms is *not* a safe harbor if the checked reasons don't reflect the actual per-applicant decision.

**Ranking rule (computable today from `cbes_breakdown`):**
`shortfall_i = weight_i × (1 − component_score_i)`, sorted descending, top 4 above a floor. This is CBES's own analogue of FICO's practice of ranking up to four or five codes by marginal score impact ([FICO reason codes](https://www.creditscoring.com/creditscore/fico/factors/reason-codes.html)).

**Reason code catalog:**

| Code | Component (weight) | Triggering input | Customer-facing statement |
|---|---|---|---|
| CB-01 | Credit (35%) | `credit_score` percentile | "Your credit-bureau risk score is in the lower range compared with other applicants." |
| CB-02 | Credit (35%) | `delinquencies` | "Your record shows more past missed or late payments than most applicants." |
| CP-01 | Capacity (30%) | `dti` | "Your existing debt is high relative to your income." |
| CP-02 | Capacity (30%) | `loan_to_income` | "The amount requested is large relative to your income." |
| BH-01 | Behaviour (20%) | `active_loans` | "You currently have more open credit accounts than most applicants." |
| ST-01 | Stability (10%) | `employment_tenure_years` | "Your length of employment is shorter than most applicants." |
| RG-01 | Region (5%) | `region` | "Regional risk rating recorded for your application." |

Each emitted statement carries `code`, `component`, `weight`, `component_score`, `shortfall`, and the plain-language text — stored in the decision log so the notice is auditable (undocumented decision mechanics are a model-risk deficiency under SR 11-7 framing).

**Percentile phrasing only ("compared with other applicants"), never absolute claims** — the thresholds are p10–p90 breakpoints from the Home Credit distribution, not real bank conventions; saying "your DTI is too high" would misdescribe the engine.

**Never shown to the customer:** raw `p_cbes`, `p_ml`, `p_blend`, thresholds, or the weight formula. Disclosing a mathematical formula does not satisfy GDPR's "meaningful information about the logic involved" duty (Arts. 13–15/22) — component-level plain-language reasons are the right altitude.

**Per-decision templates:**

- **REJECT** (`ml_reject`, `cbes_fallback_reject`) — full 4-reason treatment, plus a GDPR Art. 22 rights line (right to human intervention, to express a view, to contest). The CJEU's SCHUFA ruling (C-634/21) extends Art. 22 obligations upstream to the score producer when a downstream decision leans heavily on it — relevant since CBES's output feeds the hybrid engine, so the CBES leg needs its own explanation record.
- **APPROVE** (`ml_approve`, `cbes_fallback_approve`) — outcome line + top 2 *weakest* components as informational "factors that limited your assessment," explicitly labeled not-adverse-action.
- **DEFER** (`disagreement`, `low_confidence`, `grey_zone`) — must be stated as "under human review," **never** as an outcome; the three internal reason strings collapse to one external statement. This is also where EU AI Act Art. 14 human-oversight duties bite (credit scoring is Annex III 5(b) high-risk) — log every DEFER with its trigger and hand the reviewer the full CBES breakdown (the Art. 13 "technical measures" that make Art. 14 oversight possible). One caution for the paper: abstention thresholds are not automatically neutral — uncertainty-based deferral can disproportionately affect under-represented groups ([Unequal Uncertainty](https://arxiv.org/abs/2508.07872)) — log applicant segment alongside DEFER so this is measurable later.

**What gets added once a model is trained:** SHAP feature attribution over the ML leg — **as an internal input to the reason-code layer, never as raw customer-facing output.** Four reasons: (1) CFPB permits SHAP only if creditors can independently validate its accuracy, which may not be possible for less-interpretable models ([Circular 2022-03](https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/)); (2) SHAP can provably misrank importance, even ranking irrelevant features above relevant ones ([Huang & Marques-Silva 2024](https://www.sciencedirect.com/science/article/abs/pii/S0888613X23002438)); (3) correlated features — exactly CBES's §1.5 situation — are SHAP's worst case ([Explainable ML for Credit Risk Management When Features are Dependent](https://www.tandfonline.com/doi/abs/10.1080/15366367.2023.2261186)); (4) operational fragility — faithfulness gaps, KernelExplainer sampling instability, no causal guarantee ([arXiv:2604.14231](https://arxiv.org/pdf/2604.14231)). The future architecture: **SHAP → grouped into component-level significance → mapped onto the CB/CP/BH/ST/RG codes above → top 4 emitted** — the same pattern as [US Patent 12,050,975](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12050975)'s grouped-SHAP adverse-action codes. Optionally add an LLM rendering layer over already-validated codes ([arXiv:2409.00079](https://arxiv.org/pdf/2409.00079)) — never as the attribution step itself. On Home Credit, expect SHAP to surface EXT_SOURCE_2/3 and loan/goods-amount as dominant ([Interpretable Credit Default Prediction with Ensemble Learning and SHAP](https://arxiv.org/pdf/2505.20815)) — which maps cleanly onto CB-01/CP-02, a good consistency check against CBES's own top components.

## 3. Relearning-loop design (data capture only — no retraining trigger)

### Hard precondition

**Nothing may be retrained on deferred-case labels until the deferral rule is independently validated as better-than-random at selecting hard cases.** Retraining on cases a broken router selected is the textbook runaway feedback loop — a system using its own past decisions to choose its retraining data reinforces its initial bias regardless of the true underlying rate ([Ensign et al., Runaway Feedback Loops in Predictive Policing](https://arxiv.org/pdf/1706.09847)); more generally, a routing component determining which cases get new labels produces a non-representative, self-reinforcing sample ([A Classification of Feedback Loops and Their Relation to Biases in Automated Decision-Making Systems](https://dl.acm.org/doi/fullHtml/10.1145/3617694.3623227), FAccT 2023). This project's deferral rule is already documented as worse than random — retraining on its output now would amplify an actively harmful selection policy.

Two further reasons deferred-case labels aren't clean ground truth even once the router works: **selective labels** — outcomes are observed only for the approved subset, so training/evaluating on them yields biased risk estimates over the full population ([Lakkaraju et al., The Selective Labels Problem](https://cs.stanford.edu/~jure/pubs/contraction-kdd17.pdf), KDD 2017; [Kleinberg, Lakkaraju et al., Human Decisions and Machine Predictions](https://cs.stanford.edu/people/jure/pubs/bail-qje17.pdf), QJE) — and **human labels carry their own bias** — feeding reviewer judgments back as training labels can cause non-convergent training and amplify human bias if treated as ground truth ([Designing Closed Human-in-the-loop Deferral Pipelines](https://arxiv.org/pdf/2202.04718); [Madras et al., Predict Responsibly](https://arxiv.org/pdf/1711.06664), NeurIPS 2018), and reviewer attention degrades as the model gets *more* accurate, since scrutiny rarely changes the outcome ("Human in the Loop AI Oversight Design").

### Build now: capture only

Safe today regardless of the router's state — logging changes nothing about model behavior, and without it there's no dataset for later.

**Schema — `deferred_review` (one row per DEFER routed to a human):**

| Field | Notes |
|---|---|
| `review_id`, `application_id` (SK_ID_CURR), `created_at` | keys |
| `decision_reason` | verbatim `disagreement` / `low_confidence` / `grey_zone` |
| `p_ml`, `p_cbes`, `p_blend`, `disagreement`, `confidence`, `t_approve`, `t_reject` | full engine state at defer time |
| `cbes_breakdown_json` | the 5 component scores |
| `engine_version`, `threshold_artifact_hash`, `t_base` | lets a future analysis separate a fixed router's labels from a broken router's |
| `reviewer_id`, `reviewed_at`, `time_spent_seconds` | supports the reviewer-attention-degradation check |
| `human_decision` (APPROVE/REJECT), `human_reason_codes[]`, `human_free_text` | |
| `agreed_with_engine` (bool), `override_direction` | feeds override-rate monitoring (SR 11-7 requires this be tracked) |
| `reviewer_confidence` (1–5) | for modeling reviewer consistency (Madras et al.) |
| `applicant_segment_json` | for the disparate-deferral check |
| `realized_outcome`, `outcome_observed_at`, `outcome_censored` (bool) | nullable; explicitly encodes MNAR-ness |
| `exploration_flag` (bool) | see below |

**Also capture a control arm — `exploration_flag`.** Route a small random 2–5% of would-be-auto-decided applications into human review anyway. Small deliberate exploration rates of 2–5% are sufficient to diagnose, at near-zero cost, whether a rejection/deferral mechanism is deteriorating in its ability to screen out true defaulters ([The Illusion of Improvement](https://arxiv.org/abs/2606.18479)). This is the single most valuable thing to build now — the only source of *un-selected* labels, the escape hatch from the selective-labels trap, and what will eventually prove the deferral rule was fixed.

**Also log per batch:** decision-mix counts, override rate, accuracy on the auto-decided and deferred subsets separately against a no-deferral baseline (CoDoC pattern), and the AUC-implied natural override rate (Tasche).

### Explicitly do not build yet

- No automatic retraining trigger, scheduled or volume-based.
- No treating `human_decision` as a training label anywhere.
- No reject-inference imputation of outcomes for deferred/rejected cases.
- No online/continuous learning — scorecards are conventionally rebuilt periodically (often annually) as population-drift risk management ([Exploring Population Drift on Consumer Credit Behavioral Scoring](https://link.springer.com/chapter/10.1007/978-3-319-33003-7_7)); naive continuous retraining without safeguards can itself introduce instability rather than correct drift ([Fair and Explainable Credit-Scoring under Concept Drift](https://arxiv.org/pdf/2511.03807)).
- No CBES weight updates driven by deferral outcomes (§1.11).

### Gate to open the loop (all four must hold)

1. Deferral rule beats random at isolating hard cases (separate accuracy, automated vs. deferred subsets, against no-deferral baselines — CoDoC pattern).
2. Observed override/defer rate sits within the AUC-implied natural-rate bound (Tasche).
3. The exploration arm has accumulated enough un-selected labels for an unbiased evaluation set.
4. The retraining design explicitly models the missingness/selection mechanism (cf. RMT-Net) and reviewer bias/consistency (Madras et al.), rather than treating human decisions as ground truth.

Then rebuild periodically as a versioned scorecard redevelopment — never continuously.

## 4. Full citation list

- [An Information-Theoretic Framework for Credit Risk Modeling](https://arxiv.org/pdf/2509.09855)
- [Credit Scoring — Scorecard Development Process](https://medium.com/@yanhuiliu104/credit-scoring-scorecard-development-process-8554c3492b2b)
- [Rank-based score normalization framework (US Patent 10,235,344)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10235344)
- [Approaches to the Validation of Internal Rating Systems (Bundesbank)](https://www.bundesbank.de/resource/blob/623114/536b20aaf00fe593e6dea5faee28fbfe/mL/2003-09-approaches-data.pdf)
- [Supervisory Handbook on the Validation of IRB Rating Systems (EBA)](https://www.eba.europa.eu/sites/default/files/document_library/Publications/Reports/2023/1061495/Supervisory%20handbook%20on%20the%20validation%20of%20IRB%20rating%20systems%20revised.pdf)
- [Robustness and Sensitivity of Weighting and Aggregation in Constructing Composite Indices](https://www.sciencedirect.com/science/article/abs/pii/S1470160X13000034)
- [Expert Elicitation: Using the Classical Model to Validate Experts' Judgments](https://www.journals.uchicago.edu/doi/10.1093/reep/rex022)
- [Using a Genetic Algorithm to Optimize an Expert Credit Rating Model](https://www.sciencedirect.com/science/article/abs/pii/S095741742200834X)
- [Double-counting/VIF (US Patent 11,792,197)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11792197)
- [The Illusion of Improvement: Reject Inference Strategies in Credit Scoring](https://arxiv.org/abs/2606.18479)
- [Unequal Uncertainty: Rethinking Algorithmic Interventions for Mitigating Discrimination from AI](https://arxiv.org/abs/2508.07872)
- [Bounds for rating override rates (Tasche)](https://arxiv.org/abs/1203.2287)
- [SR 11-7 Model Risk Management](https://www.magicmirrorsecurity.com/blog/sr-11-7-model-risk-management-guidance-explained)
- [Human in the Loop AI Oversight Design](https://www.kovrr.com/blog-post/human-in-the-loop-how-to-tell-review-is-real)
- [Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians (Nature Medicine)](https://www.nature.com/articles/s41591-023-02437-x)
- [Who Should Predict? Exact Algorithms For Learning to Defer to Humans](https://arxiv.org/pdf/2301.06197)
- [RMT-Net: Reject-aware Multi-Task Network for MNAR Data in Financial Credit Scoring](https://arxiv.org/pdf/2206.00568)
- [§ 1002.9 Notifications (Regulation B), CFPB](https://www.consumerfinance.gov/rules-policy/regulations/1002/9/)
- [Comment for 1002.9 — Notifications, CFPB](https://www.consumerfinance.gov/rules-policy/regulations/1002/interp-9/)
- [CFPB Circular 2022-03](https://www.federalregister.gov/documents/2022/06/14/2022-12729/consumer-financial-protection-circular-2022-03-adverse-action-notification-requirements-in)
- [CFPB Circular 2023-03](https://www.federalregister.gov/documents/2024/04/17/2024-08003/consumer-financial-protection-circular-2023-03-adverse-action-notification-requirements-and-proper)
- [Automated Decision Making: Overview of GDPR Article 22](https://gdprlocal.com/automated-decision-making-gdpr/)
- [CJEU SCHUFA ruling C-634/21 summary](https://www.loc.gov/item/global-legal-monitor/2024-01-10/european-union-court-of-justice-rules-credit-scoring-constitutes-automated-individual-decision-making-under-gdpr/)
- [Explaining Credit Scores — The ECJ Rules on Automated Credit Assessments](https://wp.nyu.edu/compliance_enforcement/2025/03/18/explaining-credit-scores-the-ecj-rules-on-automated-credit-assessments/)
- [EU AI Act Annex III](https://artificialintelligenceact.eu/annex/3/)
- [EU AI Act Article 14 (Human Oversight)](https://artificialintelligenceact.eu/article/14/)
- [EU AI Act Article 13 (Transparency to Deployers)](https://artificialintelligenceact.eu/article/13/)
- [FICO credit risk score reason codes](https://www.creditscoring.com/creditscore/fico/factors/reason-codes.html)
- [On the failings of Shapley values for explainability](https://www.sciencedirect.com/science/article/abs/pii/S0888613X23002438)
- [CFPB Circular 2022-03 (compliance mirror)](https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/)
- [Interpretable Algorithms as a Potential Solution to CFPB's Guidance on AI-driven Credit Denials](https://www.consumerfinanceinsights.com/2024/03/28/interpretable-algorithms-as-a-potential-solution-to-cfpbs-guidance-on-ai-driven-credit-denials/)
- [Explaining Adverse Actions in Credit Decisions Using Shapley Decomposition](https://arxiv.org/abs/2204.12365)
- [Explainable Machine Learning for Credit Risk Management When Features are Dependent](https://www.tandfonline.com/doi/abs/10.1080/15366367.2023.2261186)
- [Interpretable Credit Default Prediction with Ensemble Learning and SHAP](https://arxiv.org/pdf/2505.20815)
- [A novel framework for enhancing transparency in credit scoring: Leveraging Shapley values](https://pmc.ncbi.nlm.nih.gov/articles/PMC11318906/)
- [Grouped SHAP for adverse-action reason codes (US Patent 12,050,975)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12050975)
- [Enhancing the Interpretability of SHAP Values Using Large Language Models](https://arxiv.org/pdf/2409.00079)
- [Shapley Value-Guided Adaptive Ensemble Learning for Explainable Financial Fraud Detection](https://arxiv.org/pdf/2604.14231)
- [The Selective Labels Problem (Lakkaraju, Kleinberg, Leskovec, Ludwig, Mullainathan, KDD 2017)](https://cs.stanford.edu/~jure/pubs/contraction-kdd17.pdf)
- [Human Decisions and Machine Predictions (Kleinberg, Lakkaraju et al., QJE)](https://cs.stanford.edu/people/jure/pubs/bail-qje17.pdf)
- [A Classification of Feedback Loops and Their Relation to Biases in Automated Decision-Making Systems (FAccT 2023)](https://dl.acm.org/doi/fullHtml/10.1145/3617694.3623227)
- [Designing Closed Human-in-the-loop Deferral Pipelines](https://arxiv.org/pdf/2202.04718)
- [Runaway Feedback Loops in Predictive Policing (Ensign et al.)](https://arxiv.org/pdf/1706.09847)
- [Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer (Madras et al., NeurIPS 2018)](https://arxiv.org/pdf/1711.06664)
- [Exploring Population Drift on Consumer Credit Behavioral Scoring](https://link.springer.com/chapter/10.1007/978-3-319-33003-7_7)
- [Fair and Explainable Credit-Scoring under Concept Drift](https://arxiv.org/pdf/2511.03807)

## 5. Summary

Keep CBES's percentile normalization, sigmoid shape, and expert weights (35/30/20/10/5) — all three are literature-defensible, and Basel II IRB explicitly legitimizes expert-weighted scorecards. Do **not** change the weight numbers — no finding endorses any alternative. Do add: a VIF/multicollinearity check (Capacity's dti+loan_to_income and Credit-delinquency vs. Behaviour-active_loans are real double-counting exposures), a ±5/±10-point weight sensitivity sweep, a WOE/IV corroboration pass against Home Credit's `TARGET`, and a written qualitative validation protocol. The sigmoid steepness constants (k=4, k=5) are unbacked engineering choices — label them as such in the paper, don't present them as derived. The explanation module ships today as top-4-of-5 ranked component reason codes (weight × shortfall), percentile-phrased, with DEFER surfaced only as "under human review"; SHAP gets added later but must be grouped into these same codes, never shown raw. The relearning loop: build the `deferred_review` capture table plus a 2–5% random exploration arm now — but no retraining trigger, no reject inference, no continuous learning until the deferral rule is proven better-than-random via separate automated-vs-deferred accuracy against no-deferral baselines.
