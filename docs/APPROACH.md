# The SmartLend approach, explained in plain words

This document walks through the entire project the way you would explain it to a
person, not to a reviewer: what problem we took on, what we built, what broke,
how we proved it broke, and what we did about it. Every number here comes from a
machine-readable result file in `reports/` (named in brackets), so nothing in
this story is from memory.

---

## 1. The problem

A lender gets a loan application and has three honest options: **approve** it,
**reject** it, or admit the case is genuinely hard and **send it to a human
reviewer**. Most ML projects only study the first two. Our project is about the
third one — the *deferral* decision — because in the real world review capacity
is scarce (a bank can maybe re-check 20–25% of applications, not all of them),
and sending a human the *wrong* 20% wastes the scarcest resource in the system.

So the question underneath everything: **when the machine hands a case to a
human, is it actually handing over the hard ones?**

## 2. The data

We use **307,511 real loan applications** from the Home Credit Default Risk
dataset, joined with credit-bureau records. Only **8.07%** of applicants
actually defaulted — which matters, because a model that simply approves
everyone is "92% accurate" while catching zero defaulters. That is why we judge
models by **ROC-AUC** (how well they *rank* risky above safe applicants), not by
accuracy.

## 3. What the system is

Two scorers look at every application:

- **The ML model.** A calibrated logistic regression serves live traffic
  (held-out AUC 0.6919 — `serving_model_retrain.json`). It is deliberately
  simpler than our best research model, because the thing that scores real
  applications must be calibrated, explainable, and use exactly the 15 fields
  the application form collects. Stronger models (CatBoost, XGBoost, up to
  AUC 0.787) exist on the research side as references.
- **CBES.** A hand-built, rule-based "credit bureau expert score" over 8
  portable fields with five weighted pillars. On its own it barely predicts
  (AUC 0.5650), so it gets only a 10% weight in the blend (α = 0.10, a measured
  choice — `docs/BLEND-DECISION.md`). It is kept for interpretability: every
  pillar can be explained to an applicant.

The blended score goes through thresholds: clearly safe → **approve**, clearly
risky → **reject**, and some rule decides which remaining cases go to a human.
That last rule — the **deferral router** — is where the research happened.

## 4. The finding: the deferral rule was upside down

The original (and still default) router deferred cases where the ML model and
CBES *disagreed* — which sounds sensible. We tested it the way you test a
claim, not the way you demo a feature: shuffle which cases get deferred 10,000
times at the same deferral rate, and ask whether the real router picks harder
cases than random chance would (`deferral_audit_10k.json`).

It did the opposite. The disagreement router was deferring the cases the model
was **most confident and most correct about** — tens of standard deviations
*easier* than a random pick, while deferring 52% of everything. The root cause
was measured, not guessed: the ML model and CBES sit on different probability
scales (mean 0.92 vs 0.61), so "disagreement" mostly measured that fixed offset,
which is *largest* exactly where the model is most sure.

**The fix is almost embarrassingly simple:** defer the cases the ML model is
*unsure* about — the ones near its own approve/reject boundary (Chow's rule).
At the operating rate of 22.5%, this sends humans a pile that is dramatically
harder than random (audit z ≈ −60; `deferral_audit_10k.json`). Everything
fancier we tried — repairing the disagreement signal by ranking, z-scoring, or
recalibrating it — also fixed the inversion but *lost* to plain uncertainty.
That negative result is part of the contribution.

## 5. Three ways to route, because "hard" has two meanings

The engine now has three selectable deferral modes
(`SMARTLEND_DEFERRAL_MODE`), because we found that "which cases deserve a
human" has two honest answers:

1. **Legacy disagreement** (default, kept for demo stability) — the measured-
   inverted rule, retained so the before/after story is reproducible.
2. **Uncertainty** (`uncertainty`) — defer where the model is least sure. Best
   at catching the model's *mistakes*.
3. **Expected loss** (`expected_loss`) — defer where acting would be most
   *expensive* if wrong. A false approval costs us 0.45 (money lent and lost);
   a false rejection costs 0.2644 (profit forgone) — both derived from the
   dataset's own credit amounts. This mode asks "where is the money at risk?"
   instead of "where am I confused?". At the same 22.5% review rate it cuts the
   expected cost of the auto-decided pile from **0.0731 to 0.0291** per
   applicant and gets **42% of true defaulters in front of a human** (vs 24.7%
   under the uncertainty rule) — `expected_loss_calibration.json`.

The honest summary we present: **one review budget can buy error-catching or
loss-avoidance, and those are different purchases.** Choosing between them is a
business decision, so the system exposes both and defaults to neither.

## 6. How we made sure the numbers are honest

This is the part examiners probe, so it is spelled out:

- **Cross-validation everywhere.** Research numbers are 5-fold out-of-fold over
  all 307,511 rows; the fold noise floor (±0.0036 AUC) is stated, and any gain
  smaller than it is reported as "inside noise", not as a win.
- **Tune/test discipline.** Any threshold (deferral cutoffs, θ values) is chosen
  on a tune half and *measured* on the untouched other half (fixed seed
  20260831). Nothing is ever tuned on the data it is graded on.
- **Early stopping on an inner split.** Boosted-tree capacity (how many trees)
  is not hand-picked: each model gets a budget of 2,000 trees and stops when
  100 consecutive rounds bring no improvement on a 10% slice carved *from the
  training folds only* — evaluation data never influences the choice. In
  practice validation AUC peaked at tree 439 (0.7607) and training stopped
  around 540; the other ~1,460 trees of the budget would only have memorized
  noise (`convergence.json`). One deliberate exception: the SGLB uncertainty
  model trains a fixed 1,000 iterations, because its ensemble-based uncertainty
  needs a fixed trajectory.
- **Batches, for completeness:** training has none — boosted trees fit the full
  dataset every round and the serving model's solver is full-batch. Batches
  exist only at *inference* for the transformer baselines, where GPU memory
  binds (TabPFN: 16 batches of 500 rows on an 8 GB laptop GPU).
- **Permutation audits.** Claims like "the router picks harder cases" are
  tested against a 10,000-shuffle null, with one fixed sign convention, on a
  final slice no experiment ever touched.

## 7. Where the accuracy actually came from

A plateau question drove a year of work: tuning 14 hyperparameter configs
bought +0.001 (inside noise) — so was the model done, or was the *data* done?

The **learning curve** answered it, with a catch: you must re-run early stopping
at every sample size, or a fixed tree budget underfits the large subsets and
fakes a plateau. Done honestly, the curve was still climbing at full size
(+0.010 AUC per doubling of data — ~3× the noise floor; `convergence.json`).
Translation: the model had converged, but the *information* hadn't. The
remaining headroom was in features, not in architecture.

That prediction was then confirmed twice, under one pre-registered rule
("claim a gain only if it clears the noise floor in every fold"):

- **FE-v2:** 319 engineered features (ratios + deep bureau aggregates) lifted
  every model in every fold — CatBoost 0.7603 → **0.7757**
  (`features_vs_fusion.json`).
- **FE-v3:** 26 aggregates from four more Home Credit side tables — CatBoost
  → **0.7868** (`features_v3_side_tables.json`; measured after the report
  freeze, which is why older documents still quote 0.7757 as best).

Meanwhile every model-fusion idea (stacking, blending, foundation-model
ensembles) stayed *below* the noise floor. One sentence carries the result:
**headroom lived in information, not in combination.**

## 8. The foundation-model reality check

We also scored two zero-training "foundation models for tables" as baselines:
TabPFN-2.5 reached 0.7284 ± 0.0127 on 8,000 id-tracked rows — respectable with
*no training at all* — and Google's TabFM 0.6911 ± 0.0113 on the same rows,
about equal to our trained serving model (`tabpfn_quick_run.json`,
`tabfm_quick_run.json`). Both carry non-commercial licences, so they are
research baselines, not deployable options — and the engineering cost was real
(the larger TabICLv2 needed batch size 2 plus ~18 GB of disk offload per fold
just to run on consumer hardware).

## 9. Does any of this survive outside one dataset?

Two checks:

- **Replication.** The entire audit pipeline, unmodified, ran on the UCI Taiwan
  credit-card dataset (30,000 rows, different country, different product,
  2.8× the default rate). Every router again beat random at p < 10⁻⁴ — and the
  *best* routing signal flipped with the base rate, which is itself a finding:
  routing policy should be re-validated per portfolio
  (`taiwan_replication.json`).
- **Stress tests.** Under simulated data degradation, the deferral rate at the
  frozen threshold works as a free drift alarm — it jumps 22.4% → 56.6% when
  input information is lost. Its blind spot (a pure shift in the default rate
  with features intact) is stated alongside (`shift_stress_tests.json`).

## 10. What we deliberately refused to build

Negative space is part of the design:

- **No automatic retraining.** Reviewer verdicts are captured live, but a
  four-condition gate decides whether they may ever be trained on, and today it
  fails 0/4 — so the loop stays shut (`python -m research.relearning.gate`
  exits non-zero *by design*). Retraining on labels your own router selected is
  how feedback loops poison models; a 3% random exploration arm collects the
  only unbiased labels.
- **AI stays advisory.** The optional Claude reviewer-briefing and the remote
  TabPFN second opinion can inform a human but can never write a decision.
- **No silent provenance.** Every decision records the engine version, serving
  artifact and threshold hash, and the UI shows them — so any number on screen
  can be traced to the exact model that produced it.

## 11. The one-paragraph summary

We built a working loan-decisioning system on real data, then audited the part
everyone assumes works — the handoff from machine to human — and found it was
sending humans the *easiest* cases. We proved why (a calibration offset wearing
a "disagreement" costume), fixed it two independent ways (route by uncertainty,
or route by expected money-at-risk), showed the two fixes buy different things,
replicated the audit on a second country's data, and wrapped the whole thing in
the discipline that makes the numbers trustworthy: out-of-fold evaluation,
tune/test separation, early stopping that never sees test data, permutation
audits, and a retraining gate that is proud to say no.

---

*Where to go deeper:* [`PROJECT-OVERVIEW.md`](PROJECT-OVERVIEW.md) (system +
file map) · [`STATUS.md`](STATUS.md) (current numbers) ·
[`DEFERRAL-FIX.md`](DEFERRAL-FIX.md) (the diagnosis in full) ·
[`FINAL-REPORT.md`](FINAL-REPORT.md) (consolidated results) · `reports/*.json`
(every number's source of truth).
