# Future scope: is a hybrid worth building?

**Question.** Would combining XGBoost with TabPFN-2.5, Google TabFM, or CBES
into a hybrid improve accuracy?

**Answer, measured — the question is now CLOSED for the whole roster.**
- **XGBoost + CBES: no** — every honest combination is equal to or worse than
  XGBoost alone (§2).
- **XGBoost + TabPFN-2.5: no** — the row-ID re-score (commit 12bbf4f,
  `reports/tabpfn_scored_rows.csv`) unblocked the pre-registered experiment,
  the alignment gate **verified** (AUC reproduces 0.7284 exactly, zero label
  mismatches, corr with XGBoost 0.83 — the informative-model correlation the
  broken artifact lacked), and the pre-registered decision rule returns
  **KILL**: best honest hybrid (CV stack) −0.0008 vs XGBoost, weight-sweep
  optimum w_XGB = 1.0, simple average significantly *hurts* (paired-bootstrap
  CI [−0.0098, −0.0010]), and TabPFN loses to XGBoost in **all 16 segments**
  including all four pre-named weak ones (§1).
- **Gradient-boosting family: no** — best measured combination (+0.0021 AUC)
  sits below the ±0.0036 fold-to-fold noise floor (§3).
- **XGBoost + TabFM (Google): no** — scored 2026-09-22 on the same 8,000
  id-indexed rows (`reports/tabfm_scored_rows.csv`, AUC 0.6911, gate verified
  with corr(TabFM, XGBoost) = 0.66); best honest hybrid (CV stack) **−0.0004**
  vs XGBoost, weight-sweep optimum w_XGB = 1.0, simple average significantly
  hurts (CI [−0.0219, −0.0071]), and no real segment win (its only positive
  segment is the n=13 missing-EXT_SOURCE_2 bucket at CI [0.0, 0.5] — noise).

Per the pre-registered rule in `docs/DEFENCE-DECISION-HYBRID.md` §4.4: *"the
roster is exhausted, XGBoost alone is the system, and the hybrid line of work
ends with a documented negative result — which is a finding, not a failure."*

All numbers below come from `reports/complementarity.json`, produced by
`research/analysis/complementarity.py` (metric helpers unit-tested in
`research/tests/test_complementarity.py`). Evaluation set: all **307,511
out-of-fold rows** from 5-fold CV (`backend/artifacts/prediction_outputs.csv`),
converted to a single convention: y = 1 means default, all probabilities are
P(default). Reference noise floor: XGBoost's fold-to-fold AUC standard
deviation, **±0.0036**. A "gain" smaller than that is not a gain.

---

## 1. XGBoost + TabPFN: answered — the pre-registered rule says KILL

**History (why this was ever open).** The original TabPFN artifact
(`reports/_tabpfn_probs_5000.npy`) carried no row IDs; the alignment gate
failed (AUC 0.5052 on every one of 30 attempted reconstructions, corr with
XGBoost ~0.01), so all TabPFN comparisons were skipped rather than reported
on misaligned rows. Commit 12bbf4f re-scored 8,000 rows saving `SK_ID_CURR`
with every probability (`reports/tabpfn_scored_rows.csv`), and
`research/analysis/complementarity.py` now joins on ids and verifies the join
(labels must match the dataset TARGET on every row; the recomputed AUC must
reproduce the run's own 0.7284).

**Gate result (2026-09-22): VERIFIED.** 8,000/8,000 rows joined, 0 label
mismatches, AUC reproduces exactly, corr(TabPFN, XGBoost) in P(default) =
**0.83** — the informative-model correlation the broken artifact lacked.

**Hybrid result, against the pre-registered decision rule** (fixed in
`docs/DEFENCE-DECISION-HYBRID.md` §4.4 *before* the data existed):

| Criterion | Required for BUILD | Measured | Verdict |
|---|---|---|---|
| TabPFN beats XGBoost in ≥1 pre-named weak segment (CI excluding 0) | yes | loses in **all 16 segments** (deltas −0.014 to −0.083), including age 60+, thin-file, under-30, EXT_SOURCE_2 Q2 | fails |
| Honest combination beats XGBoost by > +0.0036 | yes | best honest hybrid (CV stack) **−0.0008**; weight sweep optimum **w_XGB = 1.0**; simple average −0.0057 (CI [−0.0098, −0.0010], significantly worse) | fails |

**KILL.** The one candidate with a plausible mechanism (different model
class, 0.83 — not 0.99 — correlation) still adds nothing an honest evaluation
can claim. Source: `reports/complementarity.json → tabpfn_alignment,
hybrid_xgb_tabpfn, segments_xgb_vs_tabpfn`.

## 2. XGBoost + CBES: measured, and the answer is no

### 2a. Do they fail differently? Yes — but that alone is not enough

Error correlation (Pearson correlation of |y − p|, the quantity that limits
what averaging can recover):

| Pair | Corr of P(default) | Corr of errors |
|---|---|---|
| XGBoost · LightGBM | 0.9604 | **0.9943** |
| XGBoost · CatBoost | 0.9541 | 0.9934 |
| XGBoost · Logistic Regression | 0.8184 | 0.9761 |
| XGBoost · Random Forest | 0.8278 | 0.9721 |
| XGBoost · CBES | **0.1928** | **0.4388** |

CBES is the only genuinely decorrelated signal in the system (error
correlation 0.44 vs ≥0.97 for everything else). XGBoost and CBES land on
opposite sides of the 0.5 approval threshold on **19.8%** of applicants, and
the confusion-of-errors table shows real non-overlap:

| (threshold 0.5, approval space, n = 307,511) | count | fraction |
|---|---|---|
| both right | 228,560 | 74.3% |
| only XGBoost right | 54,255 | 17.6% |
| only CBES right | 6,512 | 2.1% |
| both wrong | 18,184 | 5.9% |

So diversity exists. But an ensemble needs a partner that is *both* diverse
*and* competent, and CBES's standalone AUC is **0.5650** — 3.5 disagreements
in favor of XGBoost for every one in favor of CBES (17.6% vs 2.1%).

### 2b. Is there any segment where CBES wins? No

Per-segment AUC on all 307,511 OOF rows, XGBoost vs CBES:

| Segmentation | CBES − XGBoost, range across segments |
|---|---|
| EXT_SOURCE_2 quartiles (+ missing) | −0.165 to −0.264 |
| Thin-file (no bureau, n=44,020) / has bureau | −0.201 / −0.208 |
| Income quartiles | −0.186 to −0.221 |
| Age bands | −0.138 (60+) to −0.194 |

XGBoost wins **every one of 16 segments**, by 0.14–0.26 AUC. CBES comes
closest among applicants aged 60+ (0.5813 vs 0.7190) — still a 0.14 gap.
There is no niche where the rule-based score adds information the trees lack.

### 2c. Does any combination beat XGBoost alone? No

| Combination (n = 307,511 OOF rows) | AUC | vs XGBoost 0.7651 |
|---|---|---|
| Simple average | 0.6850 | **−0.0801** (bootstrap 95% CI −0.0830 to −0.0772) |
| Rank average | 0.7067 | −0.0584 |
| Weighted average, weight swept 0–1 | best at **w_XGB = 1.0** → 0.7651 | ±0 — the sweep itself says: put zero weight on CBES |
| Logistic stacking (5-fold CV, honest) | 0.7650 | −0.0001 |

The stacked meta-learner, given both signals and evaluated out-of-fold,
effectively learns to ignore CBES and reproduces XGBoost to four decimals.
Every fixed-weight blend is strictly worse. **Verdict: adding CBES to
XGBoost does not help at any mixing weight; at most weights it actively
hurts.** Diversity (2a) without competence (2b) buys nothing.

## 3. Calibration for expectations: even the *best* available partner adds nothing claimable

To bound what any hybrid could plausibly deliver on this dataset, the same
suite was run for XGBoost's strongest peers:

| Pair | Best honest hybrid | AUC | Gain vs XGBoost | Exceeds ±0.0036? |
|---|---|---|---|---|
| XGBoost + CatBoost | simple average | 0.7672 | **+0.0021** (bootstrap CI +0.0016 to +0.0026) | **No** |
| XGBoost + LightGBM | CV stack | 0.7662 | +0.0012 (CI +0.0007 to +0.0015) | **No** |
| XGBoost + CBES | CV stack | 0.7650 | −0.0001 | No |

The XGBoost+CatBoost gain is *directionally real* (the paired bootstrap CI
excludes zero — the ordering is stable under resampling of these rows) but it
is **smaller than the fold-to-fold noise** of the base model itself. A future
evaluation could not reliably reproduce it, so it must not be claimed as an
improvement. This is exactly the pattern §2a predicts: error correlations of
0.99+ leave almost no independent error for averaging to cancel.

## 4. What the evidence supports proposing

1. **Do not build the CBES hybrid for accuracy.** The numbers are
   unambiguous (§2c). CBES's documented value, if any, lies elsewhere
   (interpretability / rule transparency), and any such claim should be argued
   on those grounds, not on AUC.
2. **Do not claim tree-ensemble blending gains.** The best measurable blend
   (+0.0021) is inside the ±0.0036 noise floor.
3. ~~Regenerate the TabPFN artifact with row IDs~~ — **done** (commit 12bbf4f)
   and the fusion question is answered: KILL (§1). TabPFN did break the 0.99
   error-correlation wall (corr 0.83) — diversity again without enough
   segment-local competence, the same lesson CBES taught at a lower level.
4. ~~Target XGBoost's weak segments when TabPFN is re-scored~~ — **done**:
   TabPFN loses in all four pre-named weak segments (and all 12 others).
   With the roster exhausted, the honest future-scope items are
   (a) **feature engineering on the unused Home Credit tables** — now
   **measured, and it delivers** (§5 below), and (b) foundation models on
   **richer features / larger contexts** (the Colab endpoint in
   `colab/tabpfn_colab_server.ipynb` exists precisely to lift the 8 GB VRAM
   context cap).

## 5. Features vs fusion — the closing experiment (2026-09-23)

The natural follow-up question after four KILL verdicts: if fusing models
adds nothing, where DOES headroom live? Tested by holding models, folds and
evaluation fixed and changing only the features — v1 (the current ~113-column
numeric frame) vs v2 (319 engineered features from application ratios,
EXT_SOURCE combinations and ~35 deep bureau.csv aggregates —
`research/features/engineer.py`, grounded in the published top solutions to
this dataset). Decision rules pre-registered in
`research/analysis/features_vs_fusion.py` with the same ±0.0036 noise floor
that killed every hybrid. Source: `reports/features_vs_fusion.json`.

**Feature claim: GRANTED — for every model, in every fold.**

| Model (same 5 folds, seed 42) | v1 OOF AUC | v2 OOF AUC | mean fold delta |
|---|---|---|---|
| CatBoost | 0.7603 | **0.7757** | +0.0155 |
| XGBoost | 0.7584 | 0.7749 | +0.0164 |
| LightGBM | 0.7560 | 0.7721 | +0.0160 |
| LogisticRegression | 0.7377 | 0.7560 | +0.0182 |

All 20 fold deltas positive; every mean delta ≥ 4× the noise floor. Even the
linear model gains +0.018 — the information is real, not a tree artefact.

**Fusion claim on v2: DENIED — again.** Best honest combination (simple
average of the three GBMs) reaches 0.7773 vs CatBoost's 0.7757: +0.0016,
directionally real (paired-bootstrap CI [0.0010, 0.0021]) but **below the
0.0036 floor** — the exact same pattern as the earlier XGB+CatBoost +0.0021.
OOF correlations remain 0.87–0.96: better features made every model better
*in the same way*, so there is still almost no independent error to cancel.

**The sentence this buys the paper:** under one pre-registered criterion,
feature enrichment cleared the noise floor for every model in every fold
(+0.015 AUC), while no honest model combination ever has — on this dataset,
headroom lives in information, not in fusion. Remaining known headroom to the
~0.80 ceiling: the side tables not on disk (previous_application,
installments_payments, POS_CASH, credit_card_balance).

### Honesty notes

- Every TabPFN-related claim above is a claim about *artifact provenance*,
  not model quality; TabPFN's own 0.7446 holdout AUC (5,000 training rows) is
  unaffected.
- Any future TabPFN comparison will rest on ~10,000 rows (~807 defaults);
  AUC differences there carry sampling error of roughly ±0.01, so only
  paired statistics (as implemented) can resolve gains near the noise floor.
- Bootstrap CIs quantify stability of orderings on these rows; the ±0.0036
  fold std is the stricter and binding criterion for claiming a gain.
