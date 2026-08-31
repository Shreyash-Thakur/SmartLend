# Future scope: is a hybrid worth building?

**Question.** Would combining XGBoost with TabPFN-2.5 (or with CBES) into a
hybrid improve accuracy?

**Answer, measured.** For **XGBoost + CBES: no** — every honest combination is
equal to or worse than XGBoost alone, and the data show why. For
**XGBoost + TabPFN-2.5: the question is currently unanswerable** — the saved
TabPFN probabilities cannot be matched to the rows they score, and we report
that rather than a number built on misaligned rows. What the data *do*
support is (1) regenerating the TabPFN scores with row identifiers, and
(2) not claiming ensemble gains among the gradient-boosting family, whose
best measured combination (+0.0021 AUC) sits below the ±0.0036 fold-to-fold
noise floor.

All numbers below come from `reports/complementarity.json`, produced by
`research/analysis/complementarity.py` (metric helpers unit-tested in
`research/tests/test_complementarity.py`). Evaluation set: all **307,511
out-of-fold rows** from 5-fold CV (`backend/artifacts/prediction_outputs.csv`),
converted to a single convention: y = 1 means default, all probabilities are
P(default). Reference noise floor: XGBoost's fold-to-fold AUC standard
deviation, **±0.0036**. A "gain" smaller than that is not a gain.

---

## 1. Why the TabPFN half of the question has no answer yet

`reports/_tabpfn_probs_5000.npy` holds 10,000 TabPFN-2.5 P(default) values
documented as a `default_rng(42)` subsample of the 20% holdout
(`train_test_split(test_size=0.2, random_state=42, stratify=TARGET)` on the
merged CSV). Before comparing anything we verified that reconstruction:
TabPFN's AUC on the reconstructed rows must reproduce its known **0.7446**.

It does not. Measured facts:

| Check | Expected if aligned | Measured |
|---|---|---|
| TabPFN AUC on reconstructed rows | ~0.7446 | **0.5052** (chance) |
| Corr(TabPFN P(default), XGBoost P(default)) on those rows | ~0.6 for two informative models | **0.0109** |
| 30 alternative reconstructions (legacy RNG, sorted holdout, permutation draw, flipped stratify labels, consecutive 10k chunks, full-dataset draw) | one near 0.7446 | all in **0.47–0.53** |
| Batch-of-500 permutation search (20×20 pairings vs XGBoost) | some batch corr ≫ noise | max abs corr **0.126** = null noise at n=500 |

The probabilities themselves look sane (mean 0.059 against an 8.07% default
rate), but their row order matches nothing derivable from the current CSV.
The scoring was done off-repo (GPU session); no index array was saved.

**Consequence.** Every XGBoost + TabPFN comparison — error correlation,
segment wins, hybrid AUC — was skipped. This is an *unanswered* question, not
a negative answer. The analysis script contains the full TabPFN pipeline
behind the alignment gate; it will run automatically the moment a correctly
indexed artifact exists.

**What to do next (concrete, cheap).** Re-score TabPFN on the holdout and
save `SK_ID_CURR` alongside each probability (two-column parquet/npz). The
existing run took 77 GPU-minutes for all 61,503 rows; a 10,000-row re-score
is ~13 minutes.

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
3. **Regenerate the TabPFN artifact with row IDs** (§1, ~13 GPU-minutes for
   10k rows). This is the one genuinely open question: TabPFN is the only
   competent model in the roster built on a different learning principle, so
   it is the only candidate that could plausibly break the 0.99 error-
   correlation wall. Whether it does is precisely what the missing alignment
   prevents us from knowing.
4. **When TabPFN is re-scored, target XGBoost's measured weak segments
   first**: applicants 60+ (XGB AUC 0.7190), under 30 (0.7423), thin-file/no
   bureau (0.7353 vs 0.7686 with bureau), and EXT_SOURCE_2 Q2 (0.7227).
   A hybrid case exists only if TabPFN beats XGBoost *somewhere*; these are
   the places to look, and the segment machinery in
   `research/analysis/complementarity.py` already measures it automatically
   once the alignment gate passes.

### Honesty notes

- Every TabPFN-related claim above is a claim about *artifact provenance*,
  not model quality; TabPFN's own 0.7446 holdout AUC (5,000 training rows) is
  unaffected.
- Any future TabPFN comparison will rest on ~10,000 rows (~807 defaults);
  AUC differences there carry sampling error of roughly ±0.01, so only
  paired statistics (as implemented) can resolve gains near the noise floor.
- Bootstrap CIs quantify stability of orderings on these rows; the ±0.0036
  fold std is the stricter and binding criterion for claiming a gain.
