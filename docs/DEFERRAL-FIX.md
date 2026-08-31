# Deferral Fix — measured, not asserted

**Date:** 2026-08-31
**Code:** `research/deferral/` · **Numbers:** `reports/deferral_fix.json` · **Wiring:** `backend/app/services/decision_engine.py`
**Data:** `backend/artifacts/prediction_outputs.csv` — 307,511 out-of-fold rows, `y_true == 1` = good customer, all probabilities are approval probabilities.

## What was wrong

The router defers when `D = |p_ml - p_cbes| > TAU_D` (0.43). Gate condition 1
(`research/relearning/gate.py`, unmodified throughout) says a working router
must hand humans a pile that is *harder* than random selection — negative z.
The production router scored, on the full artifact:

| metric | before |
|---|---|
| balance-distance z | **+38.11** |
| accuracy z | **+43.01** |
| deferral rate | 51.76% (AUC-implied band: 8.0%–16.0%) |
| deferred pile | 93.69% good vs 90.03% good auto-decided |

Ten-plus standard deviations the wrong way: it defers the cases the model is
*most confident* about.

## Root cause: scale offset — hypothesis CONFIRMED

Measured on all 307,511 rows (`reports/deferral_fix.json → scale_offset_hypothesis`):

- mean `p_ml` = 0.9207, mean `p_cbes` = 0.6133 → mean offset **+0.3074**
- `p_ml > p_cbes` on **98.4%** of rows; signed diff mean/sd = 2.25
- mean `|D|` = 0.3103; after subtracting the constant offset, mean residual = **0.1078** — the fixed calibration offset accounts for ~65% of the signal's magnitude
- corr(`D`, `|p_ml − 0.5|`) = **+0.38** — the offset widens with model confidence
- CBES alone: AUC 0.5650 (barely above random), vs ML 0.7651

So `D` is dominated by a fixed calibration offset between two differently-scaled
scores, and because that offset tracks confidence, `TAU_D` fires precisely on
the confident (easy) cases. Decisive cross-check: the incumbent signal *at the
corrected 22.6% rate* is still inverted (z = +18.3) — the defect is the
**signal**, not the threshold.

## Protocol

- 50/50 tune/test split (seed 20260831). Every fitted transform and every
  deferral threshold selected on TUNE only; every reported number from TEST.
- Operating point: **hard business requirement of 20–25% deferral rate**
  (underwriter capacity). Each candidate's threshold is the tune-split
  quantile targeting 22.5%, so all candidates are compared at a matched rate —
  the rate is fixed, so *which* cases get deferred is the whole question, and
  the gate z-score is the measure of success.
- Scoring: rebuild `final_decision` on TEST (DEFER per candidate, else the
  engine's hard approve/reject at `approval_threshold`) and run the unmodified
  `evaluate_condition_1` (200 trials, seed 20260830).

## Results at the matched 20–25% rate (held-out test split, n = 153,756)

Before, for reference — production router on the same test rows:
rate 51.83%, balance z **+24.36**, accuracy z **+27.85**, selective risk 0.0907
vs random 0.0835 (**worse than random abstention**, position −0.09).

| signal | rate | balance z | accuracy z | selective risk | vs random 0.0835 | position (random→oracle) |
|---|---|---|---|---|---|---|
| `current_abs_diff` (incumbent) | 22.58% | **+15.83** | **+18.32** | 0.0907 | loses | −0.09 |
| `rank_diff` (percentile-normalised) | 22.56% | −12.66 | −13.43 | 0.0785 | beats | 0.06 |
| `zscore_diff` | 22.65% | −42.51 | −49.21 | 0.0649 | beats | 0.22 |
| `isotonic_diff` (calibrated onto y) | 22.66% | −66.04 | −70.59 | 0.0583 | beats | 0.30 |
| **`ml_uncertainty`** (−\|p_ml − t\|) | **22.70%** | **−92.50** | **−99.68** | **0.0451** | **beats** | **0.46** |
| `ml_uncertainty_0.5` (−\|p_ml − 0.5\|) | 22.68% | −90.21 | −92.72 | 0.0447 | beats | 0.47 |

Oracle at matched coverage: 0.0000 selective risk (the model's total error
mass, 8.35%, is smaller than the 22.7% deferral budget, so a perfect router
could catch every error).

### Headline

**Yes — at the required 20–25% rate the best candidate achieves a strongly
NEGATIVE gate z: accuracy z = −99.7, balance z = −92.5** (winner
`ml_uncertainty`, rate 22.70%). Gate condition 1 flips FAIL → **PASS**. The
deferred pile is 79.9% good (vs 95.4% auto) and model accuracy on it drops to
78.6% (vs 95.5% auto): humans finally get the hard cases.

### The publishable negative finding

**The ML-vs-rule disagreement idea does not survive contact with real data.**
Every repair of the disagreement signal (rank, z-score, isotonic) fixes the
inversion, but all of them lose to plain model uncertainty — the standard
selective-prediction baseline (Chow's rule). Ordering: uncertainty (−99.7) ≫
isotonic (−70.6) > z-score (−49.2) > rank (−13.4) ≫ incumbent (+18.3). The
reason is visible in the data: CBES carries almost no discriminative signal
(AUC 0.565), so even perfectly calibrated "disagreement" with it is mostly
noise around the ML score. Disagreement-based deferral would only earn its
place if the rule-based score improved substantially.

## The capacity-vs-AUC tension (stated, not smoothed over)

The required 20–25% rate sits **above** the AUC-implied natural-rate bound of
[8.00%, 16.00%] (test-split AUC 0.765). At the winner's 22.70% rate:

- excess above the bound: 6.70 percentage points;
- implied cost: roughly **29.5% of all deferrals are avoidable** — cases the
  model's discriminative power says it already handles correctly, deferred
  only because capacity was set above what the model justifies;
- **gate condition 2 still FAILS at this rate, and that is expected.** It is a
  capacity decision made outside the model, not a router defect. If capacity
  is ever renegotiated, the bound says ≤16% is where every referral can be a
  case the model genuinely needs help on.

Conditions 3 (no exploration labels yet) and 4 (no retraining design) also
still fail, so the overall verdict remains DO NOT OPEN THE LOOP — condition 1
is now the only data-quality condition passing, which is exactly what this fix
set out to change.

## What was wired in

`backend/app/services/decision_engine.py` now supports two routing modes.
**The default is the old disagreement behaviour — nothing changes under the
running demo.** To switch on the measured fix:

```bash
export SMARTLEND_DEFERRAL_MODE=uncertainty   # defer on |p_ml - t_approve| < TAU_U
export SMARTLEND_TAU_U=0.2458                # optional; default is the measured
                                             # tune-split quantile for a 22.5% rate
```

or per call: `hybrid_decision(..., deferral_mode="uncertainty", tau_u=0.2458)`.
In uncertainty mode the auto decision is the hard `p_ml >= t_approve`
approve/reject — exactly the rule the evaluation scored, so the measured
z-scores carry over. When flipping the flag in production, bump
`ENGINE_VERSION` so relearning capture rows from the two routers stay
separable.

## Files

- `research/deferral/signals.py` — the six candidate signals (fit-on-tune / score-anywhere)
- `research/deferral/evaluate.py` — protocol runner; writes `reports/deferral_fix.json`
- `research/tests/test_deferral.py` — 23 tests: transforms, offset contamination, rate matching, risk-coverage with deliberately-good and deliberately-bad router fixtures, engine flag wiring
- `reports/relearning_gate_before_deferral_fix.json` — frozen full-artifact "before" gate run
- `reports/deferral_fix.json` — every candidate's z, rate, risk-coverage position, curves, and the capacity-vs-bound tension
