# Blend decision fix — slanted boundary, measured cost

**Date:** 2026-08-31 · **Report:** `reports/blend_decision.json` · **Code:** `research/blend/regenerate.py`

## What was wrong

`decision_engine.py` specifies a two-stage blend: `p_blend = (1 - α)·p_ml + α·p_cbes`
with `α = _BLEND_ALPHA = 0.25`, and thresholds compared against `p_blend` so CBES
nudges the effective decision. But `backend/artifacts/prediction_outputs.csv` had
been regenerated with decisions taken on **p_ml alone** (threshold 0.925484,
deferral band ±0.022269). Verified before the fix: every non-DEFER decision in the
artifact was a pure function of `best_model_prob`; `cbes_prob` influenced nothing.

## The geometry (why bands were vertical, and are now slanted)

The Decision Landscape scatter plots p_ml on x and p_cbes on y.

- A rule `p_ml ≥ t` ignores y entirely: its boundary is the **vertical line**
  x = t. Regions are vertical bands, whatever p_cbes is.
- A rule `(1-α)·p_ml + α·p_cbes ≥ t` has boundary
  `y = (t − (1-α)x) / α`, a **line of slope −(1-α)/α** — with α = 0.25 the slope
  is −3. Two applicants with identical p_ml can now get different decisions when
  their p_cbes differs by enough to cross that line.

## What was done

- Decisions regenerated on `p_blend` with α read from
  `decision_engine._BLEND_ALPHA` (0.25), never hardcoded.
- **Split discipline:** 50/50 tune/test split (seed 20260831, the same
  `split_tune_test` used by the deferral study). Thresholds selected on the tune
  half only; every reported metric below comes from the test half.
- **Approve/Reject:** Youden's J on p_blend → t\* = 0.828091.
- **Defer:** model uncertainty on p_blend, |p_blend − t\*| < τ_u = 0.024227
  (tune-half quantile targeting the 20–25% underwriter-capacity band).
  Realized test deferral rate: **22.59%** (in band); full artifact: 22.54%.
- Artifact columns updated in place: `final_decision`,
  `approval_threshold` = t\*+τ_u = 0.852318, `rejection_threshold` = t\*−τ_u =
  0.803863, `confidence` = clip(2·|p_blend − t\*|, 0, 1). All other columns,
  including every `prob_*`, untouched.
- `model_metrics.csv` was **not** changed: its rows are per-model operating
  points (each model at its own threshold), which remain correct. The blend is a
  decision policy, not a model row.

## The measured cost — read this before liking the slanted boundary

CBES alone scores **0.5621 AUC** on the test half (barely above coin-flip 0.5).
Blending 25% of it into a 0.7669-AUC model costs real discrimination:

| metric (test half) | p_ml alone | p_blend (α=0.25) | delta |
|---|---|---|---|
| AUC | 0.7669 | 0.7393 | **−0.0277** (95% paired-bootstrap CI [−0.0302, −0.0253]; 1000/1000 resamples negative) |
| default capture (rejected) | 61.9% | 53.5% | **−8.4 pp** |
| default capture (rejected or deferred) | 81.9% | 75.1% | −6.8 pp |
| approve rate (all rows) | 52.0% | 55.5% | +3.5 pp |
| defaulters auto-approved | 2,245 | 3,094 | **+849 (+38%)** |
| auto-decided accuracy | 71.3% | 74.7% | +3.5 pp (approving more in a 92%-good population raises raw accuracy while missing more defaulters) |

Confusion matrices (test half, decision × truth) are in
`reports/blend_decision.json → decision_policy_comparison`.

**Gate condition 1** (`research/relearning/gate.py`, unmodified, test half):

| router | balance z | accuracy z | verdict |
|---|---|---|---|
| p_ml-only | +5.98 | −39.76 | FAIL (mix side) |
| blend | +2.80 | **+29.50** | FAIL — deferred pile is now *easier* for the ML counterfactual than random |

Under the blend, deferral is centred on the blend boundary, which the weak CBES
signal drags away from where the ML model is actually uncertain — so the
deferred pile stops isolating ML-hard cases. Neither router passes condition 1
on this half, but the blend flips the accuracy-z sign from strongly right to
strongly wrong.

## Alpha sweep (test half; thresholds refit on tune per α)

| α | AUC | default capture (rej.) | capture (rej.+defer) | defer rate | approve rate |
|---|---|---|---|---|---|
| 0.00 | 0.7669 | 61.7% | 82.1% | 22.4% | 51.7% |
| 0.05 | 0.7654 | 61.4% | 81.9% | 22.5% | 51.8% |
| 0.10 | 0.7613 | 57.5% | 79.5% | 22.5% | 54.5% |
| 0.15 | 0.7553 | 59.4% | 79.8% | 22.6% | 52.5% |
| 0.20 | 0.7479 | 53.9% | 76.2% | 22.7% | 56.0% |
| **0.25** | **0.7393** | **53.5%** | **75.1%** | **22.6%** | **55.5%** |
| 0.30 | 0.7297 | 51.6% | 73.5% | 22.6% | 55.9% |
| 0.35 | 0.7193 | 52.3% | 73.4% | 22.4% | 54.2% |
| 0.40 | 0.7082 | 50.9% | 72.4% | 22.4% | 53.8% |
| 0.45 | 0.6965 | 47.9% | 70.2% | 22.4% | 54.5% |
| 0.50 | 0.6843 | 48.5% | 70.7% | 22.5% | 52.2% |

Roughly linear damage: each +0.05 of α costs ~0.008 AUC and ~1.5 pp of default
capture. α = 0.05 buys a (slightly) slanted boundary for near-zero cost
(−0.0015 AUC); α = 0.25 is visibly slanted but costs ~2.8 AUC points and ~8 pp
of caught defaulters.

## Bottom line

The boundary is now genuinely slanted — e.g. applicants HC305089 and HC248970
have p_ml 0.856784 vs 0.856785 but p_cbes 0.851 vs 0.595, and get APPROVE vs
REJECT. That is exactly what was asked for, and it is now honest to the engine's
documented blend. But at the engine's current α = 0.25 it **costs −0.028 AUC,
−8.4 pp default capture, and ~849 extra auto-approved defaulters per 153k
applications**, and it degrades the deferral router's hard-case isolation.
If the slant is wanted mostly for legibility, α ≈ 0.05–0.10 delivers it at a
fraction of the cost; keeping α = 0.25 should be a deliberate choice made with
this table in view.
