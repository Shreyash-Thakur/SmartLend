# Presentation Guide — slide order, accuracy defence, version rule

Written to implement three reviewer instructions from the September review:
1. *"Keep the CBES page in the earlier pages"* → CBES gets slides 3–4, before any ML result.
2. *"The version of the model should always be mentioned"* → the version rule below.
3. *"Accuracy must be looked at"* → the accuracy section below, with the numbers to show.

The PPT itself lives outside the repo; build or reorder it against this guide.
Speak from `docs/DEFENCE-DECISION-HYBRID.md` and `docs/DEFENCE-SHAP-CBES.md`.

---

## Recommended slide order

| # | Slide | Content and source |
|---|---|---|
| 1 | Title + problem | Loan decisioning with a human-in-the-loop deferral layer; 307,511 real applications (Home Credit + bureau). |
| 2 | System overview | The three-layer diagram: ML model, CBES, deferral router → human review → gated relearning capture. One slide, no numbers yet. |
| **3** | **CBES — what it is** *(early, per reviewer)* | 8 portable fields → five weighted pillars (0.35/0.30/0.20/0.10/0.05) → sigmoids → p_cbes. Percentile-calibrated thresholds from the real distribution, not invented cutoffs. Source: `docs/DEFENCE-SHAP-CBES.md` §2. |
| **4** | **CBES — what it is for (and not for)** | Interpretability (pillar breakdown a customer/regulator can read), portability (8 fields vs 129), bounded influence (α = 0.10 blend, ±0.0075 tilt). Standalone AUC **0.5650** — say it yourself before anyone asks; it is why CBES is an explanation tool, not a predictor. |
| 5 | Data | Home Credit verification checks (row count, default rate, bureau direction sanity); 14.3% thin-file applicants kept as explicit missingness. |
| 6 | Model leaderboard | The CV table from `docs/STATUS.md` — **always with the sample column** and each model's version/variant named (XGBoost OOF-CV; TabPFN-**2.5**; TabFM **1.0**; serving = calibrated LogisticRegression, `pipeline_v3_real.joblib`). |
| 7 | Accuracy, done honestly | The accuracy section below — this is a full slide, not a footnote. |
| 8 | **The finding** — inverted deferral | Before: 52% deferral, z = +38σ/+43σ. Mechanism: the ~0.31 calibration offset. One chart: deferral rate by model-confidence band. |
| 9 | The fix, measured | Candidate table from `docs/DEFERRAL-FIX.md`: uncertainty z = −99.7 at 22.7%; repaired disagreement variants all lose — the negative result IS the contribution. |
| 10 | Blend decision | α sweep table; 0.25 → 0.10 with the measured cost (−0.0056 vs −0.028 AUC). Decisions made with numbers, not defaults. |
| 11 | Foundation models | TabPFN-2.5 and Google TabFM: honest scores with row IDs, licences stated, fusion experiment pre-registered → verdict KILL. Colab-hosted TabPFN endpoint as the hardware story. |
| 12 | Convergence | The two plots from `backend/artifacts/plots/` (learning curve + boosting curve) and the tuning-found-nothing study — what is and is not exhausted. |
| 13 | Relearning loop + governance | Capture live (reason codes, confidence, measured time), gate shut (4 conditions, current numbers), 3% exploration arm. RBI FREE-AI framing from `docs/RBI-COMPLIANCE.md`. |
| 14 | Live demo | Submit an application (own data — reviewer item), show the deferral, the reviewer form, the decision report with engine version, optionally the agent briefing. |
| 15 | Limitations + future work | From README "Known limitations" + `docs/FUTURE-SCOPE.md` §4. Lead with them yourself. |

**Rule of thumb enforced by this order:** CBES (slides 3–4) is fully explained
before any accuracy number appears (slide 6), so its 0.565 AUC lands as
"the interpretable layer is weak as a predictor — and we measured what that
costs" rather than "why is this here?".

---

## The model-version rule (apply to every slide and every demo screen)

Any number shown must name **which model, which version, which sample**:

- **Serving**: "calibrated LogisticRegression, artifact `pipeline_v3_real.joblib`, engine `hybrid-2stage-5gate/2026-04-27+cbes-v2`" — visible live in the app header, `/health`, and every decision report.
- **Research reference**: "XGBoost, 5-fold out-of-fold CV, 307,511 rows, 0.7651 ± 0.0036".
- **TabPFN**: always "**TabPFN-2.5**" (v2 is the *Nature* paper, 2.5 is arXiv:2511.08667 — a preprint; `pip install tabpfn` gives v3, the checkpoint pins 2.5).
- **TabFM**: always "**TabFM 1.0** (google-research/tabfm, pytorch weights)".
- Never say "the model" unqualified. Never show an AUC without its evaluation sample.

If asked *"which model are you actually running right now?"* — open the org
dashboard: the header badge answers it (`Model: LogisticRegression ·
pipeline_v3_real.joblib`), and clicking any application's Decision Report shows
the engine version and threshold-artifact hash stamped at decision time.

---

## Accuracy — the defence (reviewer: "accuracy must be looked at")

Do not dodge accuracy; own it in three steps.

**1. Show why raw accuracy misleads here.** 8.07% default rate → "approve
everyone" scores 91.9% accuracy while catching zero defaulters. Concrete
project example: TabPFN once displayed accuracy 0.9193 / recall 1.0000 purely
because a missing probability column fell back to a shared 0.5 threshold
(fixed in commit 12bbf4f — tell this story, it is memorable and honest).

**2. Show accuracy done properly.** Every leaderboard model is now scored at
its **own Youden-selected operating threshold** (`model_metrics.csv`):
e.g. XGBoost accuracy 0.7128 with 76.8% specificity and 66.9% approve rate —
a real operating point, not the approve-everything artefact. Alongside it show
**default capture** (share of defaulters rejected) and **approve rate**, the
two numbers a lender actually trades off.

**3. State the ranking metric and why.** ROC-AUC (threshold-free ranking
quality) + PR-AUC (honest under imbalance). The thresholded metrics *derive
from* the ranking plus a chosen operating point; the operating point is a
business decision (see the t_base study, `reports/t_base_selection.json`,
which grounds it in Basel LGD / profit parameters rather than an arbitrary
sweep).

One-line answer for the viva: *"We looked at accuracy the way a lender must —
at a chosen operating point with its default-capture cost attached — and we
show precisely how the 92% version of the number is manufactured."*

---

## Demo checklist (before the defense)

- [ ] Backend + frontend running; org header shows the model badge.
- [ ] `SMARTLEND_DEFERRAL_MODE` decision made (legacy vs uncertainty) — and if flipped, `ENGINE_VERSION` bumped.
- [ ] Colab TabPFN endpoint live + `python -m backend.run_tabpfn_connection_check` → PASS (`reports/tabpfn_connection_run.json`).
- [ ] `ANTHROPIC_API_KEY` set → agent briefing demo on a deferred case.
- [ ] Own application submitted through the UI and adjudicated once in `/review` (shows capture + reviewer form + report end to end).
- [ ] `python -m pytest backend/tests research/tests -q` green.
- [ ] `python -m research.relearning.gate` — show the refusal live; it is a feature.
