# SmartLend — Project Status

**Updated:** 22 September 2026 · **Defense:** November 2026 · **Weeks remaining:** ~7

---

## Headlines

**1. The deferral rule is diagnosed AND fixed — measured both ways.** The original disagreement router deferred the model's *most confident* cases (z = +38σ/+43σ, 52% deferral). Root cause confirmed: a fixed ~0.31 calibration offset between p_ml and p_cbes. The measured fix (model uncertainty, Chow's rule) scores z = **−99.7** at the required 22.7% rate. The regenerated artifact runs at **22.5% deferral** on the α = 0.10 blend. (`reports/deferral_fix.json`, `reports/blend_decision.json`)

**2. The serving model is real-data and leak-free.** `pipeline_v3_real.joblib` — calibrated LogisticRegression, 15 features built by the same derivation the live API uses, two target-leak columns removed. Held-out ROC-AUC 0.6919 (P(default) framing). Live scoring of a new application goes through it — verified end to end on 22 Sep (`reports/e2e_demo_run.json`).

**3. Two foundation-model baselines, both honestly scored with row IDs.** TabPFN-2.5: 0.7284 ± 0.0127 (5,000-row context, 8,000 scored rows). **Google TabFM 1.0: 0.6911 ± 0.0113 zero-shot on the exact same 8,000 rows** — matching the *trained* serving LogisticRegression on the same 15-field vocabulary. Both licences are non-commercial → research baselines only.

**4. Model version is now stamped and shown everywhere** (reviewer requirement). Every decision records `engine_version` + serving artifact + threshold hash in `_decision_meta`; `/health`, the org dashboard header, the customer decision panel and the audit report all display it. The old hardcoded `/health` values (synthetic-era AUC 0.710) are gone.

**5. Convergence analysis exists** (reviewer requirement). Learning curve (capacity-adaptive XGBoost + LogReg) and boosting-convergence curve: `reports/convergence.json`, plots in `backend/artifacts/plots/`.

---

## Where we are

| Area | Status | Notes |
|---|---|---|
| Real dataset (307,511 rows, 8.07% default) | ✅ Verified | synthetic era fully retired |
| Serving model retrained on real data | ✅ Done | v3 artifact, leak removed, cost-based t_base study |
| Deferral inversion diagnosed + fixed | ✅ **Measured** | fix opt-in via `SMARTLEND_DEFERRAL_MODE=uncertainty` |
| Blend weight decided | ✅ α = 0.10 | measured cost table in `docs/BLEND-DECISION.md` |
| Decision artifact regenerated | ✅ 22.5% deferral | was 52% |
| Reviewer feedback capture (reason codes, confidence, timed) | ✅ Live | `docs/RELEARNING-LOOP.md` |
| Model-version provenance end to end | ✅ Done (22 Sep) | UI badge, /health, report, decision meta |
| TabPFN-2.5 honest re-score with `SK_ID_CURR` | ✅ Done | `reports/tabpfn_scored_rows.csv` |
| Google TabFM baseline (reviewer item) | ✅ **Scored 22 Sep** | 0.6911 on paired rows; leaderboard row added |
| TabPFN hosted on Colab as live endpoint (reviewer item) | ✅ Built | `colab/tabpfn_colab_server.ipynb` + backend client + connection run — needs a Colab session to go live |
| Convergence analysis (reviewer item) | ✅ Done (22 Sep) | `research/analysis/convergence.py` |
| Reviewer-briefing agent (reviewer item) | ✅ Built | Claude-powered, advisory-only; needs `ANTHROPIC_API_KEY` |
| Foundation-model fusion analysis | ⏳ Running | complementarity re-run with TabPFN + TabFM through the id-join alignment gate |
| RBI compliance mapping (reviewer item) | ✅ Documented | `docs/RBI-COMPLIANCE.md` |
| Report / paper writing | ⏳ **Now** | October is writing, not experiments |

---

## Model leaderboard (out-of-fold CV; dashboard-live numbers)

| Model | ROC-AUC | Sample |
|---|---|---|
| **XGBoost** | **0.7651 ± 0.0036** | 307,511 OOF rows |
| CatBoost | 0.7643 | 307,511 OOF |
| LightGBM | 0.7631 | 307,511 OOF |
| Logistic Regression | 0.7378 | 307,511 OOF |
| Random Forest | 0.7376 | 307,511 OOF |
| TabPFN-2.5 | 0.7284 ± 0.0127 | 8,000 id-indexed rows, 5,000-row context |
| **TabFM 1.0 (Google)** | **0.6911 ± 0.0113** | same 8,000 rows, 5,000-row context, 15-feature serving frame |
| CBES (rule-based) | 0.5650 | 307,511 OOF |

Notes to state precisely:
- Earlier single-holdout numbers (XGB 0.7670, TabPFN 0.7446 on 61,503 rows) are real but superseded by the CV numbers the dashboard serves. Quote the interval, not the point.
- TabFM ran on the 15-feature serving vocabulary with a committed script (`research/analysis/score_tabfm.py`); TabPFN's original context feature set came from an uncommitted script, so cross-foundation-model comparisons are paired-rows but possibly different-features — say so if asked.
- Serving model ≠ research reference: LogisticRegression serves (calibration + simplicity), XGBoost is the analysis reference. Both versions always displayed.

## Convergence (new)

- **Boosting converges:** held-out AUC rises monotonically and early-stops (~440 rounds at lr 0.05); no divergence, no overfitting within horizon.
- **Learning curve:** capacity-adaptive protocol (per-size early stopping on a validation split carved from the training draw). Read the verdict in `reports/convergence.json` — if the half→full gain is below the ±0.0036 noise floor the curve has plateaued; otherwise more same-kind data still helps and the honest wording in the report applies.
- Together with the 14-trial tuning study (best +0.001, inside noise — `reports/tuning_cpu.json`), the remaining path to the ~0.80 Kaggle ceiling is **feature engineering on the unused Home Credit tables**.

## Relearning gate — regenerated 22 Sep on the current artifact

**Verdict: DO NOT OPEN THE LOOP (0/4 pass)** — but the numbers moved a lot:

| # | Condition | Status | Now (was, pre-fix) |
|---|---|---|---|
| 1 | Deferral beats random at isolating hard cases | FAIL | balance z **+0.47** (was +38.1); accuracy z **−2.74** (was +43.0). Near-random rather than inverted; the uncertainty router passes this condition but is opt-in. |
| 2 | Defer rate inside AUC-implied bound [8%, 16%] | FAIL | **22.49%** (was 51.8%) — 1.4× the bound; a capacity decision, expected to fail. |
| 3 | ≥1,000 exploration labels with observed outcomes | FAIL | 0 — capture just went live; outcomes need 12–24 months. |
| 4 | Written retraining design | FAIL | deliberately not written until 1–3 hold. |

---

## What needs a human (cannot be automated from here)

1. **Go live with the Colab TabPFN endpoint once:** run `python -m backend.export_tabpfn_context` (already run — files in `models/tabpfn/`), open `colab/tabpfn_colab_server.ipynb` in Colab (GPU), upload the context CSV, copy the printed URL/token into `.env`, then `python -m backend.run_tabpfn_connection_check`.
2. **Set `ANTHROPIC_API_KEY`** in `.env` to enable the reviewer-briefing agent.
3. **Submit your own application** through the UI (reviewer item) — the demo persona run (`reports/e2e_demo_run.json`) proves the pipeline; your own submission will be captured the same way and can then be adjudicated in `/review` to demo the full loop.
4. Decide whether to flip `SMARTLEND_DEFERRAL_MODE=uncertainty` for the defense demo (bump `ENGINE_VERSION` if so).

## Plan to November

| Weeks | Dates | Work |
|---|---|---|
| — | done | Real data, retrain, deferral fix, blend decision, capture, provenance, TabFM/TabPFN, convergence, agent, docs |
| 1 | 22 – 28 Sep | Finish foundation-model fusion analysis; freeze numbers; start report skeleton |
| 2–4 | 29 Sep – 17 Oct | **Report writing** (research freeze — no new experiments) |
| 5 | 18 – 24 Oct | Demo polish: Colab endpoint live, agent key, own application submitted, deferral-mode decision |
| 6 | 25 – 31 Oct | Report submission, guide feedback |
| 7+ | 1 – 15 Nov | **Defense** — rehearse from `docs/PRESENTATION-GUIDE.md` and the two defence briefs |

---

## Reference

| Document | Contents |
|---|---|
| `docs/PROJECT-OVERVIEW.md` | master overview + file map |
| `docs/REVIEW-FEEDBACK-2026-09.md` | reviewer feedback → action taken, item by item |
| `docs/PRESENTATION-GUIDE.md` | slide order, accuracy defence, version-mention rule |
| `docs/RBI-COMPLIANCE.md` | RBI Digital Lending Directions 2025 + FREE-AI mapping |
| `docs/REFERENCE.md` | CBES logic, citations, likely questions |
| `docs/DEFERRAL-FIX.md` / `docs/BLEND-DECISION.md` | the two measured fixes |
| `reports/*.json` | machine-readable results — every doc number traces to one |
