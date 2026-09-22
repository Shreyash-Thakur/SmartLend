# SmartLend

Loan decisioning system with a human-in-the-loop deferral layer, built on **307,511 real loan applications** (Home Credit Default Risk + credit-bureau aggregates). Final-year research project; the research contribution is a measured diagnosis — and fix — of an inverted deferral rule.

New here? Read **[`docs/PROJECT-OVERVIEW.md`](docs/PROJECT-OVERVIEW.md)** (what this is) and **[`docs/DEV-GUIDE.md`](docs/DEV-GUIDE.md)** (how to work on it).

---

## Quick start

```bash
# terminal 1 — backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# terminal 2 — frontend
cd frontend && npm run dev
```

**http://localhost:5173** · API docs at **http://localhost:8000/docs**

```bash
python -m pytest backend/tests research/tests -q
```

Optional services (both degrade to a calm 503 when unconfigured):
- **Remote TabPFN** — run `colab/tabpfn_colab_server.ipynb` on a Colab GPU, set `SMARTLEND_TABPFN_URL`/`SMARTLEND_TABPFN_TOKEN`, verify with `python -m backend.run_tabpfn_connection_check`.
- **Reviewer-briefing agent** (Claude) — set `ANTHROPIC_API_KEY`.

---

## Model leaderboard (out-of-fold cross-validation, live on the dashboard)

Numbers from `backend/artifacts/model_metrics.csv` — 5-fold out-of-fold CV over all 307,511 rows (foundation models: smaller id-indexed samples, stated per row). Each model is scored at its own Youden-selected operating threshold.

| Model | ROC-AUC | Notes |
|---|---|---|
| **XGBoost** | **0.7651 ± 0.0036** | reference model for all research analyses |
| CatBoost | 0.7643 | |
| LightGBM | 0.7631 | |
| Logistic Regression | 0.7378 | same class as the serving model |
| Random Forest | 0.7376 | |
| **TabPFN-2.5** | 0.7284 ± 0.0127 | zero-training foundation model, **5,000-row context**, 8,000 scored rows with saved `SK_ID_CURR` |
| **TabFM 1.0** (Google) | see `reports/tabfm_quick_run.json` | second foundation-model baseline, scored on the same 8,000 rows |
| CBES *(rule-based)* | 0.5650 | 8 fields, untrained — kept for interpretability, not accuracy |

**Read AUC, not accuracy.** Only 8.07% of applicants default, so approving everyone scores 92% accuracy while catching no defaults. Where accuracy is shown, it is computed at each model's own operating threshold — see `docs/PRESENTATION-GUIDE.md` for the defence of this.

**Foundation models are research baselines only**: TabPFN-2.5's licence is non-commercial *including outputs*; Google TabFM's weights are `tabfm-non-commercial-v1.0`. Neither may serve commercial decisions.

**Serving model** (what scores a live application): `pipeline_v3_real.joblib` — a calibrated LogisticRegression on 15 leak-free features, held-out ROC-AUC 0.6919 in P(default) framing (`reports/serving_model_retrain.json`). It trades AUC for calibration, simplicity and an exact train/serve feature contract. `/health` and the dashboard header always show the live model + artifact version.

---

## The finding (the research contribution)

The original deferral rule — send a human the cases where ML and CBES disagree by more than τ_D — was **inverted**: it deferred the cases the model was *most confident* about (gate z = +38σ balance / +43σ accuracy against a random-router null, at a 52% deferral rate).

**Root cause, confirmed by measurement:** ~65% of the "disagreement" signal is a fixed calibration offset (mean p_ml 0.92 vs mean p_cbes 0.61) that widens with model confidence. The defect is the signal, not the threshold — the incumbent stays inverted even at a matched 22.6% rate.

**The fix, measured:** at the required 20–25% deferral rate, plain **model uncertainty** (Chow's rule, `|p_ml − t_approve| < τ_U`) scores z = **−99.7** — humans finally get the hard cases. Notably, every *repaired* disagreement signal (rank, z-score, isotonic) also fixes the inversion but **loses to plain uncertainty** — the publishable negative result. Details: [`docs/DEFERRAL-FIX.md`](docs/DEFERRAL-FIX.md).

The production default is still the legacy router (deliberate demo-stability choice); the fix is wired behind `SMARTLEND_DEFERRAL_MODE=uncertainty`. The decision artifact was regenerated on the α=0.10 blend at a **22.5% deferral rate** (was 52%).

Two companion measurements close the loop:
- **No roster hybrid helps.** Every honest XGBoost+CBES combination is ≤ XGBoost alone; the best tree-family blend (+0.0021) is inside the ±0.0036 fold noise. See [`docs/FUTURE-SCOPE.md`](docs/FUTURE-SCOPE.md) and `reports/complementarity.json`.
- **Convergence measured.** Learning curve (AUC vs training rows, capacity-adaptive) and boosting-convergence curves in `reports/convergence.json` + `backend/artifacts/plots/`.

---

## Relearning loop

Reviewer decisions on deferred cases are captured live (verdict, mandatory reason codes, confidence, measured time-on-case). **Retraining on them is deliberately gated** and all four gate conditions currently fail — verdict **DO NOT OPEN THE LOOP**:

```bash
python -m research.relearning.gate     # exits non-zero while the loop must stay shut
curl localhost:8000/api/relearning/status
```

On the regenerated 22.5%-deferral artifact, condition 1 improved from +38σ to **+0.5σ** (near-random, still not *better* than random — the uncertainty router passes it, but only behind its opt-in flag). Conditions 2–4 fail for stated reasons (capacity above the AUC-implied bound; 0 of 1,000 exploration labels; no retraining design). A **3% exploration arm** collects the only labels the router did not select. See [`docs/RELEARNING-LOOP.md`](docs/RELEARNING-LOOP.md).

---

## Architecture

```
backend/app/       FastAPI — serves, never trains
research/          experiments — may import backend; never imported by the API
frontend/src/      React + TypeScript
backend/artifacts/ trained outputs the API reads
colab/             Colab notebook hosting TabPFN as a live GPU endpoint
reports/           machine-readable experiment results (one JSON per study)
```

- **ML model** — serving artifact on 15 features; research reference (XGBoost) on the full numeric frame
- **CBES** — 8 portable fields, five weighted pillars, percentile-calibrated thresholds; blend weight α = 0.10 (measured choice, `docs/BLEND-DECISION.md`)
- **Deferral layer** — legacy disagreement router by default; measured uncertainty router opt-in
- **Provenance** — every decision records `engine_version`, serving artifact and threshold hash; shown in the UI header, decision panels and audit report
- **Agent briefing** — optional Claude-generated reviewer briefing (advisory only, never a verdict)

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/PROJECT-OVERVIEW.md`](docs/PROJECT-OVERVIEW.md) | **start here** — what the project is, findings, file map |
| [`docs/DEV-GUIDE.md`](docs/DEV-GUIDE.md) | setup, layout, working agreements, pitfalls |
| [`docs/STATUS.md`](docs/STATUS.md) | current state, results, plan to the November defense |
| [`docs/REVIEW-FEEDBACK-2026-09.md`](docs/REVIEW-FEEDBACK-2026-09.md) | reviewer feedback → what was done about each item |
| [`docs/PRESENTATION-GUIDE.md`](docs/PRESENTATION-GUIDE.md) | slide order (CBES early), accuracy defence, model-version rule |
| [`docs/RBI-COMPLIANCE.md`](docs/RBI-COMPLIANCE.md) | mapping to RBI Digital Lending Directions 2025 + FREE-AI |
| [`docs/REFERENCE.md`](docs/REFERENCE.md) | CBES logic, citations, likely review questions |
| [`docs/DEFERRAL-FIX.md`](docs/DEFERRAL-FIX.md) · [`docs/BLEND-DECISION.md`](docs/BLEND-DECISION.md) | the two measured fixes |
| [`docs/DEFENCE-DECISION-HYBRID.md`](docs/DEFENCE-DECISION-HYBRID.md) · [`docs/DEFENCE-SHAP-CBES.md`](docs/DEFENCE-SHAP-CBES.md) | viva speaking scripts |
| [`docs/RELEARNING-LOOP.md`](docs/RELEARNING-LOOP.md) | capture flow and gate conditions |
| [`models/README.md`](models/README.md) | model weights, licence and citation traps |
| [`docs/archive/`](docs/archive/README.md) | superseded documents — history, not current facts |

---

## Known limitations

- The **legacy (inverted) deferral router is still the production default**; the measured fix is opt-in via `SMARTLEND_DEFERRAL_MODE=uncertainty`. Flipping it must bump `ENGINE_VERSION`.
- The serving model (calibrated LogisticRegression, 0.6919 AUC) is deliberately weaker than the research reference (XGBoost 0.7651) — a calibration/simplicity trade-off, stated, not hidden.
- CBES at ~0.565 AUC is close to uninformative as a predictor; it exists for interpretability and its blend weight is capped at α = 0.10 for a measured cost.
- TabPFN-2.5 and Google TabFM are non-commercial-licensed research baselines, not deployable models.
- Real repayment outcomes take 12–24 months to season; the relearning gate stays shut until its four conditions pass.
