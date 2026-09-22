# SmartLend — Project Overview (the master document)

**Read this first.** One document that says what the project is, what it found,
where everything lives, and what remains — written so a future session (human
or AI) can be productive without re-deriving the project from the code.

**What it is:** a final-year research project (defense: November 2026) being
evaluated for research publication. A loan decisioning system — ML model +
interpretable rule score (CBES) + human-in-the-loop deferral + gated relearning
capture — built on 307,511 real Home Credit applications, whose research
contribution is a *measurement discipline*: an inverted deferral rule
diagnosed, fixed, and re-measured, plus a chain of pre-registered negative
results about hybrids.

---

## 1. The system in one diagram

```
application (32-field form, or short form + customer_id → bank profile merge)
    │
    ├── ML score  p_ml   = P(approve)   serving artifact: pipeline_v3_real.joblib
    │                                   (calibrated LogisticRegression, 15 leak-free
    │                                    features, held-out AUC 0.6919 P(default))
    ├── CBES      p_cbes = P(approve)   8 fields → 5 weighted pillars → sigmoids;
    │                                   percentile-calibrated; standalone AUC 0.5650
    │
    ├── blend     p_blend = 0.90·p_ml + 0.10·p_cbes   (α measured, docs/BLEND-DECISION.md)
    │
    ├── router    default: disagreement |p_ml−p_cbes| > τ_D   (the DIAGNOSED-INVERTED rule)
    │             opt-in:  uncertainty |p_ml−t_approve| < τ_U (the MEASURED FIX, z = −99.7)
    │             → APPROVE / REJECT / DEFER (≈22.5% deferral on the current artifact)
    │
    ├── DEFER (or 3% exploration coin-flip) → deferred_reviews capture row
    │             reviewer decides in /review: verdict + mandatory reason codes +
    │             confidence 1–5 + measured time-on-case
    │
    └── relearning gate (research/relearning/gate.py): 4 conditions, all FAIL →
                  "DO NOT OPEN THE LOOP" — retraining deliberately forbidden
```

Every decision is stamped with `engine_version`, serving artifact name and a
threshold-artifact hash; `/health`, the org header badge, the decision panels
and the per-application audit report all display it.

## 2. The findings (what gets published)

1. **The deferral inversion.** The ML-vs-rule disagreement router, measured on
   real data, deferred the model's MOST confident cases: z = +38σ/+43σ vs a
   random-router null, at 52% deferral. Root cause confirmed by measurement: a
   fixed ~0.31 calibration offset between p_ml (mean 0.92) and p_cbes (mean
   0.61) that widens with confidence; at a matched rate the signal stays
   inverted, so the defect is the signal, not the threshold.
   → `reports/deferral_fix.json`, `docs/DEFERRAL-FIX.md`
2. **The fix and the embedded negative result.** At the required 20–25% rate,
   plain model uncertainty (Chow's rule) scores z = −99.7. Every *repaired*
   disagreement signal (rank/z-score/isotonic) fixes the inversion but loses
   to plain uncertainty — disagreement-based deferral needs a competent
   partner, and CBES (0.565 AUC) isn't one.
3. **The hybrid chain of negative results, pre-registered.** XGBoost+CBES:
   optimal weight is zero. Tree family: best gain +0.0021 < ±0.0036 noise.
   TabPFN-2.5 (the one candidate with a plausible mechanism — corr 0.83, not
   0.99): pre-registered decision rule returned **KILL** (loses in all 16
   segments; best honest hybrid −0.0008). Google TabFM goes through the same
   gate. → `reports/complementarity.json`, `docs/FUTURE-SCOPE.md`
4. **Foundation-model baselines, honestly scored.** TabPFN-2.5: 0.7284±0.0127
   from a 5,000-row context; TabFM 1.0: 0.6911±0.0113 zero-shot ≈ a trained
   LogisticRegression on the same 15 fields. Both non-commercial licences.
5. **Convergence, honestly reported.** Boosting converges; tuning is exhausted
   (+0.001, inside noise); the capacity-adaptive learning curve is NOT yet
   flat (100k→246k ≈ +0.010 AUC) — more same-kind data would still help, and
   the rest of the ~0.80-ceiling gap is feature engineering.
   → `reports/convergence.json`
6. **Features beat fusion, under one pre-registered rule.** 319 engineered
   features (application ratios, EXT_SOURCE combinations, ~35 deep bureau
   aggregates) lift every model in every fold — best single 0.7603 → 0.7757
   OOF (+0.0155, 4× the noise floor) — while the best honest ensemble of the
   improved models gains +0.0016, below the same floor. Headroom on this
   dataset is information, not fusion. → `reports/features_vs_fusion.json`,
   `docs/FUTURE-SCOPE.md` §5
7. **Governance as architecture.** Relearning capture is live but retraining is
   gated behind four measured conditions (runaway-feedback-loop / selective-
   labels defence); a 3% exploration arm collects the only unselected labels.
   Aligned to RBI FREE-AI human-in-the-loop expectations. → `docs/RBI-COMPLIANCE.md`

**Direction conventions (the #1 way to get confused):** in serving code and
`prediction_outputs.csv`, `y_true==1` = GOOD and probabilities are P(approve);
in `reports/complementarity.json` and training labels, 1 = DEFAULT. SHAP is in
P(default) log-odds — opposite orientation to p_ml.

## 3. Repository map

| Path | What lives there |
|---|---|
| `backend/app/routers/` | HTTP API: applications, customers, relearning, tabpfn, agent, voice, public |
| `backend/app/services/` | decision_engine (blend α, gates, ENGINE_VERSION), ml_service (artifact loading, SHAP, provenance), cbes_engine + cbes_calibration, deferred_review_service (capture), decision_report_service (audit record), relearning_service (fail-closed status), remote_tabpfn_service, underwriting_agent_service, explainability, voice, parser |
| `backend/artifacts/` | serving pipelines (v3 real is live; v1/v2 rollback), `model_metrics.csv` leaderboard, `prediction_outputs.csv` (307,511 OOF rows), CBES thresholds, plots |
| `backend/retrain_serving_model_v3.py` | the only live trainer (leak-free 15-feature contract) |
| `backend/export_tabpfn_context.py` / `run_tabpfn_connection_check.py` | Colab-endpoint support |
| `research/data/` | canonical schema, dataset specs (home_credit, lending_club), missingness profiler |
| `research/deferral/` | the six candidate signals + matched-rate evaluation protocol |
| `research/relearning/gate.py` | the four-condition gate (CI-runnable, exits non-zero while shut) |
| `research/blend/regenerate.py` | regenerates decisions on p_blend (α from the engine, never hardcoded) |
| `research/thresholds/t_base.py` | threshold-method study (cost method recommended) |
| `research/analysis/` | complementarity (fusion + alignment gates), convergence, score_tabfm |
| `frontend/src/pages/` | Landing, dashboards (customer/org/models), CustomerNewApplication, ReviewPage, ApplicationReview, GeoAnalytics |
| `colab/tabpfn_colab_server.ipynb` | TabPFN on a Colab GPU as an authenticated live endpoint |
| `reports/` | one machine-readable JSON per study — **every documented number traces here** |
| `docs/` | see the index in `README.md`; `docs/archive/` = history, not facts |

**Load-bearing split:** `research/` may import `backend/`; the API never
imports `research/` and never trains. Grep-tests enforce no training code in
the capture layer.

## 4. Environment & running

- Python 3.13 venv at `.venv`; `backend/requirements-api.txt` must keep the app importable. Node/Vite frontend (`npm run dev`, proxies `/api` → 8000).
- Dataset (not in repo): `creddefer_full_merged.csv` — resolved via `SMARTLEND_CUSTOMER_DATA` env var or the default Downloads path (`customer_profile_service._resolve_source_path`). Raw Kaggle files in `data/raw/home_credit/`.
- Secrets via root `.env` through `backend/app/config.py` only. Keys: `ELEVENLABS_API_KEY`, `SARVAM_API_KEY`, `ANTHROPIC_API_KEY`, `SMARTLEND_TABPFN_URL`/`_TOKEN`, `SMARTLEND_EXPLORATION_RATE`, `SMARTLEND_DEFERRAL_MODE`/`SMARTLEND_TAU_U`, `SMARTLEND_CUSTOMER_DATA`.
- Tests: `python -m pytest backend/tests research/tests -q` (≈130 tests, all green as of 22 Sep 2026).
- GPU: RTX 4060 Laptop 8 GB — fits TabPFN at 5,000-row context; the Colab endpoint exists to go beyond.

## 5. Current state & what remains

See `docs/STATUS.md` (kept current) for the full table. In brief, as of
2026-09-22: research work is complete except the TabFM fusion verdict landing
in `reports/complementarity.json`; October is report writing; three human
steps remain (Colab session for the TabPFN endpoint, `ANTHROPIC_API_KEY`,
submitting one's own application) plus the deliberate decision whether to flip
`SMARTLEND_DEFERRAL_MODE=uncertainty` for the demo (bump `ENGINE_VERSION` if
so). Reviewer feedback disposition: `docs/REVIEW-FEEDBACK-2026-09.md`.

## 6. Rules that must survive any future session

1. **Never quote a number without its source report and sample.** Every claim
   traces to `reports/*.json`; when two numbers disagree, the newer report and
   the CV framing win, and the doc must say which slice it quotes.
2. **Never tune on the test split; never hand-edit artifacts;** regenerate via
   the committed script (and commit the script — the era of uncommitted
   one-off runs caused the TabPFN alignment failure).
3. **Never wire retraining** (the gate and grep-tests will fight you; that is
   by design). The exploration arm must stay on.
4. **Flipping the deferral mode or blend weight requires bumping
   `ENGINE_VERSION`** so capture rows stay separable.
5. **AI components stay advisory.** The agent briefing and TabPFN second
   opinion must never write a decision — licence and governance both forbid it.
6. **State limitations before the examiner finds them** — the defence docs are
   built on that habit; keep it.
