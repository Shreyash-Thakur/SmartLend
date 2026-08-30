# SmartLend

Loan decisioning system with a human-in-the-loop deferral layer, built on **307,511 real loan applications** (Home Credit Default Risk + credit-bureau aggregates).

New here? Read **[`docs/DEV-GUIDE.md`](docs/DEV-GUIDE.md)** first.

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

---

## Model performance

Trained on 246,008 rows, evaluated on a held-out **61,503 rows**. Stratified, seed 42.

| Model | ROC-AUC | PR-AUC | Training rows |
|---|---|---|---|
| **XGBoost** | **0.7670** | 0.2624 | 246,008 |
| LightGBM | 0.7667 | 0.2632 | 246,008 |
| CatBoost | 0.7666 | 0.2638 | 246,008 |
| **TabPFN-2.5** | **0.7446** | 0.2291 | **5,000** |
| Logistic Regression | 0.7405 | 0.2226 | 246,008 |
| Random Forest | 0.7388 | 0.2204 | 246,008 |
| CBES *(rule-based)* | 0.5638 | 0.0997 | — |

**Read AUC, not accuracy.** Only 8.07% of applicants default, so approving everyone scores 92% accuracy while catching no defaults. Accuracy on this dataset is not evidence.

**TabPFN-2.5 is worth noting**: a training-free tabular foundation model given 5,000 rows — 2% of the data — outranks Logistic Regression and Random Forest, both trained on all 246,008. Constrained to 5,000 by 8 GB of VRAM; Prior Labs recommend A100/H100-class hardware. Its licence is **non-commercial and covers outputs**, so it is a research baseline, not a deployable model.

---

## What we found

The system defers uncertain applications to a human reviewer. **That mechanism is inverted**, and this is the project's main finding rather than an outstanding bug.

| Model confidence | Applicants | Deferred |
|---|---|---|
| 0.4–0.6 — genuinely uncertain | 280 | **4.3%** |
| 0.8–1.0 — highly confident | 22,942 | **54.8%** |

It defers the cases the model is *sure* about and keeps the ones it is unsure about. Measured against a random-router baseline, it sits **10.5σ** the wrong way on class balance and **12.2σ** on accuracy. It also defers **52%** of applications against an AUC-implied ceiling of **15.9%**.

**Mechanism:** deferral triggers on `|p_ml − p_cbes| > 0.43`, but CBES scores 0.5638 AUC (random = 0.5) and sits ~0.31 below the ML score population-wide. That difference measures a *scale mismatch* between two differently-calibrated scores, not genuine disagreement. It fires where the offset is widest — which tracks confidence, not difficulty.

This is why three earlier fixes failed: each tuned a threshold on a signal pointing the wrong way.

---

## Relearning loop

Reviewer decisions on deferred cases are captured live. **Retraining on them is deliberately gated.**

```bash
python -m research.relearning.gate     # exits non-zero while the loop must stay shut
curl localhost:8000/api/relearning/status
```

All four gate conditions currently fail. The load-bearing one is the router itself: it selects which cases get human labels, so retraining on its output means learning from a sample its own broken policy chose — the bias compounds each cycle while metrics look healthy.

A **3% exploration arm** randomly routes would-be-auto-decided applications to human review. Those are the only labels the router did not select, and they are what will eventually prove it is fixed.

See [`docs/RELEARNING-LOOP.md`](docs/RELEARNING-LOOP.md).

---

## Architecture

```
backend/app/       FastAPI — serves, never trains
research/          experiments — never imported by the API
frontend/src/      React + TypeScript
backend/artifacts/ trained outputs the API reads
```

- **ML model** consumes all 129 native features
- **CBES** consumes 8 portable fields across five weighted pillars, thresholds calibrated from real percentiles
- **Deferral layer** routes on disagreement between them

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/DEV-GUIDE.md`](docs/DEV-GUIDE.md) | **start here** — setup, layout, pitfalls |
| [`docs/STATUS.md`](docs/STATUS.md) | current state, results, plan |
| [`docs/REFERENCE.md`](docs/REFERENCE.md) | CBES logic, citations, likely review questions |
| [`docs/RELEARNING-LOOP.md`](docs/RELEARNING-LOOP.md) | capture flow and gate conditions |
| [`docs/FORM-IMPLEMENTATION.md`](docs/FORM-IMPLEMENTATION.md) | the shortened application form |
| [`docs/VOICE-MODULE.md`](docs/VOICE-MODULE.md) | voice setup, provider swapping |
| [`models/README.md`](models/README.md) | model weights, licence and citation traps |

---

## Known limitations

- **Live scoring still uses an April artifact trained on synthetic data.** The dashboard is real; scoring a *newly submitted* application is not. Retraining the serving artifact is outstanding work.
- The deferral rule is inverted (above) and should not be trusted for decisions.
- Accuracy/precision/recall on the dashboard are computed in approval framing and mostly measure the easy direction. Use AUC.
- CBES at 0.5638 AUC is close to uninformative as a predictor; it exists for interpretability.
