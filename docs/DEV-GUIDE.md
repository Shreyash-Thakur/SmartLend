# SmartLend — Developer Guide

**For anyone picking this up.** Start here, then read `docs/STATUS.md` for where the project is.

---

## Run it

```bash
# terminal 1 — backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# terminal 2 — frontend
cd frontend && npm run dev
```

Open **http://localhost:5173**. Vite proxies `/api` → port 8000.
API docs: **http://localhost:8000/docs**

```bash
python -m pytest backend/tests research/tests -q     # full suite
```

---

## What this project actually is

A loan decisioning system **and** a research project. The two have different goals and it matters which one you're touching.

- **The system** — FastAPI + React, scores applications, routes uncertain ones to a human reviewer.
- **The research** — measuring *when a deferral guarantee is actually valid* under credit selection bias. This is the part that gets published.

The headline finding is **negative and deliberate**: the original deferral mechanism was inverted — it sent humans the *easy* cases (up to ~38σ worse than a random router). It has since been **diagnosed (a fixed calibration offset between the two scores), fixed (model uncertainty, gate z = −99.7), and re-measured** — see `docs/DEFERRAL-FIX.md`. The measured fix is opt-in (`SMARTLEND_DEFERRAL_MODE=uncertainty`); the legacy router remains the demo default. Don't "fix" the legacy signal by tuning its threshold — the defect is the signal, and that diagnosis is the contribution.

---

## Repository layout

```
backend/app/          FastAPI service - serves, never trains
  routers/            HTTP endpoints
  services/           business logic
  models.py           SQLAlchemy tables
  config.py           ALL secrets go through here
backend/artifacts/    trained model outputs the API reads
research/             experiments - never imported by the API
  data/               canonical schema, missingness profiler
  relearning/         the retraining gate
frontend/src/         React + TypeScript
docs/                 you are here
reports/              machine-readable experiment results
models/               model weights (gitignored - see models/README.md)
```

**The `backend/` ↔ `research/` split is load-bearing.** `research/` may import from `backend/`; the reverse must never happen. The API loads artifacts, it does not train.

---

## The data

**Home Credit Default Risk** — 307,511 real loan applications, 8.07% default rate, plus merged credit-bureau aggregates. Not in the repo (too large); see `docs/STATUS.md` for how to obtain it.

Two things about this dataset that trip people up:

**1. Label direction is inverted between layers.** Home Credit's `TARGET = 1` means the customer **defaulted**. The dashboard contract uses `y_true = 1` to mean **approve** (a good customer). So artifacts store `y_true = 1 - TARGET` and probabilities as `P(approve) = 1 - P(default)`. Get this backwards and every chart renders upside-down while still looking plausible.

**2. 14.3% of applicants (44,020) have no credit-bureau record at all.** These are kept as NULL plus a missingness indicator, deliberately **not** filled with zeros. "No overdue loans" and "no information" are different facts, and these thin-file applicants are exactly the population the research is about. Do not impute them away.

---

## Accuracy is a trap on this dataset

Only 8% of applicants default, so **approving everyone scores 92% accuracy**. That exact artefact happened here: TabPFN once showed accuracy 0.9193 / recall 1.0000 because a missing probability column fell back to a shared 0.5 threshold.

**Judge models by AUC and PR-AUC.** Where the dashboard shows accuracy/precision/recall, each model is now scored **at its own Youden-selected operating threshold** (commit c64c692), which is the honest version of those numbers — the defence of this is written out in `docs/PRESENTATION-GUIDE.md`.

---

## The relearning loop — read before touching

Reviewer decisions on deferred cases are captured live. **Retraining on them is deliberately gated.**

```bash
python -m research.relearning.gate      # exits non-zero while the loop must stay shut
```

Four conditions must all pass. They currently all fail — but read the current numbers, not folklore: on the regenerated 22.5%-deferral artifact the deferred pile is statistically indistinguishable from random (balance z +0.5, was +38), and the opt-in uncertainty router passes condition 1 outright. Conditions 2–4 fail for stated reasons (capacity above the AUC bound; 0 of 1,000 exploration labels; no retraining design). Current table: `docs/RELEARNING-LOOP.md` §7.

**Why this is gated rather than just built:** the router chooses which cases get human labels. Retraining on those labels means the model learns from a sample its own broken policy selected, and the bias compounds every cycle while the metrics look fine. That is the runaway feedback loop (Ensign et al.), compounded by the selective-labels problem (Lakkaraju et al., KDD 2017).

There is a test that greps the service module for `def train` / `retrain` / `.fit(` / `partial_fit` / `reject_inference`. **If you wire up retraining, that test fails on purpose.** Read `docs/RELEARNING-LOOP.md` first.

The **exploration arm** (~3% of would-be-auto-decided applications randomly routed to human review) is the escape hatch — the only source of labels the router did not select. Do not disable it; it is what will eventually prove the router is fixed.

---

## Secrets

Everything goes through `backend/app/config.py`. Never read `os.environ` directly for a secret, never log a secret value.

```bash
cp .env.example .env    # then fill in real values
```

`.env` is gitignored. `.env.example` is tracked and must never contain a real key.

---

## Working agreements

- **Never commit** model weights, datasets, `.claude/`, or `catboost_info/` — all gitignored, and there is ~800MB of it sitting untracked.
- **Never tune a threshold on the test split.** There is one locked holdout, opened at the end. The original `TAU_D` was tuned to hit a target deferral rate, and that failure is why this rule exists; every later study (deferral fix, blend decision, t_base selection) uses an explicit tune/test split discipline — copy theirs.
- **Artifacts are regenerated, not hand-edited.** If `model_metrics.csv` looks wrong, rerun the training script.
- **A stale number is worse than no number.** The dashboard once showed an April synthetic accuracy beside live values for days, because a lookup preferred a training-time file over live computation. If a value can go stale, compute it.

---

## Where to read next

| Document | Contents |
|---|---|
| `docs/PROJECT-OVERVIEW.md` | master overview + file map (start here) |
| `docs/STATUS.md` | current state, results, plan, next actions |
| `docs/REFERENCE.md` | CBES logic, citations, likely review questions |
| `docs/DEFERRAL-FIX.md` / `docs/BLEND-DECISION.md` | the two measured fixes |
| `docs/RELEARNING-LOOP.md` | data flow, gate conditions, how to open the loop |
| `docs/FORM-IMPLEMENTATION.md` | the application form |
| `docs/VOICE-MODULE.md` | voice setup and provider swapping |
| `models/README.md` | fetching model weights, licence traps |
| `colab/tabpfn_colab_server.ipynb` | hosting TabPFN on a Colab GPU as a live endpoint |
| `docs/superpowers/specs/` | full research design |

### Optional services (all degrade to a calm 503 when unconfigured)

- **Remote TabPFN**: `backend/app/services/remote_tabpfn_service.py` ← `SMARTLEND_TABPFN_URL`/`_TOKEN` from the Colab notebook; verify with `python -m backend.run_tabpfn_connection_check`.
- **Reviewer-briefing agent** (Claude): `backend/app/services/underwriting_agent_service.py` ← `ANTHROPIC_API_KEY`. Advisory only — it must never write a decision.
- **Voice**: `backend/app/services/voice_service.py` ← `ELEVENLABS_API_KEY`/`SARVAM_API_KEY`.
