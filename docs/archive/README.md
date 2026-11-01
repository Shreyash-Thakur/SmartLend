# Archived documents — historical record, NOT current facts

Everything in this directory is superseded. It is kept because the project's
story (synthetic era → real-data era → the deferral finding) is part of the
research narrative, but **no number or claim in these files should be quoted
as current**. The live documentation index is `docs/PROJECT-OVERVIEW.md`.

| File | What it was | Superseded by / why |
|---|---|---|
| `SMARTLEND_QUICK_REFERENCE.md` | April 2026 frontend build-prompt cheat sheet | Describes endpoints that never shipped (`/api/auth/login`, `/api/org/metrics`, `/:id/decide`). Real API: `backend/app/routers/*`. |
| `SMARTLEND_TECHNICAL_PROMPT.md` | April 2026 frontend build prompt | Same era; the frontend has long diverged. |
| `SMARTLEND_TYPESCRIPT_DEFINITIONS.md` | April 2026 planned TS types | Real types live in `frontend/src/types/`. |
| `FORM-REDESIGN.md` | The 14-field short-form proposal | Implemented, then deliberately superseded by the 32-field form (commit 73a7f78). Its real-bank-form/RBI rationale is still citable history. |
| `week-01-goal.md` | Week-1 plan (30 Aug – 5 Sep 2026) | Executed. Quotes pre-swap synthetic numbers (62.7% accuracy, 25.5% deferral) that are no longer meaningful. |
| `training.py` | April–Aug 2026 synthetic-era training script (was repo root) | Its data source (`synthetic_indian_loan_dataset.csv`) was deleted in the Home Credit swap; superseded by `backend/retrain_serving_model_v3.py` and the `research/` experiment scripts. Kept per its own header note so the history isn't silently lost. |
| `yuvaan_eda_cbes_v2.ipynb` | April 2026 EDA notebook on the synthetic Indian dataset (was repo root) | Dataset deleted; the real-data EDA lives in `research/` and `reports/*.json`. |
| `eda_graphs_indian.png` | Output figure of that notebook (was repo root) | Synthetic-era plot; no current number derives from it. |
