# Reviewer Suggestions → What We Did → Results

**For the presentation.** Every number below traces to a committed artifact in
`reports/*.json` (named per item). Written 2026-10-09, after all engineering
closed. Items quoted as the reviewers gave them in September.

---

## 1. "Keep the CBES page in the earlier pages of the PPT"

**Did:** Wrote the slide-order guide — `docs/PRESENTATION-GUIDE.md` places the
CBES architecture slide in the opening third, before any results.
**Status:** guide ready; the actual PPT reorder is a manual step.

## 2. "Version of model should always be mentioned"

**Did:** Every decision now records and displays `engine_version`, serving
artifact name, and a threshold-artifact hash. Visible on: `/health`, the org
sidebar badge ("Serving model"), the customer decision card, the audit report.
**Result:** live engine version string:
`hybrid-2stage-5gate/2026-04-27+cbes-v2+model=LogisticRegression`, artifact
`pipeline_v3_real.joblib`. Demonstrated working on the cloud deployment.
**Soundbite:** *"No decision leaves the system without its model's fingerprint."*

## 3. "TabFM is an open-source project on Google"

**Did:** Installed Google TabFM 1.0 (Apache-2.0 code, non-commercial weights)
and evaluated it on 8,000 id-verified rows — identical rows to TabPFN, with a
row-alignment gate (0 label mismatches, AUC reproduces to 4 decimals).
**Result (`reports/tabfm_quick_run.json`, `reports/complementarity.json`):**

| Model (same 8,000 rows) | AUC |
|---|---|
| Our XGBoost | **0.7595** |
| TabPFN-2.5 | 0.7284 ± 0.0127 |
| **Google TabFM 1.0** | **0.6911 ± 0.0113** |

TabFM was worse in **every** segment (deltas −0.04 to −0.11). Fusing it:
simple average **hurts** (bootstrap CI [−0.022, −0.007]); best honest hybrid
−0.0004 vs XGBoost alone → KILL. Notably TabFM is the most *decorrelated*
partner we ever tested (corr 0.66) — diversity did not help because it is
diverse by being wrong.
**Soundbite:** *"We tested Google's tabular foundation model on identical,
id-verified rows: 0.69 vs our 0.76 — and fusing it made things worse."*

## 4. "Accuracy must be looked at"

**Did:** The pre-registered **features-vs-fusion** experiment: same models,
same 5 folds (seed 42), change only features (v1 current frame → v2 = 319
engineered features from application ratios + deep bureau aggregates). Both
claims judged by the SAME ±0.0036 fold-noise rule that killed every hybrid.
**Result (`reports/features_vs_fusion.json`):**

| Model | v1 AUC | v2 AUC | fold-paired Δ |
|---|---|---|---|
| **CatBoost** | 0.7603 | **0.7757** | **+0.0155** |
| XGBoost | 0.7584 | 0.7749 | +0.0164 |
| LightGBM | 0.7560 | 0.7721 | +0.0160 |
| LogisticRegression | 0.7377 | 0.7560 | +0.0182 |

All 20 fold deltas positive; every gain ≥4× the noise floor. Fusion on the
improved models: best honest combo +0.0016 → **below the floor, denied again**.
Attribution: engineered features carry **56% of top-25 gain** (`APP_EXT_MEAN`
alone 30%).
**Soundbite:** *"Feature enrichment cleared the noise floor for every model in
every fold; no model combination ever has. Headroom lives in information, not
fusion."*

## 5. "Apply for a loan on your own data so it enters the system"

**Did:** Submitted real applications through the live UI — locally and on the
AWS deployment. One (₹20,000 × 36m) was DEFERred by the engine into the human
review queue, demonstrating capture end to end; the relearning capture row
recorded it (3% exploration arm also observed firing on an APPROVE).
**Soundbite:** *"Our own applications flow through the deployed system —
including one the engine refused to auto-decide."*

## 6. "Host TabPFN on Colab (it is ~10 GB)"

**Did:** `colab/tabpfn_colab_server.ipynb` — serves the pinned TabPFN-2.5
checkpoint on a Colab GPU behind a Cloudflare tunnel with session-token auth;
backend `remote_tabpfn_service` consumes it as an advisory second opinion
(never writes the final decision; licence is non-commercial).
**Result:** ran live on a **Tesla T4 with a 10,000-row context** — double the
5k cap of our local 8GB GPU.

## 7. "Build a connection run"

**Did:** `backend/run_tabpfn_connection_check.py` — four gates: configured →
reachable → scores → **sane** (recomputes AUC on 400 labelled rows the Colab
never saw, so a tunnel serving garbage fails loudly).
**Result (`reports/tabpfn_connection_run.json`):** PASS — remote AUC **0.7100**
on 400 held-out rows in 27s, consistent with TabPFN's known 0.7284 ± 0.0127.
**Soundbite:** *"The connection run doesn't just ping the endpoint — it proves
the remote model is the real one by scoring labelled rows."*

## 8. "Convergence analysis (over computational analysis)"

**Did:** Two curves (`reports/convergence.json`, plots in
`backend/artifacts/plots/`): capacity-adaptive learning curve (per-size early
stopping — a fixed-capacity curve underfits and lies) and boosting-iteration
convergence.
**Result:** iterations converge (early stopping works; tuning found nothing),
but the learning curve is **still rising at 246k rows** — doubling data buys
+0.010 AUC. Honest verdict: not yet data-converged; more of the same data
still helps.
**Soundbite:** *"Optimisation is converged; information is not — exactly where
our feature-engineering result says the headroom is."*

## 9. "Hyperparameter tuning, and fusion with Google TabFM"

**Did:** Tuning: measured exhausted (`reports/tuning_cpu.json` — no
configuration beat early-stopped defaults beyond noise). TabFM fusion: see
item 3 — measured, KILL.
Also closed the question *forward*: screened 2026 SOTA challengers under
pre-registered rules (`reports/challenger_screen.json`): **RealMLP**
(TabArena-competitive deep net) lost **every fold** to CatBoost (fold-paired
−0.0121); **TabICLv2** could not run at even 30k rows on commodity hardware
(needs ~18GB scratch per fold — a deployability finding in itself).
**Soundbite:** *"Tuning, four fusions, and two 2026 SOTA challengers: all
measured, none beat boosted trees on engineered features."*

## 10. "Agents and live data online instead of offline"

**Did:** (a) Claude-powered underwriting agent briefing — advisory panel,
never writes decisions, degrades to calm 503 without a key; the deployment
pipeline carries `ANTHROPIC_API_KEY` when configured. (b) Live data literally:
full system deployed on AWS (EC2 + RDS + ECR, Terraform-managed), public
internet, guest login, real submissions scored live and persisted.
**Soundbite:** *"Not simulated-live: the system ran on the public internet and
strangers' clicks became rows in the governed capture table."*

## 11. "Check recent RBI guidelines"

**Did:** `docs/RBI-COMPLIANCE.md` — mapped against the RBI Digital Lending
Directions 2025 and the FREE-AI framework (7 sutras / 6 pillars / 26
recommendations). Strengthened by a measured result: **21 monotonicity
constraints** on domain-signed features make directional adverse-action
explanations true *by construction* at a cost of **−0.0008 AUC (inside
noise), 0 violations in 14,400 checks** (`reports/monotone_catboost.json`).
**Soundbite:** *"Regulatory explainability as a model property, priced at
zero."*

---

## Bonus results (not requested, worth slides)

- **Epistemic-uncertainty deferral (novelty):** SGLB CatBoost virtual
  ensembles route review cases ~3× better than Chow's rule at the matched
  22.5% rate (position 0.40 vs 0.14, selective risk 0.217 vs random 0.299),
  from ONE model — no partner needed (`reports/uncertainty_deferral.json`).
- **Deferral inversion fixed:** legacy disagreement router deferred the
  model's most-confident cases (z ≈ +28σ on test); the measured fix
  (model-uncertainty routing) reaches **z = −99.7** at the required 22.7%
  rate (`reports/deferral_fix.json`).
- **Cost-sensitive training — honest null:** class-weighted training never
  beats cost-optimal thresholding (CI spans zero; Elkan 2001 confirmed) —
  the economics belong in the decision layer, where SmartLend keeps them
  (`reports/cost_sensitive_catboost.json`).
- **Full-population decision mix** (`backend/artifacts/prediction_outputs.csv`,
  307,511 rows): APPROVE 167,751 · REJECT 70,593 · DEFER 69,167. (The demo's
  500-customer sample excludes the extreme-risk tail, so live demo rejections
  come via the human-review path — which is the governance story.)
- **Deployment:** one-command rebuild (`terraform apply`, ~15 min) and
  teardown (`terraform destroy`, burn → $0); image in ECR pulled by IAM role,
  no credentials on the instance.

## One honest caveat to keep in the room

TabPFN-2.5 (0.7446 best-case) **beats the deployed LogisticRegression
(0.6919)** on identical features — it stays advisory because its weights
licence is non-commercial (outputs included), it needs a ~10GB GPU context,
and our governance rule bars AI second opinions from writing decisions. Saying
this unprompted is stronger than being asked.
