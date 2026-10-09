# SmartLend — Final Consolidated Report

One document answering: what we built, what we measured, what runs where
(Colab vs AWS), and which model is best. Every number traces to
`reports/*.json`. Companion docs: [`../SUGGESTIONS.md`](../SUGGESTIONS.md)
(reviewer items → results) and [`DEFENSE-NOTES.md`](DEFENSE-NOTES.md)
(parameters & prepared answers).

---

## 1. What SmartLend is

A loan-decisioning research system over **307,511 real applications** (Home
Credit; 8.07% default, 14.3% thin-file): an ML model and a transparent rule
engine (**CBES**) score every application; a blend (α = 0.10) decides
APPROVE/REJECT, and cases the system is not confident about are **DEFERred to
a human reviewer** (22.5% capacity band). Every decision is stamped with the
engine version, serving artifact and threshold hash. Reviewer outcomes are
*captured* for a future relearning loop that stays deliberately shut behind a
4-condition gate (verdict: DO NOT OPEN THE LOOP), with a 3% exploration arm.

## 2. Which model is best — the scoreboard

| Model | AUC | Sample / protocol | Role |
|---|---|---|---|
| **CatBoost on 319 engineered features (FE-v2)** | **0.7757** | 5-fold OOF, 307,511 rows, seed 42 | **best model overall** (research track) |
| XGBoost FE-v2 | 0.7749 | same folds | research |
| LightGBM FE-v2 | 0.7721 | same folds | research |
| XGBoost v1 features | 0.7651 | same folds | September baseline |
| TabPFN-2.5 (Prior Labs) | 0.7284 ± 0.0127 | 8,000 id-verified rows | advisory 2nd opinion (non-commercial licence) |
| RealMLP-TD (2026 challenger) | 0.7360 (30k screen) | lost all 5 folds, −0.0121 | eliminated |
| **LogisticRegression (serving)** | **0.6919 holdout** | 15-feature live contract | **the deployed decision model** |
| Google TabFM 1.0 | 0.6911 ± 0.0113 | same 8,000 rows as TabPFN | evaluated, worst |

Two different "best" answers, both correct: **best accuracy = CatBoost FE-v2
(0.7757)**; **what actually serves = the calibrated LogisticRegression
(0.6919)**, because serving obeys a 15-feature train/serve contract matching
the live derivation path. Promoting FE-v2 behind that contract is the roadmap.

Why the gain is real: the features-vs-fusion experiment (pre-registered rules,
±0.0036 fold-noise floor) — features +0.0155 fold-paired with **all 20 folds
positive**, while the best honest fusion gained +0.0016 (**denied**). The
engineered features carry 56% of the top-25 gain importance.

## 3. Convergence — the result that steered everything

Two questions, two verdicts (`reports/convergence.json`):

- **Optimisation (compute) — CONVERGED.** Boosting iterations plateau under
  early stopping; a 14-config hyperparameter sweep improved +0.001, under a
  third of the noise floor (`reports/tuning_cpu.json`). Tuning is exhausted.
- **Information (data) — NOT converged.** The learning curve (per-size early
  stopping — a fixed tree budget underfits large n: 0.7581 vs 0.7607) still
  rises at 246k training rows: **+0.010 AUC per doubling**.

Reading: *the model stopped learning from more compute, not from more data* —
which is why effort went to feature engineering (paid +0.0155) and why the
known future gain is the four unused Home Credit side tables (~+0.015 → ~0.79).

## 4. What is the Colab hosting

**TabPFN-2.5** is a tabular foundation model whose checkpoint (~10GB with its
context in GPU memory) exceeds the project laptop's 8GB GPU (context capped at
5,000 rows locally). So it is hosted on **Google Colab**:
`colab/tabpfn_colab_server.ipynb` loads the pinned checkpoint on a free
**Tesla T4** with a **10,000-row in-context set** (uploaded at runtime — Home
Credit data may not be redistributed), serves an API, and exposes it through a
**Cloudflare quick tunnel** with a session token. The backend consumes it as
an **advisory second opinion** that can never write a decision (licence is
non-commercial, governance rule regardless). A **connection run**
(`backend/run_tabpfn_connection_check.py`) proves the tunnel serves the real
model: it scores 400 locally-held labelled rows the Colab never saw and
checks the AUC — last run **PASS, remote AUC 0.7100**
(`reports/tabpfn_connection_run.json`). The tunnel URL is per-session: re-run
the notebook's last cell before a demo and update the two `.env` lines.

## 5. What is on AWS

The **entire serving system**, defined in Terraform (`terraform/`, 7 files)
and rebuilt/destroyed on command:

- **EC2 t3.micro** (Ubuntu 24.04, public subnet, Elastic IP
  **http://35.154.151.137**): runs the Docker image — FastAPI backend + the
  built React frontend + the serving model baked in. First boot
  self-configures via user_data (Docker, 2GB swap, ECR pull, container run).
- **RDS PostgreSQL 16** (db.t4g.micro, private subnets, reachable only
  through the security-group chain): applications, review queue, capture
  table, 500-customer seed.
- **ECR** (private): the image (~292MB compressed); the instance pulls it via
  an **IAM role** — no registry credentials on disk.
- **Agent briefing**: Gemini-powered (`gemini-2.5-flash`) reviewer briefing on
  deferred cases — advisory only, never writes decisions, degrades to a calm
  503 without a key.
- **Ops numbers**: `terraform apply` ≈ 15 min to a working system;
  `terraform destroy` → $0/day; running cost ≈ $1.1/day; stop scripts in
  `scripts/aws-{start,stop}.ps1`. Heavy dashboard endpoints are TTL-cached
  (65s+ → <0.3s warm) after measuring a t3.micro saturate — the Free Plan
  blocks larger instance types, so the fix was software, not hardware.

## 6. What we did — the arc in one paragraph

Took the September review's 11 items and closed them with measurements:
stamped model provenance on every decision; evaluated TabPFN **and** Google
TabFM on id-verified rows (both below our GBMs, both fusion-KILLed under a
pre-registered noise floor); fixed the inverted deferral router (z ≈ +28σ →
**z = −99.7**); ran the convergence analysis that redirected effort from
tuning to information; engineered 319 features for the single biggest gain
(**0.7651 → 0.7757**); added three model-modification studies (epistemic-
uncertainty deferral **granted** — position 0.40 vs Chow 0.14, from one
model; monotone constraints **free** — −0.0008, 0/14,400 violations; cost-
sensitive training **null**); screened 2026 SOTA challengers (RealMLP lost
every fold; TabICLv2 infeasible on commodity hardware); hosted TabPFN on a
Colab T4 with a verified connection run; mapped RBI 2025 + FREE-AI
compliance; and deployed the whole system to AWS with Terraform, a guest
login, seeded demo decisions including human-adjudicated rejections, and a
live Gemini reviewer-briefing agent.

## 7. The one honest caveat (say it before they ask)

TabPFN-2.5 (0.7446 best-case) beats the deployed LogisticRegression (0.6919)
on identical features. It stays advisory because its weights licence is
non-commercial **including outputs** (as is TabPFN-3.5's — verified), it
needs a ~10GB GPU context, and the governance rule bars advisory AI from
writing decisions. The system's answer to "use the stronger model" is the
roadmap: promote FE-v2 CatBoost behind the serving contract.
