# The relearning loop

**Status: capture is LIVE. Retraining is SHUT, and all four gate conditions currently FAIL.**

This document describes what SmartLend records about human review decisions,
why none of it is allowed to train anything yet, and what a future developer
must do before that can change.

Authoritative design:
`docs/superpowers/specs/2026-08-30-cbes-research-and-explanation-design.md` §3.
This document is the implementation record; where the two disagree, the spec wins.

---

## 1. What the loop is (and is not)

It is a **capture layer plus a gate**. It writes one row to `deferred_reviews`
for every application a human looks at, records what the human decided, and
refuses to let anyone train on it.

It is **not** a feedback loop. Nothing here changes model behaviour. There is no
retraining trigger — scheduled, volume-based, or manual — and no code path reads
`human_decision` or `realized_outcome` as a training label. That absence is the
design, not an unfinished part of it.

---

## 2. Data flow, end to end

```
  customer submits application
            │
            ▼
  POST /api/applications
   └─ _create_application_record()            backend/app/routers/applications.py
        │
        ├─ predictor.predict_application()    backend/app/services/ml_service.py
        │    └─ hybrid_decision()             backend/app/services/decision_engine.py
        │         → DecisionResult(decision = APPROVE | REJECT | DEFER)
        │    ml_service then attaches provenance to the result:
        │         t_base, tau_d, engine_version, threshold_artifact_hash
        │
        ├─ routing decision  ─────────────────────────────────────────────┐
        │     decision == DEFER        → routed to human, exploration=False │
        │     otherwise                → _should_explore(): 3% coin flip    │
        │                                 True  → routed, exploration=True  │
        │                                 False → auto-decided, no capture  │
        │                                                                   │
        ├─ COMMIT the LoanApplication  ← the decision is durable HERE ──────┘
        │
        └─ if routed: _capture_deferral()      [failure-isolated]
               └─ record_deferral()            backend/app/services/deferred_review_service.py
                    → INSERT INTO deferred_reviews

  analyst opens the case in the review screen
            │        frontend/src/pages/ApplicationReview.tsx
            │         └─ <ReviewerFeedbackForm>   ← clock starts on mount
            │              • verdict: Approve | Reject
            │              • reason codes (≥1 REQUIRED, filtered by verdict)
            │              • reviewer confidence 1-5
            │              • optional free-text note
            │              • time on case — measured, never typed
            ▼
  POST /api/applications/{id}/decision
   {status, notes, reasonCodes, reviewerConfidence, timeSpentSeconds, reviewerId}
   └─ update_manual_decision()                backend/app/routers/applications.py
        ├─ apply_manual_decision() + COMMIT    ← the decision is durable HERE
        └─ _capture_human_decision()           [failure-isolated]
               └─ record_human_decision()
                    → UPDATE the open deferred_reviews row for this application

  anyone asks "what happened on this application?"
            │
            ▼
  GET /api/applications/{id}/report           backend/app/routers/applications.py
   └─ build_decision_report()                 backend/app/services/decision_report_service.py
        → { engine: {...}, humanReview: {...} | null }
        rendered by <DecisionReportPanel> on the same review screen

  loan seasons, outcome becomes known  (not yet automated)
            │
            └─ record_outcome()  → realized_outcome / outcome_censored

  anyone asks "can we retrain yet?"
            │
            ▼
  GET /api/relearning/status                  backend/app/routers/relearning.py
   └─ get_relearning_status()                 backend/app/services/relearning_service.py
        └─ research/relearning/gate.py  → verdict: DO NOT OPEN THE LOOP
```

### Files that make up the loop

| File | Role |
|---|---|
| `backend/app/models.py` → `DeferredReview` | the 29-column capture table |
| `backend/app/services/deferred_review_service.py` | write primitives; contains no training code by test |
| `backend/app/routers/applications.py` | the three live wiring points + failure isolation |
| `backend/app/services/ml_service.py` | attaches `t_base` / `engine_version` / `threshold_artifact_hash` provenance |
| `backend/app/services/decision_engine.py` | `ENGINE_VERSION` stamp |
| `backend/app/config.py` → `exploration_rate()` | control-arm rate, 3% default |
| `backend/app/services/relearning_service.py` | status, and the one honest `attempt_retrain()` refusal |
| `backend/app/routers/relearning.py` | `GET /api/relearning/status` |
| `research/relearning/gate.py` | the real four-condition gate (imported, never forked) |
| `backend/app/services/review_reason_codes.py` | the reason-code taxonomy (source of truth) + code→label expansion |
| `backend/app/services/decision_report_service.py` | assembles the per-application audit record |
| `frontend/src/components/sections/ReviewerFeedbackForm.tsx` | the reviewer UI: reason checkboxes, confidence, measured time |
| `frontend/src/components/sections/DecisionReportPanel.tsx` | renders the audit record on the review screen |
| `frontend/src/lib/reasonCodes.ts` | frontend mirror of the taxonomy, used as an offline fallback |

---

## 3. What is captured, and when

### At decision time — on DEFER, or on an exploration-arm sample

Written by `record_deferral()` immediately **after** the application row is
committed.

| Group | Fields | Why |
|---|---|---|
| Keys | `review_id`, `application_id`, `created_at` | |
| Engine state | `decision_reason`, `p_ml`, `p_cbes`, `p_blend`, `disagreement`, `confidence`, `t_approve`, `t_reject`, `cbes_breakdown_json` | the full state that produced the referral — EU AI Act Art. 13 "technical measures" that make Art. 14 oversight possible |
| Provenance | `engine_version`, `threshold_artifact_hash`, `t_base` | lets a later analysis separate a *fixed* router's labels from a *broken* router's. Without this every row is unusable |
| Segment | `applicant_segment_json` (region, city, gender, marital status, employment type, age **band**) | uncertainty-based deferral is not automatically neutral and can fall unevenly on under-represented groups ("Unequal Uncertainty"); coarse only, since this table exists to be analysed, not to re-identify anyone |
| Control arm | `exploration_flag` | see §5 |
| Outcome | `outcome_censored = True` | every row starts censored; nothing is imputed |

### At review time — when an analyst approves or rejects

Written by `record_human_decision()` immediately **after** the manual decision
is committed.

| Field | Source | Why |
|---|---|---|
| `human_decision` | `status` (`approved`/`rejected`) | override-rate monitoring ONLY |
| `reviewer_id`, `reviewed_at` | `reviewerId` | attribution |
| `time_spent_seconds` | `timeSpentSeconds` | reviewer-attention-degradation check — scrutiny is known to decay as the model gets more accurate |
| `human_reason_codes`, `human_free_text` | `reasonCodes`, `notes` | qualitative audit trail |
| `reviewer_confidence` (1–5) | `reviewerConfidence` | reviewer-consistency modelling (Madras et al.) |
| `agreed_with_engine`, `override_direction` | derived | SR 11-7 requires override rates be tracked |

`status = "deferred"` is **not** a terminal verdict — the case is still under
review — so it records nothing rather than a fabricated decision. Likewise, when
the engine's `p_blend` sat strictly between `t_reject` and `t_approve` it had no
lean at all, so `agreed_with_engine` is left NULL: counting a true grey-zone case
as an "override" would inflate the metric.

The reviewer fields are all optional additions to `ManualDecisionRequest`. An
existing client that posts only `{status, notes}` still works; the extra columns
are simply NULL rather than guessed.

### How the reviewer actually supplies all that

`frontend/src/components/sections/ReviewerFeedbackForm.tsx`, rendered on the
analyst review screen (`/review/:applicationId`). It collects four things and
posts them through the fields above — it introduces **no new backend field**.

1. **Verdict** — Approve or Reject.
2. **Reason codes** — multi-select checkboxes, filtered to the codes that argue
   for the chosen verdict plus the direction-neutral ones. Changing the verdict
   drops any ticked code that no longer applies, so an approve-only reason can
   never be filed against a rejection.
3. **Reviewer confidence** — a 1–5 selector, labelled *very unsure … very
   confident*. Sent as `reviewerConfidence`; the backend rejects anything
   outside 1–5.
4. **Free-text note** — optional, and *required only* when `GEN-OTHER` is
   ticked. Sent as `notes`, stored in `human_free_text`.

**Submit is disabled until at least one reason code is ticked**, and the form
says why in-line. This is the one hard requirement in the UI. A human decision
recorded with no reason is a row that can never contribute to the
disparate-deferral check or to reviewer-consistency analysis; it inflates
`reviewed_count` without adding a single piece of evidence, which is strictly
worse than not capturing it. If `GEN-OTHER` is ticked, the note becomes required
too — "other" with no explanation records nothing at all.

**Time spent is measured, never typed.** A `useRef` timestamp is taken when the
form mounts (i.e. when the reviewer opens the case) and the elapsed seconds are
sent at submit. There is deliberately no input for it: the
reviewer-attention-degradation check needs a real measurement, and a
self-reported number would be a rounded guess. The elapsed time is shown live so
the measurement is visible rather than covert.

Single-case adjudication in the review queue (`ReviewPage.tsx`) routes to this
screen rather than offering bare Accept/Reject buttons, for the same reason. The
**bulk** queue actions are unchanged and remain reason-less — they are a triage
action over cases the engine already auto-decided, not an adjudication of a
deferral, and they generally act on applications that have no open capture row
at all.

### The reason-code taxonomy

Source of truth: `backend/app/services/review_reason_codes.py`. The frontend
ships a mirror in `frontend/src/lib/reasonCodes.ts` and fetches the live list
from `GET /api/review-reason-codes`, falling back to the mirror if that call
fails — a backend hiccup degrades to possibly-stale *labels*, never to a review
form with no checkboxes.

| Code | Label | Supports |
|---|---|---|
| `APR-EMP-STABLE` | Stable employment | APPROVE |
| `APR-REPAY-STRONG` | Strong repayment history | APPROVE |
| `APR-COLLATERAL` | Adequate collateral | APPROVE |
| `APR-INCOME-HEADROOM` | Sufficient income headroom | APPROVE |
| `APR-RELATIONSHIP` | Long-standing customer | APPROVE |
| `APR-GUARANTOR` | Guarantor strength | APPROVE |
| `REJ-OBLIGATIONS` | High existing obligations | REJECT |
| `REJ-THIN-FILE` | Thin or no credit file | REJECT |
| `REJ-DELINQUENCY` | Recent delinquency | REJECT |
| `REJ-EMI-BURDEN` | Income insufficient for EMI | REJECT |
| `REJ-DOCS` | Unverifiable documents | REJECT |
| `REJ-UTILISATION` | High credit utilisation | REJECT |
| `GEN-MODEL-MISMATCH` | Model score inconsistent with the file | either |
| `GEN-POLICY-EXCEPTION` | Policy exception applied | either |
| `GEN-OTHER` | Other (free text becomes required) | either |

**The codes are not validated server-side, on purpose.** `reasonCodes` accepts
any string. Two reasons: an older client, a bulk action, or a future taxonomy
revision must never cause a reviewer's *decision* to be rejected over a
bookkeeping mismatch; and codes written under an earlier taxonomy must stay
readable. `describe_reason_code()` surfaces an unrecognised code verbatim with
`direction: "unknown"` rather than dropping it — an audit record must never
quietly lose a reason a human actually recorded.

The direction split is presentation guidance only. `GEN-MODEL-MISMATCH` in
particular is the code that makes an override legible after the fact: it says
the reviewer looked at the file and disagreed with the score, which is exactly
the population condition 1 of the gate is about.

### Later — when the loan outcome is known

`record_outcome()` sets `realized_outcome` / `outcome_observed_at` /
`outcome_censored`. Censoring is stored **explicitly and never imputed**: the
missingness is MNAR, and imputing it would be reject inference, which the spec
forbids (§1.11, §3).

---

## 4. Why retraining is gated

Three independent reasons, all from spec §3.

**1. Runaway feedback loop.** A system whose own router decides which cases get
new labels, and then retrains on those labels, reinforces its initial bias
regardless of the true underlying rate (Ensign et al., *Runaway Feedback Loops in
Predictive Policing*; FAccT 2023, *A Classification of Feedback Loops … in
Automated Decision-Making Systems*). This is not theoretical here: SmartLend's
deferral rule is **measured as worse than random** at isolating hard cases (see
condition 1 below — it defers the *easier* pile). Retraining on its output would
amplify an actively harmful selection policy.

**2. Selective labels.** Outcomes are observed only for the approved subset, so
training or evaluating on them yields biased risk estimates over the full
population (Lakkaraju et al., *The Selective Labels Problem*, KDD 2017;
Kleinberg, Lakkaraju et al., *Human Decisions and Machine Predictions*, QJE). The
exploration arm is the intended escape hatch and is not yet large enough to be
one.

**3. Human labels carry their own bias.** Feeding reviewer judgments back as
training labels can cause non-convergent training and amplify human bias when
treated as ground truth (Madras et al., *Predict Responsibly*, NeurIPS 2018;
*Designing Closed Human-in-the-loop Deferral Pipelines*).

Consequently the following must never exist in this codebase: an automatic
retraining trigger; any code path reading `human_decision` or `realized_outcome`
as a training label; reject-inference imputation of outcomes for
deferred/rejected cases; online or continuous learning; or a CBES weight update
driven by deferral outcomes.

Two tests enforce the boundary by grepping the service modules for training
constructs. `test_deferred_review_service.py` greps the capture layer for
`def train`, `retrain`, `.fit(`, `partial_fit` and `reject_inference`;
`test_relearning_wiring.py` greps `relearning_service.py` for `def train`,
`.fit(`, `partial_fit`, `reject_inference` and `import sklearn` (the word
"retrain" is exempted there only because `attempt_retrain()` — whose entire job
is to refuse — is named after it). If you find yourself editing those tests,
stop and read this section again.

---

## 5. The exploration arm

`maybe_route_to_exploration()` returns True with probability
`config.exploration_rate()` — **3% by default**, settable via
`SMARTLEND_EXPLORATION_RATE` and hard-clamped to `[0.0, 0.05]` so a mistyped env
var cannot dump a large slice of production traffic into manual review.

When it fires, a would-be-**AUTO**-decided application is routed to a human
anyway and the row is written with `exploration_flag = True`, **carrying the
engine's own APPROVE/REJECT**. That is the whole point: these are the only labels
in the system *not* chosen by the router, hence the only escape hatch from the
selective-labels trap, and eventually the only way to prove the deferral rule was
fixed. Spec §3 calls this "the single most valuable thing to build now".

Two invariants, both tested:

* It is consulted **only on the non-DEFER branch**. Exploration can add a review;
  it can never turn a DEFER into an auto-decision.
* A genuine DEFER is never written with `exploration_flag = True` — the capture
  layer raises rather than let that corrupt the control arm.

Only exploration rows with an *observed* (non-censored) outcome count toward gate
condition 3. An exploration row awaiting its outcome is not yet a label.

---

## 6. Failure isolation

**A logging feature must not take down lending.**

Every capture call site — `_capture_deferral`, `_capture_human_decision`,
`_should_explore` — catches `Exception`, rolls back its own transaction, logs a
warning, and returns. None re-raise. Each runs *after* the decision it is
recording has already been committed, so the ordering itself guarantees that a
capture failure can only ever cost a research row, never a customer's decision.

If exploration sampling itself throws, the answer is "not sampled" — the safe
direction, since it leaves the engine's decision exactly as the engine made it.

Covered by `test_capture_failure_does_not_break_the_decision`,
`test_exploration_sampling_failure_does_not_break_the_decision`, and
`test_reviewer_capture_failure_does_not_break_the_manual_decision`.

---

## 7. The four gate conditions — current measured status

Run live via `GET /api/relearning/status`, or on the command line with
`python -m research.relearning.gate` (which exits non-zero while the loop must
stay shut). The numbers below are the current verdict over the 25,000-row
`backend/artifacts/prediction_outputs.csv`.

### **VERDICT: DO NOT OPEN THE LOOP** — 0 of 4 conditions pass.

| # | Condition | Status | Measured |
|---|---|---|---|
| 1 | Deferral rule beats random at isolating hard cases (CoDoC pattern) | **FAIL** | The deferred pile is **+10.5 sd MORE lopsided** than a random router's would be (93.71% good, vs 90.11% good in the auto-decided pile), and the model is **+12.2 sd MORE accurate** on it. The router defers the *easier* cases — the opposite of what a working referral rule does. |
| 2 | Observed defer rate within the AUC-implied natural-rate bound (Tasche) | **FAIL** | Observed defer rate **52.38%**, which is **3.3×** the upper bound of 15.85% implied by AUC 0.768. Most referrals are cases the model already handles correctly. |
| 3 | Exploration arm has enough un-selected labels | **FAIL** | **0** un-selected labels with observed outcomes; **1,000** required. Capture only just went live, and outcomes have not seasoned. |
| 4 | Retraining design models the selection mechanism and reviewer bias | **FAIL** | No retraining design document exists. Per spec §3 it should not be written until 1–3 hold. |

Condition 4 is a design-artifact check, not a statistic: the gate looks for a
document at `docs/retraining-design.md` (or two alternate paths). Note that
merely creating that file does **not** flip the condition to PASS — the gate
grades a found document as `UNKNOWN` and demands human sign-off, deliberately
refusing to grade prose as a pass.

Condition 1 is the load-bearing failure. Conditions 3 and 4 are matters of time
and work; condition 1 says the referral mechanism is currently pointed backwards,
and no amount of additional data collected under it fixes that.

---

## 8. Opening the loop safely

If you are here to connect this table to a trainer, this is the required order.
There is no innocent shortcut.

1. **Fix the router first.** Condition 1 is not a data-volume problem. Diagnose
   why the current gates (disagreement > `tau_d`, confidence < 0.18, grey zone)
   select the easy pile, change them, and re-run
   `python -m research.relearning.gate` until condition 1 passes on held-out
   data. Bump `ENGINE_VERSION` in `decision_engine.py` when you do, so rows from
   the old and new routers are separable.
2. **Bring the defer rate into the Tasche band** (condition 2). A 52% defer rate
   is not a referral mechanism, it is a queue.
3. **Accumulate ≥1,000 exploration-arm rows with observed, non-censored
   outcomes**, all under a single `engine_version` / `threshold_artifact_hash`,
   spanning enough calendar time for outcomes to season (condition 3). Nothing
   collected under the broken router counts.
4. **Write the retraining design** (condition 4). It must explicitly model the
   missingness/selection mechanism (cf. RMT-Net) and reviewer bias/consistency
   (Madras et al.), and must explicitly reject treating `human_decision` as
   ground truth. It requires human sign-off; the gate will not grant it.
5. **Only then**, and only as a *versioned scorecard redevelopment* — offline,
   reviewed, released as a new artifact — never as a continuous or automatic
   retrain. Spec §3: "rebuild periodically as a versioned scorecard
   redevelopment — never continuously."

Even with all four conditions passing, `attempt_retrain()` still returns
`performed: False` with reason `GATE_OPEN_BUT_NO_RETRAINING_IMPLEMENTATION`.
Passing the gate authorises a written, reviewed redevelopment. It does not
authorise a retrain triggered from a web request, and there is deliberately no
HTTP route that could do so.

---

## 9. API

### `GET /api/relearning/status`

```jsonc
{
  "rows_captured": 0,          // deferred_reviews rows
  "reviewed_count": 0,         // rows with a human_decision
  "outcomes_observed": 0,      // rows with a non-censored realized_outcome
  "exploration_rows": 0,       // rows with exploration_flag = true
  "exploration_labels": 0,     // exploration rows WITH an observed outcome (gate condition 3)
  "override_rate": null,       // overrides / rows where the engine had a lean
  "override_denominator": 0,
  "override_count": 0,
  "exploration_rate": 0.03,
  "gate": {
    "verdict": "DO NOT OPEN THE LOOP",
    "all_conditions_pass": false,
    "failing_conditions": [1, 2, 3, 4],
    "conditions": [ /* four {condition, name, status, reason} entries */ ],
    "rows_evaluated": 25000,
    "unavailable": false
  },
  "verdict": "DO NOT OPEN THE LOOP",
  "retraining_permitted": false
}
```

The endpoint **fails closed**. An unreadable database yields zeroed counts
flagged `counts_unavailable`; an unevaluable gate yields
`gate.unavailable = true` and the closed verdict. There is no failure mode that
produces a permissive answer.

### `GET /api/applications/{application_id}/report`

The per-application audit record: everything the engine did, and everything the
human did, in one payload. Read-only — it builds a dict out of rows that already
exist and writes nothing.

```jsonc
{
  "applicationId": "app-...",
  "generatedAt": "2026-08-31T09:12:44+00:00",
  "application": {
    "applicantId": "cust-...", "applicantName": "...", "email": "...", "phone": "...",
    "createdAt": "...", "status": "submitted",
    "loanAmount": 500000, "loanPurpose": "personal", "loanTenureMonths": 36,
    "data": { /* the submitted form, minus underscore-prefixed internals */ }
  },
  "engine": {
    "decision": "DEFER", "decisionReason": "disagreement_gate",
    "selectedModel": "...", "engineVersion": "...", "thresholdArtifactHash": "...",
    "pMl": 0.9, "pCbes": 0.1, "pBlend": 0.7,
    "pBlendSource": "captured",          // "captured" | "derived" | "unavailable"
    "disagreement": 0.8, "confidence": 0.42, "confidenceLabel": "low", "riskScore": 0.1,
    "thresholds": { "approve": 0.55, "reject": 0.45, "base": 0.50 },
    "cbesBreakdown": { "credit": 0.4, "capacity": 0.5, /* ... one per pillar */ },
    "cbesWeights":   { /* pillar weights, when recorded */ },
    "topFactors": [ /* SHAP top factors, [] when unavailable */ ],
    "explanation": "...",
    "routedToHumanReview": true, "explorationFlag": false
  },
  "humanReview": {                        // null until a reviewer actually rules
    "reviewId": "rev-...", "reviewerId": "analyst-11", "decision": "REJECT",
    "reviewedAt": "...",
    "reasonCodes": [ { "code": "REJ-DOCS", "label": "Unverifiable documents",
                       "direction": "reject", "description": "..." } ],
    "freeText": "...", "reviewerConfidence": 4, "timeSpentSeconds": 214.0,
    "agreedWithEngine": false,
    "overrideDirection": "engine_approve_to_human_reject",
    "explorationFlag": false, "outcomeCensored": true, "realizedOutcome": null
  },
  "analystNotes": "...", "manualDecisionApplied": true
}
```

Three behaviours worth knowing:

* **`humanReview: null` is a 200, not a 404.** An application awaiting review —
  or one that was auto-decided and never captured — returns the engine half
  alone. A report that errored whenever the case was unresolved would fail
  exactly when it is most needed. Only an unknown `application_id` is a 404.
* **Where the capture row and `_decision_meta` overlap, the capture row wins.**
  It was written from the live `DecisionResult` and carries `p_blend`, `t_base`,
  `engine_version` and `threshold_artifact_hash`, none of which the application's
  stored metadata holds.
* **`pBlendSource` is honest about reconstruction.** With no capture row there is
  nothing recorded to read `p_blend` from, so it is recomputed from the engine's
  own `_BLEND_ALPHA` (imported, never re-declared) and flagged `"derived"`. A
  reader can always tell a recorded number from a recomputed one.

Rendered by `<DecisionReportPanel>` as the "Decision Report" panel on the
analyst review screen, org role only.

### `GET /api/review-reason-codes`

`{ "reasonCodes": [ { code, label, direction, description }, ... ] }` — the
taxonomy in §3, served so the review screen and the backend that stores the
codes cannot drift apart.

---

## 10. Is the loop actually closed? (the end-to-end evidence)

`backend/tests/test_decision_report_and_reviewer_feedback.py::test_relearning_loop_closes_end_to_end`
walks the whole chain in one test, rather than leaving it inferred from four
unit tests that never meet:

1. The engine defers → exactly one `deferred_reviews` row exists, with
   `human_decision IS NULL`, and the report already answers with the engine half
   and `humanReview: null`.
2. A reviewer decision carrying three reason codes, confidence 4 and 214 measured
   seconds **updates that same row** (asserted: still exactly one row) and sets
   `agreed_with_engine = false` /
   `override_direction = engine_approve_to_human_reject`.
3. `GET /api/relearning/status` moves: `reviewed_count` +1,
   `override_denominator` 1, `override_rate` 1.0 — and
   `retraining_permitted` stays `false`. Capturing feedback unlocks nothing.
4. `GET /api/applications/{id}/report` returns both halves, with `pMl`, `pCbes`,
   `pBlend`, `confidence` and all three thresholds matching the live
   `DecisionResult`, and the reason codes expanded to readable labels.
