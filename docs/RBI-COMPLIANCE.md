# RBI alignment — Digital Lending Directions 2025 and the FREE-AI framework

Written for reviewer instruction *"check recent RBI guidelines"*. SmartLend is
an academic research system, **not** a regulated entity — this document maps
its design choices to the two most recent, relevant RBI instruments so the
report and viva can cite them accurately, and states plainly where the project
does *not* meet what a production deployment would require.

Primary sources (verify links before citing in the report):
- **Reserve Bank of India (Digital Lending) Directions, 2025** — issued 8 May 2025; consolidates the 2020/2022/2023 digital-lending circulars and DLG guidelines into one set of directions.
- **FREE-AI Committee Report** ("Framework for Responsible and Ethical Enablement of Artificial Intelligence") — published 13 August 2025; 7 principles ("sutras"), 6 strategic pillars (Infrastructure, Policy, Capacity / Governance, Protection, Assurance), 26 recommendations. Committee constituted December 2024.
- Context worth citing: RBI's own survey behind FREE-AI found only ~15% of AI-using entities employed interpretation tools, ~35% validated for bias, and ~18% kept audit logs — the gaps SmartLend's design directly addresses.

---

## Mapping to the FREE-AI sutras

| FREE-AI sutra | Where SmartLend embodies it | Evidence |
|---|---|---|
| **Trust is the Foundation** | Every number traces to a machine-readable report; stale constants were purged (`/health` now reads the live artifact). | `reports/*.json`, `docs/DEV-GUIDE.md` working agreements |
| **People First** | Human-in-the-loop by architecture: uncertain cases route to a reviewer; the model never overrides a human verdict; the AI agent briefing is advisory-only and cannot write a decision. | `backend/app/routers/applications.py` (persist-then-capture), `underwriting_agent_service.py` docstring boundaries |
| **Innovation over Restraint** | Foundation models (TabPFN, TabFM) evaluated as research baselines behind licence guards rather than banned or blindly deployed. | `models/README.md`, `research/analysis/score_tabfm.py` |
| **Fairness and Equity** | Deferral is not assumed neutral: applicant segment (coarse age band, region, gender, employment) is logged at defer time precisely so a disparate-deferral check is possible; thin-file applicants (14.3%) are kept as explicit missingness, never imputed to "average creditworthiness". | `DeferredReview.applicant_segment_json`, `cbes_engine.DEFAULTS` rationale |
| **Accountability** | Every decision carries `engine_version`, serving artifact and `threshold_artifact_hash`; reviewer overrides are tracked (`agreed_with_engine`, `override_direction` — the SR 11-7-style override-rate discipline). | `_decision_meta` provenance, `docs/RELEARNING-LOOP.md` §3 |
| **Understandable by Design** | Two explanation layers: CBES's five-pillar breakdown (the computation *is* the explanation) and SHAP top factors — with the heuristic fallback explicitly labelled "Not SHAP" so nothing impersonates an explanation method. | `docs/DEFENCE-SHAP-CBES.md` |
| **Safety, Resilience and Sustainability** | The relearning gate refuses retraining until four measured conditions pass — the direct defence against runaway feedback loops; capture failures are isolated so a research instrument can never cost a customer a decision. | `research/relearning/gate.py`, failure-isolation block |

## Mapping to FREE-AI's risk-mitigation pillars

- **Governance** — model inventory in practice: one serving artifact with recorded provenance, versioned engine string, rollback artifacts kept on disk; every threshold change is a report-backed decision (`t_base_selection.json`, `blend_decision.json`, `deferral_fix.json`).
- **Protection** — no secret is read outside `config.py`; capture rows store coarse segments ("to be analysed, not to re-identify"); customer PII is minimal and the seeded profiles carry no repayment target.
- **Assurance** — 130+ automated tests including grep-tests that *forbid* training code in the capture layer; the gate is CI-runnable (`python -m research.relearning.gate`, non-zero exit while shut); audit record per application (`GET /api/applications/{id}/report`).

## Mapping to the Digital Lending Directions 2025 (spirit, not licence)

| Directions theme | SmartLend analogue | Honest gap for production |
|---|---|---|
| Transparency to the borrower (key-fact statements, disclosure of automated processing) | Per-decision explanation payload + CBES pillar breakdown surfaced to the applicant view | No KFS, APR disclosure, or grievance flow — out of scope for a research system |
| Data minimisation ("only data essential for underwriting") | CBES deliberately consumes **8 fields**; the serving model 15; the short form asks only what the bank cannot already know | The research frame (129 columns) is used offline only |
| Accountability for automated decisions / LSP conduct | Full audit record per application; override tracking; advisory-only AI components | No regulated-entity governance structure, no board-approved policy |
| Human oversight of credit decisions | Deferral to human reviewers with mandatory reason codes; AI never finalises a deferred case | Auto-decisions are model-final; a regulated deployment would need documented oversight thresholds |
| Model risk management (periodic revalidation, no unchecked continuous learning) | **The relearning gate is exactly this**: retraining forbidden until selection bias, exploration labels and a reviewed design exist; "rebuild periodically as a versioned scorecard redevelopment — never continuously" | Outcome data (12–24 months) does not exist yet |

## What to say at the viva

> "We designed to the spirit of FREE-AI before writing the compliance page:
> human-in-the-loop routing, advisory-only AI assistance, per-decision
> provenance and override tracking, explicit fairness instrumentation at defer
> time, and a hard gate against continuous retraining. Where a production
> deployment would need more — KFS disclosure, grievance redress, board-level
> model governance, seasoned outcome data — we say so rather than claim it."

**Do not claim compliance.** Claim *alignment of design choices* with named
provisions, with the gaps stated. Verify the two primary documents (RBI
website: Digital Lending Directions 2025; FREE-AI Committee Report, Aug 2025)
and pin exact paragraph numbers before the report goes out.
