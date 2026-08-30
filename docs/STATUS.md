# SmartLend — Project Status

**Updated:** 30 August 2026 · **Defense:** November 2026 · **Weeks remaining:** ~10

---

## Two headlines

**1. The deadline is much closer than the old plan assumed.**
Final defense is **November 2026**. Report writing starts in **September**, and the report is submitted in the **second half of October**. So all research has to be finished by roughly **3 October**. We have replanned around this.

**2. We have the real dataset.** ✅
`creddefer_full_merged.csv` arrived 30 Aug — **307,511 real loan applications** with credit-bureau history merged in. Verified genuine. This was the single biggest blocker and it is now cleared.

---

## Where we are

| Area | Status | Notes |
|---|---|---|
| Research direction | ✅ Locked | Design spec written and reviewed |
| Data tooling | ✅ Done | Built, tested, 66 tests passing, committed to `main` |
| Real dataset | ✅ **In hand** | Verified 30 Aug |
| Working system (app) | ✅ Exists | Runs; has one dependency bug we found |
| The main experiment | 🔄 **Starting now** | This week |
| Report | ⏳ Not started | Begins next week |

---

## The problem we are actually solving

Our system sends **1 in 4** loan applications to a human reviewer instead of deciding itself. The idea: when the ML model and our rule-based CBES score disagree, don't trust either — ask a person.

**It doesn't work.**

| | Decides | Accuracy |
|---|---|---|
| Plain LogisticRegression | 100% of cases | **70.5%** |
| Our hybrid system | 74.5% of cases | **62.7%** |

Skipping a quarter of the work made it **worse by 7.8 points**. That should be impossible if the skipping is sensible.

Think of an exam where skipping is allowed. A good student skips what they *don't* know, so their score on answered questions goes **up**. Our system appears to be skipping what it *does* know.

**Three fixes have already been attempted** (the project's own report records the journey from 57.7% → 62.7%) and it is still behind a basic model. That tells us this isn't a bug to patch — the underlying idea is wrong.

### Why that is good news

This is the pivot that makes the project research rather than an engineering exercise:

> ❌ *"We built a hybrid system. It gets 62.7%. We tried three fixes."*
>
> ✅ *"We measured it properly and found the deferral mechanism is inverted — it sends the easy cases to humans. Here's the measurement, here's why the textbook fix cannot work in lending, and here's what would."*

The second is a defensible contribution. Getting from the first to the second is this week's job.

---

## The dataset — what we verified

We did not take the file on trust. Checks run:

| Check | Result |
|---|---|
| Row count | 307,511 — matches Home Credit exactly |
| Default rate | 0.0807 — matches the known figure exactly |
| Duplicate applicants | None (307,511 unique IDs) — the merge didn't inflate rows |
| Do the bureau columns predict default sensibly? | **Yes, all of them** |

That last check is the important one. Applicants who defaulted have **2.66× more overdue credit lines**, more active loans, fewer closed loans, and sought credit more recently. Every signal points the correct direction — which is strong evidence the data was merged correctly rather than scrambled.

### One finding worth knowing

**44,020 applicants (14.3%) have no credit-bureau history at all.**

These are "thin-file" applicants — young, first-time, or informally employed borrowers with no track record. They are exactly the population our research argument is about, so we will **not** fill their blanks with zeros. "No overdue loans" and "we have no information" are different statements, and erasing that difference would hide the effect we are studying.

### Small fixes needed (half a day)

1. Three of our column names don't match the file's naming — a rename.
2. One field (`total_credit_debt`) contains negative values from the original source data; needs cleaning before we compute credit utilisation.
3. Decide how the 14.3% blank block is flagged — decided: keep blank, add an indicator column.

### One thing still missing

The file records **only approved applicants**. It doesn't say who was *rejected* — and rejection is central to our research question.

`previous_application.csv` (a companion Home Credit file) contains the actual Approved / Refused decision for prior applications. **If someone can merge that in, it materially strengthens the core result** — it would let us *observe* the lending policy rather than estimate it. This is the highest-value thing anyone can add right now.

---

## Plan to November

We split the work into two tracks so the deadline is safe:

| Track | Deadline | Contains |
|---|---|---|
| **Defense** | **November 2026** — hard | The core result + working demo + report |
| **Paper** | ~January 2027 — flexible | Extra analysis, extra datasets, journal submission |

The defense track is deliberately the smaller half. The journal paper continues afterwards and does not put the defense at risk.

| Weeks | Dates | Work |
|---|---|---|
| 1 | 30 Aug – 5 Sep | **The main measurement**; dataset fixes; dependency bug |
| 2 | 6 – 12 Sep | Locked data splits, experiment tracking, baseline models |
| 3 | 13 – 19 Sep | Core method implementation |
| 4 | 20 – 26 Sep | **Main experiment** on real data |
| 5 | 27 Sep – 3 Oct | Second result + fairness check |
| 6–7 | 4 – 17 Oct | **Report writing** (all results in hand by now) |
| 8 | 18 – 24 Oct | Demo polish, final testing |
| 9 | 25 – 31 Oct | **Report submission**, guide feedback |
| 10 | 1 – 15 Nov | **Defense** |

**Research must be finished by 3 October.** October is writing, testing and demo — not experiments. This is the constraint that shapes every decision below.

---

## This week (30 Aug – 5 Sep)

| # | Task | Owner |
|---|---|---|
| 1 | **The main measurement** — is our deferral rule better or worse than choosing at random? | Shreyash |
| 2 | Fix the dataset column mismatches | Shreyash |
| 3 | Fix `requirements-api.txt` — it's missing 7 packages the app imports, so a fresh install crashes on startup | Shreyash |
| 4 | Stop the model using `applicant_id` as a feature (an ID number says nothing about repayment) | Shreyash |
| 5 | **Merge `previous_application.csv`** for approval/rejection data | *needs an owner* |

### What "done" looks like this week

- A chart comparing three ways of choosing which cases to defer: our rule, random selection, and a best-case reference.
- A one-line answer to: **is our rule better or worse than random?**

If those two exist by Friday, the week worked.

---

## What we have paused, and why

We have **5 weeks of research time**. These are all reasonable ideas that do not fit inside it.

| Paused | Reason | Revisit |
|---|---|---|
| New models from HuggingFace | A better model **does not change our result**. What we're measuring comes from *which applicants have outcome data*, not from the classifier. | Optional, later |
| Voice feature (Sarvam) | Was already lowest priority before the deadline moved. | Week 8, demo polish, ~3 days, only if on schedule |
| Auto-relearning loop | Blocked anyway: the database has one table and stores no reviewer decisions or repayment outcomes. Real default results take **12–24 months** to arrive, which is why banks rebuild credit models yearly, not continuously. | Future work + viva discussion |
| Production hardening (audit trails, load tests, access control) | Real engineering, but earns no research credit. | After the report |

### Specifically on the relearning loop

Building it now would make things **worse on a schedule**. Our rule defers the easy cases → humans would label the easy cases → we'd retrain on easy cases → the model gets worse on hard ones → and round again. **The rule has to be fixed before the loop can be closed.**

There is a genuinely good idea inside it that we're saving for the defense discussion: sending a case to a human is a way of **collecting data about applicants the system would otherwise always reject**. That turns deferral from a cost into the thing that makes previously-impossible cases learnable. Strong future work.

---

## For tomorrow's call

**Decisions needed:**

1. **Who owns merging `previous_application.csv`?** Highest-value open task.
2. **Confirm the November defense date** — the whole plan is built on it.
3. **Who helps with report writing** from 4 October?
4. **Agreement to hold the paused list** until research is done (3 October).

**Risks to flag:**

| Risk | Mitigation |
|---|---|
| 5 weeks is tight for the research | Plan is already cut to the two most essential results; each stands alone if the other slips |
| Report starts before all results are in | Structure and methodology can be written from week 6 while final numbers land |
| Reviewer asks why accuracy *dropped* vs the old numbers | Expected and correct — the old 71% came from made-up data. Real data gives real numbers. We say this plainly rather than hiding it. |

---

## Reference

- `docs/week-01-goal.md` — this week explained simply, with diagrams
- `docs/superpowers/specs/2026-08-18-conformal-credit-deferral-design.md` — full research design
- Code and tests: `research/` on `main` (66 tests passing)
