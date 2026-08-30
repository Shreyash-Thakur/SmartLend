# Loan Application Form — Implementation

Implements `docs/FORM-REDESIGN.md`. The design is not revisited here; this
document records what was built, what is asked versus pulled, how the profile
lookup is cached, and how backward compatibility is preserved.

**Field count: 30 asked → 14 asked (+ `customer_id`).**

---

## 1. Final field list

### Asked of the applicant — 14 fields + `customer_id`

| # | Field | Type | Section |
|---|---|---|---|
| — | `customer_id` | string | Identification |
| 1 | `loan_amount` | number | A · Loan request |
| 2 | `loan_purpose` | select (home improvement / education / medical / debt consolidation / business / vehicle / wedding / other) | A |
| 3 | `loan_tenure_months` | select (12 / 24 / 36 / 48 / 60) | A |
| 4 | `repayment_source` | select (salary / business income / rental / other) | A |
| 5 | `preferred_emi_date` | select (1 / 5 / 10) | A |
| 6 | `end_use_declaration` | checkbox — **RBI required** | A |
| 7 | `obligations_other_banks` | number | B · What the bank cannot see |
| 8 | `employment_changed_recently` | boolean | B |
| 9 | `additional_income` | number | B |
| 10 | `collateral_type` | select (none / property / gold / fixed deposit / vehicle) | B |
| 11 | `collateral_value` | number (shown only when collateral ≠ none) | B |
| 12 | `is_related_party` | boolean — **RBI required** | B |
| 13 | `guarantor_relationship` | select (spouse / parent / sibling / employer / other) | C · Guarantor |
| 14 | `guarantor_customer_id` | string | C |

`guarantor_required` is **derived, never accepted from the client**. Both the
form and `LoanApplicationInput.finalize()` compute it as
`collateral_type == "none" and loan_amount > GUARANTOR_THRESHOLD` (₹500,000).
Section C renders only when that holds; the backend recomputes it regardless.

### Removed from the form — 21 fields

```
firstName, lastName, email, phone, age, gender, maritalStatus,
city, region, dependents, monthlyIncome, annualIncome,
employmentType, yearsOfEmployment, bankBalance,
cibilScore, totalLoans, activeLoans, closedLoans,
missedPayments, creditUtilizationRatio
```

`cibilScore` mattered most: a credit score is a bureau pull, never a
self-reported number. It is now resolved server-side and shown read-only.

---

## 2. Pulled, not asked

`GET /api/customers/{customer_id}/profile` →
`backend/app/services/customer_profile_service.py::get_profile`.

| Block | Fields | Home Credit source |
|---|---|---|
| KYC / demographics | `age`, `gender`, `dependents`, `marital_status`, `region`, `region_rating`, `employment_type`, `employment_tenure_years` | `DAYS_BIRTH`, `CODE_GENDER`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS`, `REGION_RATING_CLIENT`, `NAME_INCOME_TYPE`, `DAYS_EMPLOYED` |
| Account history | `annual_income`, `monthly_income`, `existing_emis`, `dti` | `AMT_INCOME_TOTAL`, `AMT_ANNUITY`, and `AMT_ANNUITY / AMT_INCOME_TOTAL` |
| Bureau pull | `credit_score`, `credit_score_display`, `delinquencies`, `active_loans`, `closed_loans`, `total_loans`, `credit_utilisation` | `EXT_SOURCE_2`, `overdue_credits`, `active_credits`, `closed_credits`, `total_prev_credits`, `total_credit_debt / total_credit_sum` |

Caveats that are recorded in code, not glossed over:

* **`DAYS_EMPLOYED == 365243`** encodes "pensioner / not employed", not 1000
  years of service. It becomes `None`, never an imputed number.
* **`credit_score` is `EXT_SOURCE_2`**, a normalised `[0,1]` anonymised external
  score — *not* a CIBIL score. `credit_score_display` rescales it onto 300–900
  purely so the UI panel and the legacy `cibilScore` field have something to
  show. The scoring engine reads the raw value; nothing reads the rescaled one.
  Every response carries `credit_score_basis` saying so.
* **`dti` and `credit_utilisation` are derived approximations.** Home Credit has
  neither natively. DTI is the standard annuity/income stand-in and omits
  obligations to other lenders — which is exactly why the form still asks
  `obligations_other_banks`.
* **An unknown id returns `None` / HTTP 404.** No profile is ever synthesised.
  Fabricated demographics would flow straight into a credit decision.

### What this does for scoring

All eight CBES inputs now come from the bank's own systems and the bureau:

```
credit_score, delinquencies, active_loans, dti,
employment_tenure_years, annual_income, region   <- profile lookup
loan_amount                                      <- the only typed risk input
```

Previously the form invited applicants to self-report the bureau fields, and
the camelCase form vocabulary did not match the snake_case keys CBES reads, so
scores were near-constant. A resolved short-form submission now produces a real,
row-specific CBES probability.

---

## 3. Profile lookup and caching

Source: a pre-merged Home Credit extract that already carries bureau
aggregates, so no join happens at request time. `SK_ID_CURR` **is** the customer
id. Path resolution order:

1. `$SMARTLEND_CUSTOMER_DATA` (env override, for deployments and tests)
2. `C:\Users\shrey\Downloads\creddefer_full_merged.csv`
3. `<repo>/data/raw/creddefer_full_merged.csv`

Caching strategy — the extract is ~307k rows × ~130 columns:

* **Lazy.** Nothing touches disk at import time. `import backend.app.main` must
  not pay for a large CSV, and neither must a process that only imports a
  schema. The first lookup triggers the load.
* **Column-pruned.** Only the 18 columns this module reads are parsed
  (`usecols`). This, not the row count, is what keeps the resident set small.
* **Process-lifetime cache**, held as a single `DataFrame` indexed by
  `SK_ID_CURR`. That index is a hash index, so `.loc` is O(1). A `DataFrame`
  beats a dict-of-dicts here purely on memory: 307k small Python dicts cost an
  order of magnitude more than 18 typed numpy columns.
* **Lock-guarded with a double check**, since two concurrent requests can race
  to the first lookup.
* **Misses are cached too.** A missing file caches an empty frame, so a lookup
  returns `None` immediately instead of re-stat-ing the filesystem every call.
* `reset_cache()` exists for tests that repoint the data source.

Measured: ~1.3 s first load, ~0.2 ms per lookup thereafter.

---

## 4. Backward compatibility

The constraint was that existing rows, the seeded applications, the document
parser path and every reader of stored `input_data` keep working. Three shims
carry that.

**(a) The legacy field block still exists and still serialises.**
`LoanApplicationInput` keeps `firstName`, `loanAmount`, `monthlyIncome`, `age`,
`cibilScore`, `region`, etc. as declared fields, so `model_dump()` still emits
them and `build_application_response()`, the org dashboard and the stored
`input_data` blobs read them unchanged. What changed is that
**`loanAmount`, `monthlyIncome` and `age` are no longer required** — the
applicant does not type them any more.

**(b) Validators fire only when a value is present.**
`validate_age`, `validate_income`, `validate_cibil` and the EMI-vs-income check
became `if value is not None` guards. A legacy caller posting the full old
payload gets byte-for-byte the old behaviour, including the old rejections; a
short-form payload is not forced to invent values to satisfy them.
`loan_amount` remains mandatory under either name and is enforced in
`finalize()`.

**(c) A two-way name bridge.**
`sync_short_and_legacy_names` (a `mode="before"` validator) mirrors
`loan_amount ↔ loanAmount`, `loan_purpose ↔ loanPurpose`,
`loan_tenure_months ↔ loanTenure`, so both vocabularies are accepted on input
and both are populated on output. `finalize()` re-syncs them after coercion.

**(d) The profile fills the legacy names.**
`resolve_application_payload()` writes three vocabularies into one payload: the
short form's own fields, the snake_case keys CBES scores on, and the legacy
camelCase fields. That last group is the actual bridge — nothing downstream had
to change to stop reading fields the applicant stopped typing.

Two deliberate notes:

* `region` is typed `Any`. CBES scores it as a number (`REGION_RATING_CLIENT`,
  1 = best … 3 = worst) while older callers and the geography report pass a
  label like `"west"`. Coercing either way would corrupt the other, so both pass
  through verbatim; `regionLabel` carries the human-readable form.
* `obligations_other_banks` is folded into the legacy `existingEmis` in
  `finalize()`. EMIs at other lenders are real obligations, and `existingEmis`
  is the field the scoring path already reads.

**Frontend compatibility.** `LoanApplicationFormData` is untouched;
`ShortLoanApplicationFormData` was added alongside it and
`applicationStore.addApplication` accepts either. The old
`LoanApplicationForm` component is left in place and still compiles.

---

## 5. Wiring

* `GET /api/customers/{customer_id}/profile` — new, in
  `backend/app/routers/customers.py`, registered in `main.py` under `/api`.
  Returns `CustomerProfileResponse`, or 404 for an unknown id.
* `_validate_payload()` in `backend/app/routers/applications.py` calls
  `resolve_application_payload()` **before** validation whenever the payload
  carries a `customer_id`, so the scoring engine receives the full field set.
  An unresolvable id is a 404, not a silently degraded score. Payloads without a
  `customer_id` (legacy callers, seeded rows, parsed documents) pass through
  untouched.
* Frontend: `getCustomerProfile()` in `services/applications.ts`;
  `pages/CustomerNewApplication.tsx` rebuilt as the three sections plus a
  debounced customer-id lookup and a read-only "we already have this" panel.
  Submission is blocked until the id resolves.

## 6. Verification

* `python -m pytest backend/tests -q` — **89 passed** (45 at the start of this
  work; the rest were added concurrently by other work in the repo).
* `npx tsc --noEmit` — clean.
* `npm run build` — clean.
