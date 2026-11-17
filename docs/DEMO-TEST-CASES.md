# Demo test cases

Inputs that give a confirmed decision on **New Application**, for live demos
and the viva. The same 15 cases are on the in-app page **Demo → Demo Test
Cases** (`/dashboard/demo-cases`). Both read
`frontend/src/data/demoTestCases.json`, and
`backend/tests/test_demo_test_cases.py` re-checks every row against the engine,
so if the engine changes the test fails before the demo does.

Engine: `hybrid-2stage-5gate/2026-11-17+uncertainty-default+foir-gate`.

**Fill in:** customer ID, loan amount, tenure and purpose as listed. Leave
everything else at its default (EMIs at other lenders 0, additional income 0,
collateral none, employment not changed recently) and tick all declarations.
Interest is the backend default of 12% p.a.

## How a decision is made

1. **Model** (calibrated logistic regression, 15 fields): approval score
   ≥ 0.896 → approve, ≤ 0.404 → reject, in between → defer to a human.
2. **Affordability policy** (`backend/app/services/policy_rules.py`), run after
   the model. FOIR = all EMIs ÷ income:
   - FOIR above 200% → **reject**, whatever the model says (cannot be repaid);
   - FOIR 100–200% → a model **approve** becomes **defer**, so a human checks income;
   - otherwise the model's decision stands.

The policy exists because the model learned from Home Credit that bigger loans
are safer (bigger loans went to trusted clients). Without it, almost nothing
was rejected: on the 510 seeded customers it approved 65% at ₹1L and 89% at
₹25L, and rejected only 2 customers (0.4%) at every size. With it, at ₹2L/36
months: approve 67→55%, defer 33→43%, reject 0.4→2%
(`reports/affordability_gate_impact.json`, from
`python -m research.analysis.affordability_gate_impact`).

## Cases

| # | Customer ID | Loan | Tenure | Purpose | Expected | Decided by | Why |
|---|---|---|---|---|---|---|---|
| 1 | 100061 | ₹1,00,000 | 36 | Other | APPROVE | Model | Strong profile; EMI about 25% of income |
| 2 | 109515 | ₹2,00,000 | 36 | Other | APPROVE | Model | Score 0.96; EMI about 54% of income |
| 3 | 104346 | ₹50,000 | 36 | Medical | APPROVE | Model | High credit use, but a small loan |
| 4 | 103729 | ₹1,00,000 | 36 | Education | APPROVE | Model | Thin file, clean record, affordable |
| 5 | 100061 | ₹5,00,000 | 60 | Home improvement | APPROVE | Model | FOIR about 67% over 5 years |
| 6 | 102279 | ₹1,00,000 | 36 | Other | DEFER | Model | Prior delinquencies; score 0.74 (uncertain) |
| 7 | 105969 | ₹2,00,000 | 36 | Other | DEFER | Model | Weak external score (366) |
| 8 | 111514 | ₹1,00,000 | 36 | Medical | DEFER | Model | Pensioner, no employment record |
| 9 | 101672 | ₹3,00,000 | 36 | Other | DEFER | Policy | Model approves, but FOIR about 136% |
| 10 | 109515 | ₹10,00,000 | 60 | Business | DEFER | Policy | FOIR about 144% |
| 11 | 156227 | ₹1,00,000 | 36 | Other | REJECT | Model | Delinquencies, near-zero external score; score 0.24 |
| 12 | 236551 | ₹50,000 | 36 | Other | REJECT | Model | Delinquencies, very low external score; score 0.26 |
| 13 | 105969 | ₹25,00,000 | 24 | Medical | REJECT | Policy | EMIs about 5× income |
| 14 | 101672 | ₹10,00,000 | 60 | Business | REJECT | Policy | FOIR about 267% |
| 15 | 100061 | ₹25,00,000 | 24 | Home improvement | REJECT | Policy | Model score 0.98, but cannot repay |

Customers 156227 and 236551 are not in the "Try a sample customer" chips;
type the ID into the Customer ID box.

## Known limits (say these before an examiner does)

- The model alone almost never rejects; most rejections come from the policy.
- Customer IDs are existing Home Credit rows, so these customers were seen in
  training. The demo shows behaviour, not held-out accuracy.
- Interest is fixed at 12% for the FOIR check; the form does not send a rate.
