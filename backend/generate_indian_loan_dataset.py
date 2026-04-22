"""
Synthetic Indian Loan Approval & Credit Risk Dataset Generator
==============================================================
Target distributions (calibrated, not guessed):
  • default_risk = 1  (default)      : ~30–35%
  • default_risk = 0  (non-default)  : ~65–70%
  • loan_approved = 1 (approved)     : ~57–62%

25,000 rows | 34 columns | realistic Indian banking behaviour
"""

import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 25_000

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATED INTERCEPTS  (derived from systematic grid search)
# ─────────────────────────────────────────────────────────────────────────────
RISK_INTERCEPT     = -2.85   # → ~31–33% default rate
APPROVAL_INTERCEPT =  2.0    # → ~58–60% approval rate

# =============================================================================
# SECTION 1 ── REGION & EMPLOYMENT
# =============================================================================
regions   = np.random.choice(
    ["Urban", "Semi-Urban", "Rural"],
    size=N, p=[0.45, 0.35, 0.20]
)

CITY_MAP = {
    "Urban": {
        "cities": [
            "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
            "Pune", "Kolkata", "Ahmedabad", "Surat", "Jaipur",
            "Lucknow", "Kochi", "Chandigarh", "Indore", "Nagpur"
        ],
        "weights": [
            0.13, 0.12, 0.11, 0.09, 0.08,
            0.08, 0.07, 0.07, 0.05, 0.04,
            0.04, 0.04, 0.03, 0.03, 0.02
        ],
    },
    "Semi-Urban": {
        "cities": [
            "Agra", "Varanasi", "Bhopal", "Coimbatore", "Vadodara",
            "Visakhapatnam", "Patna", "Rajkot", "Mysuru", "Amritsar",
            "Jabalpur", "Meerut", "Nashik", "Aurangabad", "Jodhpur",
            "Ranchi", "Guwahati", "Dehradun", "Raipur", "Bhubaneswar"
        ],
        "weights": [
            0.07, 0.07, 0.07, 0.06, 0.06,
            0.06, 0.06, 0.05, 0.05, 0.05,
            0.04, 0.04, 0.05, 0.04, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.03
        ],
    },
    "Rural": {
        "cities": [
            "Muzaffarpur", "Gorakhpur", "Aligarh", "Moradabad", "Bareilly",
            "Saharanpur", "Bhagalpur", "Darbhanga", "Sitapur", "Hardoi",
            "Nandurbar", "Osmanabad", "Bidar", "Kolar", "Raichur",
            "Araria", "Kishanganj", "Supaul", "Madhepura", "Sheohar"
        ],
        "weights": [
            0.08, 0.08, 0.07, 0.07, 0.06,
            0.06, 0.06, 0.06, 0.05, 0.05,
            0.04, 0.04, 0.04, 0.04, 0.03,
            0.04, 0.04, 0.04, 0.04, 0.01
        ],
    },
}

city = np.empty(N, dtype=object)
for region_name, config in CITY_MAP.items():
    mask = regions == region_name
    city[mask] = np.random.choice(config["cities"], size=mask.sum(), p=config["weights"])

emp_types = np.random.choice(
    ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
    size=N, p=[0.50, 0.25, 0.18, 0.07]
)

# =============================================================================
# SECTION 2 ── APPLICANT PROFILE
# =============================================================================
applicant_ids  = [f"LOAN{str(i).zfill(6)}" for i in range(1, N + 1)]
ages           = np.random.randint(21, 66, size=N)
genders        = np.random.choice(
    ["Male", "Female", "Other"], size=N, p=[0.62, 0.36, 0.02]
)
marital_status = np.random.choice(
    ["Married", "Single", "Divorced"], size=N, p=[0.60, 0.33, 0.07]
)
dependents     = np.random.choice(
    [0, 1, 2, 3, 4, 5], size=N, p=[0.20, 0.25, 0.30, 0.15, 0.07, 0.03]
)
education      = np.random.choice(
    ["Graduate", "Non-Graduate"], size=N, p=[0.62, 0.38]
)
years_employed = np.where(
    emp_types == "Unemployed",
    0,
    np.clip(np.random.exponential(scale=6, size=N).astype(int), 0, 35)
)

# =============================================================================
# SECTION 3 ── INCOME  (log-normal, region-stratified)
# =============================================================================
_income_mean = {"Urban": 13.5, "Semi-Urban": 12.8, "Rural": 12.2}

annual_income = np.array([
    np.clip(
        np.random.lognormal(mean=_income_mean[r], sigma=0.6),
        200_000, 5_000_000
    )
    for r in regions
])
annual_income = np.where(
    emp_types == "Unemployed",
    np.clip(annual_income * np.random.uniform(0.15, 0.40, N), 200_000, 800_000),
    annual_income
)
annual_income  = annual_income.round(2)
monthly_income = (annual_income / 12).round(2)

# =============================================================================
# SECTION 4 ── LOAN DETAILS
# =============================================================================
loan_types = np.random.choice(
    ["Home", "Personal", "Gold", "Auto", "Credit Card"],
    size=N, p=[0.30, 0.28, 0.12, 0.20, 0.10]
)
loan_multiplier = np.random.uniform(0.5, 8.0, size=N)
loan_multiplier = np.where(loan_types == "Home",
                           np.random.uniform(3.0, 8.0, N), loan_multiplier)
loan_multiplier = np.where(loan_types == "Credit Card",
                           np.random.uniform(0.1, 1.0, N), loan_multiplier)

loan_amount = np.clip(
    annual_income * loan_multiplier + np.random.normal(0, 50_000, N),
    50_000, 10_000_000
).round(2)

loan_term = np.random.choice(
    [12, 24, 36, 48, 60, 84, 120, 180, 240, 360],
    size=N,
    p=[0.04, 0.06, 0.10, 0.08, 0.12, 0.10, 0.15, 0.15, 0.10, 0.10]
)
interest_rate = np.clip(
    np.random.normal(loc=12.5, scale=3.5, size=N), 8.0, 24.0
).round(2)

monthly_rate = interest_rate / 100 / 12
emi = np.where(
    monthly_rate == 0,
    loan_amount / loan_term,
    (loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term)
    / ((1 + monthly_rate) ** loan_term - 1)
).round(2)

existing_emis = np.clip(
    np.random.exponential(scale=monthly_income * 0.12, size=N),
    0, monthly_income * 0.60
).round(2)

# =============================================================================
# SECTION 5 ── ASSETS & BANK BALANCE
# =============================================================================
residential_assets = np.clip(
    np.random.lognormal(mean=np.log(annual_income * 1.5), sigma=0.8),
    0, 30_000_000
).round(2)

commercial_assets = np.where(
    np.isin(emp_types, ["Business Owner", "Self-Employed"]),
    np.clip(np.random.lognormal(mean=np.log(annual_income * 0.8 + 1), sigma=1.0),
            0, 20_000_000),
    np.clip(np.random.exponential(annual_income * 0.2), 0, 5_000_000)
).round(2)

bank_balance = np.clip(
    np.random.lognormal(mean=np.log(monthly_income * 3 + 1), sigma=0.9),
    0, 5_000_000
).round(2)

total_assets = (residential_assets + commercial_assets + bank_balance).round(2)

# =============================================================================
# SECTION 6 ── CREDIT BEHAVIOUR (CIBIL-like)
# =============================================================================
cibil_base  = (
    400
    + (annual_income / 5_000_000) * 200
    + (ages - 21) / 44 * 100
    + np.random.normal(0, 60, N)
)
cibil_score = np.clip(cibil_base, 300, 900).astype(int)

total_loans   = np.random.choice(
    [1, 2, 3, 4, 5, 6, 7], size=N,
    p=[0.20, 0.25, 0.22, 0.15, 0.10, 0.05, 0.03]
)
active_loans  = np.array([np.random.randint(0, t + 1) for t in total_loans])
closed_loans  = total_loans - active_loans

missed_base     = np.clip((800 - cibil_score) / 100, 0, 5)
missed_payments = np.clip(
    (missed_base + np.random.exponential(0.8, N)).astype(int), 0, 15
)

credit_utilization_ratio = np.clip(
    np.random.beta(a=2, b=3, size=N)
    + (missed_payments / 30)
    + np.random.normal(0, 0.05, N),
    0.0, 1.0
).round(4)

# =============================================================================
# SECTION 7 ── DERIVED FEATURES
# =============================================================================
total_emi         = (emi + existing_emis).round(2)
emi_income_ratio  = np.clip((total_emi / monthly_income).round(4), 0, 5)
loan_income_ratio = np.clip((loan_amount / annual_income).round(4), 0, 20)
debt_to_income    = np.clip(((existing_emis * 12) / annual_income).round(4), 0, 5)

# =============================================================================
# SECTION 8 ── DEFAULT RISK
# Probabilistic sigmoid model — NOT rule-based thresholds
# Calibrated intercept → ~31–33% default rate
# =============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

risk_score = (
    RISK_INTERCEPT
    - 0.012 * (cibil_score - 600)          # higher CIBIL = lower risk
    + 0.18  * emi_income_ratio             # repayment stress
    + 0.22  * credit_utilization_ratio     # credit pressure
    + 0.15  * (missed_payments / 5)        # delinquency history
    + 0.08  * debt_to_income               # existing obligations
    - 0.04  * (annual_income / 1_000_000)  # income as buffer
    + 0.05  * (dependents / 5)             # family burden
    + np.random.normal(0, 0.35, N)         # intentional noise
)

default_prob = sigmoid(risk_score)
default_risk = (np.random.uniform(0, 1, N) < default_prob).astype(int)

# =============================================================================
# SECTION 9 ── LOAN APPROVED
# Separate model from default_risk — intentionally imperfect
# Calibrated intercept → ~58–60% approval rate
# =============================================================================
approval_score = (
    APPROVAL_INTERCEPT
    + 0.010 * (cibil_score - 600)
    - 0.20  * emi_income_ratio
    + 0.04  * (annual_income / 1_000_000)
    - 0.10  * (missed_payments / 5)
    - 0.08  * credit_utilization_ratio
    + 0.06  * (years_employed / 10)
    + np.random.normal(0, 0.40, N)         # bank-level decision noise
)

approval_prob = sigmoid(approval_score)
loan_approved = (np.random.uniform(0, 1, N) < approval_prob).astype(int)

# Deliberate flips (~8%): relationship banking, errors, mis-selling
flip_mask     = np.random.uniform(0, 1, N) < 0.08
loan_approved = np.where(flip_mask, 1 - loan_approved, loan_approved)

# =============================================================================
# SECTION 10 ── CONFIDENCE SCORE
# =============================================================================
margin           = np.abs(default_prob - 0.5)
confidence_score = np.clip(
    0.5 + margin + np.random.normal(0, 0.04, N),
    0.0, 1.0
).round(4)

# =============================================================================
# SECTION 11 ── ASSEMBLE DATAFRAME
# =============================================================================
df = pd.DataFrame({
    "applicant_id"             : applicant_ids,
    "age"                      : ages,
    "gender"                   : genders,
    "marital_status"           : marital_status,
    "dependents"               : dependents,
    "education"                : education,
    "employment_type"          : emp_types,
    "years_employed"           : years_employed,
    "annual_income"            : annual_income,
    "monthly_income"           : monthly_income,
    "loan_type"                : loan_types,
    "loan_amount"              : loan_amount,
    "loan_term"                : loan_term,
    "interest_rate"            : interest_rate,
    "emi"                      : emi,
    "existing_emis"            : existing_emis,
    "residential_assets_value" : residential_assets,
    "commercial_assets_value"  : commercial_assets,
    "bank_balance"             : bank_balance,
    "total_assets"             : total_assets,
    "cibil_score"              : cibil_score,
    "total_loans"              : total_loans,
    "active_loans"             : active_loans,
    "closed_loans"             : closed_loans,
    "missed_payments"          : missed_payments,
    "credit_utilization_ratio" : credit_utilization_ratio,
    "emi_income_ratio"         : emi_income_ratio,
    "loan_income_ratio"        : loan_income_ratio,
    "debt_to_income_ratio"     : debt_to_income,
    "region"                   : regions,
    "city"                     : city,
    "default_risk"             : default_risk,
    "loan_approved"            : loan_approved,
    "confidence_score"         : confidence_score,
})

# =============================================================================
# SECTION 12 ── QUALITY CHECKS
# =============================================================================
assert df["applicant_id"].nunique() == N,               "Duplicate IDs!"
assert (df["closed_loans"] <= df["total_loans"]).all(), "closed > total!"
assert (df["active_loans"] <= df["total_loans"]).all(), "active > total!"
assert (df[["annual_income","loan_amount","emi",
            "bank_balance","total_assets"]] >= 0).all().all(), "Negatives!"
assert 0.28 <= df["default_risk"].mean() <= 0.38, \
    f"Default rate out of calibrated range: {df['default_risk'].mean():.2%}"

# =============================================================================
# SECTION 13 ── SAVE & REPORT
# =============================================================================
output_path = "synthetic_indian_loan_dataset.csv"
df.to_csv(output_path, index=False)

sep = "=" * 68
print(sep)
print("   SYNTHETIC INDIAN LOAN DATASET  —  GENERATION COMPLETE")
print(sep)
print(f"   Rows     : {len(df):,}")
print(f"   Columns  : {len(df.columns)}")
print(f"   Saved to : {output_path}")

n_default  = int(df["default_risk"].sum())
n_safe     = N - n_default
n_approved = int(df["loan_approved"].sum())
n_rejected = N - n_approved

print()
print("── Default Risk Distribution ───────────────────────────────────────")
print(f"   Safe (0)    : {n_safe:>6,}  ({n_safe/N*100:.1f}%)")
print(f"   Default (1) : {n_default:>6,}  ({n_default/N*100:.1f}%)")

print()
print("── Loan Approval Distribution ──────────────────────────────────────")
print(f"   Approved (1) : {n_approved:>6,}  ({n_approved/N*100:.1f}%)")
print(f"   Rejected (0) : {n_rejected:>6,}  ({n_rejected/N*100:.1f}%)")

cross = pd.crosstab(
    df["loan_approved"], df["default_risk"],
    rownames=["loan_approved"], colnames=["default_risk"],
    margins=True
)
print()
print("── Approval × Default (rows=approved, cols=default) ────────────────")
print(cross.to_string())

true_pos  = int(((df["loan_approved"]==0) & (df["default_risk"]==1)).sum())
true_neg  = int(((df["loan_approved"]==1) & (df["default_risk"]==0)).sum())
false_app = int(((df["loan_approved"]==1) & (df["default_risk"]==1)).sum())
false_rej = int(((df["loan_approved"]==0) & (df["default_risk"]==0)).sum())

print()
print("── Real-World Decision Quality ─────────────────────────────────────")
print(f"   Correct approvals  (approved & safe)    : {true_neg:>6,}  ({true_neg/N*100:.1f}%)")
print(f"   Correct rejections (rejected & default) : {true_pos:>6,}  ({true_pos/N*100:.1f}%)")
print(f"   FALSE approvals    (approved & default) : {false_app:>6,}  ({false_app/N*100:.1f}%)  ← risk exposure")
print(f"   FALSE rejections   (rejected & safe)    : {false_rej:>6,}  ({false_rej/N*100:.1f}%)  ← lost business")

borderline = int((df["confidence_score"] < 0.6).sum())
print()
print("── Confidence Score ────────────────────────────────────────────────")
print(f"   Mean                              : {df['confidence_score'].mean():.3f}")
print(f"   Borderline cases (conf < 0.60)    : {borderline:,}  ({borderline/N*100:.1f}%)")

print()
print("── Region | Median Income | Default Rate ───────────────────────────")
rstats = df.groupby("region").agg(
    median_income=("annual_income","median"),
    default_rate =("default_risk","mean"),
    n            =("applicant_id","count")
).reset_index()
for _, row in rstats.iterrows():
    print(f"   {row['region']:<12} ₹{row['median_income']:>10,.0f}   "
          f"default={row['default_rate']*100:.1f}%   n={row['n']:,}")

print()
print("── CIBIL Band | Default Rate ───────────────────────────────────────")
bins   = [300, 500, 600, 700, 750, 800, 900]
labels = ["300-500","500-600","600-700","700-750","750-800","800-900"]
df["_band"] = pd.cut(df["cibil_score"], bins=bins, labels=labels, include_lowest=True)
for band, grp in df.groupby("_band", observed=True):
    dr = grp["default_risk"].mean()
    print(f"   {band}   default={dr*100:.1f}%   n={len(grp):,}")
df.drop(columns="_band", inplace=True)

print()
print("── Sample Rows ─────────────────────────────────────────────────────")
cols = ["applicant_id","region","annual_income","cibil_score",
        "emi_income_ratio","missed_payments",
        "default_risk","loan_approved","confidence_score"]
print(df[cols].head(8).to_string(index=False))
print(sep)
