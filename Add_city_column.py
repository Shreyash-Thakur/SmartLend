"""
add_city_column.py
==================
Reads your existing synthetic_indian_loan_dataset.csv,
adds a 'city' column based on the existing 'region' column,
and saves it back (or to a new file — your choice).

Usage:
    python add_city_column.py
"""

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "synthetic_indian_loan_dataset.csv"   # <- your existing file
OUTPUT_FILE = "synthetic_indian_loan_dataset.csv"   # <- overwrite, or change name

SEED = 42
np.random.seed(SEED)

# ── City pools (region-aware, population-weighted) ────────────────────────────
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
        ]
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
        ]
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
        ]
    }
}

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading '{INPUT_FILE}' ...")
df = pd.read_csv(INPUT_FILE)
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# Verify the region column exists
assert "region" in df.columns, \
    "ERROR: 'region' column not found. Check your CSV file."

# Verify no unexpected region values
found_regions = set(df["region"].unique())
expected      = {"Urban", "Semi-Urban", "Rural"}
unknown       = found_regions - expected
assert not unknown, f"ERROR: Unexpected region values found: {unknown}"

# ── Assign city per row ───────────────────────────────────────────────────────
print("Assigning cities ...")

def assign_cities(region_series):
    result = np.empty(len(region_series), dtype=object)
    for region, cfg in CITY_MAP.items():
        mask = region_series == region
        n    = mask.sum()
        if n > 0:
            result[mask] = np.random.choice(
                cfg["cities"], size=n, p=cfg["weights"]
            )
    return result

df["city"] = assign_cities(df["region"])

# ── Place 'city' right after 'region' ─────────────────────────────────────────
cols = list(df.columns)
cols.remove("city")
region_idx = cols.index("region")
cols.insert(region_idx + 1, "city")
df = df[cols]

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to '{OUTPUT_FILE}'")
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# ── Quick verification ────────────────────────────────────────────────────────
print("\n── City counts by Region (top 5 per region) ──────────────────────")
for region in ["Urban", "Semi-Urban", "Rural"]:
    top = (df[df["region"] == region]["city"]
           .value_counts()
           .head(5))
    print(f"\n  {region}:")
    for city, count in top.items():
        print(f"    {city:<18} {count:>5,}")

print("\n── Sample rows (region + city) ───────────────────────────────────")
print(df[["applicant_id", "region", "city"]].sample(8, random_state=1).to_string(index=False))
print("\nDone.")