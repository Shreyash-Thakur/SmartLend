"""Append reject-tail customers to the committed demo seed.

WHY. The original 500-customer fixture was stratified by outcome x
EXT_SOURCE_2 quartile, but a /predict scan showed the SERVING model's
approval probability never drops below ~0.79 for any of them - so a live
demo can show instant APPROVE and DEFER but never an instant model REJECT,
even though the engine rejects 23% of the full population
(backend/artifacts/prediction_outputs.csv). This script appends the
population's genuine reject tail - the applicants the serving model is most
confident about rejecting - so the sample picker contains honest
clear-reject cases.

Selection is from the committed artifact (lowest best_model_prob among
final_decision == REJECT), raw fields come from the local Home Credit
extract (never redistributed; this runs only on a machine that has it), and
descriptors use the same observable-fields-only describe_profile_row as
every other seed row. TARGET is never copied.

Run:   python -m backend.append_reject_tail_seed [--count 10]
Then:  verify live decisions, and flag confirmed ones with --mark-samples.
Mark:  python -m backend.append_reject_tail_seed --mark-samples 123,456
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.services.customer_seed_service import (
    SEED_FIXTURE_PATH,
    _CSV_TO_MODEL,
    describe_profile_row,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = PROJECT_ROOT / "backend" / "artifacts" / "prediction_outputs.csv"
EXTRACT = Path.home() / "Downloads" / "creddefer_full_merged.csv"


def append_tail(count: int) -> None:
    import pandas as pd

    preds = pd.read_csv(PREDICTIONS, usecols=["applicant_id", "best_model_prob", "final_decision"])
    tail = (
        preds[preds["final_decision"] == "REJECT"]
        .nsmallest(count * 3, "best_model_prob")  # spare candidates for live verification
    )
    # applicant_id is prefixed ("HC156227"); SK_ID_CURR is the numeric part.
    ids = tail["applicant_id"].astype(str).str.removeprefix("HC").astype(int).tolist()
    print(f"[tail] {len(ids)} candidates, approval prob "
          f"{tail['best_model_prob'].min():.3f}..{tail['best_model_prob'].max():.3f}")

    frame = pd.read_csv(EXTRACT, usecols=[c for c in _CSV_TO_MODEL], low_memory=False)
    rows = frame[frame["SK_ID_CURR"].isin(ids)]

    fixture = json.loads(SEED_FIXTURE_PATH.read_text(encoding="utf-8"))
    known = {r["customer_id"] for r in fixture}
    added = 0
    for _, csv_row in rows.iterrows():
        record: dict = {}
        for csv_column, attribute in _CSV_TO_MODEL.items():
            value = csv_row[csv_column]
            if pd.isna(value):
                record[attribute] = None
            elif attribute == "customer_id":
                record[attribute] = int(value)
            elif isinstance(value, str):
                record[attribute] = value
            else:
                record[attribute] = float(value)
        if record["customer_id"] in known:
            continue
        record["descriptor"] = describe_profile_row(record)
        record["is_sample"] = False  # flipped by --mark-samples after live verification
        fixture.append(record)
        added += 1
        if added >= count:
            break

    fixture.sort(key=lambda r: r["customer_id"])
    SEED_FIXTURE_PATH.write_text(json.dumps(fixture, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[tail] appended {added}; fixture now {len(fixture)} rows")
    print("[tail] candidate ids:", [r["customer_id"] for r in fixture if r["customer_id"] in set(ids)])


def mark_samples(ids: list[int]) -> None:
    fixture = json.loads(SEED_FIXTURE_PATH.read_text(encoding="utf-8"))
    hit = 0
    for r in fixture:
        if r["customer_id"] in ids:
            r["is_sample"] = True
            hit += 1
    SEED_FIXTURE_PATH.write_text(json.dumps(fixture, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[mark] flagged {hit} as showcase samples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--mark-samples", default=None, help="comma-separated customer ids")
    args = ap.parse_args()
    if args.mark_samples:
        mark_samples([int(x) for x in args.mark_samples.split(",")])
    else:
        append_tail(args.count)
