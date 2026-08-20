"""Profile a dataset's missingness and canonical coverage.

    python -m research.data.cli synthetic
    python -m research.data.cli home_credit --csv data/raw/home_credit/application_train.csv
    python -m research.data.cli home_credit --csv <path> --out reports/

Run this FIRST on any newly downloaded dataset. It validates the spec against
the real columns (specs for undownloaded data are written from documentation and
must not be trusted until checked) and reports what needs cleaning before any
modelling begins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research.data.adapters import coverage_report, validate_spec
from research.data.canonical import DatasetSpec
from research.data.profile import profile_frame, to_frame
from research.data.specs import home_credit, lending_club, synthetic

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPECS: dict[str, tuple[DatasetSpec, str]] = {
    "synthetic": (synthetic.SPEC, synthetic.SOURCE_FILE),
    "home_credit": (home_credit.SPEC, home_credit.SOURCE_FILE),
    "lending_club": (lending_club.SPEC, lending_club.ACCEPTED_FILE),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SPECS))
    parser.add_argument("--csv", help="override the default source path")
    parser.add_argument("--out", help="directory to write CSV reports into")
    parser.add_argument("--sample", type=int, default=50_000, help="0 disables sampling")
    parser.add_argument("--nrows", type=int, default=None, help="read only N rows")
    args = parser.parse_args(argv)

    spec, default_path = SPECS[args.dataset]
    path = Path(args.csv) if args.csv else PROJECT_ROOT / default_path
    if not path.exists():
        parser.error(
            f"{path} not found.\n"
            "Home Credit: kaggle competitions download -c home-credit-default-risk\n"
            "Lending Club: kaggle datasets download -d wordsforthewise/lending-club"
        )

    print(f"Reading {path} ...")
    df = pd.read_csv(path, nrows=args.nrows, low_memory=False)
    print(f"  {len(df):,} rows x {len(df.columns)} columns\n")

    print("=== SPEC VALIDATION ===")
    problems = validate_spec(spec, df)
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
        print("\n  Specs written from documentation must be corrected before use.\n")
    else:
        print("  spec matches the file\n")

    print("=== CANONICAL COVERAGE ===")
    coverage = coverage_report(spec)
    print(coverage.to_string(index=False), "\n")

    print("=== MISSINGNESS PROFILE ===")
    target = spec.target if isinstance(spec.target, str) and spec.target in df.columns else None
    report = to_frame(
        profile_frame(df, target=target, sample_size=args.sample or None)
    )
    summary_columns = [
        "column",
        "missing_rate",
        "sentinel_rate",
        "sentinels",
        "structural_predictor",
        "block_id",
        "target_auc",
        "classification",
    ]
    needs_work = report[report["classification"] != "complete"]
    print((needs_work if not needs_work.empty else report)[summary_columns].to_string(index=False))

    print("\n=== SUMMARY ===")
    print(report["classification"].value_counts().to_string())

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_dir / f"{spec.name}_missingness.csv", index=False)
        coverage.to_csv(out_dir / f"{spec.name}_coverage.csv", index=False)
        print(f"\nWrote reports to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
