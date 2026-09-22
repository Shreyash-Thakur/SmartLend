"""Export the TabPFN in-context training set for the Colab endpoint.

The Colab server (``colab/tabpfn_colab_server.ipynb``) needs two things this
machine has and Colab does not: the real Home Credit extract, and the serving
feature derivation. This script bridges that gap by writing two small CSVs in
the serving 15-feature vocabulary — built by the SAME
``customer_profile_service._build_profile`` path the live API and
``retrain_serving_model_v3`` use, so a feature means the identical number at
export time, at serve time, and inside the Colab context.

Outputs (models/tabpfn/, gitignored — Home Credit data must not be committed):

    tabpfn_context.csv     N context rows + TARGET (upload to Colab; TabPFN
                           "trains" on these in-context)
    tabpfn_eval_rows.csv   M labelled rows DISJOINT from the context (kept
                           local; run_tabpfn_connection_check.py scores them
                           through the tunnel and checks the AUC is sane)
    tabpfn_export_manifest.json   counts, class balance, seed, feature list

Run:  python -m backend.export_tabpfn_context [--context-rows 10000] [--eval-rows 1000]

Colab's T4 (16 GB) holds roughly double the context our local 8 GB RTX 4060
managed (5,000 rows), hence the 10,000-row default. Stratified, seeded,
disjoint by construction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.services import customer_profile_service as cps
from backend.retrain_serving_model_v3 import FEATURES, build_training_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "models" / "tabpfn"
CONTEXT_PATH = OUT_DIR / "tabpfn_context.csv"
EVAL_PATH = OUT_DIR / "tabpfn_eval_rows.csv"
MANIFEST_PATH = OUT_DIR / "tabpfn_export_manifest.json"


def main(context_rows: int, eval_rows: int, seed: int) -> None:
    csv_path = cps._resolve_source_path()
    if csv_path is None:
        raise SystemExit("Home Credit extract not found; set SMARTLEND_CUSTOMER_DATA.")

    started = time.time()
    X, y = build_training_frame(csv_path)
    total_needed = context_rows + eval_rows
    if total_needed > len(X):
        raise SystemExit(f"Asked for {total_needed:,} rows but only {len(X):,} exist.")

    # One stratified draw for everything we export, then a stratified split of
    # that draw into context vs eval — the two files cannot share a row.
    X_pool, _, y_pool, _ = train_test_split(
        X, y, train_size=total_needed, random_state=seed, stratify=y
    )
    X_ctx, X_eval, y_ctx, y_eval = train_test_split(
        X_pool, y_pool, train_size=context_rows, random_state=seed, stratify=y_pool
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = X_ctx.copy()
    ctx["TARGET"] = y_ctx
    ctx.to_csv(CONTEXT_PATH, index=False)

    ev = X_eval.copy()
    ev["TARGET"] = y_eval
    ev.to_csv(EVAL_PATH, index=False)

    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": str(csv_path),
        "features": FEATURES,
        "target": "TARGET (1 = defaulted)",
        "seed": seed,
        "context_rows": int(len(ctx)),
        "context_default_rate": float(np.mean(y_ctx)),
        "eval_rows": int(len(ev)),
        "eval_default_rate": float(np.mean(y_eval)),
        "disjoint": True,
        "seconds": round(time.time() - started, 1),
        "note": (
            "Serving 15-feature vocabulary via customer_profile_service._build_profile "
            "— identical to retrain_serving_model_v3 and the live scoring path. "
            "Files are gitignored: Home Credit competition data may not be redistributed."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[export] {CONTEXT_PATH}  ({len(ctx):,} rows, default rate {np.mean(y_ctx):.4f})")
    print(f"[export] {EVAL_PATH}  ({len(ev):,} rows, default rate {np.mean(y_eval):.4f})")
    print(f"[export] {MANIFEST_PATH}")
    print()
    print("Next: open colab/tabpfn_colab_server.ipynb in Google Colab (GPU runtime),")
    print("upload tabpfn_context.csv when the notebook asks, and follow its printed")
    print("instructions to set SMARTLEND_TABPFN_URL / SMARTLEND_TABPFN_TOKEN here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, default=10_000)
    parser.add_argument("--eval-rows", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(context_rows=args.context_rows, eval_rows=args.eval_rows, seed=args.seed)
