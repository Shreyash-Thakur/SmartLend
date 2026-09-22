"""Score Google TabFM v1.0.0 as a foundation-model baseline, WITH row ids.

TabFM (github.com/google-research/tabfm, Google Research, June 2026) is a
zero-shot in-context tabular foundation model — the same model class as
TabPFN-2.5 but a different lineage (trained on SCM-generated synthetic data,
row-compression + in-context transformer). The reviewer asked for it by name
("google tabfm fusion"), and it is the second candidate that could plausibly
break the 0.99 error-correlation wall documented in reports/complementarity.json.

Design choices, stated up front:

* **Same eval rows as TabPFN.** The scored set is exactly the 8,000
  `SK_ID_CURR`s in reports/tabpfn_scored_rows.csv, so TabFM-vs-TabPFN and
  TabFM-vs-XGBoost comparisons are PAIRED on identical rows.
* **Serving 15-feature vocabulary** via retrain_serving_model_v3's
  build_training_frame — the reproducible, in-repo derivation. (TabPFN's
  original quick-run context features came from an uncommitted script; this
  one is committed, which is the lesson of that episode.)
* **Context is disjoint from the scored rows** — stratified draw, seeded.
* **Row ids saved with every probability** (reports/tabfm_scored_rows.csv), so
  research/analysis/complementarity.py's alignment gate can verify the join
  and run the pre-registered fusion suite automatically.

Licence: TabFM source is Apache-2.0, but the pretrained weights are
`tabfm-non-commercial-v1.0` (non-commercial, non-production) — the same
research-baseline status as TabPFN-2.5. Neither is deployable commercially.

Run:  python -m research.analysis.score_tabfm [--context-rows 5000]
                                              [--n-estimators 8] [--device cuda]
Outputs: reports/tabfm_scored_rows.csv, reports/tabfm_quick_run.json,
         a TabFM row appended/updated in backend/artifacts/model_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from backend.app.services import customer_profile_service as cps
from backend.retrain_serving_model_v3 import FEATURES, build_training_frame

REPO = Path(__file__).resolve().parents[2]
TABPFN_ROWS_CSV = REPO / "reports" / "tabpfn_scored_rows.csv"
OUT_ROWS_CSV = REPO / "reports" / "tabfm_scored_rows.csv"
OUT_RUN_JSON = REPO / "reports" / "tabfm_quick_run.json"
METRICS_CSV = REPO / "backend" / "artifacts" / "model_metrics.csv"

MODEL_LABEL = "TabFM-1.0"
SEED = 42


def _bootstrap_auc_std(y: np.ndarray, p: np.ndarray, n_boot: int = 1000) -> float:
    rng = np.random.default_rng(SEED)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.std(aucs))


def _operating_metrics(y_default: np.ndarray, p_approve: np.ndarray) -> dict:
    """Leaderboard metrics at the model's own Youden threshold, approval framing
    — the same convention as every other row in model_metrics.csv (see commit
    12bbf4f, which re-scored TabPFN this way after a shared 0.5 threshold had
    produced accuracy 0.9193 / recall 1.0 artefacts)."""
    y_good = 1 - y_default
    fpr, tpr, thresholds = roc_curve(y_good, p_approve)
    threshold = float(thresholds[int(np.argmax(tpr - fpr))])
    approve = p_approve >= threshold

    tp = int(np.sum(approve & (y_good == 1)))
    fp = int(np.sum(approve & (y_good == 0)))
    fn = int(np.sum(~approve & (y_good == 1)))
    tn = int(np.sum(~approve & (y_good == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "operating_threshold": round(threshold, 4),
        "accuracy": round((tp + tn) / len(y_good), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 4),
        "default_capture": round(tn / max(tn + fp, 1), 4),
        "approve_rate": round(float(approve.mean()), 4),
    }


def main(context_rows: int, n_estimators: int, device: str,
         update_leaderboard: bool) -> None:
    import tabfm
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("[tabfm] CUDA unavailable — falling back to CPU")
        device = "cpu"

    csv_path = cps._resolve_source_path()
    if csv_path is None:
        raise SystemExit("Home Credit extract not found; set SMARTLEND_CUSTOMER_DATA.")
    if not TABPFN_ROWS_CSV.exists():
        raise SystemExit(f"{TABPFN_ROWS_CSV} missing — the paired eval set is defined by it.")

    started = time.time()
    X, y = build_training_frame(csv_path)
    # Row order in build_training_frame follows the CSV, so ids align by position.
    ids = pd.read_csv(csv_path, usecols=["SK_ID_CURR"])["SK_ID_CURR"].to_numpy()
    assert len(ids) == len(X), "id column and feature frame disagree on row count"

    eval_ids = set(pd.read_csv(TABPFN_ROWS_CSV)["SK_ID_CURR"].astype(int))
    eval_mask = np.isin(ids, list(eval_ids))
    X_eval, y_eval, ids_eval = X[eval_mask], y[eval_mask], ids[eval_mask]
    print(f"[data] eval set: {len(X_eval):,} rows (paired with TabPFN's scored rows)")

    X_rest, y_rest = X[~eval_mask], y[~eval_mask]
    X_ctx, _, y_ctx, _ = train_test_split(
        X_rest, y_rest, train_size=context_rows, random_state=SEED, stratify=y_rest
    )
    print(f"[data] context: {len(X_ctx):,} rows, default rate {y_ctx.mean():.4f}, "
          f"disjoint from eval by construction")

    print(f"[tabfm] loading v1.0.0 pytorch weights (device={device})...")
    model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification", device=device)
    clf = tabfm.TabFMClassifier(model=model, n_estimators=n_estimators, random_state=SEED)

    t_fit = time.time()
    clf.fit(X_ctx, y_ctx)
    print(f"[tabfm] fit (context ingest) in {time.time() - t_fit:.1f}s")

    t_score = time.time()
    proba = clf.predict_proba(X_eval)
    seconds_scoring = time.time() - t_score
    classes = list(clf.classes_)
    p_default = proba[:, classes.index(1)].astype(float)
    print(f"[tabfm] scored {len(X_eval):,} rows in {seconds_scoring:.1f}s")

    out = pd.DataFrame({
        "SK_ID_CURR": ids_eval.astype(int),
        "p_default": np.round(p_default, 6),
        "p_approve": np.round(1.0 - p_default, 6),
        "y_target": y_eval.astype(int),
    })
    out.to_csv(OUT_ROWS_CSV, index=False)

    auc = float(roc_auc_score(y_eval, p_default))
    operating = _operating_metrics(y_eval, 1.0 - p_default)
    metrics = {
        "model": MODEL_LABEL,
        "roc_auc": round(auc, 4),
        "std_auc": round(_bootstrap_auc_std(y_eval, p_default), 5),
        "pr_auc_default": round(float(average_precision_score(y_eval, p_default)), 4),
        "brier": round(float(brier_score_loss(y_eval, p_default)), 5),
        **operating,
    }
    print("[metrics]", json.dumps(metrics, indent=2))

    OUT_RUN_JSON.write_text(json.dumps({
        "model": "TabFM v1.0.0 (google-research/tabfm, pytorch backend)",
        "weights_licence": "tabfm-non-commercial-v1.0 — research baseline only, not deployable",
        "context_rows": int(len(X_ctx)),
        "scored_rows": int(len(X_eval)),
        "n_estimators": n_estimators,
        "device": device,
        "features": FEATURES,
        "feature_note": (
            "Serving 15-feature vocabulary via customer_profile_service._build_profile "
            "(committed derivation). TabPFN's original quick-run context feature set came "
            "from an uncommitted script, so cross-foundation-model AUC comparisons should "
            "be read as paired-rows, possibly-different-features."
        ),
        "eval_rows_source": "exactly the SK_ID_CURRs of reports/tabpfn_scored_rows.csv",
        "seconds_fit": round(time.time() - t_fit - seconds_scoring, 1),
        "seconds_scoring": round(seconds_scoring, 1),
        "seconds_total": round(time.time() - started, 1),
        "metrics": metrics,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[report] wrote {OUT_RUN_JSON} and {OUT_ROWS_CSV}")

    if update_leaderboard and METRICS_CSV.exists():
        table = pd.read_csv(METRICS_CSV)
        row = {
            "model": MODEL_LABEL,
            "roc_auc": metrics["roc_auc"],
            "std_auc": metrics["std_auc"],
            "recall": metrics["recall"],
            "custom_score": metrics["roc_auc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "auc": metrics["roc_auc"],
            "pr_auc_default": metrics["pr_auc_default"],
            "brier": metrics["brier"],
            "operating_threshold": metrics["operating_threshold"],
            "default_capture": metrics["default_capture"],
            "approve_rate": metrics["approve_rate"],
        }
        table = table[table["model"] != MODEL_LABEL]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        table.to_csv(METRICS_CSV, index=False)
        print(f"[leaderboard] {MODEL_LABEL} row written to {METRICS_CSV}")

    print()
    print("Next: python -m research.analysis.complementarity  # runs the")
    print("pre-registered XGB+TabFM fusion suite through the alignment gate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, default=5_000,
                        help="in-context training rows (5,000 matches the TabPFN run)")
    parser.add_argument("--n-estimators", type=int, default=8,
                        help="TabFM ensemble passes (library default 32; 8 keeps a "
                             "laptop run tractable — recorded in the report)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-leaderboard", action="store_true",
                        help="skip updating backend/artifacts/model_metrics.csv")
    args = parser.parse_args()
    main(context_rows=args.context_rows, n_estimators=args.n_estimators,
         device=args.device, update_leaderboard=not args.no_leaderboard)
