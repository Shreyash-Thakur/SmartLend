"""The connection run: prove the Colab-hosted TabPFN endpoint works end to end.

What it checks, in order:

1. **Configured** — SMARTLEND_TABPFN_URL (and token) are set in .env.
2. **Reachable** — GET /health answers and reports what is loaded.
3. **Scores** — a batch of locally-held, labelled eval rows
   (models/tabpfn/tabpfn_eval_rows.csv, written by export_tabpfn_context.py,
   disjoint from the Colab context) comes back with per-row probabilities.
4. **Sane** — the AUC of those remote scores against the local labels is
   materially above chance. A tunnel that answers with garbage (wrong feature
   order, wrong model, shuffled rows) fails HERE rather than looking healthy.

Writes reports/tabpfn_connection_run.json and exits non-zero on any failure,
so it can gate a demo the same way research/relearning/gate.py gates the loop.

Run:  python -m backend.run_tabpfn_connection_check [--rows 400] [--batch 200]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.app.services import remote_tabpfn_service

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "models" / "tabpfn" / "tabpfn_eval_rows.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "tabpfn_connection_run.json"

# TabPFN-2.5 reached 0.7446 AUC on the full holdout and 0.7284 on an 8,000-row
# sample (reports/tabpfn_quick_run.json). On a few hundred rows the interval is
# wide, so the gate only rejects clearly-broken plumbing, not sampling noise.
MIN_SANE_AUC = 0.60


def fail(report: dict, reason: str) -> None:
    report["verdict"] = "FAIL"
    report["reason"] = reason
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[connection-run] FAIL — {reason}")
    print(f"[connection-run] wrote {REPORT_PATH}")
    sys.exit(1)


def main(rows: int, batch: int) -> None:
    report: dict = {"started_at": pd.Timestamp.utcnow().isoformat(), "requested_rows": rows}

    # 1. configured
    if not remote_tabpfn_service.configured():
        fail(report, "SMARTLEND_TABPFN_URL is not set. Start the Colab notebook and copy its URL/token into .env.")

    # 2. reachable
    status = remote_tabpfn_service.remote_status()
    report["health"] = status
    if not status.get("reachable"):
        fail(report, f"Endpoint configured but unreachable: {status.get('detail')}")
    print(f"[connection-run] health OK: {json.dumps(status.get('server', {}))[:300]}")

    # 3. scores
    if not EVAL_PATH.exists():
        fail(report, f"{EVAL_PATH} missing — run `python -m backend.export_tabpfn_context` first.")
    frame = pd.read_csv(EVAL_PATH).head(rows)
    y = frame.pop("TARGET").astype(int).to_numpy()
    feature_rows = frame.to_dict("records")

    p_default: list[float] = []
    latencies: list[float] = []
    for start in range(0, len(feature_rows), batch):
        chunk = feature_rows[start : start + batch]
        t0 = time.time()
        try:
            predictions = remote_tabpfn_service.score_rows(chunk)
        except remote_tabpfn_service.RemoteTabPFNError as exc:
            fail(report, f"Scoring failed on rows {start}-{start + len(chunk)}: {exc}")
        latencies.append(time.time() - t0)
        p_default.extend(item["p_default"] for item in predictions)
        print(f"[connection-run] scored {start + len(chunk)}/{len(feature_rows)} rows "
              f"({latencies[-1]:.1f}s for this batch)")

    # 4. sanity — remote probabilities must rank the local labels far above chance
    auc = float(roc_auc_score(y, p_default))
    report.update(
        {
            "scored_rows": len(p_default),
            "eval_default_rate": float(y.mean()),
            "remote_auc_p_default": auc,
            "min_sane_auc": MIN_SANE_AUC,
            "batches": len(latencies),
            "batch_seconds": [round(v, 2) for v in latencies],
            "total_seconds": round(sum(latencies), 1),
            "mean_p_default": float(pd.Series(p_default).mean()),
        }
    )
    if auc < MIN_SANE_AUC:
        fail(
            report,
            f"Remote AUC {auc:.4f} < {MIN_SANE_AUC} — the endpoint answers but its scores do not rank "
            "the labels. Check the uploaded context file and the feature order.",
        )

    report["verdict"] = "PASS"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[connection-run] PASS — remote AUC {auc:.4f} on {len(p_default)} rows "
          f"in {sum(latencies):.1f}s")
    print(f"[connection-run] wrote {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=400, help="eval rows to score (max = file length)")
    parser.add_argument("--batch", type=int, default=200)
    args = parser.parse_args()
    main(rows=args.rows, batch=args.batch)
