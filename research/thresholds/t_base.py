"""T_base selection study — why the F1 sweep was degenerate and what replaces it.

Run from the project root:

    python -m research.thresholds.t_base                 # OOF analysis only
    python -m research.thresholds.t_base --serving       # + pipeline_v3_real check

Writes reports/t_base_selection.json.

What this study does
--------------------
1. DOCUMENTS THE DEFECT. The old threshold fit took the F1-argmax over a
   hardcoded sweep of [0.30, 0.70). The engine score p_ml = P(approval)
   concentrates near 0.92 (8.07% default rate, calibrated model), so almost
   no applicant scores below 0.70: F1 is near zero and monotone increasing
   across the entire window, and the argmax is pinned at the top edge of the
   sweep. The selected t_base was an artifact of the range bounds. The flat
   curve is tabulated in the output JSON (`defect_evidence`).

2. COMPARES REPLACEMENT METHODS on the out-of-fold predictions in
   backend/artifacts/prediction_outputs.csv (307,511 rows; y_true = 1 means
   the customer repaid / should be APPROVED; prob_* columns are approval
   probabilities). Methods (see backend/app/services/threshold_selection.py):
     - cost           expected-cost minimum (LGD / ROI loss matrix) — RECOMMENDED
     - youden         max TPR - FPR (cost-agnostic reference)
     - f1             F1-argmax over a percentile grid of the observed p_ml
     - approval_rate  threshold hitting a target approval rate
     - f1_legacy      the old defect, for the before/after comparison

3. SPLIT DISCIPLINE. A threshold argmax is a fitted parameter. Rows are split
   50/50 (stratified on the label, random_state=42) into a SELECT half — the
   only data any method sees when choosing its threshold — and a REPORT half,
   on which every number in `methods[*].report_half_metrics` is computed.
   Nothing is tuned on the report half.

4. COST MODEL AND CITATIONS. Per applicant, with EAD normalised to 1 because
   the OOF file carries no loan amounts:
     - approving a defaulter costs LGD = 0.45  (Basel II foundation-IRB
       supervisory LGD for senior unsecured exposures; BCBS 128, June 2006,
       para 287)
     - rejecting a good customer costs ROI = 0.2644  (return on a performing
       consumer loan; Verbraken, Bravo, Weber & Baesens, EJOR 238(2), 2014 —
       the EMP credit-scoring measure's ROI parameter, estimated from real
       consumer-loan portfolios)
   For a calibrated score the Bayes-optimal threshold is
   t* = 1 - ROI/(ROI + LGD) ≈ 0.6299 with these defaults. Both parameters are
   configurable, and because the answer moves materially with the assumed
   cost ratio, the JSON includes a sensitivity table over LGD/ROI in
   [1, 20] — read that table, not just the headline number.

5. OPTIONAL SERVING-MODEL CHECK (--serving). Rebuilds the v3 training frame
   exactly as backend/retrain_serving_model_v3.py does, loads the actual
   serving artifact backend/artifacts/pipeline_v3_real.joblib, computes p_ml
   on the artifact's own held-out 20% split, and repeats the select/report
   procedure there. Requires the Home Credit extract on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.services.threshold_selection import (
    DEFAULT_LGD,
    DEFAULT_ROI,
    ENGINE_CLIP_HI,
    ENGINE_CLIP_LO,
    confusion_metrics,
    expected_cost_per_applicant,
    select_t_base,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OOF_CSV = PROJECT_ROOT / "backend" / "artifacts" / "prediction_outputs.csv"
SERVING_ARTIFACT = PROJECT_ROOT / "backend" / "artifacts" / "pipeline_v3_real.joblib"
OUT_JSON = PROJECT_ROOT / "reports" / "t_base_selection.json"

# The serving artifact (pipeline_v3_real) is a calibrated LogisticRegression,
# so the LogReg OOF column is the closest analogue of the score the decision
# engine actually thresholds. Other columns are reported as distribution
# context only.
PRIMARY_PROB_COLUMN = "prob_Logistic Regression"

SEED = 42
COST_RATIOS = [1.0, 1.7, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]
TARGET_APPROVAL_RATE = 0.85  # operating choice for the fixed-rate method, not a data estimate
N_BOOTSTRAP = 500


def _percentile_table(p: np.ndarray) -> dict:
    qs = [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99]
    return {f"p{q}": round(float(np.percentile(p, q)), 4) for q in qs}


def _legacy_f1_curve(y_default: np.ndarray, p_ml: np.ndarray) -> list[dict]:
    """Tabulate the old sweep so the flatness is on the record."""
    rows = []
    for t in np.arange(0.30, 0.70, 0.05):
        reject = p_ml < t
        tp = np.sum(reject & (y_default == 1))
        fp = np.sum(reject & (y_default == 0))
        fn = np.sum(~reject & (y_default == 1))
        denom = 2 * tp + fp + fn
        rows.append(
            {
                "t": round(float(t), 2),
                "f1": round(float(2 * tp / denom) if denom else 0.0, 5),
                "fraction_of_p_ml_below_t": round(float(reject.mean()), 5),
            }
        )
    return rows


def _bootstrap_cost_delta(
    p: np.ndarray, y: np.ndarray, t_a: float, t_b: float, n: int = N_BOOTSTRAP
) -> dict:
    """95% bootstrap CI for cost(t_a) - cost(t_b) on the report half.

    Used to say honestly whether the cost method's improvement over the
    legacy threshold is real or inside noise.
    """
    rng = np.random.RandomState(SEED)
    deltas = np.empty(n)
    idx_all = np.arange(len(p))
    for i in range(n):
        idx = rng.choice(idx_all, size=len(p), replace=True)
        deltas[i] = expected_cost_per_applicant(
            t_a, p[idx], y[idx], DEFAULT_LGD, DEFAULT_ROI
        ) - expected_cost_per_applicant(t_b, p[idx], y[idx], DEFAULT_LGD, DEFAULT_ROI)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean_delta": round(float(deltas.mean()), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "n_bootstrap": n,
        "significant": bool(lo > 0 or hi < 0),
    }


def run_method_comparison(p_ml: np.ndarray, y_default: np.ndarray) -> dict:
    """SELECT on one stratified half, REPORT on the other. Returns the block
    that lands under `methods` in the JSON."""
    p_sel, p_rep, y_sel, y_rep = train_test_split(
        p_ml, y_default, test_size=0.5, random_state=SEED, stratify=y_default
    )
    out = {
        "split": (
            "50/50 stratified split, random_state=42. Thresholds are SELECTED on "
            "the select half only; every metric below is computed on the REPORT "
            "half the selection never saw."
        ),
        "n_select": int(len(p_sel)),
        "n_report": int(len(p_rep)),
        "methods": {},
    }
    for method in ("f1_legacy", "cost", "youden", "f1", "approval_rate"):
        sel = select_t_base(
            y_sel, p_sel, method=method, target_approval_rate=TARGET_APPROVAL_RATE
        )
        rep = confusion_metrics(sel["t_base"], p_rep, y_rep)
        out["methods"][method] = {"selection": sel, "report_half_metrics": rep}

    # Honesty check: is cost-optimal actually cheaper than the legacy pick,
    # beyond resampling noise, on the report half?
    t_leg = out["methods"]["f1_legacy"]["selection"]["t_base"]
    t_cost = out["methods"]["cost"]["selection"]["t_base"]
    out["cost_legacy_minus_cost_optimal_bootstrap"] = _bootstrap_cost_delta(
        p_rep, y_rep, t_leg, t_cost
    )

    # Sensitivity: hold LGD at 0.45, vary the cost ratio LGD/ROI. The chosen
    # threshold moves a lot — this table is the real deliverable of the cost
    # method, not the single headline number.
    sens = []
    for ratio in COST_RATIOS:
        roi = DEFAULT_LGD / ratio
        s = select_t_base(y_sel, p_sel, method="cost", lgd=DEFAULT_LGD, roi=roi)
        m = confusion_metrics(s["t_base"], p_rep, y_rep, lgd=DEFAULT_LGD, roi=roi)
        sens.append(
            {
                "cost_ratio_lgd_over_roi": ratio,
                "lgd": DEFAULT_LGD,
                "roi": round(roi, 4),
                "t_base": s["t_base"],
                "approval_rate": m["approval_rate"],
                "default_rate_among_approved": m["default_rate_among_approved"],
                "expected_cost_per_applicant": m["expected_cost_per_applicant"],
                "engine_will_clip": s["engine_will_clip"],
            }
        )
    out["cost_sensitivity"] = sens
    return out


def run_serving_model_check() -> dict:
    """Repeat select/report with p_ml from the ACTUAL serving artifact on its
    own held-out split. Heavy (rebuilds the 307k-row frame), hence opt-in."""
    import joblib

    from backend.retrain_serving_model_v3 import build_training_frame
    from backend.app.services import customer_profile_service as cps

    csv_path = cps._resolve_source_path()
    if csv_path is None or not SERVING_ARTIFACT.exists():
        return {"skipped": "Home Credit extract or pipeline_v3_real.joblib not found"}

    X, y = build_training_frame(csv_path)
    # Identical split to retrain_serving_model_v3.main(): the artifact's
    # calibrator was fitted on X_train, so only X_test gives honest probs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    payload = joblib.load(SERVING_ARTIFACT)
    p_default = payload["calibrator"].predict_proba(X_test)[:, 1]
    p_ml = 1.0 - p_default
    block = run_method_comparison(p_ml, np.asarray(y_test).astype(int))
    block["artifact"] = str(SERVING_ARTIFACT.relative_to(PROJECT_ROOT)).replace("\\", "/")
    block["artifact_current_t_base"] = float(payload.get("t_base", float("nan")))
    block["p_ml_percentiles"] = _percentile_table(p_ml)
    block["note"] = (
        "p_ml from the live serving calibrator on the artifact's own 20% held-out "
        "split (same seed/stratify as the retrain script), then select/report "
        "halved within that split."
    )
    return block


def main(with_serving: bool = False) -> dict:
    df = pd.read_csv(OOF_CSV)
    # y_true = 1 means APPROVE (repaid); methods work in default space.
    y_default = (1 - df["y_true"].to_numpy()).astype(int)
    p_ml = df[PRIMARY_PROB_COLUMN].to_numpy(dtype=float)

    report: dict = {
        "generated_by": "python -m research.thresholds.t_base",
        "data": {
            "source": "backend/artifacts/prediction_outputs.csv (out-of-fold predictions)",
            "n_rows": int(len(df)),
            "default_rate": round(float(y_default.mean()), 6),
            "primary_prob_column": PRIMARY_PROB_COLUMN,
            "primary_column_reason": (
                "The serving artifact (pipeline_v3_real.joblib) is a calibrated "
                "LogisticRegression; this column is the closest OOF analogue of "
                "the p_ml the decision engine thresholds."
            ),
            "p_ml_percentiles_by_column": {
                c: _percentile_table(df[c].to_numpy(dtype=float))
                for c in df.columns
                if c.startswith("prob_") or c in ("best_model_prob", "cbes_prob")
            },
        },
        "cost_model": {
            "unit": "expected cost per applicant, EAD normalised to 1 (no loan amounts in the OOF file)",
            "false_approval_cost": {
                "value": DEFAULT_LGD,
                "meaning": "LGD x EAD for an approved defaulter",
                "source": (
                    "Basel II foundation-IRB supervisory LGD, senior unsecured "
                    "exposures — BCBS 128 (June 2006), para 287"
                ),
            },
            "false_rejection_cost": {
                "value": DEFAULT_ROI,
                "meaning": "foregone return on a rejected good customer",
                "source": (
                    "Verbraken, Bravo, Weber & Baesens (2014), 'Development and "
                    "application of consumer credit scoring models using "
                    "profit-based classification measures', EJOR 238(2) — the EMP "
                    "measure's ROI parameter"
                ),
            },
            "bayes_optimal_threshold_if_calibrated": round(
                1 - DEFAULT_ROI / (DEFAULT_ROI + DEFAULT_LGD), 4
            ),
            "caveat": (
                "The cost-optimal threshold is only as defensible as these two "
                "numbers. They are literature defaults, not SmartLend portfolio "
                "estimates; see cost_sensitivity for how the answer moves with "
                "the assumed ratio."
            ),
        },
        "engine_clip_range": [ENGINE_CLIP_LO, ENGINE_CLIP_HI],
    }

    # ── Defect evidence: the legacy sweep is flat and edge-pinned ───────────
    legacy_curve = _legacy_f1_curve(y_default, p_ml)
    report["defect_evidence"] = {
        "claim": (
            "The old fixed-range F1 sweep [0.30, 0.70) is degenerate: p_ml mass "
            "below 0.70 is ~2%, F1 is near zero and monotone increasing across "
            "the whole window, and the argmax sits at the top edge of the sweep "
            "— i.e. t_base was set by the range bounds, not the data."
        ),
        "f1_curve_full_data": legacy_curve,
        "fraction_of_p_ml_below_0.70": round(float((p_ml < 0.70).mean()), 5),
        "v3_artifact_reported": {"t_base": 0.65, "f1_at_selection": 0.0024},
    }

    report["oof_analysis"] = run_method_comparison(p_ml, y_default)

    if with_serving:
        report["serving_model_v3_check"] = run_serving_model_check()

    report["recommendation"] = {
        "method": "cost",
        "reasons": [
            "It optimises the quantity a lender actually loses money on, with an "
            "explicit, citable loss matrix — the criterion credit-scoring "
            "reviewers expect.",
            "With literature-default costs (LGD 0.45 / ROI 0.2644) the empirical "
            "optimum sits near the analytic Bayes threshold 0.63 for a "
            "calibrated score, which is independent corroboration that the "
            "method is reading the calibration correctly.",
            "It is the only y-aware method whose optimum lies inside "
            "decision_engine's hard clip range [0.30, 0.75]; Youden and "
            "percentile-F1 optima (~0.86-0.93) would be silently clamped to "
            "0.75 at decision time.",
            "The percentile grid removes the structural defect: candidates "
            "always cover the observed score mass, whatever the base rate.",
        ],
        "honesty": [
            "On the report half the cost saving vs the legacy 0.69 threshold is "
            "small; see cost_legacy_minus_cost_optimal_bootstrap for whether it "
            "clears resampling noise. The main win is that the threshold is now "
            "DERIVED from stated economics instead of being an artifact of "
            "arbitrary sweep bounds.",
            "The recommended value moves materially with the LGD/ROI ratio "
            "(cost_sensitivity); revisit the parameters once SmartLend has "
            "portfolio estimates.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[t_base study] wrote {OUT_JSON}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--serving",
        action="store_true",
        help="also evaluate against the live pipeline_v3_real.joblib (slow; "
        "needs the Home Credit extract on disk)",
    )
    main(with_serving=ap.parse_args().serving)
