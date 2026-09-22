"""Complementarity analysis: would an XGBoost + TabPFN-2.5 (or XGBoost + CBES,
or XGBoost + any other id-indexed scorer) hybrid measurably beat XGBoost alone?

Answers with measurements, never assertions.  Two data sources:

  * backend/artifacts/prediction_outputs.csv — 307,511 rows of OUT-OF-FOLD
    5-fold-CV probabilities for XGBoost, LightGBM, CatBoost, Logistic
    Regression, Random Forest and CBES.  All six-model analyses (correlations,
    error overlap, segments, hybrid suites) run on these rows.
  * reports/<model>_scored_rows.csv — per-row scores for foundation-model
    baselines, saved WITH `SK_ID_CURR` so they join exactly
    (reports/tabpfn_scored_rows.csv from commit 12bbf4f; a TabFM file plugs in
    the same way).

LABEL CONVENTION USED THROUGHOUT THIS SCRIPT: y = 1 means DEFAULT and every
probability is P(default).  prediction_outputs.csv stores the opposite
(y_true = 1 means approve, prob_* = P(approve)), so those columns are flipped
(p -> 1 - p, y -> 1 - y) immediately on load.

ALIGNMENT GATE: before any foundation-model comparison is reported, the scored
rows are joined on SK_ID_CURR and the join is verified two ways: the file's
own labels must agree with the dataset's TARGET on every joined row, and the
model's AUC recomputed on the joined rows must reproduce its known reference
value.  If the gate fails, that model's analyses are SKIPPED and the failure
is recorded in the output JSON instead — a misaligned comparison would invert
or destroy every conclusion.

HISTORY: the original TabPFN artifact (reports/_tabpfn_probs_5000.npy) carried
NO row ids.  The gate failed on 2026-08-31 — AUC 0.5052 (chance) on every one
of 30 attempted reconstructions — so all TabPFN analyses were skipped and the
question was left explicitly unanswered rather than answered with garbage.
Commit 12bbf4f re-scored 8,000 rows and saved SK_ID_CURR alongside every
probability (reports/tabpfn_scored_rows.csv), which is what this script now
consumes; the .npy is retired.

Run:  python -m research.analysis.complementarity
Outputs: reports/complementarity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
PRED_CSV = REPO / "backend" / "artifacts" / "prediction_outputs.csv"
TABPFN_CSV = REPO / "reports" / "tabpfn_scored_rows.csv"
TABFM_CSV = REPO / "reports" / "tabfm_scored_rows.csv"
OUT_JSON = REPO / "reports" / "complementarity.json"


def _merged_csv_path() -> Path | None:
    """The Home Credit extract with segmentation features (not in the repo).

    Resolved the same way the serving layer does (env var first), with the
    original hardcoded location as a fallback for this machine.
    """
    from backend.app.services.customer_profile_service import _resolve_source_path

    resolved = _resolve_source_path()
    if resolved is not None:
        return Path(resolved)
    legacy = Path(r"C:\Users\shrey\Downloads\creddefer_full_merged.csv")
    return legacy if legacy.exists() else None


# Reference AUCs for the alignment gates: the model's AUC recomputed on the
# joined rows must reproduce the value its own scoring run reported.
# TabPFN: reports/tabpfn_quick_run.json (8,000 rows, RTX 4060 re-score).
TABPFN_REF_AUC = 0.7284
TABPFN_REF_TOL = 0.002

# Fold-to-fold standard deviation of XGBoost's CV AUC (given): a hybrid "gain"
# smaller than this is inside training noise and is not a gain.
FOLD_STD = 0.0036

RNG_SEED = 42
N_BOOT = 1000

MODELS = ["XGBoost", "LightGBM", "CatBoost", "Logistic Regression",
          "Random Forest", "CBES"]


# --------------------------------------------------------------------------
# Metric helpers (unit-tested in research/tests/test_complementarity.py)
# --------------------------------------------------------------------------

def rank_average(*prob_arrays: np.ndarray) -> np.ndarray:
    """Average of rank-transformed probabilities, rescaled to (0, 1].

    Rank transform removes calibration differences between models so each
    contributes equally regardless of the scale of its probabilities.
    """
    n = len(prob_arrays[0])
    ranks = [rankdata(p) / n for p in prob_arrays]
    return np.mean(ranks, axis=0)


def weight_sweep(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                 step: float = 0.05) -> tuple[float, float, list[dict]]:
    """AUC of w*p_a + (1-w)*p_b for w in 0..1.

    Returns (best_w, best_auc, full_curve).  NOTE: the best w is selected on
    the same data it is evaluated on, so best_auc is optimistically biased —
    it is an upper bound on what a tuned weighted average could achieve.
    """
    curve = []
    best_w, best_auc = 0.0, -1.0
    for w in np.round(np.arange(0.0, 1.0 + 1e-9, step), 2):
        auc = float(roc_auc_score(y, w * p_a + (1 - w) * p_b))
        curve.append({"w_a": float(w), "auc": round(auc, 5)})
        if auc > best_auc:
            best_w, best_auc = float(w), auc
    return best_w, best_auc, curve


def cv_stack_auc(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                 n_splits: int = 5, seed: int = RNG_SEED) -> float:
    """Logistic stacking of two probability vectors, evaluated honestly.

    The meta-learner (logistic regression on the two logit-transformed
    probabilities) is fit inside a stratified K-fold loop; each row's stacked
    prediction comes from a fold whose training part did NOT contain that row,
    so the meta-learner is never scored on data it was fit on.
    Returns AUC of the out-of-fold stacked predictions.
    """
    eps = 1e-6
    X = np.column_stack([
        np.log(np.clip(p_a, eps, 1 - eps) / np.clip(1 - p_a, eps, 1 - eps)),
        np.log(np.clip(p_b, eps, 1 - eps) / np.clip(1 - p_b, eps, 1 - eps)),
    ])
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def paired_bootstrap_auc_delta(y: np.ndarray, p_new: np.ndarray,
                               p_base: np.ndarray, n_boot: int = N_BOOT,
                               seed: int = RNG_SEED) -> dict:
    """Bootstrap distribution of AUC(p_new) - AUC(p_base) computed on the SAME
    resampled rows (paired, so shared sampling noise cancels).  Returns the
    mean delta, 95% CI, and the fraction of resamples where delta > 0."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_new[idx])
                      - roc_auc_score(y[idx], p_base[idx]))
    d = np.array(deltas)
    return {
        "delta_mean": round(float(d.mean()), 5),
        "delta_ci95": [round(float(np.percentile(d, 2.5)), 5),
                       round(float(np.percentile(d, 97.5)), 5)],
        "frac_positive": round(float((d > 0).mean()), 4),
        "n_boot": int(len(d)),
    }


def error_confusion(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Confusion-of-errors table at a classification threshold.

    'Right' means the thresholded prediction (p >= threshold -> class 1)
    matches y.  Returns counts and fractions for both-right / only-A-right /
    only-B-right / both-wrong.
    """
    a_right = ((p_a >= threshold).astype(int) == y)
    b_right = ((p_b >= threshold).astype(int) == y)
    n = len(y)
    counts = {
        "both_right": int(np.sum(a_right & b_right)),
        "only_a_right": int(np.sum(a_right & ~b_right)),
        "only_b_right": int(np.sum(~a_right & b_right)),
        "both_wrong": int(np.sum(~a_right & ~b_right)),
    }
    return {"threshold": threshold, "n": n, "counts": counts,
            "fractions": {k: round(v / n, 4) for k, v in counts.items()}}


# --------------------------------------------------------------------------
# Scored-rows alignment gate (TabPFN, TabFM, any id-indexed scorer)
# --------------------------------------------------------------------------

def align_scored_rows(csv_path: Path, model_label: str, features: pd.DataFrame,
                      ref_auc: float | None = None,
                      ref_tol: float = 0.002) -> dict:
    """Join an id-indexed scored-rows CSV onto the OOF frame and VERIFY it.

    The CSV must carry `SK_ID_CURR`, `p_default` and a label column
    (`y_target`, 1 = defaulted).  Verification, in order of strength:

      1. every SK_ID_CURR must exist in the OOF frame (an id that doesn't is a
         wrong-dataset file, not sampling);
      2. the file's own labels must equal the dataset's TARGET on every joined
         row — the single strongest misalignment detector;
      3. when a reference AUC is given, the model's AUC recomputed on the
         joined rows must reproduce it within tolerance.

    Returns a status dict; on success it also carries the aligned frame under
    '_frame' (stripped before JSON output) with a `pd_{model_label}` column.
    """
    if not csv_path.exists():
        return {"status": "skipped",
                "reason": f"{csv_path.name} missing — score {model_label} with "
                          "row ids first"}

    scored = pd.read_csv(csv_path)
    required = {"SK_ID_CURR", "p_default", "y_target"}
    if not required.issubset(scored.columns):
        return {"status": "ALIGNMENT_FAILED",
                "reason": f"{csv_path.name} lacks columns {sorted(required - set(scored.columns))}"}

    sub = features.merge(
        scored[["SK_ID_CURR", "p_default", "y_target"]],
        on="SK_ID_CURR", how="inner", validate="1:1",
    ).rename(columns={"p_default": f"pd_{model_label}"})

    if len(sub) != len(scored):
        return {"status": "ALIGNMENT_FAILED",
                "scored_rows": int(len(scored)), "joined_rows": int(len(sub)),
                "reason": "some scored SK_ID_CURR ids do not exist in the OOF frame"}

    label_mismatches = int((sub["y_target"].astype(int) != sub["TARGET"].astype(int)).sum())
    auc = float(roc_auc_score(sub["TARGET"], sub[f"pd_{model_label}"]))
    corr_xgb = float(np.corrcoef(sub[f"pd_{model_label}"], sub["pd_XGBoost"])[0, 1])

    report = {
        "source": csv_path.name,
        "scored_rows": int(len(scored)),
        "label_mismatches_vs_dataset": label_mismatches,
        "auc_on_joined_rows": round(auc, 4),
        "expected_auc": ref_auc,
        "corr_with_xgb_pd": round(corr_xgb, 4),
    }
    if label_mismatches:
        return {"status": "ALIGNMENT_FAILED", **report,
                "reason": "file labels disagree with dataset TARGET on joined rows"}
    if ref_auc is not None and abs(auc - ref_auc) > ref_tol:
        return {"status": "ALIGNMENT_FAILED", **report,
                "reason": "AUC on joined rows does not reproduce the reference"}
    return {"status": "verified", **report, "_frame": sub}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_oof_with_features() -> pd.DataFrame:
    """All 307,511 out-of-fold rows in P(default) space, joined with the
    segmentation features from the merged Home Credit CSV."""
    preds = pd.read_csv(PRED_CSV)
    preds["SK_ID_CURR"] = (preds["applicant_id"].str.removeprefix("HC")
                           .astype(int))
    # Flip approval probabilities / labels into default space.
    preds["y"] = 1 - preds["y_true"]
    for m in MODELS:
        preds[f"pd_{m}"] = 1.0 - preds[f"prob_{m}"]

    merged_csv = _merged_csv_path()
    if merged_csv is None:
        raise SystemExit("Home Credit extract not found; set SMARTLEND_CUSTOMER_DATA.")
    feats = pd.read_csv(merged_csv, usecols=[
        "SK_ID_CURR", "TARGET", "EXT_SOURCE_2", "AMT_INCOME_TOTAL",
        "DAYS_BIRTH", "total_prev_credits"])
    df = preds.merge(feats, on="SK_ID_CURR", how="inner", validate="1:1")
    assert len(df) == len(preds), "feature join lost rows"
    # Label consistency gate: y_true (approve) must complement TARGET (default)
    assert (df["y"] == df["TARGET"]).all(), \
        "y_true/TARGET mismatch — label conventions are misaligned"
    return df


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def analysis_correlations(df: pd.DataFrame, models: list[str]) -> dict:
    """Pairwise Pearson/Spearman of P(default), plus Pearson of absolute
    errors |y - p|.  Error correlation is the quantity that limits ensemble
    gains: averaging only helps to the extent errors are uncorrelated."""
    y = df["y"].to_numpy()
    P = {m: df[f"pd_{m}"].to_numpy() for m in models}
    E = {m: np.abs(y - P[m]) for m in models}
    out = {"pearson_probs": {}, "spearman_probs": {}, "pearson_errors": {}}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            key = f"{a}|{b}"
            out["pearson_probs"][key] = round(float(np.corrcoef(P[a], P[b])[0, 1]), 4)
            out["spearman_probs"][key] = round(float(spearmanr(P[a], P[b]).statistic), 4)
            out["pearson_errors"][key] = round(float(np.corrcoef(E[a], E[b])[0, 1]), 4)
    # Disagreement + error overlap for the pair the architecture is built on.
    appr_xgb = 1 - P["XGBoost"]
    appr_cbes = 1 - P["CBES"]
    y_appr = 1 - y
    out["disagreement_rate_xgb_cbes_at_0.5"] = round(
        float(np.mean((appr_xgb >= 0.5) != (appr_cbes >= 0.5))), 4)
    out["error_confusion_xgb_cbes"] = error_confusion(y_appr, appr_xgb, appr_cbes)
    out["note"] = ("error_confusion is in approval space at threshold 0.5; "
                   "with an 8.07% default rate a model that approves everyone "
                   "is 'right' 91.9% of the time at this threshold, so read "
                   "the overlap pattern, not the accuracy level.")
    return out


def _segment_auc(df: pd.DataFrame, mask: np.ndarray, label: str,
                 model_a: str, model_b: str) -> dict:
    seg = df[mask]
    y = seg["y"].to_numpy()
    row = {"segment": label, "n": int(len(seg)), "n_default": int(y.sum())}
    if len(seg) == 0 or y.sum() in (0, len(y)):
        return row
    pa, pb = seg[f"pd_{model_a}"].to_numpy(), seg[f"pd_{model_b}"].to_numpy()
    row[f"auc_{model_a}"] = round(float(roc_auc_score(y, pa)), 4)
    row[f"auc_{model_b}"] = round(float(roc_auc_score(y, pb)), 4)
    delta = row[f"auc_{model_b}"] - row[f"auc_{model_a}"]
    row[f"{model_b}_minus_{model_a}"] = round(delta, 4)
    if delta > 0:
        # Only where B appears to win: is the win real? Paired bootstrap CI.
        row["bootstrap_delta"] = paired_bootstrap_auc_delta(y, pb, pa)
    return row


def analysis_segments(df: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Per-segment AUC of model_a vs model_b across the four segmentations."""
    out = {"models": [model_a, model_b]}
    q = pd.qcut(df["EXT_SOURCE_2"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    rows = [_segment_auc(df, (q == lab).to_numpy(), str(lab), model_a, model_b)
            for lab in ["Q1_low", "Q2", "Q3", "Q4_high"]]
    if df["EXT_SOURCE_2"].isna().any():
        rows.append(_segment_auc(df, df["EXT_SOURCE_2"].isna().to_numpy(),
                                 "missing", model_a, model_b))
    out["ext_source_2_quartile"] = rows

    thin = df["total_prev_credits"].isna().to_numpy()
    out["bureau_file"] = [
        _segment_auc(df, thin, "thin_file_no_bureau", model_a, model_b),
        _segment_auc(df, ~thin, "has_bureau_record", model_a, model_b)]

    qi = pd.qcut(df["AMT_INCOME_TOTAL"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    out["income_quartile"] = [
        _segment_auc(df, (qi == lab).to_numpy(), str(lab), model_a, model_b)
        for lab in ["Q1_low", "Q2", "Q3", "Q4_high"]]

    age = (-df["DAYS_BIRTH"] / 365.25)
    bands = [("<30", age < 30), ("30-40", (age >= 30) & (age < 40)),
             ("40-50", (age >= 40) & (age < 50)),
             ("50-60", (age >= 50) & (age < 60)), ("60+", age >= 60)]
    out["age_band"] = [_segment_auc(df, m.to_numpy(), lab, model_a, model_b)
                       for lab, m in bands]
    return out


def analysis_hybrid(y: np.ndarray, p_xgb: np.ndarray, p_other: np.ndarray,
                    other_name: str, n_rows_note: str) -> dict:
    """Full hybrid suite for XGBoost + one other model, in P(default) space."""
    auc_xgb = float(roc_auc_score(y, p_xgb))
    auc_other = float(roc_auc_score(y, p_other))
    p_simple = (p_xgb + p_other) / 2
    simple = float(roc_auc_score(y, p_simple))
    best_w, best_auc, curve = weight_sweep(y, p_xgb, p_other)
    rank_auc = float(roc_auc_score(y, rank_average(p_xgb, p_other)))
    stack_auc = cv_stack_auc(y, p_xgb, p_other)

    candidates = {"simple_average": (simple, p_simple),
                  "rank_average": (rank_auc, rank_average(p_xgb, p_other))}
    best_name = max(candidates, key=lambda k: candidates[k][0])
    if stack_auc > candidates[best_name][0]:
        best_name, best_probs, best_val = "cv_logistic_stack", None, stack_auc
    else:
        best_val, best_probs = candidates[best_name]

    out = {
        "pair": f"XGBoost + {other_name}",
        "evaluation_rows": n_rows_note,
        "auc_xgboost_alone": round(auc_xgb, 4),
        f"auc_{other_name.lower().replace(' ', '_')}_alone": round(auc_other, 4),
        "auc_simple_average": round(simple, 4),
        "auc_weighted_best": {
            "w_xgb": best_w, "auc": round(best_auc, 4),
            "note": "weight chosen on the evaluation data itself — an "
                    "optimistic upper bound, not an honest estimate"},
        "weight_sweep_curve": curve,
        "auc_rank_average": round(rank_auc, 4),
        "auc_cv_logistic_stack": round(stack_auc, 4),
        "best_honest_hybrid": {
            "method": best_name,
            "auc": round(best_val, 4),
            "gain_vs_xgb": round(best_val - auc_xgb, 4),
            "fold_std_reference": FOLD_STD,
            "gain_exceeds_fold_std": bool(best_val - auc_xgb > FOLD_STD),
        },
        # Paired bootstrap cancels shared sampling noise, so this CI speaks to
        # whether the ordering hybrid-vs-XGB is stable, not to fold noise.
        "paired_bootstrap_simple_avg_vs_xgb":
            paired_bootstrap_auc_delta(y, p_simple, p_xgb),
    }
    if best_probs is not None:
        out["best_honest_hybrid"]["paired_bootstrap_vs_xgb"] = \
            paired_bootstrap_auc_delta(y, best_probs, p_xgb)
    return out


def _tabfm_ref_auc() -> float | None:
    """TabFM's own reported AUC, read from its run report when it exists."""
    report = REPO / "reports" / "tabfm_quick_run.json"
    if not report.exists():
        return None
    try:
        return float(json.loads(report.read_text())["metrics"]["roc_auc"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def main() -> None:
    df = load_oof_with_features()
    y = df["y"].to_numpy()
    p_xgb = df["pd_XGBoost"].to_numpy()

    results = {
        "meta": {
            "oof_rows": int(len(df)),
            "n_defaults": int(y.sum()),
            "default_rate": round(float(y.mean()), 4),
            "label_convention": "y=1 default; all probabilities are P(default)",
            "fold_to_fold_auc_std": FOLD_STD,
            "gain_criterion": ("a hybrid 'gain' must exceed the fold-to-fold "
                               "AUC std (0.0036) to count as a gain at all"),
        },
    }

    # ---- Foundation models: alignment gate first, then the same suite -----
    # TabPFN's ~807-default sample carries AUC sampling error ~±0.01, so only
    # the PAIRED statistics inside analysis_hybrid/_segment_auc can resolve
    # gains near the noise floor — stated here so nobody quotes raw deltas.
    foundation_sources = [
        ("TabPFN", TABPFN_CSV, TABPFN_REF_AUC, TABPFN_REF_TOL),
        # TabFM plugs in the moment reports/tabfm_scored_rows.csv exists; its
        # reference AUC is read from its own run report when present.
        ("TabFM", TABFM_CSV, _tabfm_ref_auc(), 0.002),
    ]
    for label, csv_path, ref_auc, ref_tol in foundation_sources:
        gate = align_scored_rows(csv_path, label, df, ref_auc=ref_auc, ref_tol=ref_tol)
        frame = gate.pop("_frame", None)
        results[f"{label.lower()}_alignment"] = gate
        if frame is None:
            print(f"{label}: alignment gate {gate.get('status')} — analyses skipped "
                  f"({gate.get('reason', '')})")
            continue
        sub = frame
        n_note = f"{len(sub):,} {label}-scored rows (id-joined, labels verified)"
        results[f"segments_xgb_vs_{label.lower()}"] = analysis_segments(
            sub.assign(y=sub["TARGET"]), "XGBoost", label)
        results[f"hybrid_xgb_{label.lower()}"] = analysis_hybrid(
            sub["TARGET"].to_numpy(), sub["pd_XGBoost"].to_numpy(),
            sub[f"pd_{label}"].to_numpy(), label, n_note)
        results[f"correlations_{label.lower()}_rows"] = analysis_correlations(
            sub.assign(y=sub["TARGET"]), MODELS + [label])
        print(f"{label} analyses completed on {len(sub):,} verified rows.")

    # ---- Six OOF models: correlations, errors, segments, hybrids ---------
    results["correlations_full_oof"] = analysis_correlations(df, MODELS)
    results["segments_xgb_vs_cbes_full_oof"] = analysis_segments(
        df, "XGBoost", "CBES")
    results["hybrid_xgb_cbes_full_oof"] = analysis_hybrid(
        y, p_xgb, df["pd_CBES"].to_numpy(), "CBES", "all 307,511 OOF rows")
    # Context: does ensembling XGBoost with its strongest peers help either?
    # This is what the numbers CAN say about hybrids while TabPFN is unusable.
    results["hybrid_xgb_lightgbm_full_oof"] = analysis_hybrid(
        y, p_xgb, df["pd_LightGBM"].to_numpy(), "LightGBM",
        "all 307,511 OOF rows")
    results["hybrid_xgb_catboost_full_oof"] = analysis_hybrid(
        y, p_xgb, df["pd_CatBoost"].to_numpy(), "CatBoost",
        "all 307,511 OOF rows")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
