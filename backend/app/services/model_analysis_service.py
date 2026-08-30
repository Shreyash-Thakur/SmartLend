from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import logging
import math

from backend.app.services.ml_service import dynamic_hybrid_decision


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_METRICS_PATH = ARTIFACTS_DIR / "model_metrics.csv"
PREDICTION_OUTPUTS_PATH = ARTIFACTS_DIR / "prediction_outputs.csv"
PIPELINE_SUMMARY_PATH = ARTIFACTS_DIR / "pipeline_summary.json"

logger = logging.getLogger(__name__)

# Errors that reading a possibly-missing / partially-rewritten CSV can raise.
_READ_ERRORS = (OSError, csv.Error, UnicodeDecodeError, ValueError)


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce to a JSON-safe float. NaN/Inf fall back to ``default``.

    ``float("nan")`` parses fine but serialises to invalid JSON, so it is
    treated as a missing value rather than propagated into the payload.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a CSV into (fieldnames, rows), degrading to ([], []) on any problem.

    Tolerates a missing file, an empty file, a header-only file, and a file
    being rewritten underneath us (truncated/undecodable content).
    """
    if not path.exists():
        return [], []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [name for name in (reader.fieldnames or []) if name]
            if not fieldnames:
                return [], []
            rows = [row for row in reader]
    except _READ_ERRORS:
        logger.warning("Could not read artifact CSV %s; serving empty data", path, exc_info=True)
        return [], []
    return fieldnames, rows


def _to_int(value: Any, default: int = 0) -> int:
    try:
        # OverflowError guards against inf; ValueError against NaN and junk text.
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _is_cbes_baseline(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized in {"cbes", "cbes baseline"}


def _compute_metrics_from_predictions() -> dict[str, dict[str, float]]:
    """Compute per-model accuracy/precision/recall/auc/f1 from prediction_outputs.csv.
    Called when model_metrics.csv is missing those columns (retrain_pipeline_v2 path).
    """
    results: dict[str, dict[str, float]] = {}
    fieldnames, rows = _read_csv(PREDICTION_OUTPUTS_PATH)
    prob_cols = [c for c in fieldnames if c.startswith("prob_") and c != "prob_CBES"]
    if not rows or not prob_cols:
        return {}
    y_true = [_to_int(r.get("y_true")) for r in rows]
    for col in prob_cols:
        model = col.replace("prob_", "")
        probs = [_to_float(r.get(col)) for r in rows]
        preds = [1 if p >= 0.5 else 0 for p in probs]
        tp = sum(1 for yt, pd in zip(y_true, preds) if pd == 1 and yt == 1)
        fp = sum(1 for yt, pd in zip(y_true, preds) if pd == 1 and yt == 0)
        fn = sum(1 for yt, pd in zip(y_true, preds) if pd == 0 and yt == 1)
        tn = sum(1 for yt, pd in zip(y_true, preds) if pd == 0 and yt == 0)
        total = len(y_true)
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        # AUC approximation using existing roc_auc column or 0
        # Use rank-based AUC: sorted probs against y_true
        n_pos = sum(y_true)
        n_neg = total - n_pos
        if n_pos and n_neg:
            paired = sorted(zip(probs, y_true), key=lambda x: x[0])
            rank_sum = sum(i + 1 for i, (_, yt) in enumerate(paired) if yt == 1)
            auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        else:
            auc = 0.0
        entry = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
        results[model] = entry
        # Also index by normalised name (no spaces, lowercase) to bridge CSV naming mismatches
        results[model.replace(" ", "").lower()] = entry
    return results


def _load_model_metrics() -> list[dict[str, Any]]:
    _fieldnames, metric_rows = _read_csv(MODEL_METRICS_PATH)
    if not metric_rows:
        return []

    # Pre-compute per-model metrics from predictions to fill gaps
    computed = _compute_metrics_from_predictions()

    items: list[dict[str, Any]] = []
    for row in metric_rows:
        # Support both Title-Case (training.py) and lowercase (retrain_pipeline_v2.py) schemas
        model_name = str(row.get("Model") or row.get("model") or "").strip()
        if not model_name:
            # A row with no identifiable model is unusable downstream.
            continue
        # Try CSV columns first; fall back to computed values from prediction_outputs
        # Try exact name, then normalised name (handles 'LogisticRegression' vs 'Logistic Regression')
        calc = computed.get(model_name) or computed.get(model_name.replace(" ", "").lower(), {})
        accuracy  = _to_float(row.get("Accuracy") or row.get("accuracy")) or calc.get("accuracy", 0.0)
        precision = _to_float(row.get("Precision") or row.get("precision")) or calc.get("precision", 0.0)
        recall    = _to_float(row.get("Recall") or row.get("recall")) or calc.get("recall", 0.0)
        auc       = _to_float(row.get("AUC") or row.get("roc_auc")) or calc.get("auc", 0.0)
        f1        = _to_float(row.get("F1") or row.get("f1")) or calc.get("f1", 0.0)
        if not f1:
            denom = precision + recall
            f1 = (2 * precision * recall / denom) if denom else 0.0
        items.append(
            {
                "model": model_name,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "auc": round(auc, 4),
                "f1": round(f1, 4),
                "tuned": True,
            }
        )

    items.sort(key=lambda item: item.get("auc", 0.0), reverse=True)
    for idx, item in enumerate(items, start=1):
        item["rank"] = idx

    return items


def _load_pipeline_summary() -> dict[str, Any]:
    if not PIPELINE_SUMMARY_PATH.exists():
        return {}

    try:
        with PIPELINE_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        # ValueError covers JSONDecodeError and UnicodeDecodeError.
        logger.warning("Could not read %s; ignoring pipeline summary", PIPELINE_SUMMARY_PATH, exc_info=True)
        return {}


def _load_prediction_outputs() -> tuple[list[dict[str, Any]], list[str]]:
    fieldnames, raw_rows = _read_csv(PREDICTION_OUTPUTS_PATH)
    if not raw_rows:
        return [], []

    model_names = [
        name.replace("prob_", "")
        for name in fieldnames
        if name.startswith("prob_") and name != "prob_CBES"
    ]

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        y_true = _to_int(row.get("y_true"))
        expected = "APPROVE" if y_true == 1 else "REJECT"
        best_model_prob = _to_float(row.get("best_model_prob"))
        cbes_prob = _to_float(row.get("cbes_prob"))
        try:
            hybrid_decision, hybrid_confidence, approval_threshold, rejection_threshold = dynamic_hybrid_decision(
                best_model_prob,
                cbes_prob,
            )
        except Exception:
            # Never let a single unscorable row take down the whole endpoint.
            logger.warning("dynamic_hybrid_decision failed for a prediction row", exc_info=True)
            hybrid_decision, hybrid_confidence, approval_threshold, rejection_threshold = "DEFER", 0.0, 0.5, 0.5

        stored_decision = str(row.get("final_decision") or "").strip().upper()
        if stored_decision in {"APPROVE", "REJECT", "DEFER"}:
            hybrid_decision = stored_decision

        hybrid_confidence = _to_float(row.get("confidence"), hybrid_confidence)
        approval_threshold = _to_float(row.get("approval_threshold"), approval_threshold)
        rejection_threshold = _to_float(row.get("rejection_threshold"), rejection_threshold)

        model_probabilities = {
            model: _to_float(row.get(f"prob_{model}"))
            for model in model_names
        }
        model_predictions = {
            model: "APPROVE" if prob >= 0.5 else "REJECT"
            for model, prob in model_probabilities.items()
        }

        rows.append(
            {
                "applicantId": str(row.get("applicant_id") or ""),
                "yTrue": y_true,
                "expectedDecision": expected,
                "hybridDecision": hybrid_decision,
                "hybridConfidence": hybrid_confidence,
                "approvalThreshold": approval_threshold,
                "rejectionThreshold": rejection_threshold,
                "cbesProb": cbes_prob,
                "bestModelProb": best_model_prob,
                "modelProbabilities": model_probabilities,
                "modelPredictions": model_predictions,
            }
        )

    return rows, model_names


def _empty_payload() -> dict[str, Any]:
    """Shape-complete payload used whenever artifacts are unusable.

    The router turns this into a 404, so callers never see a 500.
    """
    return {
        "models": [],
        "modelsByProbabilityColumns": [],
        "summary": {
            "totalCases": 0,
            "deferredCases": 0,
            "deferralRate": 0.0,
            "automatedCoverage": 0.0,
            "automatedAccuracy": 0.0,
            "overallHybridAccuracy": 0.0,
            "bestModel": "",
            "selectedAlpha": 0.25,
        },
        "modelPredictionSummary": [],
        "confusionByModel": [],
        "probabilityBands": [],
        "cases": [],
    }


def get_model_analysis_payload(limit: int = 200) -> dict[str, Any]:
    """Build the model-analysis payload, degrading to an empty payload on error.

    Artifacts are written by a separate training process and may be missing,
    empty, mid-rewrite, or missing columns; this must never raise.
    """
    try:
        return _build_model_analysis_payload(limit)
    except Exception:
        logger.exception("Failed to build model analysis payload; serving empty payload")
        return _empty_payload()


def _build_model_analysis_payload(limit: int = 200) -> dict[str, Any]:
    metrics = _load_model_metrics()
    cases, model_names = _load_prediction_outputs()
    pipeline_summary = _load_pipeline_summary()
    display_metrics = [metric for metric in metrics if not _is_cbes_baseline(metric.get("model", ""))]

    if not display_metrics or not cases:
        return _empty_payload()

    total_cases = len(cases)
    deferred_cases = sum(1 for row in cases if row["hybridDecision"] == "DEFER")
    automated_cases = total_cases - deferred_cases

    automated_correct = sum(
        1
        for row in cases
        if row["hybridDecision"] != "DEFER" and row["hybridDecision"] == row["expectedDecision"]
    )
    # Overall accuracy = accuracy among automated (non-deferred) decisions only.
    # Counting DEFER as wrong is misleading — those go to human review by design.
    overall_hybrid_accuracy = round((automated_correct / automated_cases) * 100, 2) if automated_cases else 0.0

    model_prediction_summary: list[dict[str, Any]] = []
    confusion_by_model: list[dict[str, Any]] = []
    for model in model_names:
        approve_count = 0
        correct_count = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for row in cases:
            pred = row["modelPredictions"].get(model, "REJECT")
            if pred == "APPROVE":
                approve_count += 1
            if pred == row["expectedDecision"]:
                correct_count += 1

            expected_is_positive = row["expectedDecision"] == "APPROVE"
            pred_is_positive = pred == "APPROVE"
            if pred_is_positive and expected_is_positive:
                tp += 1
            elif pred_is_positive and not expected_is_positive:
                fp += 1
            elif not pred_is_positive and expected_is_positive:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_cases = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        model_prediction_summary.append(
            {
                "model": model,
                "approveCount": approve_count,
                "rejectCount": total_cases - approve_count,
                "accuracyFromCases": round((correct_count / total_cases) * 100, 2),
            }
        )

        confusion_by_model.append(
            {
                "model": model,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "f1FromCases": round(f1_cases * 100, 2),
            }
        )

    bands = [
        {"label": "0.0-0.2", "low": 0.0, "high": 0.2},
        {"label": "0.2-0.4", "low": 0.2, "high": 0.4},
        {"label": "0.4-0.6", "low": 0.4, "high": 0.6},
        {"label": "0.6-0.8", "low": 0.6, "high": 0.8},
        {"label": "0.8-1.0", "low": 0.8, "high": 1.01},
    ]
    probability_bands: list[dict[str, Any]] = []
    for band in bands:
        bucket = [
            row
            for row in cases
            if band["low"] <= float(row.get("bestModelProb", 0.0)) < band["high"]
        ]
        probability_bands.append(
            {
                "band": band["label"],
                "approve": sum(1 for row in bucket if row["hybridDecision"] == "APPROVE"),
                "reject": sum(1 for row in bucket if row["hybridDecision"] == "REJECT"),
                "defer": sum(1 for row in bucket if row["hybridDecision"] == "DEFER"),
                "total": len(bucket),
            }
        )

    # Pull trusted accuracy from pipeline_summary if available (written during training)
    ps_accuracy = _to_float(pipeline_summary.get("accuracy_non_deferred"))
    automated_accuracy = round(ps_accuracy * 100, 2) if ps_accuracy else (
        round((automated_correct / automated_cases) * 100, 2) if automated_cases else 0.0
    )

    safe_limit = max(1, min(limit, total_cases))
    return {
        "models": display_metrics,
        "modelsByProbabilityColumns": model_names,
        "summary": {
            "totalCases": total_cases,
            "deferredCases": deferred_cases,
            "deferralRate": round((deferred_cases / total_cases) * 100, 2),
            "automatedCoverage": round((automated_cases / total_cases) * 100, 2),
            "automatedAccuracy": automated_accuracy,
            "overallHybridAccuracy": overall_hybrid_accuracy,
            "bestModel": str(pipeline_summary.get("best_model", display_metrics[0]["model"] if display_metrics else "")),
            "selectedAlpha": round(_to_float(pipeline_summary.get("selected_alpha"), 0.25), 4),
        },
        "modelPredictionSummary": model_prediction_summary,
        "confusionByModel": confusion_by_model,
        "probabilityBands": probability_bands,
        "cases": cases[:safe_limit],
    }

