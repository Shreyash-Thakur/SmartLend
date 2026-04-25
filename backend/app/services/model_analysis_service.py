from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import csv
import json

from backend.app.services.ml_service import dynamic_hybrid_decision


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_METRICS_PATH = ARTIFACTS_DIR / "model_metrics.csv"
PREDICTION_OUTPUTS_PATH = ARTIFACTS_DIR / "prediction_outputs.csv"
PIPELINE_SUMMARY_PATH = ARTIFACTS_DIR / "pipeline_summary.json"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def _load_model_metrics() -> list[dict[str, Any]]:
    if not MODEL_METRICS_PATH.exists():
        return []

    items: list[dict[str, Any]] = []
    with MODEL_METRICS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            precision = _to_float(row.get("Precision"))
            recall = _to_float(row.get("Recall"))
            denom = precision + recall
            f1 = (2 * precision * recall / denom) if denom else 0.0
            items.append(
                {
                    "model": str(row.get("Model", "")),
                    "accuracy": _to_float(row.get("Accuracy")),
                    "precision": precision,
                    "recall": recall,
                    "auc": _to_float(row.get("AUC")),
                    "f1": f1,
                    "tuned": True,
                }
            )

    items.sort(key=lambda item: item.get("auc", 0.0), reverse=True)
    for idx, item in enumerate(items, start=1):
        item["rank"] = idx

    return items


@lru_cache(maxsize=1)
def _load_pipeline_summary() -> dict[str, Any]:
    if not PIPELINE_SUMMARY_PATH.exists():
        return {}

    try:
        with PIPELINE_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def _load_prediction_outputs() -> tuple[list[dict[str, Any]], list[str]]:
    if not PREDICTION_OUTPUTS_PATH.exists():
        return [], []

    rows: list[dict[str, Any]] = []
    model_names: list[str] = []

    with PREDICTION_OUTPUTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            model_names = [name.replace("prob_", "") for name in reader.fieldnames if name.startswith("prob_")]

        for row in reader:
            y_true = _to_int(row.get("y_true"))
            expected = "APPROVE" if y_true == 1 else "REJECT"
            best_model_prob = _to_float(row.get("best_model_prob"))
            cbes_prob = _to_float(row.get("cbes_prob"))
            hybrid_decision, hybrid_confidence, approval_threshold, rejection_threshold = dynamic_hybrid_decision(
                best_model_prob,
                cbes_prob,
            )

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
                    "applicantId": str(row.get("applicant_id", "")),
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


def get_model_analysis_payload(limit: int = 200) -> dict[str, Any]:
    metrics = _load_model_metrics()
    cases, model_names = _load_prediction_outputs()
    pipeline_summary = _load_pipeline_summary()

    if not metrics or not cases:
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

    total_cases = len(cases)
    deferred_cases = sum(1 for row in cases if row["hybridDecision"] == "DEFER")
    automated_cases = total_cases - deferred_cases

    automated_correct = sum(
        1
        for row in cases
        if row["hybridDecision"] != "DEFER" and row["hybridDecision"] == row["expectedDecision"]
    )
    overall_correct = sum(1 for row in cases if row["hybridDecision"] == row["expectedDecision"])

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
            pred = row["modelPredictions"][model]
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

    safe_limit = max(1, min(limit, total_cases))
    return {
        "models": metrics,
        "modelsByProbabilityColumns": model_names,
        "summary": {
            "totalCases": total_cases,
            "deferredCases": deferred_cases,
            "deferralRate": round((deferred_cases / total_cases) * 100, 2),
            "automatedCoverage": round((automated_cases / total_cases) * 100, 2),
            "automatedAccuracy": round((automated_correct / automated_cases) * 100, 2) if automated_cases else 0.0,
            "overallHybridAccuracy": round((overall_correct / total_cases) * 100, 2),
            "bestModel": str(pipeline_summary.get("best_model", metrics[0]["model"] if metrics else "")),
            "selectedAlpha": round(_to_float(pipeline_summary.get("selected_alpha"), 0.25), 4),
        },
        "modelPredictionSummary": model_prediction_summary,
        "confusionByModel": confusion_by_model,
        "probabilityBands": probability_bands,
        "cases": cases[:safe_limit],
    }
