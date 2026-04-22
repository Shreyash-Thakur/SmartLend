from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path(__file__).resolve().parents[2] / "synthetic_indian_loan_dataset.csv"
logger = logging.getLogger(__name__)


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(x, high))


def dynamic_hybrid_decision(ml_prob: float, cbes_prob: float, alpha: float = 0.25) -> tuple[str, float, float, float]:
    # CBES influences threshold center while ML still determines the final class.
    # Keep a non-zero defer band so REVIEW can always happen when ML is uncertain.
    alpha = clamp(alpha, 0.2, 0.4)

    # Higher CBES lowers center (easier approve, harder reject); lower CBES does the opposite.
    center = clamp(0.5 - ((cbes_prob - 0.5) * alpha), 0.42, 0.58)

    # Wider defer band near neutral CBES, narrower at strong CBES extremes.
    neutrality = 1.0 - min(1.0, abs(cbes_prob - 0.5) * 2.0)
    defer_band = 0.06 + (0.08 * neutrality)

    rejection_threshold = clamp(center - (defer_band / 2), 0.20, 0.60)
    approval_threshold = clamp(center + (defer_band / 2), 0.40, 0.80)

    # Enforce strict ordering to avoid overlap, preserving DEFER space.
    if approval_threshold <= rejection_threshold:
        midpoint = (approval_threshold + rejection_threshold) / 2
        rejection_threshold = clamp(midpoint - 0.03, 0.20, 0.60)
        approval_threshold = clamp(midpoint + 0.03, 0.40, 0.80)

    confidence = abs(ml_prob - 0.5)

    if ml_prob >= approval_threshold:
        return "APPROVE", confidence, approval_threshold, rejection_threshold
    if ml_prob <= rejection_threshold:
        return "REJECT", confidence, approval_threshold, rejection_threshold
    return "DEFER", confidence, approval_threshold, rejection_threshold


def _to_snake_case(key: str) -> str:
    normalized = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return normalized.lower()


@dataclass
class PredictionResult:
    ml_prob: float
    cbes_prob: float
    final_decision: str
    confidence: float
    approval_threshold: float
    rejection_threshold: float
    cbes_components: dict[str, float]
    engineered_features: dict[str, float]


class MLPredictor:
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.template_row: dict[str, Any] = {}
        self.numeric_columns: set[str] = set()
        self._fit_once()

    def _fit_once(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        raw_df = pd.read_csv(self.dataset_path)
        df = self._prepare_input_frame(raw_df)

        self.template_row = df.mode(dropna=True).iloc[0].to_dict()
        self.numeric_columns = set(df.select_dtypes(include=["number"]).columns)

        y = raw_df["loan_approved"].astype(int)
        X = raw_df.drop(columns=[c for c in ["loan_approved", "applicant_id", "city"] if c in raw_df.columns])
        X = self._prepare_input_frame(X)
        X = pd.get_dummies(X, drop_first=True)

        self.feature_columns = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def _prepare_input_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()

        prepared["EMI_INCOME_RATIO"] = prepared["emi"] / (prepared["monthly_income"] + 1)
        prepared["DEBT_BURDEN"] = (prepared["existing_emis"] + prepared["emi"]) / (prepared["monthly_income"] + 1)
        prepared["DEBT_TO_INCOME_RATIO"] = (prepared["existing_emis"] * 12 + prepared["loan_amount"]) / (
            prepared["annual_income"] + 1
        )
        prepared["LOAN_INCOME_RATIO"] = prepared["loan_amount"] / (prepared["annual_income"] + 1)
        prepared["ASSET_COVERAGE"] = prepared["total_assets"] / (prepared["loan_amount"] + 1)
        prepared["LIQUIDITY_RATIO"] = prepared["bank_balance"] / (prepared["loan_amount"] + 1)
        prepared["LOAN_ACTIVITY_RATIO"] = prepared["active_loans"] / (prepared["total_loans"] + 1)
        prepared["REPAYMENT_SCORE"] = prepared["closed_loans"] / (prepared["total_loans"] + 1)
        prepared["MISSED_PAYMENT_RATIO"] = prepared["missed_payments"] / (prepared["total_loans"] + 1)
        prepared["EMPLOYMENT_STABILITY"] = prepared["years_employed"] / (prepared["age"] + 1)

        return prepared

    def _normalize_request_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in input_data.items():
            normalized[_to_snake_case(key)] = value

        if "existing_emis" not in normalized:
            normalized["existing_emis"] = normalized.get("existing_emi", normalized.get("emi", 0))
        if "annual_income" not in normalized and "monthly_income" in normalized:
            normalized["annual_income"] = float(normalized["monthly_income"]) * 12

        residential_assets = float(normalized.get("residential_assets_value", 0) or 0)
        commercial_assets = float(normalized.get("commercial_assets_value", 0) or 0)
        bank_balance = float(normalized.get("bank_balance", 0) or 0)
        if "total_assets" not in normalized:
            normalized["total_assets"] = residential_assets + commercial_assets + bank_balance

        return normalized

    def _compute_cbes_components(self, row: dict[str, Any]) -> dict[str, float]:
        cibil_score = float(row.get("cibil_score", 650))
        missed_payment_ratio = float(row.get("MISSED_PAYMENT_RATIO", 0))
        credit_utilization_ratio = float(row.get("credit_utilization_ratio", 0))

        cibil_norm = clamp((cibil_score - 300) / 600, 0, 1)
        payment_penalty = clamp(1 - missed_payment_ratio, 0, 1)
        util_penalty = clamp(1 - credit_utilization_ratio, 0, 1)
        credit_component = 0.5 * cibil_norm + 0.3 * payment_penalty + 0.2 * util_penalty

        dti_score = clamp(1 - float(row.get("DEBT_TO_INCOME_RATIO", 0)), 0.2, 1)
        emi_score = clamp(1 - float(row.get("EMI_INCOME_RATIO", 0)), 0, 1)
        loan_income_score = clamp(1 - float(row.get("LOAN_INCOME_RATIO", 0)), 0, 1)
        capacity_component = 0.5 * dti_score + 0.3 * emi_score + 0.2 * loan_income_score

        asset_score = clamp(float(row.get("ASSET_COVERAGE", 0)), 0, 2) / 2
        liquidity_score = clamp(float(row.get("LIQUIDITY_RATIO", 0)), 0, 1)
        asset_component = 0.7 * asset_score + 0.3 * liquidity_score

        stability_component = clamp(float(row.get("EMPLOYMENT_STABILITY", 0)), 0, 1)

        return {
            "credit_component": clamp(credit_component, 0, 1),
            "capacity_component": clamp(capacity_component, 0, 1),
            "asset_component": clamp(asset_component, 0, 1),
            "stability_component": clamp(stability_component, 0, 1),
        }

    def _compute_cbes_prob(self, components: dict[str, float]) -> float:
        cbes_score = (
            0.35 * components["credit_component"]
            + 0.3 * components["capacity_component"]
            + 0.25 * components["asset_component"]
            + 0.1 * components["stability_component"]
        )
        return clamp(cbes_score, 0, 1)

    def predict_application(self, input_data: dict[str, Any]) -> PredictionResult:
        try:
            normalized = self._normalize_request_input(input_data)

            row = dict(self.template_row)
            row.update(normalized)

            for col in self.numeric_columns:
                if col in row and row[col] is not None:
                    row[col] = float(row[col])

            one_row = pd.DataFrame([row])
            one_row = self._prepare_input_frame(one_row)

            merged_row = {**row, **one_row.iloc[0].to_dict()}
            cbes_components = self._compute_cbes_components(merged_row)
            cbes_prob = self._compute_cbes_prob(cbes_components)

            features = pd.get_dummies(one_row, drop_first=True)
            features = features.reindex(columns=self.feature_columns, fill_value=0)

            scaled = self.scaler.transform(features)
            ml_prob = float(self.model.predict_proba(scaled)[0, 1])

            final_decision, confidence, approval_threshold, rejection_threshold = dynamic_hybrid_decision(ml_prob, cbes_prob)

            engineered_features = {
                "emi_income_ratio": float(merged_row.get("EMI_INCOME_RATIO", 0)),
                "debt_to_income_ratio": float(merged_row.get("DEBT_TO_INCOME_RATIO", 0)),
                "loan_income_ratio": float(merged_row.get("LOAN_INCOME_RATIO", 0)),
                "asset_coverage": float(merged_row.get("ASSET_COVERAGE", 0)),
                "liquidity_ratio": float(merged_row.get("LIQUIDITY_RATIO", 0)),
                "missed_payment_ratio": float(merged_row.get("MISSED_PAYMENT_RATIO", 0)),
                "employment_stability": float(merged_row.get("EMPLOYMENT_STABILITY", 0)),
            }

            return PredictionResult(
                ml_prob=ml_prob,
                cbes_prob=float(cbes_prob),
                final_decision=final_decision,
                confidence=float(confidence),
                approval_threshold=float(approval_threshold),
                rejection_threshold=float(rejection_threshold),
                cbes_components=cbes_components,
                engineered_features=engineered_features,
            )
        except Exception as exc:
            logger.exception("Prediction failed")
            raise RuntimeError(f"Prediction pipeline failed: {exc}") from exc


_predictor: MLPredictor | None = None


def get_predictor() -> MLPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor(DATASET_PATH)
    return _predictor
