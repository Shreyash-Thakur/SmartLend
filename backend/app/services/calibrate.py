"""
calibrate.py
------------
TAU_D calibration for the hybrid decision engine.

Sweeps tau in [0.15, 0.45] and selects the value whose deferral rate
falls inside the target band [0.22, 0.28].  If multiple taus satisfy
the band the first one found (ascending scan) is kept.  If no tau sits
inside the band the one closest to the band midpoint (0.25) is used.

The resulting TAU_D is persisted to artifacts/pipeline.joblib so the
MLPredictor can load it at startup without retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from backend.app.services.decision_engine import hybrid_decision

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
PIPELINE_PATH  = ARTIFACTS_DIR / "pipeline.joblib"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TauCalibrationResult:
    tau_d:                float
    deferral_rate:        float
    non_deferred_accuracy: float
    curve:                list[dict[str, float]]


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------

def calibrate_tau_d(
    p_ml:         np.ndarray,
    p_cbes:       np.ndarray,
    y_true:       np.ndarray,
    lower_target: float = 0.22,
    upper_target: float = 0.28,
) -> TauCalibrationResult:
    """Sweep tau in [0.15, 0.45] and return the calibrated TAU_D.

    Parameters
    ----------
    p_ml, p_cbes : array-like of float, shape (n,)
        Approval probabilities from the ML model and CBES engine
        respectively for a representative validation / calibration set.
    y_true : array-like of int, shape (n,)
        Ground-truth default labels (1 = default, 0 = no default).
    lower_target, upper_target : float
        Deferral-rate band.  Default [0.22, 0.28].

    Returns
    -------
    TauCalibrationResult
        Contains selected tau_d, its deferral_rate, non-deferred
        accuracy, and the full sweep curve for diagnostics.
    """
    p_ml   = np.asarray(p_ml,   dtype=float)
    p_cbes = np.asarray(p_cbes, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    band_mid = (lower_target + upper_target) / 2.0

    best_tau        = 0.30       # sensible fallback
    best_gap        = float("inf")
    best_deferral   = 1.0
    best_acc        = 0.0
    band_found      = False
    curve: list[dict[str, float]] = []

    # STEP 6 — sweep tau in [0.15, 0.45] at 0.01 resolution
    for tau in np.arange(0.15, 0.451, 0.01):
        tau = float(round(tau, 4))

        decisions = [
            hybrid_decision(float(pm), float(pc), tau)
            for pm, pc in zip(p_ml, p_cbes)
        ]
        labels = np.array([d.decision for d in decisions])

        defer_mask    = labels == "DEFER"
        deferral_rate = float(np.mean(defer_mask))

        non_defer_mask = ~defer_mask
        if np.any(non_defer_mask):
            # APPROVE maps to predicted label 0 (no-default), REJECT to 1 (default)
            preds = np.where(labels[non_defer_mask] == "APPROVE", 0, 1)
            acc   = float(np.mean(preds == y_true[non_defer_mask]))
        else:
            acc = 0.0

        curve.append({
            "tau_d":                  round(tau, 4),
            "deferral_rate":          round(deferral_rate, 6),
            "non_deferred_accuracy":  round(acc, 6),
        })

        in_band = lower_target <= deferral_rate <= upper_target
        if in_band and not band_found:
            # First tau whose deferral rate falls inside the target band
            best_tau      = tau
            best_deferral = deferral_rate
            best_acc      = acc
            band_found    = True
            break               # ascending scan — take first qualifying value

        gap = abs(deferral_rate - band_mid)
        if not band_found and gap < best_gap:
            best_gap      = gap
            best_tau      = tau
            best_deferral = deferral_rate
            best_acc      = acc

    return TauCalibrationResult(
        tau_d=best_tau,
        deferral_rate=float(best_deferral),
        non_deferred_accuracy=float(best_acc),
        curve=curve,
    )


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_tau_to_artifact(tau_d: float) -> None:
    """Persist TAU_D into the existing pipeline.joblib artifact.

    Loads the payload, injects/updates the ``tau_d`` key, and re-saves.
    Raises ``FileNotFoundError`` if the artifact does not exist — TAU_D
    must be calibrated after the pipeline artifact has been created.
    """
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline artifact not found at {PIPELINE_PATH}.  "
            "Run train_pipeline() before calibrating tau."
        )
    payload = joblib.load(PIPELINE_PATH)
    payload["tau_d"] = float(tau_d)
    joblib.dump(payload, PIPELINE_PATH)


def calibrate_and_save(
    p_ml:         np.ndarray,
    p_cbes:       np.ndarray,
    y_true:       np.ndarray,
    lower_target: float = 0.22,
    upper_target: float = 0.28,
) -> TauCalibrationResult:
    """Convenience wrapper: calibrate tau and persist it in one call."""
    result = calibrate_tau_d(p_ml, p_cbes, y_true, lower_target, upper_target)
    save_tau_to_artifact(result.tau_d)
    return result
