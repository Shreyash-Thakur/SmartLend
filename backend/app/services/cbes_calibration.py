"""CBES calibration: compute percentile thresholds from training data.

This module is a placeholder for Task 4. Task 5 will replace it with a full
calibration script that loads Home Credit training data and computes real
percentile breakpoints. For now, load_thresholds() raises FileNotFoundError
to signal that no real artifact exists yet.
"""

from __future__ import annotations


def load_thresholds(path: str | None = None) -> dict[str, list[float]]:
    """Load CBES percentile thresholds from a calibration artifact.

    Args:
        path: Optional path to a custom thresholds file. If None, looks for
              the default artifact produced by the calibration pipeline.

    Returns:
        Dictionary mapping field names to 5-element lists of percentile
        breakpoints (p10, p30, p50, p70, p90).

    Raises:
        FileNotFoundError: If no artifact exists. Task 5 must be run first
                          to generate the calibration artifact via
                          cbes_calibration.py's compute_thresholds() function.
    """
    raise FileNotFoundError(
        "No CBES thresholds artifact found. Run the cbes_calibration.py pipeline "
        "(Task 5) to compute thresholds from Home Credit training data and create "
        "the calibration artifact."
    )
