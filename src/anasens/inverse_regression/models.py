"""Typed models for inverse estimation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


RegressionType = Literal["quantile", "least_squares"]


@dataclass(slots=True)
class VoltammogramCalibrationInput:
    """Represents one pickled voltammogram object used for inverse estimation."""

    name: str
    source_path: str
    calibration_table: pd.DataFrame


@dataclass(slots=True)
class EstimateRow:
    """Represents one inverse estimate for a measured calibrated metric value."""

    name: str
    measured_value: float
    estimate: float | None
    low_limit: float | None
    upper_limit: float | None
    error: str | None = None


@dataclass(slots=True)
class InverseEstimationResult:
    """Container for all inverse regression outputs."""

    metric_column: str
    regression_type: RegressionType
    rows: list[EstimateRow]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result rows to a tabular DataFrame for display/export."""
        return pd.DataFrame(
            {
                "Name": [row.name for row in self.rows],
                "Measured Value": [row.measured_value for row in self.rows],
                "Estimate": [row.estimate for row in self.rows],
                "Low Limit": [row.low_limit for row in self.rows],
                "Upper Limit": [row.upper_limit for row in self.rows],
                "Error": [row.error for row in self.rows],
            }
        )
