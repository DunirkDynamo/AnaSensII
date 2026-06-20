"""Typed domain models for the calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


CalibrationMethod = Literal["subcal", "divcal"]
CalibrationMode = Literal["data_table", "measurement_objects"]


@dataclass(slots=True)
class CalibrationRequest:
    """Fully describes a calibration run: what to compare, how, and on which columns to match."""

    group_column: str
    analyte_label: str
    control_label: str
    method: CalibrationMethod
    match_filters: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CalibrationResult:
    """Holds the result table and metadata for a completed calibration run."""

    result_table: pd.DataFrame
    method: str
    analyte_label: str
    control_label: str
    source_mode: CalibrationMode


@dataclass(slots=True)
class MeasurementObject:
    """Represents a single loaded measurement pickle with its extracted parameters."""

    measurement_name: str
    extracted_parameters: pd.DataFrame
    source_path: str
    calibration_table: pd.DataFrame | None = None


@dataclass(slots=True)
class BatchCalibrationResult:
    """Holds per-measurement and stacked results from a multi-object calibration run."""

    stacked_result: CalibrationResult
    per_measurement_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
