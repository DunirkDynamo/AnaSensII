"""Domain models for the AnaSens2 analysis workflow."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class PeakSet:
    peak_indices: Any
    peak_heights: Any
    peak_potentials: Any
    fwhm: Any
    fw01m: Any
    left_edge_indices: Any
    right_edge_indices: Any
    left_edge_01_indices: Any
    right_edge_01_indices: Any


@dataclass(slots=True)
class SweepData:
    file_name: str
    raw_frame: pd.DataFrame
    processed_frame: pd.DataFrame | None = None
    qualifiers: dict[str, Any] = field(default_factory=dict)
    extracted_peak_parameters: dict[str, Any] | None = None
    average_baseline: float | None = None


@dataclass(slots=True)
class Measurement:
    measurement_name: str
    sweeps: list[SweepData]
    extracted_parameters: pd.DataFrame | None = None
    calibration_table: pd.DataFrame | None = None
