"""Runtime configuration models for pipeline execution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


AnalysisMode = Literal["single", "batch"]
CalibrationMethod = Literal["subcal", "divcal"]
InputSource = Literal["legacy_csv", "json", "pssession"]


@dataclass(slots=True)
class PreprocessConfig:
    cut_start_points: int = 0
    cut_end_points: int = 0
    smooth_window_length: int = 17
    smooth_polyorder: int = 3
    baseline_algo: str = "snip"
    baseline_snip_expected_fwhm: int = 35
    baseline_snip_window_size: int = 10
    baseline_moving_average_window_size: int = 10
    baseline_moving_average_forward: bool = True
    baseline_moving_average_max_iter: int = 1000


@dataclass(slots=True)
class VisualizationConfig:
    pairplot: bool = True
    drift_2d: bool = True
    drift_1d: bool = True
    overlay_scans: bool = True


@dataclass(slots=True)
class PipelineConfig:
    input_path: Path
    output_path: Path | None = None
    analysis_mode: AnalysisMode = "single"
    input_source: InputSource = "legacy_csv"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
