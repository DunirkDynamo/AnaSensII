"""Runtime configuration models for the AnaSens2 analysis pipeline.

This module defines frozen-field dataclasses used to pass user-selected
parameters from the GUI (or CLI) to the native analysis pipeline.  All
decision-making lives in the pipeline; these classes are pure data
containers with no behaviour of their own.

Typical usage::

    from anasens.config import PipelineConfig, PreprocessConfig
    from pathlib import Path

    config = PipelineConfig(
        input_path=Path("/data/session.json"),
        input_source="json",
        analysis_mode="single",
        preprocess=PreprocessConfig(smooth_window_length=21),
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ── Type aliases ────────────────────────────────────────────────────────────
# These narrow string parameters to a fixed vocabulary so typos are caught
# at analysis time rather than deep inside pipeline logic.
AnalysisMode = Literal["single", "batch"]
"""Pipeline analysis scope: one measurement folder or many."""

CalibrationMethod = Literal["subcal", "divcal"]
"""Arithmetic method applied during peak-height calibration."""

InputSource = Literal["legacy_csv", "json", "pssession"]
"""Source format of the input data being analysed."""


@dataclass(slots=True)
class PreprocessConfig:
    """Per-sweep preprocessing parameters.

    Every field has a sensible default so the pipeline can run without
    any explicit configuration.  The GUI exposes these fields through the
    :class:`~anasens.interfaces.gui_views.pipeline_tab.PreprocessSettingsDialog`.

    Attributes:
        cut_start_points: Number of data points to discard from the
            beginning of each scan before processing.
        cut_end_points: Number of data points to discard from the end
            of each scan before processing.
        smooth_window_length: Savitzky-Golay filter window length
            (must be odd; automatically enforced at runtime).
        smooth_polyorder: Savitzky-Golay polynomial order.  Must be
            less than ``smooth_window_length``.
        baseline_algo: Baseline subtraction algorithm identifier.
            ``"snip"`` uses pybaselines SNIP; ``"moving_average"`` uses
            the Ekidna moving-average implementation.
        baseline_snip_expected_fwhm: Expected full-width at half-maximum
            of peaks for the SNIP estimator (arbitrary units).
        baseline_snip_window_size: Half-window size passed to the SNIP
            ``max_half_window`` parameter.
        baseline_moving_average_window_size: Rolling window width for
            the Ekidna moving-average baseline.
        baseline_moving_average_forward: Direction of the rolling
            average; ``True`` for a forward (causal) pass.
        baseline_moving_average_max_iter: Maximum iterations allowed
            before the Ekidna estimator stops converging.
    """

    # ── Cut stage ────────────────────────────────────────────────────────
    cut_start_points: int = 0   # Points removed from scan start.
    cut_end_points: int = 0     # Points removed from scan end.

    # ── Smoothing stage ──────────────────────────────────────────────────
    smooth_window_length: int = 17  # SG window; coerced to odd at runtime.
    smooth_polyorder: int = 3       # SG polynomial order < window_length.

    # ── Baseline stage ───────────────────────────────────────────────────
    baseline_algo: str = "snip"                     # "snip" or "moving_average".
    baseline_snip_expected_fwhm: int = 35           # SNIP peak FWHM estimate.
    baseline_snip_window_size: int = 10             # SNIP half-window.
    baseline_moving_average_window_size: int = 10   # MA rolling window.
    baseline_moving_average_forward: bool = True    # MA forward pass flag.
    baseline_moving_average_max_iter: int = 1000    # MA convergence cap.


@dataclass(slots=True)
class VisualizationConfig:
    """Flags controlling which optional plots are rendered per measurement.

    All flags default to ``True`` so a default run produces the full set
    of standard diagnostic plots.

    Attributes:
        pairplot: Render seaborn pairplot and peak-coordinate density
            scatter for the extracted-parameter table.
        drift_2d: Render 2-D potential-vs-current scatter coloured by
            scan order to visualise measurement drift.
        drift_1d: Render dual-axis line plot of peak potential and
            peak current vs scan order.
        overlay_scans: Render all scans in one measurement overlaid on
            a single axes with peak markers.
    """

    pairplot: bool = True                 # Seaborn pairplot + density scatter.
    drift_2d: bool = True                 # 2-D drift scatter coloured by scan order.
    drift_1d: bool = True                 # Dual-axis drift line plot.
    overlay_scans: bool = True            # All-scans overlay with peak markers.
    preprocessing_overview: bool = True   # Side-by-side raw/smoothed/baseline + corrected.


@dataclass(slots=True)
class PipelineConfig:
    """Top-level runtime configuration for one pipeline execution.

    Combines source selection, analysis scope, preprocessing settings,
    and visualization flags into a single object passed to
    :class:`~anasens.pipeline.native_pipeline.NativeAnalysisPipeline`.

    Attributes:
        input_path: Path to the input file or directory selected in the
            UI.  Interpretation depends on ``input_source`` and
            ``analysis_mode``.
        output_path: Optional explicit output root.  When ``None`` the
            pipeline writes outputs adjacent to the input path.
        analysis_mode: ``"single"`` for one measurement, ``"batch"``
            for a folder of measurements.
        input_source: Data format of the input.  Controls which loader
            implementation is used.
        preprocess: Preprocessing stage parameters.
        visualization: Flags selecting which plots to render.
    """

    input_path: Path                    # Resolved at launch from GUI/CLI.
    output_path: Path | None = None     # None → output beside input.
    analysis_mode: AnalysisMode = "single"
    input_source: InputSource = "legacy_csv"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
