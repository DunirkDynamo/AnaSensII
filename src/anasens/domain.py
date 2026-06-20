"""Core domain models for the AnaSens2 analysis workflow.

These dataclasses represent the fundamental data objects that flow
through the pipeline from ingestion to final output.  They are
intentionally free of analysis logic; computation lives in the
feature, preprocess, and pipeline layers.

Object hierarchy::

    Measurement
    └── SweepData  (one per electrochemical scan)
        ├── raw_frame          (V, I columns; unmodified source data)
        ├── processed_frame    (cut, smoothed, baseline-corrected columns)
        ├── qualifiers         (metadata dict from source or filename)
        └── extracted_peak_parameters  (peak indices, heights, integrals …)
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class PeakSet:
    """Structured container for peak detection results (currently unused in main path).

    This dataclass mirrors the dict returned by
    :func:`~anasens.features.peaks.find_peaks_with_widths` and is
    provided for downstream typed access when needed.

    Attributes:
        peak_indices: Integer array of sample indices where peaks occur.
        peak_heights: Current values at detected peak positions.
        peak_potentials: Potential values at detected peak positions.
        fwhm: Full-width at half-maximum for each peak.
        fw01m: Full-width at 10 % of maximum for each peak.
        left_edge_indices: Left FWHM edge sample indices.
        right_edge_indices: Right FWHM edge sample indices.
        left_edge_01_indices: Left FW0.1M edge sample indices.
        right_edge_01_indices: Right FW0.1M edge sample indices.
    """

    peak_indices: Any           # int array — sample positions of peaks.
    peak_heights: Any           # float array — current at each peak.
    peak_potentials: Any        # float array — potential at each peak.
    fwhm: Any                   # float array — width at half-maximum.
    fw01m: Any                  # float array — width at 10 % maximum.
    left_edge_indices: Any      # int array — left FWHM edge positions.
    right_edge_indices: Any     # int array — right FWHM edge positions.
    left_edge_01_indices: Any   # int array — left FW0.1M edge positions.
    right_edge_01_indices: Any  # int array — right FW0.1M edge positions.


@dataclass(slots=True)
class SweepData:
    """All data associated with a single electrochemical scan.

    A sweep is the atomic unit of analysis.  The pipeline populates
    fields progressively: ``raw_frame`` is set at load time;
    ``processed_frame`` and ``extracted_peak_parameters`` are filled by
    preprocessing and extraction stages respectively.

    Attributes:
        file_name: Stable identifier string derived from source metadata
            (not necessarily a real filesystem path for JSON/pssession
            sources).
        raw_frame: DataFrame with columns ``V`` (potential) and ``I``
            (current) exactly as loaded from the source file.
        processed_frame: DataFrame produced by the preprocessing stage.
            Contains the original columns plus ``I_processing``,
            ``I_smoothed``, ``baseline``, and ``I_corrected``.
        qualifiers: Free-form metadata dict.  For JSON/pssession sources
            this includes ``MeasurementTitle``, ``Date``, ``SourceTime``,
            ``MeasurementIndex``, ``CurveIndex``, and ``ScanOrder``.
            For legacy CSV sources it contains ``Date``, ``Q1``,
            ``Label``, and ``Timestamp``.
        extracted_peak_parameters: Dict of numpy arrays produced by
            :func:`~anasens.features.peaks.find_peaks_with_widths`
            and enriched with integrals by the pipeline.
        average_baseline: Mean baseline value over the scan, stored
            after the baseline-subtraction stage.
    """

    # ── Set at load time ─────────────────────────────────────────────────
    file_name: str              # Stable sweep identifier string.
    raw_frame: pd.DataFrame     # Raw V/I data; never mutated after loading.

    # ── Set by preprocessing stage ───────────────────────────────────────
    processed_frame: pd.DataFrame | None = None  # None until preprocessed.

    # ── Set at load time from source metadata ────────────────────────────
    qualifiers: dict[str, Any] = field(default_factory=dict)  # Metadata dict.

    # ── Set by extraction stage ──────────────────────────────────────────
    extracted_peak_parameters: dict[str, Any] | None = None  # None until extracted.
    average_baseline: float | None = None                    # None until extracted.


@dataclass(slots=True)
class Measurement:
    """Container grouping all sweeps that belong to one measurement session.

    In pssession/JSON sources a single file may contain multiple
    measurements; each becomes its own :class:`Measurement` instance.  In
    legacy CSV mode each sub-folder maps to one measurement.

    Attributes:
        measurement_name: Human-readable stable identifier derived from
            source metadata or folder name.  Used as the output folder
            name and as the ``MeasurementName`` column in exported tables.
        sweeps: Ordered list of :class:`SweepData` objects comprising
            this measurement.
        extracted_parameters: Combined extracted-parameter DataFrame
            produced after all sweeps have been processed.  Contains one
            row per detected peak, with metadata columns appended.
        calibration_table: DataFrame populated by
            :class:`~anasens.calibrator.calibrator.DataTableCalibrator`
            after an optional calibration run.  ``None`` until calibration
            is performed.
    """

    measurement_name: str                       # Stable output identifier.
    sweeps: list[SweepData]                     # All scans in this session.
    extracted_parameters: pd.DataFrame | None = None  # None until pipeline runs.
    calibration_table: pd.DataFrame | None = None     # None until calibrated.
