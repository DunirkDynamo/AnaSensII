"""Native object-oriented analysis pipeline implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from anasens.config import PipelineConfig
from anasens.domain import Measurement, SweepData
from anasens.features.integration import integrate_simpson
from anasens.features.peaks import find_peaks_with_widths
from anasens.features.tables import build_analyte_data_table
from anasens.io.measurement_loader import CsvMeasurementLoader
from anasens.io.measurement_store import save_pickle
from anasens.preprocess.baseline import baseline_subtraction
from anasens.preprocess.cutoff import cut_points
from anasens.preprocess.smoothing import smooth_voltammogram
from anasens.visualization.pipeline_visualizations import (
    render_1d_peak_drift,
    render_2d_peak_drift,
    render_overlay_scans,
    render_pairplot,
    render_peak_coordinate_density,
)


@dataclass(slots=True)
class NativeAnalysisPipeline:
    """Run the end-to-end native analysis workflow on typed measurements."""

    measurement_loader: CsvMeasurementLoader = field(default_factory=CsvMeasurementLoader)

    def run_from_config(self, config: PipelineConfig) -> list[Measurement]:
        """Load measurements using IO services and execute analysis stages."""
        measurements = self.measurement_loader.load_measurements(config.input_path, config.analysis_mode)
        return self.run_measurements(measurements, config)

    def run_measurements(self, measurements: list[Measurement], config: PipelineConfig) -> list[Measurement]:
        """Execute preprocessing, extraction, persistence, and visualization stages."""
        all_extracted_tables: list[pd.DataFrame] = []

        for measurement in measurements:
            output_root = self._resolve_measurement_output_root(config, measurement)
            analysis_dir = output_root / "Analysis"
            processed_dir = analysis_dir / "Processed Voltammograms"
            extracted_dir = analysis_dir / "Extracted Peak Parameters"
            sweep_info_dir = analysis_dir / "Sweep Info"
            voltammogram_list_pickle_dir = analysis_dir / "Pickled Voltammogram Info List"
            measurement_pickle_dir = analysis_dir / "Full Measurement Info"
            visualization_dir = analysis_dir / "Data Visualization"

            processed_dir.mkdir(parents=True, exist_ok=True)
            extracted_dir.mkdir(parents=True, exist_ok=True)
            sweep_info_dir.mkdir(parents=True, exist_ok=True)
            voltammogram_list_pickle_dir.mkdir(parents=True, exist_ok=True)
            measurement_pickle_dir.mkdir(parents=True, exist_ok=True)
            visualization_dir.mkdir(parents=True, exist_ok=True)

            for sweep in measurement.sweeps:
                self._preprocess_sweep(sweep, config)
                self._extract_sweep_parameters(sweep)
                self._save_processed_sweep(processed_dir, sweep)
                self._save_sweep_info_pickle(sweep_info_dir, sweep)

            extracted_table = self._build_measurement_table(measurement)
            extracted_table = self._add_scan_order(extracted_table, rank_column="Timestamp")
            measurement.extracted_parameters = extracted_table
            extracted_table.to_csv(extracted_dir / "extracted parameters.csv", index=False)
            all_extracted_tables.append(extracted_table)

            save_pickle(measurement.sweeps, voltammogram_list_pickle_dir / "voltammogram_info_list.pkl")
            save_pickle(
                {
                    "voltammogram_list": measurement.sweeps,
                    "extracted_parameters": extracted_table,
                },
                measurement_pickle_dir / "measurement_info.pkl",
            )

            self._render_selected_visualizations(measurement, extracted_table, visualization_dir, config)

        if config.analysis_mode == "batch" and all_extracted_tables:
            self._save_batch_outputs(config, measurements, all_extracted_tables)

        return measurements

    def _resolve_measurement_output_root(self, config: PipelineConfig, measurement: Measurement) -> Path:
        if config.output_path is not None:
            root = config.output_path / measurement.measurement_name
            root.mkdir(parents=True, exist_ok=True)
            return root

        if config.analysis_mode == "single":
            return config.input_path
        return config.input_path / measurement.measurement_name

    def _preprocess_sweep(self, sweep: SweepData, config: PipelineConfig) -> None:
        frame = cut_points(
            sweep.raw_frame,
            cut_start=config.preprocess.cut_start_points,
            cut_end=config.preprocess.cut_end_points,
        )
        if "I" not in frame.columns or "V" not in frame.columns:
            raise ValueError(f"Sweep {sweep.file_name} must contain V and I columns.")

        smoothed = smooth_voltammogram(
            frame["I"].to_numpy(dtype=float),
            window_length=config.preprocess.smooth_window_length,
            polyorder=config.preprocess.smooth_polyorder,
        )
        corrected, baseline = baseline_subtraction(
            frame["V"].to_numpy(dtype=float),
            smoothed,
            algo=config.preprocess.baseline_algo,
            snip_expected_fwhm=config.preprocess.baseline_snip_expected_fwhm,
            snip_window_size=config.preprocess.baseline_snip_window_size,
            moving_average_window_size=config.preprocess.baseline_moving_average_window_size,
            moving_average_forward=config.preprocess.baseline_moving_average_forward,
            moving_average_max_iter=config.preprocess.baseline_moving_average_max_iter,
        )

        processed = frame.copy()
        processed["I_smoothed"] = smoothed
        processed["baseline"] = baseline
        processed["I_corrected"] = corrected
        sweep.processed_frame = processed

    def _extract_sweep_parameters(self, sweep: SweepData) -> None:
        if sweep.processed_frame is None:
            raise ValueError("Processed frame is required before extraction.")

        frame = sweep.processed_frame
        x_values = frame["V"].to_numpy(dtype=float)
        y_values = frame["I_corrected"].to_numpy(dtype=float)

        peak_info = find_peaks_with_widths(
            x_values,
            y_values,
            lower_bound=float(np.min(x_values)),
            upper_bound=float(np.max(x_values)),
            width=20,
        )

        peak_integrals: list[float] = []
        baseline_integrals: list[float] = []

        left_edges = peak_info["left_edge_indices"]
        right_edges = peak_info["right_edge_indices"]
        baseline_values = frame["baseline"].to_numpy(dtype=float)

        for left_idx, right_idx in zip(left_edges, right_edges, strict=False):
            if right_idx <= left_idx:
                peak_integrals.append(0.0)
                baseline_integrals.append(0.0)
                continue

            peak_integrals.append(integrate_simpson(x_values, y_values, int(left_idx), int(right_idx)))
            baseline_integrals.append(integrate_simpson(x_values, baseline_values, int(left_idx), int(right_idx)))

        peak_info["integrated_peak_areas"] = np.asarray(peak_integrals, dtype=float)
        peak_info["integrated_baseline_areas"] = np.asarray(baseline_integrals, dtype=float)
        peak_info["integrated_peak_minus_baseline"] = (
            peak_info["integrated_peak_areas"] - peak_info["integrated_baseline_areas"]
        )

        sweep.average_baseline = float(np.mean(baseline_values))
        sweep.extracted_peak_parameters = peak_info

    def _build_measurement_table(self, measurement: Measurement) -> pd.DataFrame:
        labels = np.asarray([str(sweep.qualifiers.get("Label", "Unknown")) for sweep in measurement.sweeps])
        timestamps = np.asarray([float(sweep.qualifiers.get("Timestamp", 0.0)) for sweep in measurement.sweeps])

        peak_info_list: list[dict[str, np.ndarray]] = []
        for sweep in measurement.sweeps:
            if sweep.extracted_peak_parameters is None:
                raise ValueError("All sweeps must have extracted parameters before table build.")
            peak_info_list.append(sweep.extracted_peak_parameters)

        return build_analyte_data_table(labels, peak_info_list, timestamps)

    def _add_scan_order(self, extracted_table: pd.DataFrame, rank_column: str) -> pd.DataFrame:
        if rank_column not in extracted_table.columns:
            return extracted_table

        table = extracted_table.copy()
        table["Scan Order"] = table[rank_column].rank(method="dense").astype(int)
        return table

    def _render_selected_visualizations(
        self,
        measurement: Measurement,
        extracted_table: pd.DataFrame,
        visualization_dir: Path,
        config: PipelineConfig,
    ) -> None:
        if config.visualization.pairplot:
            render_pairplot(extracted_table, visualization_dir)
            render_peak_coordinate_density(extracted_table, visualization_dir)

        if config.visualization.drift_2d:
            render_2d_peak_drift(extracted_table, visualization_dir)

        if config.visualization.drift_1d:
            render_1d_peak_drift(extracted_table, visualization_dir)

        if config.visualization.overlay_scans:
            render_overlay_scans(measurement, visualization_dir)

    def _save_processed_sweep(self, processed_dir: Path, sweep: SweepData) -> None:
        if sweep.processed_frame is None:
            return
        stem = Path(sweep.file_name).stem
        sweep.processed_frame.to_csv(processed_dir / f"{stem}_PROCESSED.csv", index=False)

    def _save_sweep_info_pickle(self, sweep_info_dir: Path, sweep: SweepData) -> None:
        stem = Path(sweep.file_name).stem
        save_pickle(
            {
                "file": sweep.file_name,
                "qualifiers": sweep.qualifiers,
                "average_baseline": sweep.average_baseline,
                "extracted_peak_parameters": sweep.extracted_peak_parameters,
            },
            sweep_info_dir / f"{stem}_sweep_info.pkl",
        )

    def _save_batch_outputs(
        self,
        config: PipelineConfig,
        measurements: list[Measurement],
        all_extracted_tables: list[pd.DataFrame],
    ) -> None:
        batch_root = config.output_path if config.output_path is not None else config.input_path
        batch_analysis_folder = batch_root / "Batch Analysis"
        pooled_params_folder = batch_analysis_folder / "pooled_extracted_params"
        pooled_params_folder.mkdir(parents=True, exist_ok=True)

        tagged_tables: list[pd.DataFrame] = []
        for measurement, table in zip(measurements, all_extracted_tables, strict=False):
            tagged = table.copy()
            tagged["Measurement"] = np.repeat(measurement.measurement_name, len(tagged))
            tagged_tables.append(tagged)

        pooled = pd.concat(tagged_tables, ignore_index=True)
        pooled.to_csv(pooled_params_folder / "pooled_extracted_params.csv", index=False)
