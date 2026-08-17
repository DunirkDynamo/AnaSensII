"""Native object-oriented analysis pipeline implementation.

This module orchestrates the end-to-end workflow:
1. Load normalized measurements.
2. Preprocess each sweep.
3. Extract peak parameters.
4. Persist tables/artifacts.
5. Render selected visualizations.

The analysis math is implemented in dedicated feature/preprocess modules;
this class coordinates those stages and manages I/O boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from anasens.config import PipelineConfig
from anasens.domain import Measurement, SweepData
from anasens.features.integration import integrate_simpson
from anasens.features.peaks import find_peaks_with_widths
from anasens.features.tables import build_analyte_data_table
from anasens.io.measurement_loader import MeasurementLoader
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
    render_preprocessing_overview,
)


@dataclass(slots=True)
class NativeAnalysisPipeline:
    """Run the end-to-end native analysis workflow on typed measurements."""

    measurement_loader: MeasurementLoader = field(default_factory=MeasurementLoader)

    def run_from_config(self, config: PipelineConfig) -> list[Measurement]:
        """Run the full workflow starting from runtime configuration.

        Args:
            config: Runtime configuration including source/mode/preprocess flags.

        Returns:
            list[Measurement]: Fully processed measurements with extracted tables.
        """
        # Stream session-style batch folders one source file at a time so the
        # pipeline no longer has to materialize every pssession/JSON payload in
        # memory before preprocessing begins.
        if (
            config.analysis_mode == "batch"
            and config.input_source in ("json", "pssession")
            and config.input_path.is_dir()
        ):
            return self._run_streaming_session_batch(config)

        measurements = self.measurement_loader.load_measurements(
            config.input_path,
            config.analysis_mode,
            config.input_source,
        )
        return self.run_measurements(measurements, config)

    def _run_streaming_session_batch(self, config: PipelineConfig) -> list[Measurement]:
        """Process one JSON/pssession source file at a time in batch mode.

        This path reduces peak memory usage by loading a single source file,
        fully processing its measurements, saving outputs, and then dropping
        the heavy sweep objects before advancing to the next source file.

        Args:
            config: Runtime configuration for a directory-based JSON/pssession batch.

        Returns:
            list[Measurement]: Processed measurements with sweep payloads released.
        """
        suffixes = (".json",) if config.input_source == "json" else (".pssession",)
        batch_files = self.measurement_loader.session_loader._resolve_batch_files(config.input_path, suffixes)

        completed_measurements: list[Measurement] = []
        measurement_names: list[str] = []
        all_extracted_tables: list[pd.DataFrame] = []

        for file_index, batch_file in enumerate(batch_files, start=1):
            print(
                f"[PROGRESS] Loading source file {file_index}/{len(batch_files)}: {batch_file.name}",
                flush=True,
            )

            # Force per-file execution through the existing single-file loader
            # so each source is fully processed and released before the next.
            file_config = replace(config, input_path=batch_file, analysis_mode="single")
            measurements = self.measurement_loader.load_measurements(
                batch_file,
                "single",
                config.input_source,
            )
            processed_measurements = self.run_measurements(measurements, file_config)

            # Keep only the compact measurement table and name needed for the
            # final pooled export; the large sweep traces can now be discarded.
            for measurement in processed_measurements:
                measurement_names.append(measurement.measurement_name)
                if measurement.extracted_parameters is not None:
                    all_extracted_tables.append(measurement.extracted_parameters)
                measurement.sweeps = []
                completed_measurements.append(measurement)

        if all_extracted_tables:
            self._save_batch_outputs(config, measurement_names, all_extracted_tables)

        print("[PROGRESS] Analysis complete.", flush=True)
        return completed_measurements

    def run_measurements(self, measurements: list[Measurement], config: PipelineConfig) -> list[Measurement]:
        """Execute full analysis for all supplied measurements.

        The method intentionally groups work into four stages for each
        measurement:

        1. Resolve/create output folders.
        2. Preprocess and extract each sweep.
        3. Persist tabular and pickle artifacts.
        4. Render selected visualization products.

        Args:
            measurements: Pre-loaded measurement objects.
            config: Runtime execution configuration.

        Returns:
            list[Measurement]: Updated measurements containing extracted tables.
        """
        # Retain each measurement table so batch mode can also emit a pooled
        # cross-measurement export at the end of the run.
        all_extracted_tables: list[pd.DataFrame] = []
        total_measurements = len(measurements)

        for index, measurement in enumerate(measurements, start=1):
            print(
                f"[PROGRESS] Analyzing measurement {index}/{total_measurements}: {measurement.measurement_name}",
                flush=True,
            )

            # Output folder layout: JSON/pssession now uses one shared master
            # folder per source file; legacy CSV keeps compatibility behavior.
            master_root = self._resolve_measurement_output_root(config, measurement)
            analysis_dir = master_root / "Analysis"
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

            # Filenames are tagged so all artifacts can coexist safely inside
            # shared folders without collisions across measurement/curve pairs.
            measurement_index = int(
                measurement.sweeps[0].qualifiers.get("MeasurementIndex", 0)
            ) if measurement.sweeps else 0

            # Sweep stage: preprocess -> extract -> persist sweep-level outputs.
            for sweep in measurement.sweeps:
                self._preprocess_sweep(sweep, config)
                self._extract_sweep_parameters(sweep)
                curve_index = int(sweep.qualifiers.get("CurveIndex", 0))
                self._save_processed_sweep(processed_dir, sweep, measurement_index, curve_index)
                self._save_sweep_info_pickle(sweep_info_dir, sweep, measurement_index, curve_index)

            # Measurement stage: build one extracted parameter table and save it.
            extracted_table = self._build_measurement_table(measurement)
            extracted_table = self._add_scan_order(extracted_table, rank_column="Timestamp")
            measurement.extracted_parameters = extracted_table
            extracted_table.to_csv(
                extracted_dir / f"{measurement_index}_extracted parameters.csv",
                index=False,
            )
            all_extracted_tables.append(extracted_table)

            # Persist pickled intermediate objects used by legacy downstream tools.
            save_pickle(
                measurement.sweeps,
                voltammogram_list_pickle_dir / f"{measurement_index}_voltammogram_info_list.pkl",
            )
            save_pickle(
                {
                    "voltammogram_list": measurement.sweeps,
                    "extracted_parameters": extracted_table,
                },
                measurement_pickle_dir / f"{measurement_index}_measurement_info.pkl",
            )

            # Visualization stage: each renderer writes filename-tagged outputs.
            self._render_selected_visualizations(
                measurement,
                extracted_table,
                visualization_dir,
                config,
                measurement_index,
            )

            print(
                f"[PROGRESS] Completed measurement {index}/{total_measurements}: {measurement.measurement_name}",
                flush=True,
            )

        # Optional pooled output is generated only for explicit batch mode.
        if config.analysis_mode == "batch" and all_extracted_tables:
            self._save_batch_outputs(
                config,
                [measurement.measurement_name for measurement in measurements],
                all_extracted_tables,
            )

        print("[PROGRESS] Analysis complete.", flush=True)
        return measurements

    def _resolve_measurement_output_root(self, config: PipelineConfig, measurement: Measurement) -> Path:
        """Resolve the filesystem root where analysis artifacts are written.

        The folder strategy is source-aware:

        1. ``json``/``pssession``: return a shared master folder (one per
           input source stem) so all measurement artifacts land under one
           ``Analysis`` tree.
        2. Legacy CSV single mode: keep historical behavior and write directly
           in the selected input folder.
        3. Legacy CSV batch mode: keep one subfolder per measurement.

        :param config: Runtime pipeline configuration.
        :type config: PipelineConfig
        :param measurement: Measurement currently being processed.
        :type measurement: Measurement
        :return: Output root path used by downstream persistence stages.
        :rtype: Path
        """
        def source_stem_for_measurement() -> str:
            if measurement.sweeps:
                source_stem = measurement.sweeps[0].qualifiers.get("SourceStem")
                if isinstance(source_stem, str) and source_stem.strip():
                    return source_stem.strip()

            return measurement.measurement_name.split("__", 1)[0]

        def master_folder(base_path: Path) -> Path:
            return base_path / source_stem_for_measurement()

        # When an explicit output root is provided, honour it but still nest
        # JSON/pssession outputs under a source-named master folder.
        if config.output_path is not None:
            if config.input_source in ("json", "pssession") and config.input_path.is_file():
                master = master_folder(config.output_path)
            elif config.input_source in ("json", "pssession"):
                master = master_folder(config.output_path)
            else:
                master = config.output_path
            master.mkdir(parents=True, exist_ok=True)
            return master

        # For JSON/pssession inputs, nest under a master folder adjacent to the
        # source directory or file so each source file keeps its own outputs together.
        if config.input_source in ("json", "pssession"):
            master_parent = config.input_path.parent if config.input_path.is_file() else config.input_path
            master = master_folder(master_parent)
            master.mkdir(parents=True, exist_ok=True)
            return master

        # Legacy CSV single mode: output goes directly into the input folder.
        if config.analysis_mode == "single":
            return config.input_path

        # Legacy CSV batch mode: each measurement gets a subfolder inside the
        # input folder, named after the measurement.
        measurement_subfolder = config.input_path / measurement.measurement_name
        measurement_subfolder.mkdir(parents=True, exist_ok=True)
        return measurement_subfolder

    def _preprocess_sweep(self, sweep: SweepData, config: PipelineConfig) -> None:
        """Apply cut, smoothing, and baseline correction to one sweep.

        Args:
            sweep: Sweep to preprocess.
            config: Runtime preprocessing parameters.

        Raises:
            ValueError: If required raw sweep columns are absent.
        """
        # Stage 0: preserve raw current values and build a separate working
        # current used exclusively for preprocessing math.
        frame = sweep.raw_frame.copy()
        if "I" not in frame.columns or "V" not in frame.columns:
            raise ValueError(f"Sweep {sweep.file_name} must contain V and I columns.")

        # Stage 1: remove configured point ranges from start/end.
        frame = cut_points(
            frame,
            cut_start=config.preprocess.cut_start_points,
            cut_end=config.preprocess.cut_end_points,
        )

        # Capture the original unmodified current for output persistence.
        original_current = frame["I"].to_numpy(dtype=float)
        # Build a processing current that can be sign-normalized without
        # changing the saved raw current trace.
        processing_current = original_current.copy()

        # If the net current is negative, invert the processing trace only.
        current_sum = float(processing_current.sum())
        if current_sum < 0:
            processing_current = -processing_current

        # Stage 2: smooth current for robust baseline and peak operations.
        smoothed = smooth_voltammogram(
            processing_current,
            window_length=config.preprocess.smooth_window_length,
            polyorder=config.preprocess.smooth_polyorder,
        )
        # Stage 3: compute baseline and corrected signal using selected algorithm.
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

        # Persist one raw-current column only: the original cut trace remains
        # in ``I``, while derived processing stages are written separately.
        processed = frame.copy()
        processed["I_processing"] = processing_current
        processed["I_smoothed"] = smoothed
        processed["baseline"] = baseline
        processed["I_corrected"] = corrected
        sweep.processed_frame = processed

    def _extract_sweep_parameters(self, sweep: SweepData) -> None:
        """Extract peak descriptors and integrals from one processed sweep.

        Args:
            sweep: Sweep with ``processed_frame`` populated.

        Raises:
            ValueError: If preprocessing has not run yet.
        """
        if sweep.processed_frame is None:
            raise ValueError("Processed frame is required before extraction.")

        frame = sweep.processed_frame
        x_values = frame["V"].to_numpy(dtype=float)
        y_values = frame["I_corrected"].to_numpy(dtype=float)

        # Detect candidate peaks and width descriptors over the full scan range.
        peak_info = find_peaks_with_widths(
            x_values,
            y_values,
            lower_bound=float(np.min(x_values)),
            upper_bound=float(np.max(x_values)),
            width=20,
        )

        # Integrate signal and baseline between detected edge pairs.
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
        """Build the extracted-parameter table for one measurement.

        Args:
            measurement: Measurement containing sweeps with extracted peaks.

        Returns:
            pd.DataFrame: Measurement-level extracted parameter table enriched
            with workflow metadata columns.
        """
        labels = np.asarray([str(sweep.qualifiers.get("Label", "Unknown")) for sweep in measurement.sweeps])
        timestamps = np.asarray([float(sweep.qualifiers.get("Timestamp", 0.0)) for sweep in measurement.sweeps])

        peak_info_list: list[dict[str, np.ndarray]] = []
        for sweep in measurement.sweeps:
            if sweep.extracted_peak_parameters is None:
                raise ValueError("All sweeps must have extracted parameters before table build.")
            peak_info_list.append(sweep.extracted_peak_parameters)

        # Build base table from per-sweep peak dictionaries.
        table = build_analyte_data_table(labels, peak_info_list, timestamps)

        # Replicate sweep-level metadata across each peak row for retro-analysis.
        metadata_columns = {
            "MeasurementName": [],
            "SweepName": [],
            "Date": [],
            "SourceTime": [],
            "MeasurementIndex": [],
            "CurveIndex": [],
            "MeasurementTitle": [],
            "SourceStem": [],
            "ScanOrder": [],
        }

        for sweep in measurement.sweeps:
            if sweep.extracted_peak_parameters is None:
                continue

            row_count = self._peak_row_count(sweep.extracted_peak_parameters)
            if row_count <= 0:
                continue

            qualifiers = sweep.qualifiers
            metadata_columns["MeasurementName"].extend([measurement.measurement_name] * row_count)
            metadata_columns["SweepName"].extend([sweep.file_name] * row_count)
            metadata_columns["Date"].extend([str(qualifiers.get("Date", ""))] * row_count)
            metadata_columns["SourceTime"].extend([str(qualifiers.get("SourceTime", ""))] * row_count)
            metadata_columns["MeasurementIndex"].extend([int(qualifiers.get("MeasurementIndex", 0))] * row_count)
            metadata_columns["CurveIndex"].extend([int(qualifiers.get("CurveIndex", 0))] * row_count)
            metadata_columns["MeasurementTitle"].extend([str(qualifiers.get("MeasurementTitle", ""))] * row_count)
            metadata_columns["SourceStem"].extend([str(qualifiers.get("SourceStem", ""))] * row_count)
            metadata_columns["ScanOrder"].extend([int(qualifiers.get("ScanOrder", 0))] * row_count)

        for column_name, values in metadata_columns.items():
            if len(values) == len(table):
                table[column_name] = values

        return table

    def _add_scan_order(self, extracted_table: pd.DataFrame, rank_column: str) -> pd.DataFrame:
        """Attach a stable scan-order column used by drift visualizations.

        :param extracted_table: Extracted parameter table.
        :type extracted_table: pd.DataFrame
        :param rank_column: Fallback column used to derive a dense rank if
            ``ScanOrder`` is unavailable.
        :type rank_column: str
        :return: Table with ``Scan Order`` column added when possible.
        :rtype: pd.DataFrame
        """
        # Prefer loader-provided deterministic order when available so scan
        # sequence remains stable across re-runs and timestamp formats.
        if "ScanOrder" in extracted_table.columns:
            table = extracted_table.copy()
            table["Scan Order"] = table["ScanOrder"].astype(int)
            return table

        # If no suitable ranking column exists, return unchanged input table.
        if rank_column not in extracted_table.columns:
            return extracted_table

        # Fallback path: derive a dense integer order from the chosen column.
        table = extracted_table.copy()
        table["Scan Order"] = table[rank_column].rank(method="dense").astype(int)
        return table

    @staticmethod
    def _peak_row_count(peak_info: dict[str, np.ndarray]) -> int:
        """Return the first available row-count in a peak-info payload.

        :param peak_info: Dictionary of per-peak arrays produced by feature
            extraction.
        :type peak_info: dict[str, np.ndarray]
        :return: Number of rows represented by the first sized value.
        :rtype: int
        """
        for values in peak_info.values():
            try:
                return len(values)
            except TypeError:
                continue
        return 0

    def _render_selected_visualizations(
        self,
        measurement: Measurement,
        extracted_table: pd.DataFrame,
        visualization_dir: Path,
        config: PipelineConfig,
        measurement_index: int,
    ) -> None:
        """Dispatch optional visualization renderers for one measurement.

        :param measurement: Measurement with processed sweeps.
        :type measurement: Measurement
        :param extracted_table: Extracted parameter table for plotting.
        :type extracted_table: pd.DataFrame
        :param visualization_dir: Destination directory for image outputs.
        :type visualization_dir: Path
        :param config: Runtime visualization switches.
        :type config: PipelineConfig
        :param measurement_index: Filename tag used to prevent collisions in
            shared output folders.
        :type measurement_index: int
        :return: None
        :rtype: None
        """
        # Pairplot mode intentionally emits two related visual products:
        # the pairwise matrix and the 2D peak-coordinate density view.
        if config.visualization.pairplot:
            render_pairplot(extracted_table, visualization_dir, measurement_index)
            render_peak_coordinate_density(extracted_table, visualization_dir, measurement_index)

        # Drift views are independent toggles so users can run either form
        # (2D scatter or 1D dual-axis) without requiring the other.
        if config.visualization.drift_2d:
            render_2d_peak_drift(extracted_table, visualization_dir, measurement_index)

        if config.visualization.drift_1d:
            render_1d_peak_drift(extracted_table, visualization_dir, measurement_index)

        # Overlay plots consume sweep-level processed traces, so they are
        # rendered from the measurement object rather than extracted_table.
        if config.visualization.overlay_scans:
            render_overlay_scans(measurement, visualization_dir, measurement_index)

        # Preprocessing overview shows raw/smoothed/baseline and corrected side
        # by side for diagnostic review of signal processing quality.
        if config.visualization.preprocessing_overview:
            render_preprocessing_overview(measurement, visualization_dir, measurement_index)

    def _save_processed_sweep(self, processed_dir: Path, sweep: SweepData, measurement_index: int, curve_index: int) -> None:
        """Persist one processed sweep CSV with a collision-safe filename tag.

        :param processed_dir: Directory for processed sweep CSV files.
        :type processed_dir: Path
        :param sweep: Sweep containing ``processed_frame``.
        :type sweep: SweepData
        :param measurement_index: Measurement index tag for output naming.
        :type measurement_index: int
        :param curve_index: Curve index tag for output naming.
        :type curve_index: int
        :return: None
        :rtype: None
        """
        # No-op for sweeps that were skipped upstream or failed preprocessing.
        if sweep.processed_frame is None:
            return

        # Keep the historical stem while prefixing measurement/curve tags to
        # support shared-folder output layouts.
        stem = Path(sweep.file_name).stem
        tagged_filename = f"{measurement_index}_{curve_index}_{stem}_PROCESSED.csv"
        sweep.processed_frame.to_csv(processed_dir / tagged_filename, index=False)

    def _save_sweep_info_pickle(self, sweep_info_dir: Path, sweep: SweepData, measurement_index: int, curve_index: int) -> None:
        """Persist one sweep metadata pickle with tagged filename.

        :param sweep_info_dir: Directory for per-sweep metadata pickle files.
        :type sweep_info_dir: Path
        :param sweep: Sweep whose metadata/extraction payload is persisted.
        :type sweep: SweepData
        :param measurement_index: Measurement index tag for output naming.
        :type measurement_index: int
        :param curve_index: Curve index tag for output naming.
        :type curve_index: int
        :return: None
        :rtype: None
        """
        # Build deterministic filename tags first so we can safely write into
        # shared folders across all measurements in one source file.
        stem = Path(sweep.file_name).stem
        tagged_filename = f"{measurement_index}_{curve_index}_{stem}_sweep_info.pkl"

        # Persist enough context to reconstruct sweep provenance and debug
        # extraction outputs without re-running preprocessing.
        save_pickle(
            {
                "file": sweep.file_name,
                "qualifiers": sweep.qualifiers,
                "average_baseline": sweep.average_baseline,
                "extracted_peak_parameters": sweep.extracted_peak_parameters,
            },
            sweep_info_dir / tagged_filename,
        )

    def _save_batch_outputs(
        self,
        config: PipelineConfig,
        measurement_names: list[str],
        all_extracted_tables: list[pd.DataFrame],
    ) -> None:
        """Persist pooled extracted parameters for batch runs.

        Notes:
            ``pandas.concat`` uses the union of all input columns. Columns
            missing from individual measurement tables are filled with ``NaN``
            in the pooled output.
        """
        batch_root = config.output_path if config.output_path is not None else config.input_path
        batch_analysis_folder = batch_root / "Batch Analysis"
        pooled_params_folder = batch_analysis_folder / "pooled_extracted_params"
        pooled_params_folder.mkdir(parents=True, exist_ok=True)

        tagged_tables: list[pd.DataFrame] = []
        for measurement_name, table in zip(measurement_names, all_extracted_tables, strict=False):
            tagged = table.copy()
            tagged["Measurement"] = np.repeat(measurement_name, len(tagged))
            tagged_tables.append(tagged)

        pooled = pd.concat(tagged_tables, ignore_index=True)
        pooled.to_csv(pooled_params_folder / "pooled_extracted_params.csv", index=False)
