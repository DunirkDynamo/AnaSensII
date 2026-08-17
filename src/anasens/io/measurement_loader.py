"""Measurement loading services for native pipeline execution.

This module centralizes source-specific ingestion and normalizes loaded data
into shared domain models (:class:`Measurement` and :class:`SweepData`).
The native pipeline can therefore stay source-agnostic.

The normalization contract is:
1. Resolve input paths according to source and mode (single or batch).
2. Parse raw source payloads (CSV, JSON, or pssession-derived JSON).
3. Build typed measurements and sweeps with workflow-relevant qualifiers.

The output of this module is consumed directly by
``anasens.pipeline.native_pipeline.NativeAnalysisPipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from anasens.config import InputSource
from anasens.domain import Measurement, SweepData
from anasens.errors import ConfigurationError
from anasens.io.csv_loader import load_csv_measurement
from anasens.io.pssession_json import PSSessionConversionService


@dataclass(slots=True)
class FilenameQualifierParser:
    """Parse qualifiers from legacy CSV sweep filenames.

    This parser exists strictly for backward compatibility with legacy CSV
    layouts that encode metadata in file names.

    Example:
           A filename like ``2025-01-01_batch-Control-36s_RAW.csv`` can be parsed into
           a qualifier dictionary containing ``Date``, ``Label``, and
        ``Timestamp``.
    """

    def parse(self, file_name: str) -> dict[str, object]:
        """Parse one legacy filename into a qualifier dictionary.

        Args:
            file_name: Sweep filename string from legacy CSV workflow.

        Returns:
            dict[str, object]: Parsed qualifiers. Missing or malformed parts
            fall back to empty strings and ``0.0`` timestamps.
        """
        # Split the filename into legacy date and qualifier segments.
        parts = file_name.split("_")
        if len(parts) < 2:
            return {"Date": "", "Label": "", "Timestamp": 0.0}

        # Parse the expected qualifier segment ``legacy-tag-Label-Time`` and keep only
        # the label and timestamp fields still used by the native pipeline.
        subparts = parts[1].split("-")
        if len(subparts) < 3:
            return {"Date": parts[0], "Label": "", "Timestamp": 0.0}

        timestamp_text = subparts[2]
        if timestamp_text.endswith("s"):
            timestamp_text = timestamp_text[:-1]

        try:
            timestamp_value = float(timestamp_text)
        except ValueError:
            timestamp_value = 0.0

        return {
            "Date": parts[0],
            "Label": subparts[1],
            "Timestamp": timestamp_value,
        }


@dataclass(slots=True)
class CsvMeasurementLoader:
    """Load legacy CSV measurement directories into typed domain objects.

    The legacy CSV structure is directory-based and treated differently from
    JSON/pssession sources. This class preserves that legacy behavior while
    mapping to the same typed output consumed by the native pipeline.
    """

    qualifier_parser: FilenameQualifierParser = field(default_factory=FilenameQualifierParser)

    def load_measurements(self, input_folder: Path, analysis_mode: str) -> list[Measurement]:
        """Load measurements from a legacy CSV directory structure.

        Args:
            input_folder: Root directory for CSV inputs.
            analysis_mode: ``"single"`` for one measurement directory,
                ``"batch"`` for a directory containing measurement subfolders.

        Returns:
            list[Measurement]: Normalized measurements for pipeline processing.

        Raises:
            ConfigurationError: If the input path or mode is invalid.
        """
        if not input_folder.exists() or not input_folder.is_dir():
            raise ConfigurationError(f"Input path is not a directory: {input_folder}")

        if analysis_mode == "single":
            return [self._load_one_measurement(input_folder, input_folder.name)]

        if analysis_mode == "batch":
            measurements: list[Measurement] = []
            subfolders = sorted([path for path in input_folder.iterdir() if path.is_dir()])
            if not subfolders:
                raise ConfigurationError("Batch mode requires subfolders, each containing CSV sweeps.")

            for folder in subfolders:
                measurements.append(self._load_one_measurement(folder, folder.name))
            return measurements

        raise ConfigurationError(f"Unsupported analysis mode: {analysis_mode}")

    def _load_one_measurement(self, folder: Path, measurement_name: str) -> Measurement:
        """Load one legacy CSV measurement folder.

        Args:
            folder: Directory containing sweep CSV files.
            measurement_name: Name assigned to the resulting measurement.

        Returns:
            Measurement: Measurement with normalized sweeps.
        """
        sweep_rows = load_csv_measurement(folder)
        if not sweep_rows:
            raise ConfigurationError(f"No CSV sweeps found in: {folder}")

        sweeps: list[SweepData] = []
        for row in sweep_rows:
            file_name = str(row["file"])
            qualifiers = self.qualifier_parser.parse(file_name)
            sweeps.append(
                SweepData(
                    file_name=file_name,
                    raw_frame=row["voltammogram"],
                    qualifiers=qualifiers,
                )
            )

        return Measurement(measurement_name=measurement_name, sweeps=sweeps)


@dataclass(slots=True)
class SessionMeasurementLoader:
    """Load JSON and pssession sources into typed domain objects.

    This loader implements a metadata-first ingestion strategy: it uses source
    metadata when available, but does not require metadata to make analysis
    decisions. The output remains structurally stable even when metadata fields
    vary across files.

    Example:
        In single mode, a single JSON file containing N measurements returns a
        list of N :class:`Measurement` objects.
    """

    pssession_service: PSSessionConversionService = field(default_factory=PSSessionConversionService)

    def load_from_json_path(self, input_path: Path, analysis_mode: str) -> list[Measurement]:
        """Load measurements from extracted JSON payload files.

        Args:
            input_path: JSON file path or directory of JSON files.
            analysis_mode: ``"single"`` or ``"batch"``.

        Returns:
            list[Measurement]: Parsed and normalized measurements.
        """
        if analysis_mode == "single":
            payload = self._load_single_json_payload(input_path)
            source_stem = self._source_stem_for_path(input_path)
            return self._payload_to_measurements(payload, source_stem)

        if analysis_mode == "batch":
            json_files = self._resolve_batch_files(input_path, suffixes=(".json",))
            measurements: list[Measurement] = []
            for json_file in json_files:
                payload = self._load_single_json_payload(json_file)
                measurements.extend(self._payload_to_measurements(payload, json_file.stem))
            return measurements

        raise ConfigurationError(f"Unsupported analysis mode: {analysis_mode}")

    def load_from_pssession_path(self, input_path: Path, analysis_mode: str) -> list[Measurement]:
        """Load measurements from direct ``.pssession`` files.

        Args:
            input_path: Pssession file path or directory of pssession files.
            analysis_mode: ``"single"`` or ``"batch"``.

        Returns:
            list[Measurement]: Parsed and normalized measurements.
        """
        if analysis_mode == "single":
            payload = self._parse_single_pssession_payload(input_path)
            source_stem = self._source_stem_for_path(input_path)
            return self._payload_to_measurements(payload, source_stem)

        if analysis_mode == "batch":
            session_files = self._resolve_batch_files(input_path, suffixes=(".pssession",))
            measurements: list[Measurement] = []
            for session_file in session_files:
                payload = self._parse_single_pssession_payload(session_file)
                measurements.extend(self._payload_to_measurements(payload, session_file.stem))
            return measurements

        raise ConfigurationError(f"Unsupported analysis mode: {analysis_mode}")

    def _load_single_json_payload(self, input_path: Path) -> dict[str, Any]:
        """Load one JSON payload object.

        In single mode, callers may provide either a direct JSON file path or a
        directory containing exactly one JSON file.
        """
        if input_path.is_dir():
            json_files = sorted(input_path.glob("*.json"))
            if not json_files:
                raise ConfigurationError(f"No JSON files found in: {input_path}")
            if len(json_files) > 1:
                raise ConfigurationError(
                    "Single mode with a JSON folder requires exactly one JSON file."
                )
            input_path = json_files[0]

        if input_path.suffix.lower() != ".json":
            raise ConfigurationError(f"Expected a .json file, got: {input_path}")

        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigurationError(f"Failed to parse JSON file {input_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ConfigurationError(f"JSON payload is not an object in file: {input_path}")
        return payload

    def _parse_single_pssession_payload(self, input_path: Path) -> dict[str, Any]:
        """Parse one pssession file into canonical payload format.

        In single mode, callers may provide either a direct pssession file path
        or a directory containing exactly one pssession file.
        """
        if input_path.is_dir():
            pssession_files = sorted(input_path.glob("*.pssession"))
            if not pssession_files:
                raise ConfigurationError(f"No .pssession files found in: {input_path}")
            if len(pssession_files) > 1:
                raise ConfigurationError(
                    "Single mode with a pssession folder requires exactly one .pssession file."
                )
            input_path = pssession_files[0]

        if input_path.suffix.lower() != ".pssession":
            raise ConfigurationError(f"Expected a .pssession file, got: {input_path}")

        try:
            return self.pssession_service.parse_file(input_path)
        except Exception as exc:
            raise ConfigurationError(f"Failed to parse pssession file {input_path}: {exc}") from exc

    def _resolve_batch_files(self, input_path: Path, suffixes: tuple[str, ...]) -> list[Path]:
        """Resolve batch inputs to concrete files matching allowed suffixes.

        Args:
            input_path: File or directory selected by user.
            suffixes: Allowed suffix tuple, e.g. ``(".json",)``.

        Returns:
            list[Path]: Ordered list of files to process.
        """
        if not input_path.exists():
            raise ConfigurationError(f"Input path does not exist: {input_path}")

        if input_path.is_file():
            if input_path.suffix.lower() not in suffixes:
                raise ConfigurationError(f"Unsupported file type for batch mode: {input_path}")
            return [input_path]

        files: list[Path] = []
        for suffix in suffixes:
            files.extend(sorted(input_path.glob(f"*{suffix}")))

        if not files:
            raise ConfigurationError(f"No matching files found in: {input_path}")

        return files

    @staticmethod
    def _source_stem_for_path(input_path: Path) -> str:
        if input_path.is_file():
            return input_path.stem
        return input_path.name

    def _payload_to_measurements(self, payload: dict[str, Any], source_stem: str) -> list[Measurement]:
        """Convert canonical payload to measurement domain objects.

        Args:
            payload: Canonical payload with ``measurements`` list.
            source_stem: Source stem used for stable naming.

        Returns:
            list[Measurement]: Normalized measurement list.
        """
        raw_measurements = payload.get("measurements", [])
        if not isinstance(raw_measurements, list) or not raw_measurements:
            raise ConfigurationError("Payload does not contain any measurements.")

        measurements: list[Measurement] = []
        for index, raw_measurement in enumerate(raw_measurements):
            if not isinstance(raw_measurement, dict):
                continue

            measurement_name = self._build_measurement_name(raw_measurement, source_stem, index)
            sweeps = self._measurement_to_sweeps(raw_measurement, source_stem, index)
            if sweeps:
                measurements.append(Measurement(measurement_name=measurement_name, sweeps=sweeps))

        if not measurements:
            raise ConfigurationError("No valid measurement objects could be created from payload.")

        return measurements

    def _measurement_to_sweeps(
        self,
        raw_measurement: dict[str, Any],
        source_stem: str,
        measurement_index: int,
    ) -> list[SweepData]:
        """Convert one raw measurement dictionary into sweep objects.

        The resulting sweeps include both structural identifiers
        (measurement/curve index) and optional metadata context (title/time).

        Args:
            raw_measurement: Source measurement payload.
            source_stem: Source stem used for stable sweep naming.
            measurement_index: Fallback index when metadata is missing.

        Returns:
            list[SweepData]: Parsed sweeps ready for preprocessing/extraction.
        """
        title = str(raw_measurement.get("title") or "").strip()
        time_text = raw_measurement.get("time")
        date_text = self._date_from_time(time_text)
        measurement_order = self._to_int(raw_measurement.get("measurement_index"), measurement_index)
        label = title if title else f"measurement_{measurement_order:03d}"

        curves = raw_measurement.get("curves", [])
        sweep_candidates: list[tuple[list[float], list[float], int]] = []

        if isinstance(curves, list) and curves:
            for curve in curves:
                if not isinstance(curve, dict):
                    continue
                potential = self._to_float_list(curve.get("potential", []))
                current = self._to_float_list(curve.get("current", []))
                curve_index = int(curve.get("curve_index", len(sweep_candidates)))
                sweep_candidates.append((potential, current, curve_index))
        else:
            potential = self._to_float_list(raw_measurement.get("potential", []))
            current = self._to_float_list(raw_measurement.get("current", []))
            sweep_candidates.append((potential, current, 0))

        sweeps: list[SweepData] = []
        for potential, current, curve_index in sweep_candidates:
            # Keep only aligned data pairs and skip undersized traces that cannot
            # support downstream smoothing/peak extraction reliably.
            min_len = min(len(potential), len(current))
            if min_len < 3:
                continue

            frame = pd.DataFrame(
                {
                    "V": potential[:min_len],
                    "I": current[:min_len],
                }
            )

            # Use a deterministic scan order so ordering remains stable even
            # when time metadata is absent or non-numeric.
            scan_order = measurement_order * 1000 + curve_index
            qualifiers = {
                "Date": date_text,
                "Label": label,
                "Timestamp": float(scan_order),
                "SourceTime": time_text,
                "MeasurementIndex": measurement_order,
                "CurveIndex": curve_index,
                "MeasurementTitle": title,
                "SourceStem": source_stem,
                "ScanOrder": scan_order,
            }

            sweeps.append(
                SweepData(
                    file_name=self._build_sweep_name(source_stem, title, measurement_order, curve_index),
                    raw_frame=frame,
                    qualifiers=qualifiers,
                )
            )

        return sweeps

    def _build_measurement_name(self, raw_measurement: dict[str, Any], source_stem: str, fallback_index: int) -> str:
        """Build a stable, metadata-aware measurement name.

        Example:
            ``sample_file__square_wave_voltammetry__2026_04_29_13_21_27__m003``
        """
        measurement_order = self._to_int(raw_measurement.get("measurement_index"), fallback_index)
        title = str(raw_measurement.get("title") or "").strip()
        time_text = str(raw_measurement.get("time") or "").strip()

        parts = [self._slugify(source_stem) or "source"]
        title_slug = self._slugify(title)
        if title_slug:
            parts.append(title_slug)
        time_slug = self._slugify(time_text)
        if time_slug:
            parts.append(time_slug)
        parts.append(f"m{measurement_order:03d}")
        return "__".join(parts)

    def _build_sweep_name(self, source_stem: str, title: str, measurement_index: int, curve_index: int) -> str:
        stem_parts = [self._slugify(source_stem) or "source"]
        title_slug = self._slugify(title)
        if title_slug:
            stem_parts.append(title_slug)
        stem_parts.append(f"m{measurement_index:03d}")
        stem_parts.append(f"curve_{curve_index:03d}")
        return "__".join(stem_parts)

    @staticmethod
    def _to_float_list(values: Any) -> list[float]:
        if not isinstance(values, list):
            return []

        result: list[float] = []
        for value in values:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _date_from_time(time_text: Any) -> str:
        if not isinstance(time_text, str) or not time_text.strip():
            return ""

        text = time_text.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                return parsed.date().isoformat()
            except ValueError:
                continue

        return text.split(" ")[0]

    @staticmethod
    def _slugify(value: str) -> str:
        text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
        return text.strip("_")

    @staticmethod
    def _to_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback


@dataclass(slots=True)
class MeasurementLoader:
    """Source-aware measurement loader facade for pipeline execution."""

    csv_loader: CsvMeasurementLoader = field(default_factory=CsvMeasurementLoader)
    session_loader: SessionMeasurementLoader = field(default_factory=SessionMeasurementLoader)

    def load_measurements(self, input_path: Path, analysis_mode: str, input_source: InputSource) -> list[Measurement]:
        """Dispatch measurement loading based on selected input source.

        Args:
            input_path: File or directory selected in the UI.
            analysis_mode: ``"single"`` or ``"batch"``.
            input_source: ``"legacy_csv"``, ``"json"``, or ``"pssession"``.

        Returns:
            list[Measurement]: Source-normalized measurement objects.

        Raises:
            ConfigurationError: If source/mode/path combination is unsupported.
        """
        if input_source == "legacy_csv":
            if not input_path.is_dir():
                raise ConfigurationError("Legacy CSV mode expects a directory input path.")
            return self.csv_loader.load_measurements(input_path, analysis_mode)

        if input_source == "json":
            return self.session_loader.load_from_json_path(input_path, analysis_mode)

        if input_source == "pssession":
            return self.session_loader.load_from_pssession_path(input_path, analysis_mode)

        raise ConfigurationError(f"Unsupported input source: {input_source}")
