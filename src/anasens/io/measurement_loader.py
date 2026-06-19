"""Measurement loading services for native pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from anasens.domain import Measurement, SweepData
from anasens.errors import ConfigurationError
from anasens.io.csv_loader import load_csv_measurement


@dataclass(slots=True)
class FilenameQualifierParser:
    """Parse metadata qualifiers from legacy-formatted sweep filenames."""

    def parse(self, file_name: str) -> dict[str, object]:
        parts = file_name.split("_")
        if len(parts) < 2:
            return {"Date": "", "Q1": "", "Label": "", "Timestamp": 0.0}

        subparts = parts[1].split("-")
        if len(subparts) < 3:
            return {"Date": parts[0], "Q1": "", "Label": "", "Timestamp": 0.0}

        timestamp_text = subparts[2]
        if timestamp_text.endswith("s"):
            timestamp_text = timestamp_text[:-1]

        try:
            timestamp_value = float(timestamp_text)
        except ValueError:
            timestamp_value = 0.0

        return {
            "Date": parts[0],
            "Q1": subparts[0],
            "Label": subparts[1],
            "Timestamp": timestamp_value,
        }


@dataclass(slots=True)
class CsvMeasurementLoader:
    """Load measurement directories into typed domain objects."""

    qualifier_parser: FilenameQualifierParser = field(default_factory=FilenameQualifierParser)

    def load_measurements(self, input_folder: Path, analysis_mode: str) -> list[Measurement]:
        """Load single or batch measurements from an input folder."""
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
