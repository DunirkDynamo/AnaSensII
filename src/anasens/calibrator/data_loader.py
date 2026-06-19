"""Data loading utilities for the calibrator module."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from anasens.calibrator.models import MeasurementObject


def load_data_table(file_path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        file_path: Path to a CSV extracted-parameters file.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If the file cannot be read or is not a CSV.
    """
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {file_path.suffix}")
    return pd.read_csv(file_path)


def load_measurement_objects(parent_dir: Path) -> list[MeasurementObject]:
    """Load calibration-ready measurement objects from a parent directory.

    Expects the directory layout produced by the native pipeline:
        parent_dir/
            measurement_name/
                Full Measurement Info/
                    measurement_info.pkl

    Args:
        parent_dir: Root directory containing per-measurement subdirectories.

    Returns:
        List of MeasurementObject instances with extracted_parameters populated.

    Raises:
        ValueError: If no valid measurement objects are found.
    """
    measurement_objects: list[MeasurementObject] = []

    for subdir in sorted(parent_dir.iterdir()):
        if not subdir.is_dir():
            continue

        pkl_path = subdir / "Full Measurement Info" / "measurement_info.pkl"
        if not pkl_path.exists():
            continue

        with pkl_path.open("rb") as handle:
            data = pickle.load(handle)

        if not isinstance(data, dict) or "extracted_parameters" not in data:
            continue

        extracted = data["extracted_parameters"]
        if not isinstance(extracted, pd.DataFrame):
            continue

        measurement_objects.append(
            MeasurementObject(
                measurement_name=subdir.name,
                extracted_parameters=extracted.copy(),
                source_path=str(pkl_path),
            )
        )

    if not measurement_objects:
        raise ValueError(f"No valid measurement objects found in: {parent_dir}")

    return measurement_objects


def save_measurement_objects(measurements: list[MeasurementObject]) -> list[str]:
    """Write calibration_table back into each measurement's source pickle.

    Args:
        measurements: List of MeasurementObject instances with calibration_table set.

    Returns:
        List of file paths that were updated.
    """
    updated: list[str] = []
    for measurement in measurements:
        if measurement.calibration_table is None:
            continue
        pkl_path = Path(measurement.source_path)
        with pkl_path.open("rb") as handle:
            data = pickle.load(handle)
        data["calibration_table"] = measurement.calibration_table
        with pkl_path.open("wb") as handle:
            pickle.dump(data, handle)
        updated.append(str(pkl_path))
    return updated
