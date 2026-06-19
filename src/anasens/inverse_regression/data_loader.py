"""Data loading services for inverse regression workflows."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from anasens.inverse_regression.models import VoltammogramCalibrationInput


class InverseDataLoader:
    """Load inverse-estimation inputs from pickle files."""

    def load_voltammograms_from_folder(self, folder: Path) -> list[VoltammogramCalibrationInput]:
        """Load voltammograms that contain calibration tables from a folder of pickle files."""
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid voltammogram folder: {folder}")

        voltammograms: list[VoltammogramCalibrationInput] = []
        for file_path in sorted(folder.iterdir()):
            if file_path.suffix.lower() not in {".pkl", ".pickle"}:
                continue

            with file_path.open("rb") as file_handle:
                loaded = pickle.load(file_handle)

            if not isinstance(loaded, dict):
                continue

            calibration_table = loaded.get("calibration_table")
            if not isinstance(calibration_table, pd.DataFrame):
                continue

            display_name = str(loaded.get("name", file_path.stem))
            voltammograms.append(
                VoltammogramCalibrationInput(
                    name=display_name,
                    source_path=str(file_path),
                    calibration_table=calibration_table.copy(),
                )
            )

        if not voltammograms:
            raise ValueError("No valid voltammogram pickle files with calibration_table were found.")

        return voltammograms

    def load_standard_curve(self, file_path: Path):
        """Load and return the pickled standard curve object."""
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(f"Invalid standard curve file: {file_path}")

        with file_path.open("rb") as file_handle:
            return pickle.load(file_handle)
