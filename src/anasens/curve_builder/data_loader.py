"""Data loading utilities for Plotter Fitter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import LoadedDataset


class DataLoader:
    """Load supported tabular file formats into pandas DataFrames.

    The loader is isolated from GUI concerns so the same code can be reused
    by tests, scripts, and future non-GUI interfaces.
    """

    # Supported suffixes are kept in one place so validation and file-dialog
    # configuration can stay aligned as the package evolves.
    SUPPORTED_SUFFIXES = {".csv", ".xls", ".xlsx", ".json", ".pkl", ".pickle"}

    def load(self, file_path: str) -> LoadedDataset:
        """Load a file into a DataFrame-backed dataset object.

        Args:
            file_path: Path to a supported tabular data file.

        Returns:
            A LoadedDataset containing the original path and normalized DataFrame.

        Raises:
            ValueError: If the file extension is unsupported.
            TypeError: If a pickle file does not contain a pandas DataFrame.
        """
        # The path object centralizes suffix handling and creates a clear place
        # to validate file-type support before attempting any I/O operation.
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Each branch maps exactly one storage format into a DataFrame. This
        # keeps the function explicit and easy to extend for future formats.
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".xls", ".xlsx"}:
            frame = pd.read_excel(path)
        elif suffix == ".json":
            frame = pd.read_json(path)
        else:
            frame = pd.read_pickle(path)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Pickle file must contain a pandas DataFrame.")

        # A defensive copy prevents later GUI mutations from aliasing the object
        # returned by pandas' file readers in surprising ways.
        normalized_frame = pd.DataFrame(frame).copy()

        return LoadedDataset(source_path=str(path), frame=normalized_frame)
