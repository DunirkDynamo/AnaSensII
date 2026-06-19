"""CSV loading utilities."""

from pathlib import Path

import pandas as pd


def load_csv_measurement(folder: Path) -> list[dict[str, object]]:
    """Load all CSV sweeps from a measurement directory."""
    rows: list[dict[str, object]] = []
    for file in sorted(folder.glob("*.csv")):
        rows.append({"voltammogram": pd.read_csv(file), "file": file.name})
    return rows
