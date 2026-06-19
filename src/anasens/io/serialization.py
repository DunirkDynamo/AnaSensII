"""Serialization helpers."""

from pathlib import Path

import pandas as pd


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV with index disabled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
