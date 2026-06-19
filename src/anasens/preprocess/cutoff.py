"""Leading-point cutoff operations."""

import pandas as pd


def cut_points(df: pd.DataFrame, cut_start: int = 0, cut_end: int = 0) -> pd.DataFrame:
    """Return DataFrame after removing points from start and end."""
    num_rows = len(df)
    start = max(0, int(cut_start))
    end = max(0, int(cut_end))

    if num_rows == 0:
        return df.copy()

    if start >= num_rows:
        return df.iloc[0:0].copy().reset_index(drop=True)

    stop_index = num_rows - end
    if stop_index <= start:
        return df.iloc[0:0].copy().reset_index(drop=True)

    return df.iloc[start:stop_index].reset_index(drop=True)
