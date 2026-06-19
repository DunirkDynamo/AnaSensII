"""Data table construction utilities."""

import numpy as np
import pandas as pd


def build_analyte_data_table(
    labels: np.ndarray,
    peak_info: list[dict[str, np.ndarray]],
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """Build extracted-parameter table from per-sweep peak dictionaries."""
    data_table = pd.DataFrame(peak_info[0])
    n0 = len(data_table)
    data_table["Timestamp"] = np.repeat(timestamps[0], n0)
    data_table["Label"] = np.repeat(labels[0], n0)

    for i in range(1, len(labels)):
        temp_df = pd.DataFrame(peak_info[i])
        n_i = len(temp_df)
        temp_df["Timestamp"] = np.repeat(timestamps[i], n_i)
        temp_df["Label"] = np.repeat(labels[i], n_i)
        data_table = pd.concat([data_table, temp_df], ignore_index=True)

    return data_table
