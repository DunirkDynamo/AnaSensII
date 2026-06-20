"""Extracted-parameter table construction utilities.

Builds the per-measurement DataFrame that aggregates peak dictionaries
from all sweeps with their associated labels and timestamps.  This table
is the primary output of the extraction stage and feeds visualizations,
Calibrator, and batch pooling.
"""

import numpy as np
import pandas as pd


def build_analyte_data_table(
    labels: np.ndarray,
    peak_info: list[dict[str, np.ndarray]],
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """Concatenate per-sweep peak dictionaries into a single measurement table.

    Each sweep may produce zero or more detected peaks.  Every peak row
    in the output table is tagged with the sweep's label and timestamp
    so rows can be traced back to their originating sweep.

    Args:
        labels: 1-D string array, one element per sweep, used to
            populate the ``Label`` column.
        peak_info: List of peak-parameter dicts (one per sweep) as
            returned by
            :func:`~anasens.features.peaks.find_peaks_with_widths` and
            enriched with integral arrays by the pipeline.
        timestamps: 1-D float array of sweep timestamps (or scan-order
            proxies), one element per sweep.

    Returns:
        pd.DataFrame: Combined table with one row per detected peak.
        Columns are the union of all keys in ``peak_info`` plus
        ``Timestamp`` and ``Label``.

    Note:
        Metadata columns such as ``MeasurementName``, ``SourceTime``, and
        ``CurveIndex`` are appended by the pipeline *after* this function
        returns, not inside it, to keep this utility source-agnostic.
    """
    # ── Seed the table from the first sweep ───────────────────────────────
    data_table = pd.DataFrame(peak_info[0])
    n0 = len(data_table)  # Row count for sweep 0 (one row per peak).
    data_table["Timestamp"] = np.repeat(timestamps[0], n0)
    data_table["Label"] = np.repeat(labels[0], n0)

    # ── Append remaining sweeps ────────────────────────────────────────
    # pd.concat takes the union of columns so sweeps with different peak
    # counts or missing keys produce NaN entries rather than errors.
    for i in range(1, len(labels)):
        temp_df = pd.DataFrame(peak_info[i])
        n_i = len(temp_df)  # Row count for sweep i.
        temp_df["Timestamp"] = np.repeat(timestamps[i], n_i)
        temp_df["Label"] = np.repeat(labels[i], n_i)
        data_table = pd.concat([data_table, temp_df], ignore_index=True)

    return data_table
