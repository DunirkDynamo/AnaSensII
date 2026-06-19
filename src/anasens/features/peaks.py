"""Peak detection utilities."""

import numpy as np
from scipy.signal import find_peaks, peak_widths


def find_peaks_with_widths(
    x: np.ndarray,
    y: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    width: int = 20,
) -> dict[str, np.ndarray]:
    """Find peaks and return widths and edge indices."""
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_filtered = x[mask]
    y_filtered = y[mask]

    peaks, _ = find_peaks(
        y_filtered,
        width=width,
        rel_height=0.5,
        prominence=0.1 * (max(y) - min(y)),
        distance=10,
    )

    corrected_peaks = peaks + np.where(mask)[0][0]
    widths, _, left_edge, right_edge = peak_widths(y, corrected_peaks, rel_height=0.5)
    widths_01, _, left_edge_01, right_edge_01 = peak_widths(y, corrected_peaks, rel_height=0.9)

    return {
        "peak_indices": corrected_peaks.astype(int),
        "peak_heights": y_filtered[peaks],
        "peak_potentials": x_filtered[peaks],
        "FWHM": widths,
        "FW0.1M": widths_01,
        "left_edge_indices": np.round(left_edge).astype(int),
        "right_edge_indices": np.round(right_edge).astype(int),
        "left_edge_01_indices": np.round(left_edge_01).astype(int),
        "right_edge_01_indices": np.round(right_edge_01).astype(int),
    }
