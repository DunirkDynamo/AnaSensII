"""Smoothing operations."""

import numpy as np
from scipy.signal import savgol_filter


def smooth_voltammogram(current: np.ndarray, window_length: int = 17, polyorder: int = 6) -> np.ndarray:
    """Apply Savitzky-Golay smoothing."""
    n_points = int(len(current))
    if n_points < 3:
        return current.copy()

    window = max(3, int(window_length))
    if window % 2 == 0:
        window += 1
    window = min(window, n_points if n_points % 2 == 1 else n_points - 1)
    window = max(3, window)

    order = max(1, int(polyorder))
    if order >= window:
        order = window - 1

    return savgol_filter(current, window, order)
