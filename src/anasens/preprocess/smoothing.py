"""Savitzky-Golay smoothing for voltammogram current traces.

Savitzky-Golay filtering is applied to the current signal before
baseline estimation and peak detection.  It reduces high-frequency
noise while preserving peak shape better than simple moving-average
filters because it fits a local polynomial rather than averaging.

Reference:
    Savitzky, A.; Golay, M.J.E. (1964). Analytical Chemistry, 36 (8),
    1627–1639. https://doi.org/10.1021/ac60214a047
"""

import numpy as np
from scipy.signal import savgol_filter


def smooth_voltammogram(
    current: np.ndarray,
    window_length: int = 17,
    polyorder: int = 6,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to one current trace.

    The function enforces scipy's requirements (odd window, window > order,
    window <= n_points) automatically so it never raises on edge-case
    scan lengths or parameter combinations coming from the GUI.

    Args:
        current: 1-D array of current values for one scan.
        window_length: Number of data points in the smoothing window.
            Must be odd and at least 3; automatically incremented when
            even and clamped to the trace length when too large.
        polyorder: Polynomial order for the local fit.  Must be less
            than ``window_length``; automatically decremented when
            violated.

    Returns:
        np.ndarray: Smoothed current array with the same length as
        ``current``.  A copy of the input is returned unchanged when
        the trace has fewer than 3 points.

    Example::

        >>> import numpy as np
        >>> raw = np.array([0.1, 0.3, 0.2, 0.8, 0.7, 0.9, 0.4])
        >>> smoothed = smooth_voltammogram(raw, window_length=5, polyorder=2)
        >>> smoothed.shape
        (7,)
    """
    n_points = int(len(current))

    # Traces shorter than 3 points cannot support any SG window; return as-is.
    if n_points < 3:
        return current.copy()

    # Enforce minimum window size of 3.
    window = max(3, int(window_length))

    # scipy requires an odd window length.
    if window % 2 == 0:
        window += 1

    # Window cannot exceed the trace length (also must stay odd).
    window = min(window, n_points if n_points % 2 == 1 else n_points - 1)
    window = max(3, window)  # Re-apply minimum after clamping.

    # Polynomial order must be positive and strictly less than the window.
    order = max(1, int(polyorder))
    if order >= window:
        order = window - 1

    return savgol_filter(current, window, order)
