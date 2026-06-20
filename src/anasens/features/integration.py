"""Numerical integration utilities for peak area calculation.

Simpson's rule is used because it is more accurate than the trapezoidal
rule for smooth peak shapes while remaining fast for the array sizes
typical in voltammogram scans (~100-2000 points).
"""

import numpy as np
from scipy.integrate import simpson


def integrate_simpson(
    x: np.ndarray,
    y: np.ndarray,
    x_min: int,
    x_max: int,
) -> float:
    """Integrate a signal slice using composite Simpson's rule.

    The slice boundaries are inclusive sample indices (i.e.
    ``x[x_min : x_max + 1]`` is used).  When the slice length is even,
    the last point is dropped to satisfy the odd-point requirement of
    Simpson's rule; this matches the legacy pipeline behaviour.

    Args:
        x: Full potential array for the scan.
        y: Full current array for the scan (may be raw, smoothed, or
           baseline-corrected depending on caller context).
        x_min: Left (inclusive) sample index of the integration window,
            typically a peak left-edge index.
        x_max: Right (inclusive) sample index of the integration window,
            typically a peak right-edge index.

    Returns:
        float: Signed area under the curve in the slice window
        (units: volt × current units).

    Raises:
        ValueError: If the slice has fewer than 3 points after any
            even-length trimming.

    Example::

        >>> import numpy as np
        >>> x = np.linspace(0, 1, 101)
        >>> y = np.ones(101)         # area = 1.0
        >>> integrate_simpson(x, y, 0, 100)
        1.0
    """
    # Extract the inclusive slice from both arrays.
    x_subset = x[x_min : (x_max + 1)]
    y_subset = y[x_min : (x_max + 1)]

    # Simpson's rule requires at least 3 points.
    if len(x_subset) < 3:
        raise ValueError("Simpson rule requires at least 3 data points.")

    # Simpson's composite rule requires an odd number of points.
    # Drop the last point when even to match legacy pipeline behaviour.
    if len(x_subset) % 2 == 0:
        x_subset = x_subset[:-1]
        y_subset = y_subset[:-1]

    return float(simpson(y_subset, x_subset))
