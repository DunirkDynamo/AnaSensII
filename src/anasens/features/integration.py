"""Integration utilities."""

import numpy as np
from scipy.integrate import simpson


def integrate_simpson(x: np.ndarray, y: np.ndarray, x_min: int, x_max: int) -> float:
    """Integrate array slice using Simpson's rule, matching legacy index slicing."""
    x_subset = x[x_min : (x_max + 1)]
    y_subset = y[x_min : (x_max + 1)]

    if len(x_subset) < 3:
        raise ValueError("Simpson rule requires at least 3 data points.")

    if len(x_subset) % 2 == 0:
        x_subset = x_subset[:-1]
        y_subset = y_subset[:-1]

    return float(simpson(y_subset, x_subset))
