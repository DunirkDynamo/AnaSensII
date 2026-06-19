"""Baseline subtraction operations."""

import Ekidna as ek
import numpy as np
from pybaselines import Baseline


def baseline_subtraction(
    potential: np.ndarray,
    current: np.ndarray,
    algo: str = "snip",
    snip_expected_fwhm: int = 35,
    snip_window_size: int = 10,
    moving_average_window_size: int = 10,
    moving_average_forward: bool = True,
    moving_average_max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract baseline using configured pybaselines algorithm."""
    if len(current) == 0:
        return current.copy(), current.copy()

    method = str(algo).strip().lower()

    if method == "moving_average":
        result = ek.moving_average_baseline_subtraction(
            current,
            max(1, int(moving_average_window_size)),
            forward=bool(moving_average_forward),
            max_iter=max(1, int(moving_average_max_iter)),
        )

        if isinstance(result, tuple) and len(result) >= 2:
            corrected = np.asarray(result[0], dtype=float)
            baseline = np.asarray(result[1], dtype=float)
        else:
            corrected = np.asarray(result, dtype=float)
            baseline = current - corrected

        return current - baseline, baseline

    baseline_fitter = Baseline(x_data=potential)

    baseline, _ = baseline_fitter.snip(
        current,
        max_half_window=max(1, int(snip_window_size)),
        decreasing=True,
    )

    return current - baseline, baseline
