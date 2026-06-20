"""Baseline subtraction for voltammogram current traces.

Two algorithms are supported, selected via the ``algo`` argument:

* ``"snip"`` — SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
  via :mod:`pybaselines`.  Robust for SWV scans with well-separated peaks.
* ``"moving_average"`` — iterative moving-average subtraction via :mod:`Ekidna`.
  Suited to asymmetric baselines or longer scans.

The caller receives both the baseline-corrected signal and the baseline
array itself so both can be stored in the processed sweep DataFrame.
"""

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
    """Subtract the estimated baseline from a smoothed current trace.

    Args:
        potential: 1-D potential array (V) for the scan.  Required by the
            pybaselines SNIP estimator as the x-axis data.
        current: 1-D smoothed current array (A or nA) to be corrected.
        algo: Algorithm selector.  ``"snip"`` or ``"moving_average"``.
        snip_expected_fwhm: Approximate FWHM of peaks in data-point units,
            used by the SNIP estimator to set its internal filter scale.
        snip_window_size: Half-window size (``max_half_window``) for SNIP.
        moving_average_window_size: Rolling window width for Ekidna.
        moving_average_forward: Whether Ekidna uses a forward-direction
            rolling average.
        moving_average_max_iter: Convergence iteration cap for Ekidna.

    Returns:
        tuple[np.ndarray, np.ndarray]: A 2-tuple of
        ``(corrected_current, baseline)`` where ``corrected_current =
        current - baseline``.

    Example::

        >>> import numpy as np
        >>> v = np.linspace(-0.5, 0.5, 200)
        >>> i_smooth = np.sin(v * np.pi) + 0.1 * v
        >>> corrected, baseline = baseline_subtraction(v, i_smooth, algo="snip")
        >>> corrected.shape == i_smooth.shape
        True
    """
    # Guard against empty inputs that would crash the estimators.
    if len(current) == 0:
        return current.copy(), current.copy()

    # Normalise the algorithm name for case-insensitive comparison.
    method = str(algo).strip().lower()

    if method == "moving_average":
        # ── Ekidna moving-average path ────────────────────────────────────
        # Ekidna returns a baseline estimate (and may include an additional
        # plotting object/output). We only use the baseline array here.
        result = ek.moving_average_baseline_subtraction(
            current,
            max(1, int(moving_average_window_size)),
            forward=bool(moving_average_forward),
            max_iter=max(1, int(moving_average_max_iter)),
        )

        if isinstance(result, tuple) and len(result) >= 2:
            baseline = np.asarray(result[0], dtype=float)
        else:
            baseline = np.asarray(result, dtype=float)

        # Keep the identity explicit for downstream processed-frame columns:
        # corrected voltammogram is always smoothed current minus baseline.
        corrected = current - baseline

        # User-requested edge correction for non-integral window coverage:
        # points not covered by complete moving-average windows are zeroed in
        # the corrected trace to avoid trailing edge artifacts.
        window = max(1, int(moving_average_window_size))
        covered_points = (len(corrected) // window) * window
        if covered_points < len(corrected):
            corrected = corrected.copy()
            corrected[covered_points:] = 0.0

        return corrected, baseline

    # ── pybaselines SNIP path (default) ─────────────────────────────────
    # Initialise the Baseline estimator with the potential axis so the
    # estimator can apply x-aware peak-width scaling.
    baseline_fitter = Baseline(x_data=potential)

    # decreasing=True runs the SNIP filter in the conventional
    # decreasing-window mode recommended for peak-containing signals.
    baseline, _ = baseline_fitter.snip(
        current,
        max_half_window=max(1, int(snip_window_size)),
        decreasing=True,
    )

    return current - baseline, baseline
