"""Peak detection and width characterisation for voltammogram sweeps.

This module wraps scipy's peak-finding utilities and returns a
dictionary of numpy arrays that the pipeline stores in each
:attr:`~anasens.domain.SweepData.extracted_peak_parameters`.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths


def find_peaks_with_widths(
    x: np.ndarray,
    y: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    width: int = 20,
) -> dict[str, np.ndarray]:
    """Detect peaks within a potential window and characterise their widths.

    Peak detection is constrained to the ``[lower_bound, upper_bound]``
    potential range so edge artefacts outside the region of interest do
    not generate spurious hits.  Width measurement is performed on the
    full (unmasked) array so that edge indices reference the original
    sample positions.

    Args:
        x: Potential array (V), same length as ``y``.
        y: Baseline-corrected current array (A or nA).
        lower_bound: Left potential limit of the search window (inclusive).
        upper_bound: Right potential limit of the search window (inclusive).
        width: Minimum peak width in samples passed to
            :func:`scipy.signal.find_peaks`.  Widens the search to reject
            very narrow noise spikes.

    Returns:
        dict[str, np.ndarray]: Dictionary with the following keys:

        * ``peak_indices`` — sample indices in the original array.
        * ``peak_heights`` — current values at peak positions.
        * ``peak_potentials`` — potential values at peak positions.
        * ``FWHM`` — full-width at half-maximum (samples).
        * ``FW0.1M`` — full-width at 10 % of maximum (samples).
        * ``left_edge_indices`` — left FWHM edge positions (samples).
        * ``right_edge_indices`` — right FWHM edge positions (samples).
        * ``left_edge_01_indices`` — left FW0.1M edge positions.
        * ``right_edge_01_indices`` — right FW0.1M edge positions.

    Example::

        >>> import numpy as np
        >>> x = np.linspace(-0.5, 0.5, 500)
        >>> y = np.exp(-((x - 0.1) ** 2) / 0.005)  # Gaussian peak at 0.1 V
        >>> result = find_peaks_with_widths(x, y, lower_bound=-0.5, upper_bound=0.5)
        >>> len(result["peak_indices"])  # Should find one peak
        1
    """
    # ── Mask the scan to the potential window of interest ──────────────────
    # Peak detection runs on the masked sub-array to avoid edge artefacts.
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # ── Detect peaks in the masked current ──────────────────────────────
    # prominence threshold is 10 % of the signal dynamic range so noisy
    # flat baselines do not generate peaks.
    peaks, _ = find_peaks(
        y_filtered,
        width=width,
        rel_height=0.5,
        prominence=0.1 * (max(y) - min(y)),
        distance=10,
    )

    # ── Map masked indices back to original array positions ───────────────
    # Adding the offset of the first True mask position aligns the peak
    # indices with the full x/y arrays so width calculations are correct.
    corrected_peaks = peaks + np.where(mask)[0][0]

    # ── Compute widths on the full (unmasked) array ─────────────────────
    # rel_height=0.5 → FWHM;  rel_height=0.9 → FW at 10 % of max.
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
