"""
Quantile utilities.

This module contains helper functions for computing interval limits from
Monte Carlo samples.

Currently, only equal-tail intervals are implemented. Additional interval
construction strategies (e.g. highest-density intervals) may be added in
future versions.
"""

from __future__ import annotations

import numpy as np


def equal_tail_interval(samples, alpha=0.05):
    """
    Compute an equal-tail interval from Monte Carlo samples.

    Parameters
    ----------
    samples : array-like
        One-dimensional array of Monte Carlo samples.

    alpha : float, optional
        Significance level.

        For a 95% interval, use ``alpha=0.05``.

    Returns
    -------
    tuple
        Pair ``(lower, upper)`` containing the equal-tail interval.

    Notes
    -----
    The interval is computed using empirical quantiles:

        lower = alpha / 2

        upper = 1 - alpha / 2

    No assumption is made about the underlying probability distribution.
    """
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional.")

    if samples.size == 0:
        raise ValueError("samples must not be empty.")

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1.")

    lower = np.quantile(samples, alpha / 2.0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0)

    return float(lower), float(upper)