"""
Confidence band representation.

This module defines :class:`ConfidenceBand`, a lightweight storage and
evaluation object for confidence intervals around the fitted model mean.

A confidence band represents uncertainty in

    f(x; theta_hat)

only. It does not include future observation noise. Prediction bands are
represented separately by ``PredictionBand``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class ConfidenceBand:
    """
    Represent a confidence band for the fitted model mean.

    This class stores and evaluates confidence limits for the fitted mean
    response

        f(x; theta_hat)

    across a domain of x values.

    The class supports grid-based confidence bands. Confidence limits are
    computed elsewhere, then stored here as arrays and interpolated for
    convenient evaluation.

    Parameters
    ----------
    x_grid : array-like
        Grid of x values at which the confidence limits were computed.
    mean : array-like
        Fitted model mean evaluated on ``x_grid``.
    lower : array-like
        Lower confidence limit evaluated on ``x_grid``.
    upper : array-like
        Upper confidence limit evaluated on ``x_grid``.
    alpha : float, optional
        Significance level. For a 95% confidence band, use ``alpha=0.05``.
    method : str, optional
        Name of the method used to construct the band.
    interpolation : str, optional
        Interpolation method passed to :class:`scipy.interpolate.interp1d`.
    extrapolate : bool, optional
        If True, allow extrapolation outside ``x_grid``. If False, values
        outside the grid return ``nan``.

    Notes
    -----
    This class does not compute confidence intervals. It only stores and
    evaluates intervals that have already been computed by another class.
    """

    def __init__(
        self,
        x_grid,
        mean,
        lower,
        upper,
        alpha=0.05,
        method="unknown",
        interpolation="linear",
        extrapolate=True,
    ):
        # Validate confidence level before storing anything.
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

        # Store metadata describing the confidence band.
        self.alpha = float(alpha)
        self.method = str(method)

        # Convert user inputs into numeric arrays for reliable computation.
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.mean_values = np.asarray(mean, dtype=float)
        self.lower_values = np.asarray(lower, dtype=float)
        self.upper_values = np.asarray(upper, dtype=float)

        # Ensure all arrays represent values on the same grid.
        self._validate_shapes()

        # Ensure lower bounds never exceed upper bounds.
        if np.any(self.lower_values > self.upper_values):
            raise ValueError("lower confidence limit exceeds upper limit.")

        # Choose how scipy should handle evaluations outside the grid.
        fill_value = "extrapolate" if extrapolate else np.nan

        # Interpolator for fitted mean response.
        self._mean_interp = interp1d(
            self.x_grid,
            self.mean_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )

        # Interpolator for lower confidence limit.
        self._lower_interp = interp1d(
            self.x_grid,
            self.lower_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )

        # Interpolator for upper confidence limit.
        self._upper_interp = interp1d(
            self.x_grid,
            self.upper_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )

    def _validate_shapes(self):
        """
        Validate that all stored arrays have matching one-dimensional shape.

        Raises
        ------
        ValueError
            If any input array is not one-dimensional or if the arrays do
            not all have the same length.
        """
        arrays = {
            "x_grid": self.x_grid,
            "mean": self.mean_values,
            "lower": self.lower_values,
            "upper": self.upper_values,
        }

        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")

        n_points = len(self.x_grid)

        for name, array in arrays.items():
            if len(array) != n_points:
                raise ValueError(
                    f"{name} must have the same length as x_grid."
                )

    def mean(self, x):
        """
        Evaluate the fitted mean response at x.

        Parameters
        ----------
        x : float or array-like
            Point or points at which to evaluate the fitted mean.

        Returns
        -------
        float or numpy.ndarray
            Interpolated fitted mean value.
        """
        return self._mean_interp(x)

    def interval(self, x):
        """
        Evaluate confidence limits at x.

        Parameters
        ----------
        x : float or array-like
            Point or points at which to evaluate the confidence interval.

        Returns
        -------
        tuple
            Pair ``(lower, upper)`` containing the confidence limits.
        """
        lower = self._lower_interp(x)
        upper = self._upper_interp(x)

        return lower, upper

    def evaluate(self, x_grid):
        """
        Evaluate the confidence band on a supplied grid.

        Parameters
        ----------
        x_grid : array-like
            Grid of x values where the band should be evaluated.

        Returns
        -------
        dict
            Dictionary containing evaluated arrays:

            ``x``
                Evaluation grid.

            ``mean``
                Fitted mean values.

            ``lower``
                Lower confidence limits.

            ``upper``
                Upper confidence limits.

            ``alpha``
                Significance level.

            ``method``
                Method used to construct the confidence band.
        """
        x_grid = np.asarray(x_grid, dtype=float)
        lower, upper = self.interval(x_grid)

        return {
            "x": x_grid,
            "mean": self.mean(x_grid),
            "lower": lower,
            "upper": upper,
            "alpha": self.alpha,
            "method": self.method,
        }