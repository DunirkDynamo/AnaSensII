"""
Prediction band representation.

This module defines :class:`PredictionBand`, a lightweight storage and
evaluation object for prediction intervals around future observations.

A prediction band represents uncertainty in

    y_new | x

and therefore includes both:

1. uncertainty in the fitted model mean, and
2. random observation noise.

Prediction bands are distinct from confidence bands. A confidence band
describes uncertainty in the fitted mean function only.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class PredictionBand:
    """
    Represent a prediction band for future observations.

    This class stores and evaluates prediction limits for future measured
    values at input locations x.

    The prediction limits are computed elsewhere, then stored here as
    arrays and interpolated for convenient evaluation.

    Parameters
    ----------
    x_grid : array-like
        Grid of x values at which prediction limits were computed.

    mean : array-like
        Central fitted location value evaluated on ``x_grid``.

        This is usually ``f(x; theta_hat)`` and is not necessarily the
        empirical mean of Monte Carlo prediction samples.

        For additive error models, this is the predictive location parameter.

    lower : array-like
        Lower prediction limit evaluated on ``x_grid``.

    upper : array-like
        Upper prediction limit evaluated on ``x_grid``.

    alpha : float, optional
        Significance level.

        For a 95% prediction band, use ``alpha=0.05``.

    method : str, optional
        Name of the method used to construct the prediction band.

        Expected first-version methods are:

        - ``"gaussian_analytic"``
        - ``"monte_carlo"``

    interpolation : str, optional
        Interpolation method passed to :class:`scipy.interpolate.interp1d`.

    extrapolate : bool, optional
        If True, allow extrapolation outside ``x_grid``. If False, values
        outside the grid return ``nan``.

    Notes
    -----
    This class does not compute prediction intervals.

    It only stores prediction intervals that have already been computed by
    a prediction method.

    In this package's first iteration, prediction bands should be produced
    only by:

    - analytic Gaussian propagation, when the error model is Gaussian, or
    - Monte Carlo simulation, for arbitrary additive error models.

    Profile likelihood is not used for prediction bands. It is reserved for
    confidence intervals of the fitted model mean.
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
        # Validate significance level before storing any user data.
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

        # Store metadata describing how the band was constructed.
        self.alpha = float(alpha)
        self.method = str(method)

        # Convert all curve data to floating-point arrays.
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.mean_values = np.asarray(mean, dtype=float)
        self.lower_values = np.asarray(lower, dtype=float)
        self.upper_values = np.asarray(upper, dtype=float)

        # Validate array dimensions and lengths.
        self._validate_shapes()

        # A prediction interval must always have lower <= upper.
        if np.any(self.lower_values > self.upper_values):
            raise ValueError("lower prediction limit exceeds upper limit.")

        # Determine how interpolation handles evaluation outside the grid.
        fill_value = "extrapolate" if extrapolate else np.nan

        # Interpolator for the central fitted location curve.
        self._mean_interp = interp1d(
            self.x_grid,
            self.mean_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )

        # Interpolator for the lower prediction limit.
        self._lower_interp = interp1d(
            self.x_grid,
            self.lower_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )

        # Interpolator for the upper prediction limit.
        self._upper_interp = interp1d(
            self.x_grid,
            self.upper_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )

    def _validate_shapes(self):
        """
        Validate that stored arrays are one-dimensional and length-matched.

        Raises
        ------
        ValueError
            If any input array is not one-dimensional, or if ``x_grid``,
            ``mean``, ``lower``, and ``upper`` do not have the same length.
        """
        arrays = {
            "x_grid": self.x_grid,
            "mean": self.mean_values,
            "lower": self.lower_values,
            "upper": self.upper_values,
        }

        # Every curve must be represented as a one-dimensional sequence.
        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")

        # Every curve must be evaluated on the same x-grid.
        n_points = len(self.x_grid)

        for name, array in arrays.items():
            if len(array) != n_points:
                raise ValueError(
                    f"{name} must have the same length as x_grid."
                )

    def mean(self, x):
        """
        Evaluate the central fitted location curve.

        Parameters
        ----------
        x : float or array-like
            Point or points at which to evaluate the central curve.

        Returns
        -------
        float or numpy.ndarray
            Interpolated central fitted model value.
        """
        return self._mean_interp(x)

    def location(self, x):
        """
        Evaluate the predictive location curve at x.

        Parameters
        ----------
        x : float or array-like
            Point or points at which to evaluate the predictive location.

        Returns
        -------
        float or numpy.ndarray
            Interpolated predictive location value.
        """
        return self.mean(x)

    def interval(self, x):
        """
        Evaluate prediction limits at x.

        Parameters
        ----------
        x : float or array-like
            Point or points at which to evaluate the prediction interval.

        Returns
        -------
        tuple
            Pair ``(lower, upper)`` containing the prediction limits.
        """
        lower = self._lower_interp(x)
        upper = self._upper_interp(x)

        return lower, upper

    def evaluate(self, x_grid):
        """
        Evaluate the prediction band on a supplied grid.

        Parameters
        ----------
        x_grid : array-like
            Grid of x values where the prediction band should be evaluated.

        Returns
        -------
        dict
            Dictionary containing evaluated arrays:

            ``x``
                Evaluation grid.

            ``mean``
                Central fitted model values.

            ``lower``
                Lower prediction limits.

            ``upper``
                Upper prediction limits.

            ``alpha``
                Significance level.

            ``method``
                Method used to construct the prediction band.
        """
        x_grid = np.asarray(x_grid, dtype=float)
        lower, upper = self.interval(x_grid)

        return {
            "x": x_grid,
            "mean": self.mean(x_grid),
            "location": self.location(x_grid),
            "lower": lower,
            "upper": upper,
            "alpha": self.alpha,
            "method": self.method,
        }