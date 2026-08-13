"""Inverse-calibration band representation for profile-likelihood results."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class ProfileLikelihoodBand:
    """Represent a profile-likelihood inverse-calibration band."""

    def __init__(
        self,
        y_grid,
        x_mle,
        x_lower,
        x_upper,
        alpha=0.05,
        method="profile_likelihood",
        interpolation="linear",
        extrapolate=True,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

        self.alpha = float(alpha)
        self.method = str(method)
        self.y_grid = np.asarray(y_grid, dtype=float)
        self.mle_values = np.asarray(x_mle, dtype=float)
        self.lower_values = np.asarray(x_lower, dtype=float)
        self.upper_values = np.asarray(x_upper, dtype=float)

        self._validate_shapes()

        if np.any(self.lower_values > self.upper_values):
            raise ValueError("lower inverse-calibration limit exceeds upper limit.")

        fill_value = "extrapolate" if extrapolate else np.nan
        self._mle_interp = interp1d(
            self.y_grid,
            self.mle_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )
        self._lower_interp = interp1d(
            self.y_grid,
            self.lower_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )
        self._upper_interp = interp1d(
            self.y_grid,
            self.upper_values,
            kind=interpolation,
            bounds_error=False,
            fill_value=fill_value,
        )

    @property
    def x_mle(self):
        return self.mle_values

    @property
    def x_lower(self):
        return self.lower_values

    @property
    def x_upper(self):
        return self.upper_values

    def _validate_shapes(self):
        arrays = {
            "y_grid": self.y_grid,
            "x_mle": self.mle_values,
            "x_lower": self.lower_values,
            "x_upper": self.upper_values,
        }

        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")

        n_points = len(self.y_grid)
        for name, array in arrays.items():
            if len(array) != n_points:
                raise ValueError(f"{name} must have the same length as y_grid.")

    def estimate(self, y):
        return self._mle_interp(y)

    def interval(self, y):
        lower = self._lower_interp(y)
        upper = self._upper_interp(y)
        return lower, upper

    def resolution(self, y):
        mle = self.estimate(y)
        lower, upper = self.interval(y)
        return (upper - lower) / mle

    def evaluate(self, y_grid):
        y_grid = np.asarray(y_grid, dtype=float)
        lower, upper = self.interval(y_grid)
        return {
            "y_grid": y_grid,
            "x_mle": self.estimate(y_grid),
            "x_lower": lower,
            "x_upper": upper,
            "alpha": self.alpha,
            "method": self.method,
        }

    def as_dict(self):
        return {
            "y_grid": self.y_grid,
            "x_mle": self.mle_values,
            "x_lower": self.lower_values,
            "x_upper": self.upper_values,
        }