"""
Delta Method confidence intervals for fitted model means.

This module implements confidence intervals for the fitted model mean

    f(x; theta_hat)

using first-order or higher-order Delta Method uncertainty propagation.

These intervals quantify uncertainty in the fitted model mean only.
They do not include future observation noise.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..bands.confidence_band import ConfidenceBand


class DeltaMethod:
    """
    Delta Method confidence interval estimator.

    This class propagates parameter uncertainty through a deterministic
    model to estimate uncertainty in the fitted model mean.

    The fitted parameters are assumed to be asymptotically normal:

        theta_hat ~ Normal(theta, Cov(theta_hat))

    For a scalar model response

        q = f(x; theta_hat),

    the first-order Delta Method estimates

        Var[q] ≈ J Cov(theta_hat) J^T

    where J is the gradient of ``f`` with respect to ``theta``.

    Optionally, higher-order corrections can be included if Hessian and
    third-derivative tensor functions are supplied.

    Parameters
    ----------
    model : callable
        Deterministic model with signature ``model(x, theta)``.

    theta_hat : array-like
        Maximum likelihood estimate of the parameter vector.

    cov : array-like
        Covariance matrix of ``theta_hat``.

    grad : callable
        Gradient function with signature ``grad(x, theta)``.

        Must return a one-dimensional array with length equal to the number
        of model parameters.

    hess : callable or None, optional
        Hessian function with signature ``hess(x, theta)``.

        Required for ``order=2``.

    third : callable or None, optional
        Third-derivative tensor function with signature ``third(x, theta)``.

        Required for ``order=2``.
    """

    def __init__(self, model, theta_hat, cov, grad, hess=None, third=None):
        self.model = model
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.grad = grad
        self.hess = hess
        self.third = third

        if self.theta_hat.ndim != 1:
            raise ValueError("theta_hat must be one-dimensional.")

        n_params = len(self.theta_hat)
        if self.cov.shape != (n_params, n_params):
            raise ValueError(
                "cov must have shape (p, p), where p = len(theta_hat)."
            )

    @staticmethod
    def _validate_alpha(alpha):
        """
        Validate the significance level.

        Parameters
        ----------
        alpha : float
            Significance level.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

    def variance(self, x, order=1):
        """
        Estimate variance of the fitted model mean at x.

        Parameters
        ----------
        x : float
            Input location.

        order : int, optional
            Delta Method order.

            ``order=1`` uses first-order propagation.

            ``order=2`` includes higher-order correction terms using the
            Hessian and third-derivative tensor.

        Returns
        -------
        float
            Estimated variance of ``f(x; theta_hat)``.
        """
        theta = self.theta_hat
        sigma = self.cov

        # Gradient of the model with respect to model parameters.
        gradient = np.asarray(self.grad(x, theta), dtype=float)

        # First-order Delta Method variance.
        variance = float(gradient @ sigma @ gradient)

        if order == 1:
            return variance

        if order != 2:
            raise ValueError("Only order=1 and order=2 are currently supported.")

        if self.hess is None or self.third is None:
            raise ValueError(
                "order=2 requires both hess and third derivative functions."
            )

        # Hessian matrix of f with respect to theta.
        hessian = np.asarray(self.hess(x, theta), dtype=float)

        # Third-derivative tensor of f with respect to theta.
        third_tensor = np.asarray(self.third(x, theta), dtype=float)

        n_params = len(theta)

        # Correction term involving gradient and third derivatives.
        third_order_correction = 0.0
        for i in range(n_params):
            for j in range(n_params):
                for k in range(n_params):
                    for l in range(n_params):
                        third_order_correction += (
                            gradient[i]
                            * third_tensor[j, k, l]
                            * sigma[i, j]
                            * sigma[k, l]
                        )

        # Correction term involving Hessian-Hessian contractions.
        hessian_correction = 0.0
        for i in range(n_params):
            for j in range(n_params):
                for k in range(n_params):
                    for l in range(n_params):
                        hessian_correction += (
                            hessian[i, j]
                            * hessian[k, l]
                            * sigma[i, k]
                            * sigma[j, l]
                        )

        return float(variance + third_order_correction + 0.5 * hessian_correction)

    def std(self, x, order=1):
        """
        Estimate standard error of the fitted model mean at x.

        Parameters
        ----------
        x : float
            Input location.

        order : int, optional
            Delta Method order.

        Returns
        -------
        float
            Estimated standard error.
        """
        return float(np.sqrt(self.variance(x, order=order)))

    def interval(self, x, alpha=0.05, order=1):
        """
        Compute a Delta Method confidence interval at one x.

        Parameters
        ----------
        x : float
            Input location.

        alpha : float, optional
            Significance level.

        order : int, optional
            Delta Method order.

        Returns
        -------
        tuple
            ``(lower, upper)`` confidence interval for the fitted mean.
        """
        self._validate_alpha(alpha)

        mean = self.model(x, self.theta_hat)
        z_value = norm.ppf(1.0 - alpha / 2.0)
        standard_error = self.std(x, order=order)

        return (
            float(mean - z_value * standard_error),
            float(mean + z_value * standard_error),
        )

    def band(self, x_grid, alpha=0.05, order=1):
        """
        Construct a Delta Method confidence band over an x-grid.

        Parameters
        ----------
        x_grid : array-like
            Grid of x values.

        alpha : float, optional
            Significance level.

        order : int, optional
            Delta Method order.

        Returns
        -------
        ConfidenceBand
            Stored confidence band.
        """
        self._validate_alpha(alpha)

        x_grid = np.asarray(x_grid, dtype=float)
        z_value = norm.ppf(1.0 - alpha / 2.0)

        mean = np.zeros_like(x_grid, dtype=float)
        lower = np.zeros_like(x_grid, dtype=float)
        upper = np.zeros_like(x_grid, dtype=float)

        for i, x in enumerate(x_grid):
            mean_i = self.model(x, self.theta_hat)
            se_i = self.std(x, order=order)

            mean[i] = mean_i
            lower[i] = mean_i - z_value * se_i
            upper[i] = mean_i + z_value * se_i

        return ConfidenceBand(
            x_grid=x_grid,
            mean=mean,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method=f"delta_order_{order}",
        )