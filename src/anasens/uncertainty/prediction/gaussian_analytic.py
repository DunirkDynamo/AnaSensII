"""
Analytical Gaussian prediction intervals.

This module implements prediction bands for additive Gaussian error models.

It should be used only when the observation model is:

    y = f(x; theta) + epsilon

with

    epsilon ~ Normal(0, sigma^2)

For non-Gaussian error models, use Monte Carlo prediction instead.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t

from ..bands.prediction_band import PredictionBand


class GaussianAnalyticPrediction:
    """
    Analytical prediction band for Gaussian additive noise.

    This class combines:

    1. model uncertainty from parameter covariance
    2. observation noise from a Gaussian error model

    The prediction variance is

        Var[y_new | x] = Var[f(x; theta_hat)] + sigma^2

    where

        Var[f(x; theta_hat)] ≈ J(x) Cov(theta_hat) J(x)^T

    Parameters
    ----------
    model : callable
        Deterministic model with signature ``model(x, theta)``.

    theta_hat : array-like
        Maximum likelihood estimate of the model parameters.

    cov : array-like
        Covariance matrix of ``theta_hat``.

    error_model : GaussianError
        Gaussian error model. Must expose a ``sigma`` attribute.

    grad : callable, optional
        Gradient function with signature ``grad(x, theta)``.

        If not supplied, gradients are estimated numerically.

    dof : int or None, optional
        Degrees of freedom used for the t multiplier.

        If None, a standard normal multiplier is used.

    step_size : float, optional
        Finite-difference step size for numerical gradients.
    """

    def __init__(
        self,
        model,
        theta_hat,
        cov,
        error_model,
        grad=None,
        dof=None,
        step_size=1e-6,
    ):
        if not hasattr(error_model, "sigma"):
            raise TypeError(
                "GaussianAnalyticPrediction requires a Gaussian error model "
                "with a sigma attribute."
            )

        self.model = model
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.error_model = error_model
        self.grad = grad
        self.dof = dof
        self.step_size = float(step_size)

    def _gradient(self, x):
        """
        Evaluate or estimate the gradient of the model with respect to theta.

        Parameters
        ----------
        x : float
            Input location.

        Returns
        -------
        numpy.ndarray
            Gradient vector J(x).
        """
        if self.grad is not None:
            return np.asarray(self.grad(x, self.theta_hat), dtype=float)

        gradient = np.zeros_like(self.theta_hat, dtype=float)
        f0 = self.model(x, self.theta_hat)

        for i in range(len(self.theta_hat)):
            step = np.zeros_like(self.theta_hat)
            step[i] = self.step_size

            gradient[i] = (
                self.model(x, self.theta_hat + step) - f0
            ) / self.step_size

        return gradient

    def model_variance(self, x):
        """
        Compute variance of the fitted location at x.

        Parameters
        ----------
        x : float
            Input location.

        Returns
        -------
        float
            Approximate variance of ``f(x; theta_hat)``.
        """
        J = self._gradient(x)
        return float(J @ self.cov @ J)

    def prediction_variance(self, x):
        """
        Compute total prediction variance at x.

        Parameters
        ----------
        x : float
            Input location.

        Returns
        -------
        float
            Total variance including model uncertainty and Gaussian noise.
        """
        return self.model_variance(x) + self.error_model.sigma**2

    def multiplier(self, alpha):
        """
        Compute the interval multiplier for the requested coverage.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        float
            Normal or t critical value.
        """
        if self.dof is None:
            return norm.ppf(1.0 - alpha / 2.0)

        return t.ppf(1.0 - alpha / 2.0, df=self.dof)

    def band(self, x_grid, alpha=0.05):
        """
        Construct an analytical Gaussian prediction band.

        Parameters
        ----------
        x_grid : array-like
            Grid of x values.

        alpha : float, optional
            Significance level.

        Returns
        -------
        PredictionBand
            Prediction band evaluated on ``x_grid``.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

        x_grid = np.asarray(x_grid, dtype=float)
        critical = self.multiplier(alpha)

        mean = np.zeros_like(x_grid, dtype=float)
        lower = np.zeros_like(x_grid, dtype=float)
        upper = np.zeros_like(x_grid, dtype=float)

        for i, x in enumerate(x_grid):
            mu = self.model(x, self.theta_hat)
            var_pred = self.prediction_variance(x)
            sigma_pred = np.sqrt(var_pred)

            mean[i] = mu
            lower[i] = mu - critical * sigma_pred
            upper[i] = mu + critical * sigma_pred

        return PredictionBand(
            x_grid=x_grid,
            mean=mean,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method="gaussian_analytic",
        )