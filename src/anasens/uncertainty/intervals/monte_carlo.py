"""
Monte Carlo confidence intervals for fitted model means.

This module implements confidence intervals for

    f(x; theta_hat)

by sampling model parameters from their asymptotic joint normal
distribution and propagating those samples through the deterministic model.

These intervals quantify model uncertainty only. They do not include
future observation noise.
"""

from __future__ import annotations

import numpy as np

from ..bands.confidence_band import ConfidenceBand


class MonteCarloCI:
    """
    Monte Carlo confidence interval estimator for fitted model means.

    The fitted parameter vector is assumed to be asymptotically normal:

        theta ~ Normal(theta_hat, Cov(theta_hat))

    Each sampled parameter vector is propagated through the deterministic
    model to produce samples of the fitted mean response.

    Parameters
    ----------
    model : callable
        Deterministic model with signature ``model(x, theta)``.

    theta_hat : array-like
        Maximum likelihood estimate of the parameter vector.

    cov : array-like
        Covariance matrix of ``theta_hat``.

    random_state : int or None, optional
        Optional random seed for reproducibility.
    """

    def __init__(self, model, theta_hat, cov, random_state=None):
        self.model = model
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.rng = np.random.default_rng(random_state)

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

    def sample_parameters(self, n_samples):
        """
        Draw parameter vectors from the asymptotic joint normal distribution.

        Parameters
        ----------
        n_samples : int
            Number of parameter samples.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_parameters)``.
        """
        return self.rng.multivariate_normal(
            mean=self.theta_hat,
            cov=self.cov,
            size=n_samples,
        )

    def sample_model(self, x, n_samples=10000):
        """
        Sample fitted model mean values at a single x.

        Parameters
        ----------
        x : float
            Input location.

        n_samples : int, optional
            Number of parameter samples.

        Returns
        -------
        numpy.ndarray
            Samples of ``f(x; theta)`` induced by parameter uncertainty.
        """
        theta_samples = self.sample_parameters(n_samples)

        return np.asarray(
            [self.model(x, theta) for theta in theta_samples],
            dtype=float,
        )

    def interval(self, x, alpha=0.05, n_samples=10000):
        """
        Compute a Monte Carlo confidence interval at one x.

        Parameters
        ----------
        x : float
            Input location.

        alpha : float, optional
            Significance level.

        n_samples : int, optional
            Number of Monte Carlo parameter samples.

        Returns
        -------
        tuple
            ``(lower, upper)`` confidence interval for the fitted mean.
        """
        self._validate_alpha(alpha)

        samples = self.sample_model(x, n_samples=n_samples)

        lower = np.quantile(samples, alpha / 2.0)
        upper = np.quantile(samples, 1.0 - alpha / 2.0)

        return float(lower), float(upper)

    def band(self, x_grid, alpha=0.05, n_samples=10000):
        """
        Construct a Monte Carlo confidence band over an x-grid.

        Parameters
        ----------
        x_grid : array-like
            Grid of x values.

        alpha : float, optional
            Significance level.

        n_samples : int, optional
            Number of Monte Carlo parameter samples per x value.

        Returns
        -------
        ConfidenceBand
            Stored confidence band.
        """
        self._validate_alpha(alpha)

        x_grid = np.asarray(x_grid, dtype=float)

        mean = np.zeros_like(x_grid, dtype=float)
        lower = np.zeros_like(x_grid, dtype=float)
        upper = np.zeros_like(x_grid, dtype=float)

        for i, x in enumerate(x_grid):
            samples = self.sample_model(x, n_samples=n_samples)

            mean[i] = self.model(x, self.theta_hat)
            lower[i] = np.quantile(samples, alpha / 2.0)
            upper[i] = np.quantile(samples, 1.0 - alpha / 2.0)

        return ConfidenceBand(
            x_grid=x_grid,
            mean=mean,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method="monte_carlo",
        )