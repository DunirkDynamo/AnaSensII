"""
Monte Carlo prediction intervals.

This module implements prediction bands for arbitrary additive error models.

Unlike the analytical Gaussian prediction method, this class makes no
normality assumption about the observation noise. It estimates the full
predictive distribution by simulation.

For each input x, the procedure is:

1. Draw model parameters from the asymptotic sampling distribution

       theta ~ Normal(theta_hat, Cov(theta_hat))

2. Evaluate the deterministic location model

       mu = f(x; theta)

3. Draw an observation residual from the error model

       epsilon ~ p_epsilon(epsilon | x)

4. Form a simulated future observation

       y_new = mu + epsilon

5. Use empirical equal-tail quantiles to construct the prediction interval.
"""

from __future__ import annotations

import numpy as np

from ..bands.prediction_band import PredictionBand


class MonteCarloPrediction:
    """
    Monte Carlo prediction band for arbitrary additive residual models.

    This class computes prediction intervals by sampling both:

    - model parameter uncertainty, and
    - future observation noise.

    The method assumes that the fitted parameters are asymptotically normal:

        theta ~ Normal(theta_hat, Cov(theta_hat))

    and that the error model defines additive residuals:

        y_new = f(x; theta) + epsilon

    Parameters
    ----------
    model : callable
        Deterministic model with signature ``model(x, theta)``.

    theta_hat : array-like
        Maximum likelihood estimate of the model parameters.

    cov : array-like
        Covariance matrix of ``theta_hat``.

    error_model : ErrorModel
        Additive residual model implementing ``sample(x, rng)``.

    random_state : int or None, optional
        Optional random seed for reproducibility.
    """

    def __init__(self, model, theta_hat, cov, error_model, random_state=None):
        self.model = model
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.error_model = error_model
        self.rng = np.random.default_rng(random_state)

        # Validate that theta_hat is a parameter vector.
        if self.theta_hat.ndim != 1:
            raise ValueError("theta_hat must be a one-dimensional array.")

        # Validate covariance matrix shape.
        n_params = len(self.theta_hat)
        if self.cov.shape != (n_params, n_params):
            raise ValueError(
                "cov must have shape (p, p), where p = len(theta_hat)."
            )

    @staticmethod
    def _validate_alpha(alpha):
        """
        Validate significance level.

        Parameters
        ----------
        alpha : float
            Significance level.

        Raises
        ------
        ValueError
            If ``alpha`` is outside the open interval (0, 1).
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

    def sample_parameters(self, n_samples):
        """
        Draw parameter vectors from the asymptotic normal distribution.

        Parameters
        ----------
        n_samples : int
            Number of parameter vectors to draw.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_samples, p)`` containing sampled
            parameter vectors.
        """
        return self.rng.multivariate_normal(
            mean=self.theta_hat,
            cov=self.cov,
            size=n_samples,
        )

    def sample_observations(self, x, n_samples):
        """
        Generate Monte Carlo samples of future observations at one x.

        Parameters
        ----------
        x : float
            Input location at which to simulate future observations.

        n_samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        numpy.ndarray
            Simulated future observations at ``x``.
        """
        # Draw all parameter vectors jointly from the asymptotic
        # multivariate normal distribution.
        theta_samples = self.sample_parameters(n_samples)

        # Allocate output array for simulated future observations.
        observations = np.zeros(n_samples, dtype=float)

        for i, theta in enumerate(theta_samples):
            # Compute location of the error model from sampled parameters.
            mu = self.model(x, theta)

            # Draw additive residual from the selected error model.
            epsilon = self.error_model.sample(x, self.rng)

            # Combine deterministic model value and stochastic residual.
            observations[i] = mu + epsilon

        return observations

    def interval(self, x, alpha=0.05, n_samples=10000):
        """
        Compute a Monte Carlo prediction interval at a single x.

        Parameters
        ----------
        x : float
            Input location.

        alpha : float, optional
            Significance level.

            For a 95% prediction interval, use ``alpha=0.05``.

        n_samples : int, optional
            Number of Monte Carlo samples.

        Returns
        -------
        tuple
            Pair ``(lower, upper)`` containing equal-tail prediction limits.
        """
        self._validate_alpha(alpha)

        samples = self.sample_observations(x, n_samples=n_samples)

        lower = np.quantile(samples, alpha / 2.0)
        upper = np.quantile(samples, 1.0 - alpha / 2.0)

        return float(lower), float(upper)

    def band(self, x_grid, alpha=0.05, n_samples=10000):
        """
        Construct a Monte Carlo prediction band over an x-grid.

        Parameters
        ----------
        x_grid : array-like
            Grid of x values at which prediction limits are computed.

        alpha : float, optional
            Significance level.

        n_samples : int, optional
            Number of Monte Carlo samples per x value.

        Returns
        -------
        PredictionBand
            Prediction band storing central curve and equal-tail prediction
            limits on ``x_grid``.
        """
        self._validate_alpha(alpha)

        x_grid = np.asarray(x_grid, dtype=float)

        # These arrays store the central curve and prediction limits.
        mean = np.zeros_like(x_grid, dtype=float)
        lower = np.zeros_like(x_grid, dtype=float)
        upper = np.zeros_like(x_grid, dtype=float)

        for i, x in enumerate(x_grid):
            # Generate predictive samples for this x value.
            samples = self.sample_observations(x, n_samples=n_samples)

            # Store the fitted-location value at the MLE as the central curve.
            # This keeps the visual center tied to f(x; theta_hat), not
            # necessarily the Monte Carlo sample mean.
            mean[i] = self.model(x, self.theta_hat)

            # Equal-tail prediction interval.
            lower[i] = np.quantile(samples, alpha / 2.0)
            upper[i] = np.quantile(samples, 1.0 - alpha / 2.0)

        return PredictionBand(
            x_grid=x_grid,
            mean=mean,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method="monte_carlo",
        )