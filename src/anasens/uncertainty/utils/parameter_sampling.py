"""
Parameter sampling utilities.

This module contains helper functions for sampling fitted model parameters
from their asymptotic joint normal distribution.

These utilities are shared by both:

- Monte Carlo confidence intervals
- Monte Carlo prediction intervals
"""

from __future__ import annotations

import numpy as np


def sample_parameters(theta_hat, cov, n_samples, rng):
    """
    Draw model parameter samples from a joint multivariate normal distribution.

    Parameters
    ----------
    theta_hat : array-like
        Maximum likelihood estimate of the model parameter vector.

        This is used as the mean of the asymptotic sampling distribution.

    cov : array-like
        Covariance matrix of ``theta_hat``.

        This must be the full joint covariance matrix, not only the diagonal
        variances. Off-diagonal entries encode parameter correlations and are
        essential for correct uncertainty propagation.

    n_samples : int
        Number of parameter vectors to sample.

    rng : numpy.random.Generator
        Random number generator used to draw samples.

    Returns
    -------
    numpy.ndarray
        Array of sampled parameter vectors with shape
        ``(n_samples, n_parameters)``.

    Notes
    -----
    The sampling model is

        theta ~ Normal(theta_hat, Cov(theta_hat))

    This is the standard asymptotic approximation used after maximum
    likelihood estimation.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    cov = np.asarray(cov, dtype=float)

    # Validate that theta_hat is a one-dimensional parameter vector.
    if theta_hat.ndim != 1:
        raise ValueError("theta_hat must be one-dimensional.")

    # Validate that covariance is a square matrix matching theta_hat.
    n_parameters = len(theta_hat)

    if cov.shape != (n_parameters, n_parameters):
        raise ValueError(
            "cov must have shape (p, p), where p = len(theta_hat)."
        )

    # Validate requested number of samples.
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    # Draw from the full joint multivariate normal distribution.
    return rng.multivariate_normal(
        mean=theta_hat,
        cov=cov,
        size=n_samples,
    )