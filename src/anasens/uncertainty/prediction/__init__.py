"""
Prediction interval methods.

This package implements methods for constructing prediction intervals for
future observations.

Unlike confidence intervals, prediction intervals account for both:

1. uncertainty in the fitted model parameters, and
2. random observation noise described by the selected error model.

Available methods
-----------------
GaussianAnalyticPrediction
    Analytical prediction intervals for additive Gaussian error models.

    This method combines the variance of the fitted model with the
    Gaussian observation variance and constructs prediction intervals
    analytically.

MonteCarloPrediction
    Monte Carlo prediction intervals for arbitrary additive error models.

    Future observations are simulated by repeatedly sampling both the
    model parameters and the observation noise before estimating equal-tail
    prediction intervals.

Notes
-----
Profile likelihood is intentionally **not** implemented for prediction
intervals.

Profile likelihood quantifies uncertainty in the fitted model mean rather
than uncertainty in future observations. Consequently, it belongs in the
``intervals`` package rather than the ``prediction`` package.

The central prediction curve should be interpreted as the predictive
location parameter; for Gaussian errors, this is also the mean.
"""

from .gaussian_analytic import GaussianAnalyticPrediction
from .monte_carlo import MonteCarloPrediction
from .polynomial_specialized import (
    GeneralizedInverseProfileCalibration,
    exact_polynomial_gaussian_prediction_interval,
    fit_poly_quantile_regression,
    profile_likelihood_modified_map,
    quantile_polynomial_prediction_interval,
)

__all__ = [
    "GaussianAnalyticPrediction",
    "MonteCarloPrediction",
    "GeneralizedInverseProfileCalibration",
    "fit_poly_quantile_regression",
    "exact_polynomial_gaussian_prediction_interval",
    "quantile_polynomial_prediction_interval",
    "profile_likelihood_modified_map",
]