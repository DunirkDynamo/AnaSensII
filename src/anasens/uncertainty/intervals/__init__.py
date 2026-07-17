"""
Confidence interval methods for fitted models.

This package implements methods for quantifying uncertainty in the fitted
model mean

    f(x; theta_hat)

after maximum likelihood estimation.

Available methods
-----------------
DeltaMethod
    Analytical uncertainty propagation using the multivariate Delta Method.

    Supports both first-order and higher-order propagation.

MonteCarloCI
    Estimates confidence intervals by sampling parameter vectors from the
    asymptotic joint normal distribution of the maximum likelihood
    estimator and propagating them through the fitted model.

ProfileLikelihoodCI
    Computes likelihood-based confidence intervals using constrained
    optimization and Wilks' theorem.

Notes
-----
These classes compute confidence intervals for the fitted model mean only.

They do not account for observation noise.

Prediction intervals are implemented separately in the
``prediction`` package.
"""

from .delta import DeltaMethod
from .monte_carlo import MonteCarloCI
from .profile import ProfileLikelihoodCI

__all__ = [
    "DeltaMethod",
    "MonteCarloCI",
    "ProfileLikelihoodCI",
]