"""
mc_inference.inference

This package contains routines for fitting deterministic models to data
using likelihood-based inference.

The inference layer is responsible for:

- constructing log-likelihoods from a model and an error model
- optimizing model parameters by maximum likelihood
- estimating parameter covariance from likelihood curvature

This package does not define deterministic models or error distributions.
Those are provided separately by:

    mc_inference.models
    mc_inference.error_models
"""

from .mle import MLE
from .covariance import covariance_from_loglik, numerical_hessian

__all__ = [
    "MLE",
    "covariance_from_loglik",
    "numerical_hessian",
]