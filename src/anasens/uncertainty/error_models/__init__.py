"""
mc_inference.error_models

This package defines stochastic residual models used in the framework:

    y = f(x; θ) + ε

where ε represents random noise drawn from a specified probability distribution.

These models define the entire stochastic structure of the system.

Role in the framework
---------------------
Error models are responsible for defining:

1. The probability distribution of residuals:
       r = y - f(x; θ)

2. A sampling procedure for residuals:
       ε ~ p(ε | x)

Together, these define:

- the likelihood used in maximum likelihood estimation
- the uncertainty structure used in covariance estimation
- the noise model used in Monte Carlo prediction

Design principles
-----------------
Error models are intentionally independent of:

- the functional form of f(x; θ)
- the dimensionality or interpretation of θ
- the inference method being used

They operate only on residual values and optional input conditioning x.
"""

from .gaussian import GaussianError
from .hyperbolic_secant import HyperbolicSecantError
from .laplace import LaplaceError
from .student_t import StudentTError
from .logistic import LogisticError
from .skew_normal import SkewNormalError

__all__ = [
    "GaussianError",
    "HyperbolicSecantError",
    "LaplaceError",
    "StudentTError",
    "LogisticError",
    "SkewNormalError",
]