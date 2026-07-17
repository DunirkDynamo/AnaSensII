import numpy as np
from scipy.stats import logistic
from .base import ErrorModel


class LogisticError(ErrorModel):
    """
    Logistic residual model.

    Assumes additive noise:

        ε ~ Logistic(0, s)

    so that:

        y = f(x; θ) + ε

    The logistic distribution provides a smooth, symmetric,
    moderately heavy-tailed alternative to Gaussian noise.

    Role in the framework
    ---------------------
    This model defines:

    1. Likelihood contribution:
           log p(y | x, θ)
           = log Logistic(y - f(x; θ) | 0, s)

    2. Residual sampling:
           ε ~ Logistic(0, s)

    3. Smooth robust inference behavior:
           heavier tails than Gaussian
           smoother than Laplace

    Parameters
    ----------
    s : float
        Scale parameter controlling dispersion of residuals.
    """

    def __init__(self, s: float):
        if s <= 0:
            raise ValueError("Scale parameter s must be positive")
        self.s = float(s)

    def sample(self, x, rng):
        """
        Generate a logistic residual.

        Parameters
        ----------
        x : float or array-like
            Input location (unused in this homoscedastic model).
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled residual ε ~ Logistic(0, s)
        """
        return logistic.rvs(loc=0.0, scale=self.s, random_state=rng)

    def logpdf(self, residual, x):
        """
        Compute log-density of a residual under logistic noise.

        Parameters
        ----------
        residual : float
            Residual r = y - f(x; θ)
        x : float or array-like
            Input location (unused in this model).

        Returns
        -------
        float
            log p(ε)
        """
        return logistic.logpdf(residual, loc=0.0, scale=self.s)