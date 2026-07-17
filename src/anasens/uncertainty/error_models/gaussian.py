import numpy as np
from scipy.stats import norm
from .base import ErrorModel


class GaussianError(ErrorModel):
    """
    Gaussian residual model.

    Assumes additive noise of the form:

        ε ~ N(0, σ²)

    so that the observation model becomes:

        y = f(x; θ) + ε

    This is the standard homoscedastic Gaussian noise model.

    Role in the framework
    ---------------------
    This model defines:

    1. Likelihood contribution:
           log p(y | x, θ)
           = log N(y - f(x; θ) | 0, σ²)

    2. Residual sampling:
           ε ~ N(0, σ²)

    3. Analytical prediction intervals:
           enables closed-form variance propagation

    Parameters
    ----------
    sigma : float
        Standard deviation of the residual noise.
    """

    def __init__(self, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)

    def sample(self, x, rng):
        """
        Draw a Gaussian residual.

        Parameters
        ----------
        x : float or array-like
            Input location (unused in this homoscedastic model,
            but included for interface consistency).
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled residual ε ~ N(0, σ²)
        """
        return rng.normal(loc=0.0, scale=self.sigma)

    def logpdf(self, residual, x):
        """
        Compute log-density of a residual under Gaussian noise.

        Parameters
        ----------
        residual : float
            Residual value r = y - f(x; θ)
        x : float or array-like
            Input location (unused in this model).

        Returns
        -------
        float
            log p(ε)
        """
        return norm.logpdf(residual, loc=0.0, scale=self.sigma)