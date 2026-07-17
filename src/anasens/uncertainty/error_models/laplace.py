import numpy as np
from scipy.stats import laplace
from .base import ErrorModel


class LaplaceError(ErrorModel):
    """
    Laplace (double-exponential) residual model.

    Assumes additive noise of the form:

        ε ~ Laplace(0, b)

    so that:

        y = f(x; θ) + ε

    This corresponds to an L1 (least absolute deviations) noise model.

    Role in the framework
    ---------------------
    This model defines:

    1. Likelihood contribution:
           log p(y | x, θ)
           = log Laplace(y - f(x; θ) | 0, b)

    2. Robust residual sampling:
           ε ~ Laplace(0, b)

    3. Robust inference behavior:
           heavier tails than Gaussian, less sensitivity to outliers

    Parameters
    ----------
    b : float
        Scale parameter (diversity of residuals).
    """

    def __init__(self, b: float):
        if b <= 0:
            raise ValueError("Scale parameter b must be positive")
        self.b = float(b)

    def sample(self, x, rng):
        """
        Draw a Laplace residual.

        Parameters
        ----------
        x : float or array-like
            Input location (unused in this homoscedastic model).
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled residual ε ~ Laplace(0, b)
        """
        return rng.laplace(loc=0.0, scale=self.b)

    def logpdf(self, residual, x):
        """
        Compute log-density of a residual under Laplace noise.

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
        return laplace.logpdf(residual, loc=0.0, scale=self.b)