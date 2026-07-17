import numpy as np
from scipy.stats import skewnorm

from .base import ErrorModel


class SkewNormalError(ErrorModel):
    """
    Skew-normal residual model.

    Assumes additive noise:

        ε ~ SkewNormal(a, loc=0, scale)

    so that:

        y = f(x; θ) + ε

    The skew-normal distribution generalizes the Gaussian distribution by
    introducing a shape parameter that controls the direction and degree
    of skewness.

    When the shape parameter is zero, the distribution reduces exactly to
    a Gaussian distribution.

    Role in the framework
    ---------------------
    This model defines:

    1. Likelihood contribution:
           log p(y | x, θ)
           = log SkewNormal(y - f(x; θ))

    2. Residual sampling:
           ε ~ SkewNormal(a, 0, scale)

    3. Asymmetric residual behavior.

    Parameters
    ----------
    a : float
        Shape (skewness) parameter.

        a > 0 produces right skew.

        a < 0 produces left skew.

        a = 0 recovers the Gaussian distribution.

    scale : float
        Scale parameter of the distribution.

    Notes
    -----
    This implementation uses the location parameter of the skew-normal
    distribution directly (loc = 0).

    Consequently, for nonzero values of ``a``, the mean of the residual
    distribution is generally not zero. Future versions of the library
    may instead reparameterize the distribution so that the residual
    mean is identically zero while preserving skewness.
    """

    def __init__(self, a: float, scale: float):
        """
        Initialize the skew-normal error model.

        Parameters
        ----------
        a : float
            Shape (skewness) parameter.

        scale : float
            Positive scale parameter.
        """
        if scale <= 0:
            raise ValueError("Scale parameter must be positive.")

        self.a = float(a)
        self.scale = float(scale)

    def sample(self, x, rng):
        """
        Generate a random skew-normal residual.

        Parameters
        ----------
        x : float or array-like
            Input location.

            This parameter is currently unused because this model is
            homoscedastic, but it is included to maintain a consistent
            interface with heteroscedastic error models.

        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            A sampled residual.
        """
        return skewnorm.rvs(
            a=self.a,
            loc=0.0,
            scale=self.scale,
            random_state=rng,
        )

    def logpdf(self, residual, x):
        """
        Evaluate the log probability density of a residual.

        Parameters
        ----------
        residual : float
            Residual value

                r = y - f(x; θ)

        x : float or array-like
            Input location.

            Currently unused but retained for interface consistency.

        Returns
        -------
        float
            Log probability density of the residual.
        """
        return skewnorm.logpdf(
            residual,
            a=self.a,
            loc=0.0,
            scale=self.scale,
        )