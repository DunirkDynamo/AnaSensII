import numpy as np
from scipy.stats import hypsecant

from .base import ErrorModel


class HyperbolicSecantError(ErrorModel):
    """
    Hyperbolic secant residual model.

    Assumes additive noise:

        ε ~ HyperbolicSecant(0, scale)

    so that:

        y = f(x; θ) + ε

    The hyperbolic secant distribution is symmetric with moderately heavy
    tails. It provides an alternative to Gaussian and Laplace residual
    models and has theoretical motivations in several stochastic and
    physical processes.

    Role in the framework
    ---------------------
    This model defines:

    1. Likelihood contribution:
           log p(y | x, θ)
           = log HyperbolicSecant(y - f(x; θ))

    2. Residual sampling:
           ε ~ HyperbolicSecant(0, scale)

    3. Symmetric, moderately heavy-tailed residual behavior.

    Parameters
    ----------
    scale : float
        Scale parameter controlling the dispersion of the residual
        distribution.
    """

    def __init__(self, scale: float):
        """
        Initialize the hyperbolic secant error model.

        Parameters
        ----------
        scale : float
            Positive scale parameter.
        """
        if scale <= 0:
            raise ValueError("Scale parameter must be positive.")

        self.scale = float(scale)

    def sample(self, x, rng):
        """
        Generate a random hyperbolic secant residual.

        Parameters
        ----------
        x : float or array-like
            Input location.

            Currently unused because this model is homoscedastic, but
            retained for interface consistency.

        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled residual.
        """
        return hypsecant.rvs(
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
        return hypsecant.logpdf(
            residual,
            loc=0.0,
            scale=self.scale,
        )