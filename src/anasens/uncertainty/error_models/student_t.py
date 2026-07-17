import numpy as np
from scipy.stats import t
from .base import ErrorModel


class StudentTError(ErrorModel):
    """
    Student's t residual model.

    Assumes additive noise:

        ε ~ t_ν(0, scale)

    so that:

        y = f(x; θ) + ε

    This model introduces heavy-tailed residual behavior controlled by
    the degrees of freedom parameter ν.

    Role in the framework
    ---------------------
    This model defines:

    1. Likelihood contribution:
           log p(y | x, θ)
           = log t_ν(y - f(x; θ))

    2. Robust residual sampling:
           ε ~ t_ν(0, scale)

    3. Outlier robustness:
           as ν → ∞, recovers Gaussian behavior
           small ν increases tail heaviness

    Parameters
    ----------
    df : float
        Degrees of freedom (ν > 0). Controls tail heaviness.
    scale : float
        Scale parameter controlling spread of residuals.
    """

    def __init__(self, df: float, scale: float):
        if df <= 0:
            raise ValueError("Degrees of freedom df must be positive")
        if scale <= 0:
            raise ValueError("Scale must be positive")

        self.df = float(df)
        self.scale = float(scale)

    def sample(self, x, rng):
        """
        Draw a Student's t residual.

        Parameters
        ----------
        x : float or array-like
            Input location (unused in this homoscedastic model).
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled residual ε ~ t_ν(0, scale)
        """
        return t.rvs(df=self.df, scale=self.scale, random_state=rng)

    def logpdf(self, residual, x):
        """
        Compute log-density of a residual under Student's t noise.

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
        return t.logpdf(residual, df=self.df, scale=self.scale)