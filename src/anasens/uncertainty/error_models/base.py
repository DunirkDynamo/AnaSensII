from abc import ABC, abstractmethod


class ErrorModel(ABC):
    """
    Abstract base class for residual (error) models in the framework.

    The observation model is assumed to take the form:

        y = f(x; θ) + ε

    where ε is a random variable drawn from a probability distribution
    specified by an ErrorModel.

    This class defines the required interface for all residual models.

    Role in the framework
    ---------------------
    Error models define the stochastic component of the system and are
    used in three key places:

    1. Maximum likelihood estimation (MLE)
       Defines the likelihood:
           log p(y | x, θ) = log p_ε(y - f(x; θ) | x)

    2. Uncertainty propagation
       Defines the noise contribution in prediction intervals.

    3. Simulation / Monte Carlo methods
       Provides sampling of residuals ε.

    Design principles
    -----------------
    Error models must be:

    - independent of the deterministic model f(x; θ)
    - independent of parameterization of θ
    - independent of inference or optimization method

    They operate only on residuals:
        r = y - f(x; θ)

    and optional conditioning variables x.
    """

    @abstractmethod
    def sample(self, x, rng):
        """
        Generate a random residual ε conditioned on input x.

        Parameters
        ----------
        x : float or array-like
            Input location. Some models may allow x-dependent
            (heteroscedastic) noise structure.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float or array-like
            Sampled residual ε.
        """
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, residual, x):
        """
        Evaluate log probability density of a residual.

        Parameters
        ----------
        residual : float
            Residual value:
                r = y - f(x; θ)
        x : float or array-like
            Input location (may affect distribution).

        Returns
        -------
        float
            Log probability density log p(ε | x).
        """
        raise NotImplementedError