import numpy as np


class LinearModel:
    """
    Linear regression model of the form:

        y = m x + b

    This class defines a deterministic mapping from inputs x and
    parameters θ to an expected output y.

    No stochastic assumptions are included here. All uncertainty
    is introduced externally via an error model.

    Parameters
    ----------
    None

    Notes
    -----
    The parameter vector is defined as:

        θ = [m, b]

    where:
        m : slope
        b : intercept
    """

    def __call__(self, x, theta):
        """
        Evaluate the model at a single input.

        Parameters
        ----------
        x : float or array-like
            Input variable.
        theta : array-like of shape (2,)
            Model parameters [m, b].

        Returns
        -------
        float or array-like
            Model prediction: m * x + b
        """
        m, b = theta
        return m * x + b

    def batch(self, x, theta):
        """
        Evaluate the model over a vector of inputs.

        Parameters
        ----------
        x : array-like
            Input data points.
        theta : array-like of shape (2,)
            Model parameters [m, b].

        Returns
        -------
        numpy.ndarray
            Predictions evaluated at each x.
        """
        x = np.asarray(x)
        m, b = theta
        return m * x + b