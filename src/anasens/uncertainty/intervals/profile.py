"""
Profile-likelihood confidence intervals.

This module implements Wilks-theorem confidence intervals for the fitted
model mean

    f(x; theta_hat)

using profile likelihood.

Profile likelihood is used only for confidence intervals of the fitted
model mean. It is not used for prediction intervals.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from ..bands.confidence_band import ConfidenceBand


class ProfileLikelihoodCI:
    """
    Profile-likelihood confidence interval engine.

    This class constructs confidence intervals for the fitted model mean by
    solving constrained likelihood maximization problems.

    For a fixed input value ``x`` and candidate model value ``q``, the
    profile log-likelihood is

        ell_p(q; x) = max_theta ell(theta)

    subject to

        f(x; theta) = q

    Wilks' theorem is then used to identify values of ``q`` satisfying

        2 [ell(theta_hat) - ell_p(q; x)] <= chi2_{1, 1-alpha}

    Parameters
    ----------
    model : callable
        Deterministic model with signature ``model(x, theta)``.

    loglik : callable
        Log-likelihood function with signature ``loglik(theta)``.

    theta_hat : array-like
        Maximum likelihood estimate of the model parameters.

    bounds : sequence or None, optional
        Bounds passed to :func:`scipy.optimize.minimize`.

    optimizer_method : str, optional
        Optimization method used for constrained minimization.

        Default is ``"SLSQP"``, which supports equality constraints.

    Notes
    -----
    This method uses the likelihood surface directly. It can capture
    asymmetric confidence intervals caused by nonlinear model geometry.

    It is generally more computationally expensive than the Delta Method
    or Monte Carlo parameter propagation.
    """

    def __init__(
        self,
        model,
        loglik,
        theta_hat,
        bounds=None,
        optimizer_method="SLSQP",
    ):
        self.model = model
        self.loglik = loglik
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        self.bounds = bounds
        self.optimizer_method = optimizer_method

        if self.theta_hat.ndim != 1:
            raise ValueError("theta_hat must be a one-dimensional array.")

    @staticmethod
    def _validate_alpha(alpha):
        """
        Validate significance level.

        Parameters
        ----------
        alpha : float
            Significance level.

        Raises
        ------
        ValueError
            If ``alpha`` is outside the open interval (0, 1).
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

    def profile_loglik(self, x, q):
        """
        Compute the profile log-likelihood for a candidate model value.

        Parameters
        ----------
        x : float
            Input location.

        q : float
            Candidate fitted model value.

        Returns
        -------
        float
            Profile log-likelihood value.

        Raises
        ------
        RuntimeError
            If constrained optimization fails.
        """

        # Minimize the negative log-likelihood, equivalent to maximizing
        # the log-likelihood.
        def objective(theta):
            return -self.loglik(theta)

        # Enforce f(x; theta) = q.
        def constraint(theta):
            return self.model(x, theta) - q

        result = minimize(
            objective,
            x0=self.theta_hat,
            method=self.optimizer_method,
            bounds=self.bounds,
            constraints={"type": "eq", "fun": constraint},
        )

        if not result.success:
            raise RuntimeError(
                "Profile likelihood optimization failed: "
                f"{result.message}"
            )

        return -float(result.fun)

    def interval(self, x, q_grid, alpha=0.05):
        """
        Compute a profile-likelihood confidence interval at one x.

        Parameters
        ----------
        x : float
            Input location.

        q_grid : array-like
            Candidate model values to scan.

            The returned interval is limited by this grid. Use a sufficiently
            wide and fine grid for accurate results.

        alpha : float, optional
            Significance level.

        Returns
        -------
        tuple
            Pair ``(lower, upper)`` containing confidence limits.

        Notes
        -----
        This method evaluates the profile likelihood over ``q_grid`` and
        returns the smallest and largest accepted values according to
        Wilks' theorem.
        """
        self._validate_alpha(alpha)

        q_grid = np.asarray(q_grid, dtype=float)

        if q_grid.ndim != 1:
            raise ValueError("q_grid must be one-dimensional.")

        if q_grid.size < 2:
            raise ValueError("q_grid must contain at least two values.")

        # Wilks threshold for a one-dimensional parameter of interest.
        cutoff = chi2.ppf(1.0 - alpha, df=1) / 2.0

        # Global maximum log-likelihood at the MLE.
        max_loglik = self.loglik(self.theta_hat)

        profile_values = np.zeros_like(q_grid, dtype=float)

        for i, q in enumerate(q_grid):
            profile_values[i] = self.profile_loglik(x, q)

        # Accept all q values whose likelihood-ratio statistic falls within
        # the Wilks threshold.
        accepted = profile_values >= (max_loglik - cutoff)

        if not np.any(accepted):
            raise RuntimeError(
                "No q values satisfied the Wilks threshold. "
                "Try widening q_grid."
            )

        accepted_q = q_grid[accepted]

        return float(np.min(accepted_q)), float(np.max(accepted_q))

    def band(self, x_grid, q_grid, alpha=0.05):
        """
        Construct a profile-likelihood confidence band.

        Parameters
        ----------
        x_grid : array-like
            Grid of x values at which confidence limits are computed.

        q_grid : array-like
            Candidate model values scanned at each x.

        alpha : float, optional
            Significance level.

        Returns
        -------
        ConfidenceBand
            Interpolated confidence band storing fitted mean and confidence
            limits over ``x_grid``.
        """
        self._validate_alpha(alpha)

        x_grid = np.asarray(x_grid, dtype=float)

        mean = np.zeros_like(x_grid, dtype=float)
        lower = np.zeros_like(x_grid, dtype=float)
        upper = np.zeros_like(x_grid, dtype=float)

        for i, x in enumerate(x_grid):
            # Fitted model mean at the MLE.
            mean[i] = self.model(x, self.theta_hat)

            # Profile-likelihood confidence interval for this x.
            lower[i], upper[i] = self.interval(
                x=x,
                q_grid=q_grid,
                alpha=alpha,
            )

        return ConfidenceBand(
            x_grid=x_grid,
            mean=mean,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method="profile_likelihood",
        )