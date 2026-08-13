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
from scipy.optimize import brentq, minimize
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
            options={"ftol": 1e-8},
        )

        if not result.success:
            raise RuntimeError(
                "Profile likelihood optimization failed: "
                f"{result.message}"
            )

        return -float(result.fun)

    def _expand_bracket(self, objective, center: float, direction: float, initial_span: float, max_expand: int = 12) -> tuple[float, float]:
        """Expand a one-sided bracket until the profile threshold is crossed."""
        span = max(float(initial_span), 1e-9)
        left = float(center)
        left_value = float(objective(left))

        if np.isclose(left_value, 0.0):
            return left, left

        for _ in range(max_expand):
            candidate = left + direction * span
            candidate_value = float(objective(candidate))

            if np.isnan(candidate_value) or np.isinf(candidate_value):
                span *= 2.0
                continue

            if left_value * candidate_value <= 0.0:
                return (min(left, candidate), max(left, candidate))

            left = candidate
            left_value = candidate_value
            span *= 2.0

        raise RuntimeError("Unable to bracket a profile-likelihood root.")

    def _expand_from_previous(self, objective, previous_root: float, direction: float, initial_span: float, max_expand: int = 12) -> tuple[float, float]:
        """Expand a bracket using the previous root as the anchored endpoint."""
        span = max(float(initial_span), 1e-9)
        left = float(previous_root)
        left_value = float(objective(left))

        if np.isclose(left_value, 0.0):
            return left, left

        for _ in range(max_expand):
            candidate = left + direction * span
            candidate_value = float(objective(candidate))

            if np.isnan(candidate_value) or np.isinf(candidate_value):
                span *= 2.0
                continue

            if left_value * candidate_value <= 0.0:
                return (min(left, candidate), max(left, candidate))

            left = candidate
            left_value = candidate_value
            span *= 2.0

        raise RuntimeError("Unable to bracket a profile-likelihood root from the previous solution.")

    def _expand_with_fallback(
        self,
        objective,
        previous_root: float,
        center: float,
        direction: float,
        initial_span: float,
        fallback_span: float,
    ) -> tuple[float, float]:
        try:
            return self._expand_from_previous(
                objective,
                previous_root=previous_root,
                direction=direction,
                initial_span=initial_span,
            )
        except RuntimeError:
            return self._expand_bracket(
                objective,
                center=center,
                direction=direction,
                initial_span=fallback_span,
            )

    def interval(self, x, alpha=0.05):
        """
        Compute a profile-likelihood confidence interval at one x.

        Parameters
        ----------
        x : float
            Input location.

        alpha : float, optional
            Significance level.

        Returns
        -------
        tuple
            Pair ``(lower, upper)`` containing confidence limits.

        Notes
        -----
        This method solves for the lower and upper q values directly using
        Wilks' theorem and one-dimensional root finding.
        """
        self._validate_alpha(alpha)

        # Wilks threshold for a one-dimensional parameter of interest.
        cutoff = chi2.ppf(1.0 - alpha, df=1) / 2.0

        # Global maximum log-likelihood at the MLE.
        max_loglik = self.loglik(self.theta_hat)
        target_loglik = max_loglik - cutoff

        mean_q = float(self.model(x, self.theta_hat))
        center_value = self.profile_loglik(x, mean_q) - target_loglik
        if center_value < 0.0:
            raise RuntimeError("Profile likelihood at the fitted mean is below the Wilks threshold.")

        y_span = max(abs(mean_q) * 0.25, 1.0)

        def objective(q: float) -> float:
            return self.profile_loglik(x, q) - target_loglik

        lower_bounds = self._expand_with_fallback(
            objective,
            previous_root=mean_q,
            center=mean_q,
            direction=-1.0,
            initial_span=y_span,
            fallback_span=max(y_span, abs(mean_q) * 0.5 + 1.0),
        )
        upper_bounds = self._expand_with_fallback(
            objective,
            previous_root=mean_q,
            center=mean_q,
            direction=1.0,
            initial_span=y_span,
            fallback_span=max(y_span, abs(mean_q) * 0.5 + 1.0),
        )

        lower = brentq(objective, lower_bounds[0], lower_bounds[1], xtol=1e-6)
        upper = brentq(objective, upper_bounds[0], upper_bounds[1], xtol=1e-6)

        return float(lower), float(upper)

    def band(self, x_grid, alpha=0.05):
        """
        Construct a profile-likelihood confidence band.

        Parameters
        ----------
        x_grid : array-like
            Grid of x values at which confidence limits are computed.

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

            lower[i], upper[i] = self.interval(x=x, alpha=alpha)

        return ConfidenceBand(
            x_grid=x_grid,
            mean=mean,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method="profile_likelihood",
        )