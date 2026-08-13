import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm


class ConfidenceBand:
    """
    Unified confidence band representation.

    Supports:
    - analytic bands, e.g. Delta method
    - interpolated grid-based bands, e.g. Monte Carlo / Profile Likelihood
    """

    def __init__(
        self,
        *,
        alpha=0.05,
        method="unknown",
        x_grid=None,
        mean=None,
        lower=None,
        upper=None,
        mean_func=None,
        variance_func=None,
        interpolation="linear",
        extrapolate=True,
    ):
        if not (0 < alpha < 1):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

        self.alpha = alpha
        self.method = method
        self.z = norm.ppf(1 - alpha / 2)

        self.x_grid = None if x_grid is None else np.asarray(x_grid, dtype=float)
        self.mean_vals = None if mean is None else np.asarray(mean, dtype=float)
        self.lower_vals = None if lower is None else np.asarray(lower, dtype=float)
        self.upper_vals = None if upper is None else np.asarray(upper, dtype=float)

        self.mean_func = mean_func
        self.variance_func = variance_func

        self._mean_interp = None
        self._lower_interp = None
        self._upper_interp = None

        fill_value = "extrapolate" if extrapolate else np.nan

        if self.x_grid is not None:
            if lower is None or upper is None:
                raise ValueError("Grid-based bands require lower and upper arrays.")

            if len(self.x_grid) != len(self.lower_vals) or len(self.x_grid) != len(self.upper_vals):
                raise ValueError("x_grid, lower, and upper must have the same length.")

            if np.any(self.lower_vals > self.upper_vals):
                raise ValueError("lower exceeds upper at one or more grid points.")

            self._lower_interp = interp1d(
                self.x_grid,
                self.lower_vals,
                kind=interpolation,
                bounds_error=False,
                fill_value=fill_value,
            )

            self._upper_interp = interp1d(
                self.x_grid,
                self.upper_vals,
                kind=interpolation,
                bounds_error=False,
                fill_value=fill_value,
            )

            if self.mean_vals is not None:
                if len(self.x_grid) != len(self.mean_vals):
                    raise ValueError("x_grid and mean must have the same length.")

                self._mean_interp = interp1d(
                    self.x_grid,
                    self.mean_vals,
                    kind=interpolation,
                    bounds_error=False,
                    fill_value=fill_value,
                )

    def mean(self, x):
        """Return mean fitted value at x."""
        if self.mean_func is not None:
            return self.mean_func(x)

        if self._mean_interp is None:
            raise ValueError("No mean function or mean interpolation is available.")

        return self._mean_interp(x)

    def interval(self, x):
        """Return lower and upper confidence limits at x."""
        if self.variance_func is not None:
            mu = self.mean_func(x)
            sigma = np.sqrt(self.variance_func(x))
            return mu - self.z * sigma, mu + self.z * sigma

        if self._lower_interp is None or self._upper_interp is None:
            raise ValueError("No confidence band information is available.")

        return self._lower_interp(x), self._upper_interp(x)

    def evaluate(self, x_grid):
        """Evaluate mean, lower, and upper curves on a grid."""
        x_grid = np.asarray(x_grid, dtype=float)
        lower, upper = self.interval(x_grid)

        try:
            mean = self.mean(x_grid)
        except ValueError:
            mean = None

        return {
            "x": x_grid,
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "alpha": self.alpha,
            "method": self.method,
        }
    
class DeltaMethodEngine:
    """
    Multivariate delta method with optional second-order corrections.

    order=1 → linear propagation
    order=2 → includes Hessian + third derivatives
    """

    def __init__(self, func, theta_hat, cov, grad, hess=None, third=None):
        self.func = func
        self.theta_hat = np.asarray(theta_hat)
        self.cov = np.asarray(cov)
        self.grad = grad
        self.hess = hess
        self.third = third

    def variance(self, x, order=1):
        theta = self.theta_hat
        Sigma = self.cov

        g = self.grad(x, theta)

        # first-order term
        var = g @ Sigma @ g

        if order == 1:
            return var

        if self.hess is None or self.third is None:
            raise ValueError("Second order requires Hessian and third derivatives")

        H = self.hess(x, theta)
        T = self.third(x, theta)

        p = len(theta)

        # second-order term 1
        term2 = 0.0
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    for l in range(p):
                        term2 += g[i] * T[j, k, l] * Sigma[i, j] * Sigma[k, l]

        # second-order term 2
        term3 = 0.0
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    for l in range(p):
                        term3 += H[i, j] * H[k, l] * Sigma[i, k] * Sigma[j, l]

        return var + term2 + 0.5 * term3

    def std(self, x, order=1):
        return np.sqrt(self.variance(x, order))

    def interval(self, x, alpha=0.05, order=1):
        mu = self.func(x, self.theta_hat)
        z = norm.ppf(1 - alpha / 2)
        sigma = self.std(x, order)

        return mu - z * sigma, mu + z * sigma

    def band(self, x_grid, alpha=0.05, order=1):
        x_grid = np.asarray(x_grid)

        mu = np.array([self.func(x, self.theta_hat) for x in x_grid])
        var = np.array([self.variance(x, order) for x in x_grid])

        z = norm.ppf(1 - alpha / 2)

        lower = mu - z * np.sqrt(var)
        upper = mu + z * np.sqrt(var)

        return ConfidenceBand(
            alpha=alpha,
            method=f"delta_order_{order}",
            mean_func=lambda x: self.func(x, self.theta_hat),
            variance_func=lambda x: self.variance(x, order=order),
        )
    


class MonteCarloConfidence:
    """
    Monte Carlo confidence intervals for fitted models.

    This class assumes the fitted parameter vector is asymptotically normal:

        theta ~ N(theta_hat, cov)

    It estimates the distribution of f(x; theta) by drawing parameter
    samples and propagating them through the model.

    Parameters
    ----------
    func : callable
        Model function with signature ``func(x, theta)``.

    theta_hat : array-like
        Maximum likelihood estimate of the parameter vector.

    cov : array-like
        Estimated covariance matrix of ``theta_hat``.
        This should already be Cov(theta_hat), not per-observation
        Fisher information.

    random_state : int or None
        Optional random seed for reproducibility.
    """

    def __init__(self, func, theta_hat, cov, random_state=None):
        self.func = func
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.rng = np.random.default_rng(random_state)

        if self.theta_hat.ndim != 1:
            raise ValueError("theta_hat must be a one-dimensional parameter vector.")

        if self.cov.shape != (len(self.theta_hat), len(self.theta_hat)):
            raise ValueError("cov must have shape (p, p), where p = len(theta_hat).")

    @staticmethod
    def _validate_alpha(alpha):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

    def sample_parameters(self, n_samples=10000):
        """
        Draw parameter samples from N(theta_hat, cov).
        """
        return self.rng.multivariate_normal(
            mean=self.theta_hat,
            cov=self.cov,
            size=n_samples,
        )

    def sample_predictions(self, x, n_samples=10000):
        """
        Draw parameter samples and evaluate f(x; theta) for each draw.
        """
        theta_samples = self.sample_parameters(n_samples)

        return np.asarray(
            [self.func(x, theta_i) for theta_i in theta_samples],
            dtype=float,
        )

    def interval(self, x, alpha=0.05, n_samples=10000):
        """
        Compute a Monte Carlo confidence interval at a single x value.

        Parameters
        ----------
        x : float
            Evaluation point.

        alpha : float
            Significance level. For a 95% confidence interval, use
            ``alpha=0.05``.

        n_samples : int
            Number of Monte Carlo parameter samples.

        Returns
        -------
        tuple
            ``(lower, upper)`` confidence interval.
        """
        self._validate_alpha(alpha)

        samples = self.sample_predictions(x, n_samples=n_samples)

        lower = np.quantile(samples, alpha / 2.0)
        upper = np.quantile(samples, 1.0 - alpha / 2.0)

        return float(lower), float(upper)

    def fit_band(
        self,
        x_data=None,
        x_grid=None,
        alpha=0.05,
        n_points=100,
        n_samples=10000,
    ):
        """
        Precompute a Monte Carlo confidence band over many x values.

        If ``x_grid`` is supplied, that grid is used directly.
        Otherwise, ``x_data`` is used to define an evenly spaced grid
        from min(x_data) to max(x_data).

        Parameters
        ----------
        x_data : array-like or None
            Original fitted x data. Used only if ``x_grid`` is not supplied.

        x_grid : array-like or None
            Explicit x values at which exact Monte Carlo intervals are computed.

        alpha : float
            Significance level. For a 95% confidence band, use ``alpha=0.05``.

        n_points : int
            Number of x points used when constructing a grid from ``x_data``.

        n_samples : int
            Number of Monte Carlo parameter samples per x value.

        Returns
        -------
        ConfidenceBand
            Interpolated confidence band object.
        """
        self._validate_alpha(alpha)

        if x_grid is None:
            if x_data is None:
                raise ValueError("Either x_data or x_grid must be supplied.")

            x_data = np.asarray(x_data, dtype=float)
            x_grid = np.linspace(np.min(x_data), np.max(x_data), n_points)
        else:
            x_grid = np.asarray(x_grid, dtype=float)

        lower = np.zeros_like(x_grid, dtype=float)
        upper = np.zeros_like(x_grid, dtype=float)
        mean = np.zeros_like(x_grid, dtype=float)

        for i, x in enumerate(x_grid):
            samples = self.sample_predictions(x, n_samples=n_samples)

            lower[i] = np.quantile(samples, alpha / 2.0)
            upper[i] = np.quantile(samples, 1.0 - alpha / 2.0)
            mean[i] = np.mean(samples)

        return ConfidenceBand(
            x_grid=x_grid,
            mean=mean,
            lower=lower,
            upper=upper,
            method="monte_carlo",
            alpha=alpha,
        )
    

from scipy.optimize import minimize


class ProfileLikelihoodEngine:
    """
    Wilks-based profile likelihood CI.
    """

    def __init__(self, func, loglik, theta_hat):
        self.func = func
        self.loglik = loglik
        self.theta_hat = np.asarray(theta_hat)

    def profile_likelihood(self, x, q, bounds):
        def objective(theta):
            return -self.loglik(theta)

        def constraint(theta):
            return self.func(x, theta) - q

        res = minimize(
            objective,
            self.theta_hat,
            constraints={'type': 'eq', 'fun': constraint},
            bounds=bounds,
            options={'ftol': 1e-8}
        )

        return -res.fun

    def band(self, x_grid, q_grid, bounds, alpha=0.05):

        cutoff = 0.5 * 3.84  # chi-square(1) approx for 95%

        lower = []
        upper = []

        for x in x_grid:

            profile_vals = []

            for q in q_grid:
                lp = self.profile_likelihood(x, q, bounds)
                profile_vals.append(lp)

            profile_vals = np.array(profile_vals)

            best = np.max(profile_vals)

            mask = profile_vals >= (best - cutoff)

            qs = np.array(q_grid)

            lower.append(np.min(qs[mask]))
            upper.append(np.max(qs[mask]))

        mu = (np.array(lower) + np.array(upper)) / 2

        return ConfidenceBand(
            x_grid=x_grid,
            mean=mu,
            lower=np.array(lower),
            upper=np.array(upper)
        )
    
