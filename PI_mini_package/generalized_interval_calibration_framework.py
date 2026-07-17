"""
Generalized prediction-interval and inverse-calibration framework.

The framework separates:

1. Deterministic models.
2. Additive error distributions.
3. Parameter fitting.
4. Confidence- and prediction-interval engines.
5. Direct inverse calibration by profile likelihood.

The following methods are included:

- Exact analytical OLS intervals for polynomial models with Gaussian errors.
- General first-order delta-method confidence and prediction intervals.
- General parametric Monte Carlo confidence and prediction intervals.
- Least-squares residual-bootstrap Monte Carlo intervals.
- Polynomial quantile regression.
- General profile-likelihood inverse calibration.

Only ``AnalyticalPolynomialGaussian`` is intrinsically restricted to
polynomials with homoscedastic Gaussian errors. Quantile regression is also a
specialized method because it estimates conditional quantiles directly rather
than fitting an additive error distribution.

All other likelihood-based methods operate through the ``RegressionModel`` and
``ErrorModel`` interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq, linprog, minimize
from scipy.stats import chi2, laplace, logistic, norm, t as student_t


# ============================================================================
# 1. NUMERICAL UTILITIES
# ============================================================================


def _as_1d_finite(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    """Convert input to a finite one-dimensional floating-point array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_xy(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired one-dimensional regression data."""
    x_array = _as_1d_finite(x, "x")
    y_array = _as_1d_finite(y, "y")
    if x_array.size != y_array.size:
        raise ValueError("x and y must contain the same number of observations.")
    return x_array, y_array


def numerical_hessian(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    relative_step: float = 1e-4,
) -> np.ndarray:
    """Estimate a Hessian using second-order central finite differences.

    Args:
        function:
            Scalar objective function.
        point:
            Point at which to evaluate the Hessian.
        relative_step:
            Relative finite-difference step.

    Returns:
        Symmetric Hessian matrix.
    """
    x0 = np.asarray(point, dtype=float)
    n_parameters = x0.size
    steps = relative_step * np.maximum(1.0, np.abs(x0))
    hessian = np.empty((n_parameters, n_parameters), dtype=float)
    f0 = float(function(x0))

    for i in range(n_parameters):
        ei = np.zeros(n_parameters)
        ei[i] = steps[i]
        f_plus = float(function(x0 + ei))
        f_minus = float(function(x0 - ei))
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / steps[i] ** 2

        for j in range(i + 1, n_parameters):
            ej = np.zeros(n_parameters)
            ej[j] = steps[j]
            mixed = (
                function(x0 + ei + ej)
                - function(x0 + ei - ej)
                - function(x0 - ei + ej)
                + function(x0 - ei - ej)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = mixed
            hessian[j, i] = mixed

    return 0.5 * (hessian + hessian.T)


def stable_inverse(matrix: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Return a symmetric inverse, falling back to a pseudoinverse."""
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(matrix, rcond=rcond)
    return 0.5 * (inverse + inverse.T)


# ============================================================================
# 2. DETERMINISTIC REGRESSION MODELS
# ============================================================================


class RegressionModel(ABC):
    """Interface for a deterministic mean function ``f(x; theta)``."""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of model parameters."""

    @abstractmethod
    def evaluate(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Evaluate the deterministic mean function."""

    @abstractmethod
    def jacobian(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return ``df/dtheta`` with shape ``(n_points, n_parameters)``."""

    @abstractmethod
    def initial_parameters(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return a sensible starting point for optimization."""


class PolynomialModel(RegressionModel):
    """Polynomial model with coefficients ordered from low to high power."""

    def __init__(self, degree: int = 1):
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")
        self.degree = degree

    @property
    def n_parameters(self) -> int:
        return self.degree + 1

    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.vander(np.asarray(x, dtype=float), self.degree + 1, increasing=True)

    def evaluate(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.design_matrix(np.asarray(x, dtype=float)) @ np.asarray(theta, dtype=float)

    def jacobian(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        del theta
        return self.design_matrix(np.asarray(x, dtype=float))

    def initial_parameters(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = self.design_matrix(x)
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return coefficients


# ============================================================================
# 3. ADDITIVE ERROR MODELS
# ============================================================================


class ErrorModel(ABC):
    """Interface for an additive residual distribution.

    Error parameters are optimized in an unconstrained internal coordinate
    system ``eta``. For example, a positive scale may be represented by
    ``scale = exp(eta[0])``.
    """

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of unconstrained error parameters."""

    @abstractmethod
    def initial_eta(self, residuals: np.ndarray) -> np.ndarray:
        """Return initial unconstrained error parameters."""

    @abstractmethod
    def logpdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Evaluate residual log densities."""

    @abstractmethod
    def cdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Evaluate the residual CDF."""

    @abstractmethod
    def ppf(self, probabilities: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Evaluate residual quantiles."""

    @abstractmethod
    def sample(
        self,
        rng: np.random.Generator,
        size: tuple[int, ...],
        eta: np.ndarray,
    ) -> np.ndarray:
        """Generate additive residual draws."""

    @abstractmethod
    def variance(self, eta: np.ndarray) -> float:
        """Return residual variance, or infinity if it does not exist."""

    @abstractmethod
    def describe(self, eta: np.ndarray) -> dict[str, float]:
        """Return user-facing fitted error parameters."""


class GaussianError(ErrorModel):
    """Zero-mean Gaussian additive errors with unknown scale."""

    @property
    def n_parameters(self) -> int:
        return 1

    def initial_eta(self, residuals: np.ndarray) -> np.ndarray:
        scale = max(float(np.sqrt(np.mean(np.square(residuals)))), 1e-8)
        return np.array([np.log(scale)])

    def _scale(self, eta: np.ndarray) -> float:
        return float(np.exp(eta[0]))

    def logpdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return norm.logpdf(residuals, loc=0.0, scale=self._scale(eta))

    def cdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return norm.cdf(residuals, loc=0.0, scale=self._scale(eta))

    def ppf(self, probabilities: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return norm.ppf(probabilities, loc=0.0, scale=self._scale(eta))

    def sample(
        self,
        rng: np.random.Generator,
        size: tuple[int, ...],
        eta: np.ndarray,
    ) -> np.ndarray:
        return rng.normal(0.0, self._scale(eta), size=size)

    def variance(self, eta: np.ndarray) -> float:
        return self._scale(eta) ** 2

    def describe(self, eta: np.ndarray) -> dict[str, float]:
        return {"scale": self._scale(eta)}


class LaplaceError(ErrorModel):
    """Zero-location Laplace additive errors with unknown scale."""

    @property
    def n_parameters(self) -> int:
        return 1

    def initial_eta(self, residuals: np.ndarray) -> np.ndarray:
        scale = max(float(np.mean(np.abs(residuals))), 1e-8)
        return np.array([np.log(scale)])

    def _scale(self, eta: np.ndarray) -> float:
        return float(np.exp(eta[0]))

    def logpdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return laplace.logpdf(residuals, loc=0.0, scale=self._scale(eta))

    def cdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return laplace.cdf(residuals, loc=0.0, scale=self._scale(eta))

    def ppf(self, probabilities: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return laplace.ppf(probabilities, loc=0.0, scale=self._scale(eta))

    def sample(
        self,
        rng: np.random.Generator,
        size: tuple[int, ...],
        eta: np.ndarray,
    ) -> np.ndarray:
        return rng.laplace(0.0, self._scale(eta), size=size)

    def variance(self, eta: np.ndarray) -> float:
        return 2.0 * self._scale(eta) ** 2

    def describe(self, eta: np.ndarray) -> dict[str, float]:
        return {"scale": self._scale(eta)}


class LogisticError(ErrorModel):
    """Zero-location logistic additive errors with unknown scale."""

    @property
    def n_parameters(self) -> int:
        return 1

    def initial_eta(self, residuals: np.ndarray) -> np.ndarray:
        standard_deviation = max(float(np.std(residuals)), 1e-8)
        scale = standard_deviation * np.sqrt(3.0) / np.pi
        return np.array([np.log(max(scale, 1e-8))])

    def _scale(self, eta: np.ndarray) -> float:
        return float(np.exp(eta[0]))

    def logpdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return logistic.logpdf(residuals, loc=0.0, scale=self._scale(eta))

    def cdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return logistic.cdf(residuals, loc=0.0, scale=self._scale(eta))

    def ppf(self, probabilities: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return logistic.ppf(probabilities, loc=0.0, scale=self._scale(eta))

    def sample(
        self,
        rng: np.random.Generator,
        size: tuple[int, ...],
        eta: np.ndarray,
    ) -> np.ndarray:
        return rng.logistic(0.0, self._scale(eta), size=size)

    def variance(self, eta: np.ndarray) -> float:
        return (np.pi**2 / 3.0) * self._scale(eta) ** 2

    def describe(self, eta: np.ndarray) -> dict[str, float]:
        return {"scale": self._scale(eta)}


class StudentTError(ErrorModel):
    """Zero-location Student-t errors with unknown scale and degrees of freedom.

    The transformation

        df = 2 + exp(eta[1])

    constrains the fitted distribution to finite variance.
    """

    @property
    def n_parameters(self) -> int:
        return 2

    def initial_eta(self, residuals: np.ndarray) -> np.ndarray:
        scale = max(float(np.std(residuals)), 1e-8)
        initial_df = 8.0
        return np.array([np.log(scale), np.log(initial_df - 2.0)])

    def _parameters(self, eta: np.ndarray) -> tuple[float, float]:
        scale = float(np.exp(eta[0]))
        degrees_of_freedom = float(2.0 + np.exp(eta[1]))
        return scale, degrees_of_freedom

    def logpdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        scale, df = self._parameters(eta)
        return student_t.logpdf(residuals, df=df, loc=0.0, scale=scale)

    def cdf(self, residuals: np.ndarray, eta: np.ndarray) -> np.ndarray:
        scale, df = self._parameters(eta)
        return student_t.cdf(residuals, df=df, loc=0.0, scale=scale)

    def ppf(self, probabilities: np.ndarray, eta: np.ndarray) -> np.ndarray:
        scale, df = self._parameters(eta)
        return student_t.ppf(probabilities, df=df, loc=0.0, scale=scale)

    def sample(
        self,
        rng: np.random.Generator,
        size: tuple[int, ...],
        eta: np.ndarray,
    ) -> np.ndarray:
        scale, df = self._parameters(eta)
        return scale * rng.standard_t(df, size=size)

    def variance(self, eta: np.ndarray) -> float:
        scale, df = self._parameters(eta)
        return scale**2 * df / (df - 2.0)

    def describe(self, eta: np.ndarray) -> dict[str, float]:
        scale, df = self._parameters(eta)
        return {"scale": scale, "degrees_of_freedom": df}


# ============================================================================
# 4. FIT RESULTS AND FITTERS
# ============================================================================


@dataclass(frozen=True)
class FitResult:
    """Container shared by confidence- and prediction-interval engines."""

    model: RegressionModel
    theta_hat: np.ndarray
    covariance_theta: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    fitted_values: np.ndarray
    raw_residuals: np.ndarray
    fitting_method: str
    error_model: ErrorModel | None = None
    eta_hat: np.ndarray | None = None
    joint_hat: np.ndarray | None = None
    joint_covariance: np.ndarray | None = None
    leverage: np.ndarray | None = None
    bootstrap_residuals: np.ndarray | None = None

    @property
    def n_model_parameters(self) -> int:
        return self.theta_hat.size


class MaximumLikelihoodFitter:
    """Fit a deterministic model under an explicit additive error law."""

    def __init__(
        self,
        optimizer_method: str = "BFGS",
        optimizer_options: dict | None = None,
        hessian_relative_step: float = 1e-4,
    ):
        self.optimizer_method = optimizer_method
        self.optimizer_options = optimizer_options or {"maxiter": 5000}
        self.hessian_relative_step = hessian_relative_step

    @staticmethod
    def _negative_log_likelihood(
        joint_parameters: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        model: RegressionModel,
        error_model: ErrorModel,
    ) -> float:
        p = model.n_parameters
        theta = joint_parameters[:p]
        eta = joint_parameters[p:]
        residuals = y - model.evaluate(x, theta)
        log_density = error_model.logpdf(residuals, eta)
        if not np.all(np.isfinite(log_density)):
            return np.inf
        return float(-np.sum(log_density))

    def fit(
        self,
        x: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[float],
        model: RegressionModel,
        error_model: ErrorModel,
    ) -> FitResult:
        """Fit model and error parameters jointly by maximum likelihood."""
        x_array, y_array = _validate_xy(x, y)
        theta_initial = model.initial_parameters(x_array, y_array)
        residuals_initial = y_array - model.evaluate(x_array, theta_initial)
        eta_initial = error_model.initial_eta(residuals_initial)
        joint_initial = np.concatenate([theta_initial, eta_initial])

        objective = lambda parameters: self._negative_log_likelihood(
            parameters,
            x_array,
            y_array,
            model,
            error_model,
        )

        result = minimize(
            objective,
            joint_initial,
            method=self.optimizer_method,
            options=self.optimizer_options,
        )

        # Smooth likelihoods are usually handled efficiently by BFGS. Error
        # laws with a cusp, such as Laplace errors, can trigger a precision-loss
        # warning even near the optimum. Retry with Powell, which does not
        # require derivatives.
        if not result.success:
            fallback = minimize(
                objective,
                np.asarray(result.x, dtype=float),
                method="Powell",
                options=self.optimizer_options,
            )
            if fallback.success or (
                np.isfinite(fallback.fun) and fallback.fun < result.fun
            ):
                result = fallback

        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"Maximum-likelihood fit failed: {result.message}")

        joint_hat = np.asarray(result.x, dtype=float)
        p = model.n_parameters
        theta_hat = joint_hat[:p]
        eta_hat = joint_hat[p:]

        observed_hessian = numerical_hessian(
            objective,
            joint_hat,
            relative_step=self.hessian_relative_step,
        )
        joint_covariance = stable_inverse(observed_hessian)
        covariance_theta = joint_covariance[:p, :p]

        fitted_values = model.evaluate(x_array, theta_hat)
        residuals = y_array - fitted_values

        return FitResult(
            model=model,
            theta_hat=theta_hat,
            covariance_theta=covariance_theta,
            x_train=x_array,
            y_train=y_array,
            fitted_values=fitted_values,
            raw_residuals=residuals,
            fitting_method="maximum_likelihood",
            error_model=error_model,
            eta_hat=eta_hat,
            joint_hat=joint_hat,
            joint_covariance=joint_covariance,
        )


class LeastSquaresFitter:
    """Fit a mean model by nonlinear least squares for residual bootstrapping.

    This branch intentionally does not require a parametric error distribution.
    It assumes independent, homoscedastic, zero-mean errors.

    The fitted parameter estimator is asymptotically normal under standard
    nonlinear least-squares regularity conditions. The covariance estimate is

        s^2 (J.T J)^(-1),

    and the residuals are leverage adjusted before bootstrap resampling.
    """

    def __init__(
        self,
        optimizer_method: str = "BFGS",
        optimizer_options: dict | None = None,
    ):
        self.optimizer_method = optimizer_method
        self.optimizer_options = optimizer_options or {"maxiter": 5000}

    @staticmethod
    def _sum_squared_residuals(
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        model: RegressionModel,
    ) -> float:
        residuals = y - model.evaluate(x, theta)
        return float(residuals @ residuals)

    def fit(
        self,
        x: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[float],
        model: RegressionModel,
    ) -> FitResult:
        """Fit by least squares and calculate leverage-adjusted residuals."""
        x_array, y_array = _validate_xy(x, y)
        n_observations = x_array.size
        p = model.n_parameters
        if n_observations <= p:
            raise ValueError("The number of observations must exceed model parameters.")

        theta_initial = model.initial_parameters(x_array, y_array)
        objective = lambda theta: self._sum_squared_residuals(
            theta,
            x_array,
            y_array,
            model,
        )

        # Preserve the exact linear least-squares solution when the model's
        # initializer already solves the problem, while still supporting
        # genuinely nonlinear models.
        result = minimize(
            objective,
            theta_initial,
            method=self.optimizer_method,
            options=self.optimizer_options,
        )
        theta_hat = np.asarray(result.x if result.success else theta_initial, dtype=float)

        fitted_values = model.evaluate(x_array, theta_hat)
        residuals = y_array - fitted_values
        degrees_of_freedom = n_observations - p
        residual_variance = float(residuals @ residuals) / degrees_of_freedom

        jacobian = model.jacobian(x_array, theta_hat)
        information = jacobian.T @ jacobian
        inverse_information = stable_inverse(information)
        covariance_theta = residual_variance * inverse_information

        leverage = np.sum((jacobian @ inverse_information) * jacobian, axis=1)
        leverage = np.clip(leverage, 0.0, 1.0)
        adjusted = residuals / np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
        bootstrap_residuals = adjusted - np.mean(adjusted)

        return FitResult(
            model=model,
            theta_hat=theta_hat,
            covariance_theta=covariance_theta,
            x_train=x_array,
            y_train=y_array,
            fitted_values=fitted_values,
            raw_residuals=residuals,
            fitting_method="least_squares",
            leverage=leverage,
            bootstrap_residuals=bootstrap_residuals,
        )


# ============================================================================
# 5. EXACT ANALYTICAL POLYNOMIAL-GAUSSIAN METHOD
# ============================================================================


class AnalyticalPolynomialGaussian:
    """Exact finite-sample OLS prediction intervals.

    This method is deliberately specialized. It is exact only for a polynomial
    mean model, independent homoscedastic Gaussian errors, and a full-rank
    design matrix.
    """

    def __init__(self, degree: int = 1):
        self.model = PolynomialModel(degree)
        self.theta_hat: np.ndarray | None = None
        self.residual_scale: float | None = None
        self.inverse_xtx: np.ndarray | None = None
        self.n = 0
        self.p = self.model.n_parameters

    def fit(
        self,
        x: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[float],
    ) -> "AnalyticalPolynomialGaussian":
        x_array, y_array = _validate_xy(x, y)
        design = self.model.design_matrix(x_array)
        if np.linalg.matrix_rank(design) < self.p:
            raise np.linalg.LinAlgError("Polynomial design matrix is rank deficient.")
        self.n = x_array.size
        if self.n <= self.p:
            raise ValueError("The number of observations must exceed parameters.")

        self.theta_hat, _, _, _ = np.linalg.lstsq(design, y_array, rcond=None)
        residuals = y_array - design @ self.theta_hat
        self.residual_scale = float(
            np.sqrt((residuals @ residuals) / (self.n - self.p))
        )
        self.inverse_xtx = np.linalg.inv(design.T @ design)
        return self

    def confidence_band(
        self,
        x: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_fitted()
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        design = self.model.design_matrix(x_array)
        fitted = design @ self.theta_hat
        leverage = np.sum((design @ self.inverse_xtx) * design, axis=1)
        critical = student_t.ppf(
            1.0 - (1.0 - confidence) / 2.0,
            df=self.n - self.p,
        )
        half_width = critical * self.residual_scale * np.sqrt(leverage)
        return fitted, fitted - half_width, fitted + half_width

    def prediction_interval(
        self,
        x: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_fitted()
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        design = self.model.design_matrix(x_array)
        fitted = design @ self.theta_hat
        leverage = np.sum((design @ self.inverse_xtx) * design, axis=1)
        critical = student_t.ppf(
            1.0 - (1.0 - confidence) / 2.0,
            df=self.n - self.p,
        )
        half_width = critical * self.residual_scale * np.sqrt(1.0 + leverage)
        return fitted, fitted - half_width, fitted + half_width

    def _check_fitted(self) -> None:
        if (
            self.theta_hat is None
            or self.residual_scale is None
            or self.inverse_xtx is None
        ):
            raise RuntimeError("The analytical model must be fitted first.")


# ============================================================================
# 6. GENERAL DELTA METHOD
# ============================================================================


class DeltaMethodEngine:
    """First-order delta-method confidence and prediction intervals.

    Parameter uncertainty is approximated locally as Gaussian:

        theta_hat ~ N(theta, covariance_theta).

    For prediction intervals with a non-Gaussian error law, the engine
    numerically convolves the local Gaussian uncertainty in the fitted mean
    with the fitted additive error distribution. Thus it does not replace a
    Laplace, logistic, or Student-t error law by a Gaussian error law.

    Uncertainty in fitted error-distribution parameters is not propagated by
    this first-order delta implementation. Parametric Monte Carlo does propagate
    it through joint parameter draws.
    """

    def __init__(self, quadrature_nodes: int = 40):
        if quadrature_nodes < 5:
            raise ValueError("quadrature_nodes must be at least 5.")
        nodes, weights = hermgauss(quadrature_nodes)
        self._normal_nodes = np.sqrt(2.0) * nodes
        self._normal_weights = weights / np.sqrt(np.pi)

    @staticmethod
    def _mean_variance(
        fit_result: FitResult,
        x: np.ndarray,
    ) -> np.ndarray:
        jacobian = fit_result.model.jacobian(x, fit_result.theta_hat)
        return np.einsum(
            "ij,jk,ik->i",
            jacobian,
            fit_result.covariance_theta,
            jacobian,
        )

    def confidence_band(
        self,
        fit_result: FitResult,
        x: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a first-order normal confidence band for the fitted mean."""
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        fitted = fit_result.model.evaluate(x_array, fit_result.theta_hat)
        variance = np.clip(self._mean_variance(fit_result, x_array), 0.0, None)
        critical = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
        half_width = critical * np.sqrt(variance)
        return fitted, fitted - half_width, fitted + half_width

    def prediction_interval(
        self,
        fit_result: FitResult,
        x: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convolve delta-method mean uncertainty with the fitted error law."""
        if fit_result.error_model is None or fit_result.eta_hat is None:
            raise ValueError(
                "A fitted parametric error model is required for delta prediction."
            )
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1.")

        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        fitted = fit_result.model.evaluate(x_array, fit_result.theta_hat)
        standard_errors = np.sqrt(
            np.clip(self._mean_variance(fit_result, x_array), 0.0, None)
        )
        alpha = 1.0 - confidence
        lower_probability = alpha / 2.0
        upper_probability = 1.0 - alpha / 2.0

        lower = np.empty_like(fitted)
        upper = np.empty_like(fitted)

        for index, (mean, mean_se) in enumerate(zip(fitted, standard_errors)):
            lower[index] = self._convolved_quantile(
                lower_probability,
                mean,
                mean_se,
                fit_result.error_model,
                fit_result.eta_hat,
            )
            upper[index] = self._convolved_quantile(
                upper_probability,
                mean,
                mean_se,
                fit_result.error_model,
                fit_result.eta_hat,
            )

        return fitted, lower, upper

    def _convolved_quantile(
        self,
        probability: float,
        mean: float,
        mean_standard_error: float,
        error_model: ErrorModel,
        eta: np.ndarray,
    ) -> float:
        """Invert the Gaussian-plus-error convolution CDF."""
        if mean_standard_error <= 1e-14:
            return float(mean + error_model.ppf(np.array([probability]), eta)[0])

        def convolution_cdf(value: float) -> float:
            shifted = value - mean - mean_standard_error * self._normal_nodes
            cdf_values = error_model.cdf(shifted, eta)
            return float(np.sum(self._normal_weights * cdf_values))

        error_sd = np.sqrt(error_model.variance(eta))
        scale = max(mean_standard_error + error_sd, 1.0)
        lower_bracket = mean - 12.0 * scale
        upper_bracket = mean + 12.0 * scale

        # Expand defensively for unusually heavy tails.
        while convolution_cdf(lower_bracket) > probability:
            lower_bracket -= 2.0 * scale
        while convolution_cdf(upper_bracket) < probability:
            upper_bracket += 2.0 * scale

        return float(
            brentq(
                lambda value: convolution_cdf(value) - probability,
                lower_bracket,
                upper_bracket,
            )
        )


# ============================================================================
# 7. GENERAL PARAMETRIC MONTE CARLO
# ============================================================================


class ParametricMonteCarloEngine:
    """Monte Carlo confidence and prediction intervals for fitted likelihoods.

    Joint model and error parameters are sampled from the fitted asymptotic
    multivariate normal distribution. Future errors are then sampled from the
    selected error model using each simulation's error parameters.

    No Student-t parameter sampling is used. Parameter draws are multivariate
    normal; the residual error distribution is whatever ``ErrorModel`` defines.
    """

    def __init__(
        self,
        n_simulations: int = 20_000,
        random_state: int | np.random.Generator | None = None,
    ):
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")
        self.n_simulations = int(n_simulations)
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    def _joint_draws(self, fit_result: FitResult) -> np.ndarray:
        if fit_result.joint_hat is None or fit_result.joint_covariance is None:
            raise ValueError("Parametric Monte Carlo requires a likelihood FitResult.")
        return self.rng.multivariate_normal(
            fit_result.joint_hat,
            fit_result.joint_covariance,
            size=self.n_simulations,
            check_valid="warn",
        )

    def confidence_band(
        self,
        fit_result: FitResult,
        x: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        draws = self._joint_draws(fit_result)
        p = fit_result.n_model_parameters
        simulated_means = np.vstack(
            [
                fit_result.model.evaluate(x_array, theta)
                for theta in draws[:, :p]
            ]
        )
        fitted = fit_result.model.evaluate(x_array, fit_result.theta_hat)
        lower, upper = self._central_quantiles(simulated_means, confidence)
        return fitted, lower, upper

    def prediction_interval(
        self,
        fit_result: FitResult,
        x: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if fit_result.error_model is None:
            raise ValueError("A fitted parametric error model is required.")

        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        draws = self._joint_draws(fit_result)
        p = fit_result.n_model_parameters
        n_points = x_array.size

        simulated_responses = np.empty((self.n_simulations, n_points))
        for index, draw in enumerate(draws):
            theta = draw[:p]
            eta = draw[p:]
            simulated_mean = fit_result.model.evaluate(x_array, theta)
            simulated_error = fit_result.error_model.sample(
                self.rng,
                size=(n_points,),
                eta=eta,
            )
            simulated_responses[index] = simulated_mean + simulated_error

        fitted = fit_result.model.evaluate(x_array, fit_result.theta_hat)
        lower, upper = self._central_quantiles(simulated_responses, confidence)
        return fitted, lower, upper

    @staticmethod
    def _central_quantiles(
        simulations: np.ndarray,
        confidence: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1.")
        alpha = 1.0 - confidence
        return (
            np.quantile(simulations, alpha / 2.0, axis=0),
            np.quantile(simulations, 1.0 - alpha / 2.0, axis=0),
        )


# ============================================================================
# 8. LEAST-SQUARES RESIDUAL-BOOTSTRAP MONTE CARLO
# ============================================================================


class ResidualBootstrapMonteCarloEngine:
    """Monte Carlo intervals using a least-squares fit and empirical errors.

    Parameter uncertainty is sampled from the asymptotic normal distribution
    supplied by ``LeastSquaresFitter``. Future measurement errors are sampled
    from centered, leverage-adjusted residuals.

    This branch intentionally does not use a parametric error model.
    """

    def __init__(
        self,
        n_simulations: int = 20_000,
        random_state: int | np.random.Generator | None = None,
    ):
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")
        self.n_simulations = int(n_simulations)
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    def _theta_draws(self, fit_result: FitResult) -> np.ndarray:
        if fit_result.fitting_method != "least_squares":
            raise ValueError("Residual bootstrap requires a least-squares FitResult.")
        return self.rng.multivariate_normal(
            fit_result.theta_hat,
            fit_result.covariance_theta,
            size=self.n_simulations,
            check_valid="warn",
        )

    def confidence_band(
        self,
        fit_result: FitResult,
        x: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        theta_draws = self._theta_draws(fit_result)
        simulated_means = np.vstack(
            [fit_result.model.evaluate(x_array, theta) for theta in theta_draws]
        )
        fitted = fit_result.model.evaluate(x_array, fit_result.theta_hat)
        lower, upper = ParametricMonteCarloEngine._central_quantiles(
            simulated_means,
            confidence,
        )
        return fitted, lower, upper

    def prediction_interval(
        self,
        fit_result: FitResult,
        x: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if fit_result.bootstrap_residuals is None:
            raise ValueError("FitResult does not contain bootstrap residuals.")

        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        theta_draws = self._theta_draws(fit_result)
        simulated_means = np.vstack(
            [fit_result.model.evaluate(x_array, theta) for theta in theta_draws]
        )
        simulated_errors = self.rng.choice(
            fit_result.bootstrap_residuals,
            size=simulated_means.shape,
            replace=True,
        )
        simulated_responses = simulated_means + simulated_errors
        fitted = fit_result.model.evaluate(x_array, fit_result.theta_hat)
        lower, upper = ParametricMonteCarloEngine._central_quantiles(
            simulated_responses,
            confidence,
        )
        return fitted, lower, upper


# ============================================================================
# 9. POLYNOMIAL QUANTILE REGRESSION
# ============================================================================


class PolynomialQuantileRegression:
    """Polynomial quantile regression solved as a linear program.

    Quantile regression directly estimates selected conditional quantiles and
    therefore does not use an additive error-model object or a parameter
    covariance from the likelihood framework.
    """

    def __init__(self, degree: int = 1):
        self.model = PolynomialModel(degree)
        self.beta_lower: np.ndarray | None = None
        self.beta_median: np.ndarray | None = None
        self.beta_upper: np.ndarray | None = None
        self.quantiles: tuple[float, float, float] | None = None

    @staticmethod
    def _fit_quantile(
        design: np.ndarray,
        y: np.ndarray,
        quantile: float,
    ) -> np.ndarray:
        n, p = design.shape

        # Variables are beta_plus, beta_minus, positive residuals, and
        # negative residuals. Splitting beta permits unrestricted coefficients.
        objective = np.concatenate(
            [
                np.zeros(2 * p),
                quantile * np.ones(n),
                (1.0 - quantile) * np.ones(n),
            ]
        )
        equality_matrix = np.hstack(
            [
                design,
                -design,
                np.eye(n),
                -np.eye(n),
            ]
        )
        equality_rhs = y
        bounds = [(0.0, None)] * (2 * p + 2 * n)

        result = linprog(
            objective,
            A_eq=equality_matrix,
            b_eq=equality_rhs,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Quantile-regression fit failed: {result.message}")

        return result.x[:p] - result.x[p : 2 * p]

    def fit(
        self,
        x: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> "PolynomialQuantileRegression":
        x_array, y_array = _validate_xy(x, y)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1.")

        alpha = 1.0 - confidence
        lower_q, median_q, upper_q = alpha / 2.0, 0.5, 1.0 - alpha / 2.0
        design = self.model.design_matrix(x_array)

        self.beta_lower = self._fit_quantile(design, y_array, lower_q)
        self.beta_median = self._fit_quantile(design, y_array, median_q)
        self.beta_upper = self._fit_quantile(design, y_array, upper_q)
        self.quantiles = (lower_q, median_q, upper_q)
        return self

    def predict(
        self,
        x: np.ndarray | Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.beta_lower is None
            or self.beta_median is None
            or self.beta_upper is None
        ):
            raise RuntimeError("Quantile regression must be fitted first.")
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        return (
            self.model.evaluate(x_array, self.beta_median),
            self.model.evaluate(x_array, self.beta_lower),
            self.model.evaluate(x_array, self.beta_upper),
        )


# ============================================================================
# 10. GENERAL PROFILE-LIKELIHOOD INVERSE CALIBRATION
# ============================================================================


class ProfileLikelihoodCalibrator:
    """Direct inverse calibration under a general additive error model.

    Calibration data are fitted by maximum likelihood. For a new observed
    response ``y0``, the unknown input ``x0`` is treated as the parameter of
    interest. At each candidate ``x0``, model and error parameters are
    re-optimized as nuisance parameters using the joint likelihood of:

    - the original calibration observations; and
    - the additional observation ``(x0, y0)``.

    The confidence set is obtained from the profile likelihood-ratio statistic
    and Wilks' theorem. It is therefore calculated directly in x-space rather
    than by intersecting a forward prediction band.
    """

    def __init__(
        self,
        optimizer_method: str = "BFGS",
        optimizer_options: dict | None = None,
    ):
        self.optimizer_method = optimizer_method
        self.optimizer_options = optimizer_options or {"maxiter": 5000}
        self.fit_result: FitResult | None = None

    def fit(self, fit_result: FitResult) -> "ProfileLikelihoodCalibrator":
        if (
            fit_result.error_model is None
            or fit_result.eta_hat is None
            or fit_result.joint_hat is None
        ):
            raise ValueError("Profile likelihood requires an MLE FitResult.")
        self.fit_result = fit_result
        return self

    def _joint_nll(
        self,
        nuisance: np.ndarray,
        x0: float,
        y0: float,
    ) -> float:
        assert self.fit_result is not None
        fit = self.fit_result
        p = fit.n_model_parameters
        theta = nuisance[:p]
        eta = nuisance[p:]

        calibration_residuals = fit.y_train - fit.model.evaluate(
            fit.x_train,
            theta,
        )
        new_residual = np.array(
            [y0 - fit.model.evaluate(np.array([x0]), theta)[0]]
        )
        all_residuals = np.concatenate([calibration_residuals, new_residual])
        log_density = fit.error_model.logpdf(all_residuals, eta)
        if not np.all(np.isfinite(log_density)):
            return np.inf
        return float(-np.sum(log_density))

    def _profile_nll(
        self,
        x0: float,
        y0: float,
        initial_nuisance: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        assert self.fit_result is not None
        initial = (
            self.fit_result.joint_hat
            if initial_nuisance is None
            else initial_nuisance
        )
        objective = lambda nuisance: self._joint_nll(nuisance, x0, y0)
        result = minimize(
            objective,
            initial,
            method=self.optimizer_method,
            options=self.optimizer_options,
        )
        if not result.success:
            fallback = minimize(
                objective,
                np.asarray(result.x, dtype=float),
                method="Powell",
                options=self.optimizer_options,
            )
            if fallback.success or (
                np.isfinite(fallback.fun) and fallback.fun < result.fun
            ):
                result = fallback
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"Profile optimization failed: {result.message}")
        return float(result.fun), np.asarray(result.x, dtype=float)

    def calibrate(
        self,
        y0: float,
        confidence: float = 0.95,
        x_bounds: tuple[float, float] | None = None,
        grid_size: int = 121,
    ) -> tuple[float, float, float]:
        """Estimate ``x0`` and return its profile-likelihood confidence interval."""
        if self.fit_result is None:
            raise RuntimeError("The calibrator must be fitted first.")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1.")
        if grid_size < 21:
            raise ValueError("grid_size must be at least 21.")

        fit = self.fit_result
        if x_bounds is None:
            x_min = float(np.min(fit.x_train))
            x_max = float(np.max(fit.x_train))
            width = x_max - x_min
            x_bounds = (x_min - 0.5 * width, x_max + 0.5 * width)

        if x_bounds[0] >= x_bounds[1]:
            raise ValueError("x_bounds must be increasing.")

        # Jointly optimize x0 and all nuisance parameters.
        objective = lambda full: self._joint_nll(full[:-1], full[-1], y0)
        x_initial = float(np.mean(fit.x_train))
        initial = np.concatenate([fit.joint_hat, [x_initial]])
        global_result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(None, None)] * fit.joint_hat.size + [x_bounds],
            options=self.optimizer_options,
        )
        if not global_result.success:
            raise RuntimeError(
                f"Global inverse-calibration optimization failed: "
                f"{global_result.message}"
            )

        x_mle = float(global_result.x[-1])
        minimum_nll = float(global_result.fun)
        critical = float(chi2.ppf(confidence, df=1))

        def likelihood_ratio(x_candidate: float) -> float:
            profiled_nll, _ = self._profile_nll(x_candidate, y0)
            return 2.0 * (profiled_nll - minimum_nll) - critical

        grid = np.linspace(x_bounds[0], x_bounds[1], grid_size)
        values = np.array([likelihood_ratio(value) for value in grid])

        # The accepted profile set has LR <= critical, so the root function is
        # non-positive inside the confidence interval.
        inside = values <= 0.0
        if not np.any(inside):
            raise RuntimeError("No profile-likelihood confidence region found.")

        mle_index = int(np.argmin(np.abs(grid - x_mle)))

        lower_bracket = None
        for index in range(mle_index, 0, -1):
            if values[index] <= 0.0 and values[index - 1] > 0.0:
                lower_bracket = (grid[index - 1], grid[index])
                break

        upper_bracket = None
        for index in range(mle_index, grid.size - 1):
            if values[index] <= 0.0 and values[index + 1] > 0.0:
                upper_bracket = (grid[index], grid[index + 1])
                break

        if lower_bracket is None or upper_bracket is None:
            raise RuntimeError(
                "The confidence interval reached x_bounds. Supply wider bounds."
            )

        lower = brentq(likelihood_ratio, *lower_bracket)
        upper = brentq(likelihood_ratio, *upper_bracket)
        return x_mle, float(lower), float(upper)


# ============================================================================
# 11. DEMONSTRATION
# ============================================================================


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    degree = 2
    model = PolynomialModel(degree)
    error_model = GaussianError()

    x_calibration = np.linspace(0.1, 2.5, 250)
    true_theta = np.array([5.0, 5.0, -1.18])
    y_calibration = (
        model.evaluate(x_calibration, true_theta)
        + rng.normal(0.0, 0.12, size=x_calibration.size)
    )
    x_grid = np.linspace(0.0, 2.7, 160)

    # Explicit-error likelihood fit.
    mle_fit = MaximumLikelihoodFitter().fit(
        x_calibration,
        y_calibration,
        model,
        error_model,
    )

    # Distribution-free least-squares fit used only by residual bootstrap.
    least_squares_fit = LeastSquaresFitter().fit(
        x_calibration,
        y_calibration,
        model,
    )

    exact = AnalyticalPolynomialGaussian(degree).fit(
        x_calibration,
        y_calibration,
    )
    delta = DeltaMethodEngine()
    parametric_mc = ParametricMonteCarloEngine(
        n_simulations=10_000,
        random_state=123,
    )
    bootstrap_mc = ResidualBootstrapMonteCarloEngine(
        n_simulations=10_000,
        random_state=456,
    )
    quantile = PolynomialQuantileRegression(degree).fit(
        x_calibration,
        y_calibration,
        confidence=0.95,
    )
    profile = ProfileLikelihoodCalibrator().fit(mle_fit)

    exact_fit, exact_lower, exact_upper = exact.prediction_interval(x_grid)
    delta_fit, delta_lower, delta_upper = delta.prediction_interval(
        mle_fit,
        x_grid,
    )
    mc_fit, mc_lower, mc_upper = parametric_mc.prediction_interval(
        mle_fit,
        x_grid,
    )
    boot_fit, boot_lower, boot_upper = bootstrap_mc.prediction_interval(
        least_squares_fit,
        x_grid,
    )
    qr_fit, qr_lower, qr_upper = quantile.predict(x_grid)

    print("Fitted model parameters:", mle_fit.theta_hat)
    print("Fitted error parameters:", error_model.describe(mle_fit.eta_hat))

    selected_y0 = float(model.evaluate(np.array([1.25]), true_theta)[0])
    x0_hat, x0_lower, x0_upper = profile.calibrate(
        selected_y0,
        x_bounds=(-0.5, 3.0),
    )
    print(
        "Profile-likelihood inverse calibration:",
        f"x0={x0_hat:.5f}, 95% CI=({x0_lower:.5f}, {x0_upper:.5f})",
    )

    plt.figure(figsize=(11, 7))
    plt.scatter(
        x_calibration,
        y_calibration,
        s=12,
        alpha=0.35,
        label="Calibration data",
    )
    plt.plot(x_grid, exact_fit, linewidth=2.0, label="Fitted mean")
    plt.plot(x_grid, exact_lower, linestyle="--", label="Exact Gaussian OLS")
    plt.plot(x_grid, exact_upper, linestyle="--")
    plt.plot(x_grid, delta_lower, linestyle=":", label="General delta")
    plt.plot(x_grid, delta_upper, linestyle=":")
    plt.plot(x_grid, mc_lower, linestyle="-.", label="Parametric Monte Carlo")
    plt.plot(x_grid, mc_upper, linestyle="-.")
    plt.plot(x_grid, boot_lower, label="Residual bootstrap")
    plt.plot(x_grid, boot_upper)
    plt.plot(x_grid, qr_lower, alpha=0.7, label="Quantile regression")
    plt.plot(x_grid, qr_upper, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Generalized Forward Prediction Intervals")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
