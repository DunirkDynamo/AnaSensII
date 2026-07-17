"""Confidence and prediction band computation for curve-builder fits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import re

import numpy as np

from .models import FittingResult
from anasens.uncertainty.bands import ConfidenceBand, PredictionBand
from anasens.uncertainty.error_models import (
    GaussianError,
    HyperbolicSecantError,
    LaplaceError,
    LogisticError,
    SkewNormalError,
    StudentTError,
)
from anasens.uncertainty.intervals import DeltaMethod, MonteCarloCI, ProfileLikelihoodCI
from anasens.uncertainty.prediction import (
    MonteCarloPrediction,
    profile_likelihood_modified_map,
)


@dataclass(slots=True)
class BandComputationResult:
    """Container for fitted-mean confidence and observation prediction bands."""

    confidence_bands: dict[str, ConfidenceBand]
    prediction_bands: dict[str, PredictionBand]
    confidence_alpha: float
    prediction_alpha: float
    delta_order: int
    dof: float
    sigma: float
    confidence_errors: dict[str, str] = field(default_factory=dict)
    prediction_errors: dict[str, str] = field(default_factory=dict)
    profile_prediction_maps: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def confidence_band(self) -> ConfidenceBand | None:
        """Return the first available confidence band, for compatibility."""
        return next(iter(self.confidence_bands.values()), None)

    @property
    def prediction_band(self) -> PredictionBand | None:
        """Return the first available prediction band, for compatibility."""
        return next(iter(self.prediction_bands.values()), None)


class BandService:
    """Compute uncertainty bands for a fitted curve-builder model."""

    def compute_gaussian_bands(
        self,
        fitting_result: FittingResult,
        x_values,
        y_values,
        alpha: float = 0.05,
        grid_size: int = 400,
    ) -> BandComputationResult:
        """Build default confidence and prediction bands using Gaussian assumptions."""
        return self.compute_confidence_bands(
            fitting_result=fitting_result,
            x_values=x_values,
            y_values=y_values,
            methods=("delta",),
            prediction_methods=("exact_analytic",),
            alpha=alpha,
            confidence_alpha=alpha,
            prediction_alpha=alpha,
            error_model_name="gaussian",
            grid_size=grid_size,
        )

    def compute_confidence_bands(
        self,
        fitting_result: FittingResult,
        x_values,
        y_values,
        methods: Iterable[str] | None = None,
        prediction_methods: Iterable[str] | None = None,
        delta_order: int = 1,
        alpha: float = 0.05,
        confidence_alpha: float | None = None,
        prediction_alpha: float | None = None,
        error_model_name: str = "gaussian",
        grid_size: int = 400,
        monte_carlo_samples: int = 2500,
        prediction_monte_carlo_samples: int = 3000,
        prediction_delta_samples: int = 2500,
        profile_grid_points: int = 121,
        profile_y_points: int = 35,
        monte_carlo_mode: str = "parametric",
    ) -> BandComputationResult:
        """Build selected confidence and prediction intervals for a fitted model."""
        x_numeric = np.asarray(x_values, dtype=float)
        y_numeric = np.asarray(y_values, dtype=float)

        theta_hat = np.asarray(fitting_result.parameter_values, dtype=float)
        n_samples = x_numeric.size
        n_params = theta_hat.size
        dof = float(max(n_samples - n_params, 1))

        model = self._build_model_function(fitting_result)
        covariance = self._resolve_covariance(fitting_result, x_numeric, theta_hat)
        gradient = self._resolve_gradient_function(fitting_result)

        predictions = np.asarray(fitting_result.model_function(x_numeric, *theta_hat), dtype=float)
        residuals = y_numeric - predictions
        sigma = float(np.sqrt(max(np.sum(np.square(residuals)) / dof, 0.0)))

        confidence_alpha = float(alpha if confidence_alpha is None else confidence_alpha)
        prediction_alpha = float(confidence_alpha if prediction_alpha is None else prediction_alpha)

        x_grid = np.linspace(float(np.min(x_numeric)), float(np.max(x_numeric)), int(grid_size))
        selected_confidence_methods = tuple(methods) if methods else ("delta",)
        selected_prediction_methods = tuple(prediction_methods) if prediction_methods else ("exact_analytic",)

        if self._is_quantile_fit(fitting_result):
            selected_confidence_methods = tuple()
            if selected_prediction_methods != ("quantile_regression",):
                selected_prediction_methods = ("quantile_regression",)

        confidence_bands: dict[str, ConfidenceBand] = {}
        confidence_errors: dict[str, str] = {}

        hessian = self._resolve_hessian_function(fitting_result, model)
        third_derivative = self._resolve_third_derivative_function(fitting_result, model)
        polynomial_degree = self._extract_polynomial_degree(fitting_result)

        for method in selected_confidence_methods:
            try:
                confidence_bands[method] = self._compute_confidence_band(
                    method=method,
                    fitting_result=fitting_result,
                    model=model,
                    theta_hat=theta_hat,
                    covariance=covariance,
                    grad=gradient,
                    x_grid=x_grid,
                    x_numeric=x_numeric,
                    y_numeric=y_numeric,
                    sigma=sigma,
                    delta_order=delta_order,
                    hessian=hessian,
                    third_derivative=third_derivative,
                    alpha=confidence_alpha,
                    monte_carlo_samples=monte_carlo_samples,
                    profile_grid_points=profile_grid_points,
                    error_model_name=error_model_name,
                    polynomial_degree=polynomial_degree,
                )
            except Exception as exc:
                confidence_errors[method] = str(exc)

        error_model = self._build_error_model(error_model_name, sigma)

        prediction_bands: dict[str, PredictionBand] = {}
        prediction_errors: dict[str, str] = {}
        profile_prediction_maps: dict[str, dict[str, np.ndarray]] = {}

        for method in selected_prediction_methods:
            try:
                band, profile_map = self._compute_prediction_result(
                    method=method,
                    fitting_result=fitting_result,
                    model=model,
                    theta_hat=theta_hat,
                    covariance=covariance,
                    gradient=gradient,
                    x_numeric=x_numeric,
                    y_numeric=y_numeric,
                    x_grid=x_grid,
                    sigma=sigma,
                    dof=int(dof),
                    alpha=prediction_alpha,
                    error_model_name=error_model_name,
                    error_model=error_model,
                    polynomial_degree=polynomial_degree,
                    prediction_monte_carlo_samples=prediction_monte_carlo_samples,
                    prediction_delta_samples=prediction_delta_samples,
                    profile_y_points=profile_y_points,
                    monte_carlo_mode=monte_carlo_mode,
                    residuals=residuals,
                )
                if band is not None:
                    prediction_bands[method] = band
                if profile_map is not None:
                    profile_prediction_maps[method] = profile_map
            except Exception as exc:
                prediction_errors[method] = str(exc)

        return BandComputationResult(
            confidence_bands=confidence_bands,
            prediction_bands=prediction_bands,
            confidence_alpha=confidence_alpha,
            prediction_alpha=prediction_alpha,
            delta_order=int(delta_order),
            dof=dof,
            sigma=sigma,
            confidence_errors=confidence_errors,
            prediction_errors=prediction_errors,
            profile_prediction_maps=profile_prediction_maps,
        )

    def _compute_confidence_band(
        self,
        method: str,
        fitting_result: FittingResult,
        model,
        theta_hat: np.ndarray,
        covariance: np.ndarray,
        grad,
        x_grid: np.ndarray,
        x_numeric: np.ndarray,
        y_numeric: np.ndarray,
        sigma: float,
        delta_order: int,
        hessian,
        third_derivative,
        alpha: float,
        monte_carlo_samples: int,
        profile_grid_points: int,
        error_model_name: str,
        polynomial_degree: int | None,
    ) -> ConfidenceBand:
        method_key = method.strip().lower()

        if method_key in {"exact_analytic", "gaussian_analytic"}:
            if polynomial_degree is None:
                raise ValueError("Exact analytical CI is only available for polynomial fits.")
            if error_model_name.strip().lower() != "gaussian":
                raise ValueError("Exact analytical CI requires Gaussian error model.")

            mean, lower, upper = self._exact_confidence_band_from_fit(
                fitting_result=fitting_result,
                x_numeric=x_numeric,
                x_grid=x_grid,
                alpha=alpha,
                degree=polynomial_degree,
            )
            return ConfidenceBand(
                x_grid=x_grid,
                mean=mean,
                lower=lower,
                upper=upper,
                alpha=alpha,
                method="exact_analytic",
            )

        if method_key == "delta":
            if delta_order not in (1, 2):
                raise ValueError("delta_order must be 1 or 2.")

            return DeltaMethod(
                model=model,
                theta_hat=theta_hat,
                cov=covariance,
                grad=grad,
                hess=hessian if delta_order == 2 else None,
                third=third_derivative if delta_order == 2 else None,
            ).band(x_grid=x_grid, alpha=alpha, order=delta_order)

        if method_key == "monte_carlo":
            return MonteCarloCI(
                model=model,
                theta_hat=theta_hat,
                cov=covariance,
            ).band(x_grid=x_grid, alpha=alpha, n_samples=monte_carlo_samples)

        if method_key == "profile_likelihood":
            loglik = self._build_gaussian_loglik_function(x_numeric, y_numeric, model, sigma)
            q_grid = self._build_profile_q_grid(
                model=model,
                theta_hat=theta_hat,
                covariance=covariance,
                gradient=grad,
                x_grid=x_grid,
                sigma=sigma,
                profile_grid_points=profile_grid_points,
            )
            return ProfileLikelihoodCI(
                model=model,
                loglik=loglik,
                theta_hat=theta_hat,
            ).band(x_grid=x_grid, q_grid=q_grid, alpha=alpha)

        raise ValueError(
            "Unsupported confidence method: "
            f"{method}. Supported methods are exact_analytic, delta, monte_carlo, and profile_likelihood."
        )

    def _compute_prediction_result(
        self,
        method: str,
        fitting_result: FittingResult,
        model,
        theta_hat: np.ndarray,
        covariance: np.ndarray,
        gradient,
        x_numeric: np.ndarray,
        y_numeric: np.ndarray,
        x_grid: np.ndarray,
        sigma: float,
        dof: int,
        alpha: float,
        error_model_name: str,
        error_model,
        polynomial_degree: int | None,
        prediction_monte_carlo_samples: int,
        prediction_delta_samples: int,
        profile_y_points: int,
        monte_carlo_mode: str,
        residuals: np.ndarray,
    ) -> tuple[PredictionBand | None, dict[str, np.ndarray] | None]:
        method_key = method.strip().lower()

        if self._is_quantile_fit(fitting_result):
            if method_key != "quantile_regression":
                raise ValueError("Quantile fit mode supports only quantile_regression PI.")
            band = self._quantile_prediction_band_from_fit(fitting_result, x_grid, alpha)
            return band, None

        if method_key in {"exact_analytic", "gaussian_analytic"}:
            if polynomial_degree is None:
                raise ValueError("Exact analytical PI is only available for polynomial fits.")
            if error_model_name.strip().lower() != "gaussian":
                raise ValueError("Exact analytical PI requires Gaussian error model.")

            mean, lower, upper = self._exact_prediction_band_from_fit(
                fitting_result=fitting_result,
                x_numeric=x_numeric,
                x_grid=x_grid,
                alpha=alpha,
                degree=polynomial_degree,
            )
            band = PredictionBand(
                x_grid=x_grid,
                mean=mean,
                lower=lower,
                upper=upper,
                alpha=alpha,
                method="exact_analytic",
            )
            return band, None

        if method_key == "delta":
            band = self._delta_prediction_band(
                model=model,
                theta_hat=theta_hat,
                covariance=covariance,
                gradient=gradient,
                error_model=error_model,
                x_grid=x_grid,
                alpha=alpha,
                n_samples=prediction_delta_samples,
            )
            return band, None

        if method_key == "monte_carlo":
            monte_carlo_mode_key = monte_carlo_mode.strip().lower()
            if monte_carlo_mode_key not in {"parametric", "residual_bootstrap"}:
                raise ValueError("Monte Carlo mode must be either 'parametric' or 'residual_bootstrap'.")

            if monte_carlo_mode_key == "residual_bootstrap":
                band = self._bootstrap_prediction_band(
                    model=model,
                    theta_hat=theta_hat,
                    covariance=covariance,
                    gradient=gradient,
                    x_numeric=x_numeric,
                    x_grid=x_grid,
                    residuals=residuals,
                    alpha=alpha,
                    n_samples=prediction_monte_carlo_samples,
                )
                return band, None

            band = MonteCarloPrediction(
                model=model,
                theta_hat=theta_hat,
                cov=covariance,
                error_model=error_model,
                random_state=42,
            ).band(
                x_grid=x_grid,
                alpha=alpha,
                n_samples=prediction_monte_carlo_samples,
            )
            return band, None

        if method_key == "quantile_regression":
            raise ValueError(
                "Quantile-regression PI requires selecting quantile fit type so quantiles are fitted once during model fit."
            )

        if method_key in {"profile_likelihood_modified", "profile_likelihood"}:
            if polynomial_degree is None:
                raise ValueError("Modified profile-likelihood PI is currently available only for polynomial fits.")
            profile_map = profile_likelihood_modified_map(
                x_cal=x_numeric,
                y_cal=y_numeric,
                degree=polynomial_degree,
                confidence=1.0 - float(alpha),
                n_y_points=profile_y_points,
            )
            return None, profile_map

        raise ValueError(
            "Unsupported prediction method: "
            f"{method}. Supported methods are exact_analytic, delta, monte_carlo, profile_likelihood_modified, and quantile_regression."
        )

    def _is_quantile_fit(self, fitting_result: FittingResult) -> bool:
        return (fitting_result.fit_method or "").strip().lower() == "quantile_regression"

    def _quantile_prediction_band_from_fit(
        self,
        fitting_result: FittingResult,
        x_grid: np.ndarray,
        alpha: float,
    ) -> PredictionBand:
        extras = fitting_result.extras or {}
        beta_lower = np.asarray(extras.get("beta_lower"), dtype=float)
        beta_median = np.asarray(extras.get("beta_median"), dtype=float)
        beta_upper = np.asarray(extras.get("beta_upper"), dtype=float)

        if beta_lower.size == 0 or beta_median.size == 0 or beta_upper.size == 0:
            raise ValueError("Quantile fit result is missing stored quantile coefficients.")

        def eval_poly(x_data: np.ndarray, beta: np.ndarray) -> np.ndarray:
            x_array = np.asarray(x_data, dtype=float)
            return sum(b * np.power(x_array, idx) for idx, b in enumerate(beta))

        mean = eval_poly(x_grid, beta_median)
        lower = eval_poly(x_grid, beta_lower)
        upper = eval_poly(x_grid, beta_upper)

        return PredictionBand(
            x_grid=x_grid,
            mean=mean,
            lower=lower,
            upper=upper,
            alpha=alpha,
            method="quantile_regression",
        )

    def _exact_prediction_band_from_fit(
        self,
        fitting_result: FittingResult,
        x_numeric: np.ndarray,
        x_grid: np.ndarray,
        alpha: float,
        degree: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Uses fitted parameters and uncertainty outputs from the existing fit.
        beta = np.asarray(fitting_result.parameter_values, dtype=float)
        p = int(degree) + 1
        if beta.size != p:
            raise ValueError("Polynomial fit parameter count does not match requested degree.")

        def eval_poly(x_data: np.ndarray, coeffs_desc: np.ndarray) -> np.ndarray:
            x_array = np.asarray(x_data, dtype=float)
            y_values = np.zeros_like(x_array, dtype=float)
            for exponent, coefficient in zip(range(coeffs_desc.size - 1, -1, -1), coeffs_desc, strict=False):
                y_values = y_values + coefficient * np.power(x_array, exponent)
            return y_values

        y_fit = eval_poly(x_grid, beta)

        s_err = float(max(fitting_result.residual_std or 0.0, 1e-12))
        covariance = fitting_result.covariance
        if covariance is not None and np.asarray(covariance).shape == (p, p):
            inv_xtx = np.asarray(covariance, dtype=float) / (s_err**2)
        else:
            design = np.vander(x_numeric, p, increasing=False)
            inv_xtx = np.linalg.pinv(design.T @ design)

        grid_design = np.vander(x_grid, p, increasing=False)
        leverage = np.sum((grid_design @ inv_xtx) * grid_design, axis=1)

        dof = int(max(fitting_result.dof or 1.0, 1.0))
        from scipy.stats import t

        t_val = float(t.ppf(1.0 - alpha / 2.0, df=dof))
        pi_half_width = t_val * s_err * np.sqrt(1.0 + leverage)
        return y_fit, y_fit - pi_half_width, y_fit + pi_half_width

    def _exact_confidence_band_from_fit(
        self,
        fitting_result: FittingResult,
        x_numeric: np.ndarray,
        x_grid: np.ndarray,
        alpha: float,
        degree: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        beta = np.asarray(fitting_result.parameter_values, dtype=float)
        p = int(degree) + 1
        if beta.size != p:
            raise ValueError("Polynomial fit parameter count does not match requested degree.")

        def eval_poly(x_data: np.ndarray, coeffs_desc: np.ndarray) -> np.ndarray:
            x_array = np.asarray(x_data, dtype=float)
            y_values = np.zeros_like(x_array, dtype=float)
            for exponent, coefficient in zip(range(coeffs_desc.size - 1, -1, -1), coeffs_desc, strict=False):
                y_values = y_values + coefficient * np.power(x_array, exponent)
            return y_values

        y_fit = eval_poly(x_grid, beta)

        s_err = float(max(fitting_result.residual_std or 0.0, 1e-12))
        covariance = fitting_result.covariance
        if covariance is not None and np.asarray(covariance).shape == (p, p):
            inv_xtx = np.asarray(covariance, dtype=float) / (s_err**2)
        else:
            design = np.vander(x_numeric, p, increasing=False)
            inv_xtx = np.linalg.pinv(design.T @ design)

        grid_design = np.vander(x_grid, p, increasing=False)
        leverage = np.sum((grid_design @ inv_xtx) * grid_design, axis=1)

        dof = int(max(fitting_result.dof or 1.0, 1.0))
        from scipy.stats import t

        t_val = float(t.ppf(1.0 - alpha / 2.0, df=dof))
        ci_half_width = t_val * s_err * np.sqrt(leverage)
        return y_fit, y_fit - ci_half_width, y_fit + ci_half_width

    def _delta_prediction_band(
        self,
        model,
        theta_hat: np.ndarray,
        covariance: np.ndarray,
        gradient,
        error_model,
        x_grid: np.ndarray,
        alpha: float,
        n_samples: int,
    ) -> PredictionBand:
        # This is a direct convolution implementation of first-order delta
        # uncertainty with the selected additive error model.
        rng = np.random.default_rng(42)
        means = np.zeros_like(x_grid, dtype=float)
        lowers = np.zeros_like(x_grid, dtype=float)
        uppers = np.zeros_like(x_grid, dtype=float)

        for index, x_value in enumerate(x_grid):
            mu = float(model(float(x_value), theta_hat))
            grad_vec = np.asarray(gradient(float(x_value), theta_hat), dtype=float)
            var_mean = float(grad_vec @ covariance @ grad_vec)
            std_mean = float(np.sqrt(max(var_mean, 0.0)))

            sampled_means = rng.normal(loc=mu, scale=std_mean, size=n_samples)
            sampled_noise = np.asarray(
                [error_model.sample(float(x_value), rng) for _ in range(n_samples)],
                dtype=float,
            )
            sampled_y = sampled_means + sampled_noise

            means[index] = mu
            lowers[index] = float(np.quantile(sampled_y, alpha / 2.0))
            uppers[index] = float(np.quantile(sampled_y, 1.0 - alpha / 2.0))

        return PredictionBand(
            x_grid=x_grid,
            mean=means,
            lower=lowers,
            upper=uppers,
            alpha=alpha,
            method="delta",
        )

    def _bootstrap_prediction_band(
        self,
        model,
        theta_hat: np.ndarray,
        covariance: np.ndarray,
        gradient,
        x_numeric: np.ndarray,
        x_grid: np.ndarray,
        residuals: np.ndarray,
        alpha: float,
        n_samples: int,
    ) -> PredictionBand:
        rng = np.random.default_rng(42)
        x_grid = np.asarray(x_grid, dtype=float)
        residuals = np.asarray(residuals, dtype=float)
        if residuals.size == 0:
            raise ValueError("Residual bootstrap requires non-empty residuals.")

        jacobian = np.asarray(
            [np.ravel(np.asarray(gradient(float(x_value), theta_hat), dtype=float)) for x_value in x_numeric],
            dtype=float,
        )
        if jacobian.ndim != 2:
            raise ValueError("Failed to build Jacobian matrix for residual bootstrap.")

        information = jacobian.T @ jacobian
        inverse_information = np.linalg.pinv(information)
        leverage = np.sum((jacobian @ inverse_information) * jacobian, axis=1)
        leverage = np.clip(leverage, 0.0, 1.0)

        adjusted_residuals = residuals / np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
        bootstrap_residuals = adjusted_residuals - np.mean(adjusted_residuals)

        sampled_thetas = rng.multivariate_normal(mean=theta_hat, cov=covariance, size=n_samples)

        means = np.zeros_like(x_grid, dtype=float)
        lowers = np.zeros_like(x_grid, dtype=float)
        uppers = np.zeros_like(x_grid, dtype=float)

        for index, x_value in enumerate(x_grid):
            x_scalar = float(x_value)
            sampled_means = np.asarray([model(x_scalar, theta) for theta in sampled_thetas], dtype=float)
            sampled_residuals = rng.choice(bootstrap_residuals, size=n_samples, replace=True)
            sampled_y = sampled_means + sampled_residuals

            means[index] = float(model(x_scalar, theta_hat))
            lowers[index] = float(np.quantile(sampled_y, alpha / 2.0))
            uppers[index] = float(np.quantile(sampled_y, 1.0 - alpha / 2.0))

        return PredictionBand(
            x_grid=x_grid,
            mean=means,
            lower=lowers,
            upper=uppers,
            alpha=alpha,
            method="monte_carlo",
        )

    def _build_error_model(self, error_model_name: str, sigma: float):
        model_key = error_model_name.strip().lower()
        sigma = max(float(sigma), 1e-12)

        if model_key == "gaussian":
            return GaussianError(sigma=sigma)
        if model_key == "laplace":
            return LaplaceError(b=sigma / np.sqrt(2.0))
        if model_key == "logistic":
            return LogisticError(s=sigma * np.sqrt(3.0) / np.pi)
        if model_key == "student_t":
            return StudentTError(df=5.0, scale=sigma * np.sqrt((5.0 - 2.0) / 5.0))
        if model_key == "hyperbolic_secant":
            return HyperbolicSecantError(scale=sigma)
        if model_key == "skew_normal":
            return SkewNormalError(a=4.0, scale=sigma)

        raise ValueError(
            "Unsupported error model: "
            f"{error_model_name}. Supported models are gaussian, laplace, logistic, student_t, hyperbolic_secant, and skew_normal."
        )

    def _extract_polynomial_degree(self, fitting_result: FittingResult) -> int | None:
        match = re.search(r"Polynomial \(order\s+(\d+)\)", fitting_result.model_name)
        if match:
            return int(match.group(1))
        return None

    def _build_model_function(self, fitting_result: FittingResult):
        def model(x, theta):
            x_numeric = np.asarray(x, dtype=float)
            values = np.asarray(fitting_result.model_function(x_numeric, *theta), dtype=float)

            if np.asarray(x).ndim == 0:
                return float(np.ravel(values)[0])
            return values

        return model

    def _build_gaussian_loglik_function(self, x_numeric: np.ndarray, y_numeric: np.ndarray, model, sigma: float):
        error_model = GaussianError(sigma=max(float(sigma), 1e-12))

        def loglik(theta):
            theta = np.asarray(theta, dtype=float)
            predictions = np.asarray(model(x_numeric, theta), dtype=float)
            residuals = np.asarray(y_numeric, dtype=float) - predictions

            total = 0.0
            for residual, x_value in zip(residuals, x_numeric, strict=False):
                total += error_model.logpdf(float(residual), float(x_value))

            return float(total)

        return loglik

    def _build_profile_q_grid(
        self,
        model,
        theta_hat: np.ndarray,
        covariance: np.ndarray,
        gradient,
        x_grid: np.ndarray,
        sigma: float,
        profile_grid_points: int,
    ) -> np.ndarray:
        means = np.asarray([model(x_value, theta_hat) for x_value in x_grid], dtype=float)
        standard_errors = np.asarray(
            [
                np.sqrt(max(float(gradient(x_value, theta_hat) @ covariance @ gradient(x_value, theta_hat)), 0.0))
                for x_value in x_grid
            ],
            dtype=float,
        )

        width = max(float(np.max(standard_errors)) * 6.0, float(sigma) * 6.0, 1e-9)
        lower = float(np.min(means) - width)
        upper = float(np.max(means) + width)
        return np.linspace(lower, upper, int(profile_grid_points))

    def _build_gradient_function(self, fitting_result: FittingResult, eps: float = 1e-6):
        def gradient(x, theta):
            theta = np.asarray(theta, dtype=float)
            grad = np.zeros_like(theta)

            for index in range(theta.size):
                delta = np.zeros_like(theta)
                delta[index] = eps

                plus = np.asarray(
                    fitting_result.model_function(np.asarray([x], dtype=float), *(theta + delta)),
                    dtype=float,
                )
                minus = np.asarray(
                    fitting_result.model_function(np.asarray([x], dtype=float), *(theta - delta)),
                    dtype=float,
                )

                grad[index] = (float(np.ravel(plus)[0]) - float(np.ravel(minus)[0])) / (2.0 * eps)

            return grad

        return gradient

    def _resolve_gradient_function(self, fitting_result: FittingResult):
        if fitting_result.gradient_function is not None:
            return fitting_result.gradient_function
        return self._build_gradient_function(fitting_result)

    def _resolve_hessian_function(self, fitting_result: FittingResult, model):
        if fitting_result.hessian_function is not None:
            return fitting_result.hessian_function
        return self._build_hessian_function(model)

    def _resolve_third_derivative_function(self, fitting_result: FittingResult, model):
        if fitting_result.third_derivative_function is not None:
            return fitting_result.third_derivative_function
        return self._build_third_derivative_function(model)

    def _build_hessian_function(self, model, eps: float = 1e-4):
        def hessian(x, theta):
            theta = np.asarray(theta, dtype=float)
            n_params = theta.size
            hess = np.zeros((n_params, n_params), dtype=float)

            for i in range(n_params):
                for j in range(n_params):
                    step_i = np.zeros_like(theta)
                    step_j = np.zeros_like(theta)
                    step_i[i] = eps
                    step_j[j] = eps

                    f_pp = float(model(x, theta + step_i + step_j))
                    f_pm = float(model(x, theta + step_i - step_j))
                    f_mp = float(model(x, theta - step_i + step_j))
                    f_mm = float(model(x, theta - step_i - step_j))

                    hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)

            return hess

        return hessian

    def _build_third_derivative_function(self, model, eps: float = 1e-4):
        hessian = self._build_hessian_function(model, eps=eps)

        def third(x, theta):
            theta = np.asarray(theta, dtype=float)
            n_params = theta.size
            tensor = np.zeros((n_params, n_params, n_params), dtype=float)

            for k in range(n_params):
                step_k = np.zeros_like(theta)
                step_k[k] = eps
                hess_plus = np.asarray(hessian(x, theta + step_k), dtype=float)
                hess_minus = np.asarray(hessian(x, theta - step_k), dtype=float)
                tensor[k, :, :] = (hess_plus - hess_minus) / (2.0 * eps)

            return tensor

        return third

    def _resolve_covariance(
        self,
        fitting_result: FittingResult,
        x_values: np.ndarray,
        theta_hat: np.ndarray,
    ) -> np.ndarray:
        covariance = fitting_result.covariance
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=float)
            if covariance.shape == (theta_hat.size, theta_hat.size) and np.all(np.isfinite(covariance)):
                return covariance

        # Fallback: numeric Jacobian + pseudo-inverse if SciPy covariance is absent.
        n_points = x_values.size
        n_params = theta_hat.size
        jacobian = np.zeros((n_points, n_params), dtype=float)
        eps = 1e-6

        for index in range(n_params):
            delta = np.zeros_like(theta_hat)
            delta[index] = eps

            plus = np.asarray(
                fitting_result.model_function(x_values, *(theta_hat + delta)),
                dtype=float,
            )
            minus = np.asarray(
                fitting_result.model_function(x_values, *(theta_hat - delta)),
                dtype=float,
            )
            jacobian[:, index] = (plus - minus) / (2.0 * eps)

        fisher = jacobian.T @ jacobian
        return np.linalg.pinv(fisher)
