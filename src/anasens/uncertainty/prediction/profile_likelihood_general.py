"""General profile-likelihood prediction helpers.

This module implements the profile-likelihood forward prediction band and the
inverse-calibration map used by the curve-builder profile-likelihood flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.stats import chi2

from anasens.curve_builder.models import FittingResult
from anasens.uncertainty.bands import PredictionBand, ProfileLikelihoodBand


@dataclass(slots=True)
class ProfileLikelihoodPredictionResult:
    """Store the profile-likelihood band and inverse-calibration map."""

    band: PredictionBand
    inverse_map: Optional[ProfileLikelihoodBand]


class GeneralizedProfileLikelihoodPrediction:
    """Construct profile-likelihood prediction and inverse-calibration outputs."""

    def __init__(self, fitting_result: FittingResult, x_values, y_values) -> None:
        self.fitting_result = fitting_result
        self.x_values = np.asarray(x_values, dtype=float)
        self.y_values = np.asarray(y_values, dtype=float)
        self.theta_hat = np.asarray(fitting_result.parameter_values, dtype=float)
        self.sigma_hat = self._resolve_sigma_hat(fitting_result, self.x_values, self.y_values)
        self._init_params = np.concatenate([self.theta_hat, [self.sigma_hat]])
        self._bounds = [(None, None)] * self.theta_hat.size + [(1e-12, None)]

    def build(
        self,
        x_grid,
        confidence: float = 0.95,
        n_y_points: int = 20,
        generate_inverse_map: bool = False,
    ) -> ProfileLikelihoodPredictionResult:
        """Compute the forward band and inverse calibration map."""
        x_grid = np.asarray(x_grid, dtype=float)
        band = self._forward_prediction_band(x_grid, confidence=confidence)
        inverse_map = None
        if generate_inverse_map:
            inverse_map = self._inverse_calibration_map(confidence=confidence, n_y_points=n_y_points)
        return ProfileLikelihoodPredictionResult(band=band, inverse_map=inverse_map)

    def _resolve_sigma_hat(self, fitting_result: FittingResult, x_values: np.ndarray, y_values: np.ndarray) -> float:
        sigma_hat = float(fitting_result.residual_std or 0.0)
        if sigma_hat > 0.0:
            return max(sigma_hat, 1e-12)

        theta_hat = np.asarray(fitting_result.parameter_values, dtype=float)
        residuals = y_values - np.asarray(fitting_result.model_function(x_values, *theta_hat), dtype=float)
        dof = float(max(len(y_values) - len(theta_hat), 1))
        return max(float(np.sqrt(max(np.sum(np.square(residuals)) / dof, 0.0))), 1e-12)

    def _model_values(self, x_values, theta: np.ndarray) -> np.ndarray:
        x_numeric = np.asarray(x_values, dtype=float)
        return np.asarray(self.fitting_result.model_function(x_numeric, *theta), dtype=float)

    def _scalar_model_value(self, x_value: float, theta: np.ndarray) -> float:
        values = np.asarray(self.fitting_result.model_function(np.asarray([x_value], dtype=float), *theta), dtype=float)
        return float(np.ravel(values)[0])

    def _calibration_loglik(self, theta: np.ndarray, sigma: float) -> float:
        sigma = max(float(sigma), 1e-12)
        residuals = self.y_values - self._model_values(self.x_values, theta)
        n_samples = self.y_values.size
        return float(-0.5 * n_samples * np.log(2.0 * np.pi * sigma**2) - np.sum(np.square(residuals)) / (2.0 * sigma**2))

    def _joint_loglik(self, theta: np.ndarray, sigma: float, x_target: float, y_target: float) -> float:
        sigma = max(float(sigma), 1e-12)
        calibration = self._calibration_loglik(theta, sigma)
        residual_extra = float(y_target - self._scalar_model_value(x_target, theta))
        return float(calibration - 0.5 * np.log(2.0 * np.pi * sigma**2) - (residual_extra**2) / (2.0 * sigma**2))

    def _profile_loglik_with_theta(
        self,
        x_target: float,
        y_target: float,
        initial_guess: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        def objective(params: np.ndarray) -> float:
            theta = np.asarray(params[:-1], dtype=float)
            sigma = float(params[-1])
            return -self._joint_loglik(theta, sigma, x_target, y_target)

        starting_point = self._init_params if initial_guess is None else np.asarray(initial_guess, dtype=float)
        result = minimize(
            objective,
            starting_point,
            method="L-BFGS-B",
            bounds=self._bounds,
            options={"maxiter": 2000, "gtol": 5e-4, "ftol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"Profile likelihood optimization failed: {result.message}")
        return -float(result.fun), np.asarray(result.x, dtype=float)

    def _profile_loglik(self, x_target: float, y_target: float) -> float:
        value, _ = self._profile_loglik_with_theta(x_target, y_target)
        return value

    def _target_loglik(self, confidence: float) -> float:
        cutoff = 0.5 * chi2.ppf(float(confidence), df=1)
        joint_max = self._calibration_loglik(self.theta_hat, self.sigma_hat) - 0.5 * np.log(2.0 * np.pi * self.sigma_hat**2)
        return float(joint_max - cutoff)

    def _expand_root(self, objective, center: float, direction: float, initial_span: float, max_expand: int = 10) -> tuple[float, float]:
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

    def _expand_from_previous(self, objective, previous_root: float, direction: float, initial_span: float, max_expand: int = 10) -> tuple[float, float]:
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
            return self._expand_root(
                objective,
                center=center,
                direction=direction,
                initial_span=fallback_span,
            )

    def _forward_prediction_band(self, x_grid: np.ndarray, confidence: float = 0.95) -> PredictionBand:
        target_loglik = self._target_loglik(confidence)
        y_center = self._model_values(x_grid, self.theta_hat)

        lower = np.zeros_like(x_grid, dtype=float)
        upper = np.zeros_like(x_grid, dtype=float)
        previous_lower = None
        previous_lower_prev = None
        previous_lower_theta = None
        previous_upper = None
        previous_upper_prev = None
        previous_upper_theta = None

        for index, x_target in enumerate(x_grid):
            center = float(y_center[index])
            lower_seed = [np.asarray(previous_lower_theta if previous_lower_theta is not None else self._init_params, dtype=float)]
            upper_seed = [np.asarray(previous_upper_theta if previous_upper_theta is not None else self._init_params, dtype=float)]

            def lower_objective(y_target: float) -> float:
                try:
                    value, theta = self._profile_loglik_with_theta(
                        float(x_target),
                        float(y_target),
                        initial_guess=lower_seed[0],
                    )
                    lower_seed[0] = theta
                    return value - target_loglik
                except RuntimeError:
                    return float("nan")

            def upper_objective(y_target: float) -> float:
                try:
                    value, theta = self._profile_loglik_with_theta(
                        float(x_target),
                        float(y_target),
                        initial_guess=upper_seed[0],
                    )
                    upper_seed[0] = theta
                    return value - target_loglik
                except RuntimeError:
                    return float("nan")

            if index < 2 or previous_lower_prev is None:
                lower_bracket = [center - 2.0 * self.sigma_hat, center]
            else:
                lower_span = max(1e-3, 1.5 * abs(previous_lower - previous_lower_prev))
                lower_bracket = self._expand_with_fallback(
                    lower_objective,
                    previous_root=previous_lower,
                    center=center,
                    direction=1.0,
                    initial_span=lower_span,
                    fallback_span=max(lower_span, 2.0 * self.sigma_hat),
                )
            if index < 2 or previous_upper_prev is None:
                upper_bracket = [center, center + 2.0 * self.sigma_hat]
            else:
                upper_span = max(1e-3, 1.5 * abs(previous_upper - previous_upper_prev))
                upper_bracket = self._expand_with_fallback(
                    upper_objective,
                    previous_root=previous_upper,
                    center=center,
                    direction=1.0,
                    initial_span=upper_span,
                    fallback_span=max(upper_span, 2.0 * self.sigma_hat),
                )

            lower[index] = root_scalar(
                lower_objective,
                bracket=lower_bracket,
                method="brentq",
                xtol=1e-6,
            ).root
            upper[index] = root_scalar(
                upper_objective,
                bracket=upper_bracket,
                method="brentq",
                xtol=1e-6,
            ).root
            previous_lower_prev = previous_lower
            previous_lower = float(lower[index])
            previous_lower_theta = np.asarray(lower_seed[0], dtype=float)
            previous_upper_prev = previous_upper
            previous_upper = float(upper[index])
            previous_upper_theta = np.asarray(upper_seed[0], dtype=float)

        return PredictionBand(
            x_grid=x_grid,
            mean=y_center,
            lower=lower,
            upper=upper,
            alpha=1.0 - float(confidence),
            method="profile_likelihood",
        )

    def _inverse_calibration_map(self, confidence: float = 0.95, n_y_points: int = 35) -> ProfileLikelihoodBand:
        target_loglik = self._target_loglik(confidence)
        x_min = float(np.min(self.x_values))
        x_max = float(np.max(self.x_values))
        x_infer_min = x_min
        x_infer_max = x_max
        y_grid = np.linspace(
            float(self._scalar_model_value(x_infer_min + 0.02 * max(x_max - x_min, 1.0), self.theta_hat)),
            float(self._scalar_model_value(x_infer_max - 0.02 * max(x_max - x_min, 1.0), self.theta_hat)),
            int(n_y_points),
        )

        x_mle_list: list[float] = []
        x_lower_list: list[float] = []
        x_upper_list: list[float] = []
        y_valid: list[float] = []
        previous_mle = None
        previous_mle_prev = None
        previous_lower = None
        previous_lower_prev = None
        previous_lower_theta = None
        previous_upper = None
        previous_upper_prev = None
        previous_upper_theta = None

        for index, y_target in enumerate(y_grid):
            def mle_invert(x: float) -> float:
                return self._scalar_model_value(float(x), self.theta_hat) - float(y_target)

            lower_seed = [np.asarray(previous_lower_theta if previous_lower_theta is not None else self._init_params, dtype=float)]
            upper_seed = [np.asarray(previous_upper_theta if previous_upper_theta is not None else self._init_params, dtype=float)]

            try:
                if index < 2 or previous_mle_prev is None:
                    mle_bracket = [-1.0, 15.0]
                else:
                    mle_span = max(1e-3, 1.5 * abs(previous_mle - previous_mle_prev))
                    mle_bracket = self._expand_with_fallback(
                        mle_invert,
                        previous_root=previous_mle,
                        center=float(x_hat if 'x_hat' in locals() else np.mean(self.x_values)),
                        direction=1.0,
                        initial_span=mle_span,
                        fallback_span=max(mle_span, 2.0),
                    )
                x_hat = root_scalar(mle_invert, bracket=mle_bracket, method="brentq", xtol=1e-6).root
            except ValueError:
                continue

            def lower_objective_x(x_target: float) -> float:
                try:
                    value, theta = self._profile_loglik_with_theta(
                        float(x_target),
                        float(y_target),
                        initial_guess=lower_seed[0],
                    )
                    lower_seed[0] = theta
                    return value - target_loglik
                except RuntimeError:
                    return float("nan")

            def upper_objective_x(x_target: float) -> float:
                try:
                    value, theta = self._profile_loglik_with_theta(
                        float(x_target),
                        float(y_target),
                        initial_guess=upper_seed[0],
                    )
                    upper_seed[0] = theta
                    return value - target_loglik
                except RuntimeError:
                    return float("nan")

            try:
                if index < 2 or previous_lower_prev is None:
                    lower_bracket = [max(-2.0, x_hat - 2.0 * self.sigma_hat), x_hat]
                else:
                    lower_span = max(1e-3, 1.5 * abs(previous_lower - previous_lower_prev))
                    lower_bracket = self._expand_with_fallback(
                        lower_objective_x,
                        previous_root=previous_lower,
                        center=x_hat,
                        direction=1.0,
                        initial_span=lower_span,
                        fallback_span=max(lower_span, 2.0 * self.sigma_hat),
                    )
                if index < 2 or previous_upper_prev is None:
                    upper_bracket = [x_hat, min(15.0, x_hat + 2.0 * self.sigma_hat)]
                else:
                    upper_span = max(1e-3, 1.5 * abs(previous_upper - previous_upper_prev))
                    upper_bracket = self._expand_with_fallback(
                        upper_objective_x,
                        previous_root=previous_upper,
                        center=x_hat,
                        direction=1.0,
                        initial_span=upper_span,
                        fallback_span=max(upper_span, 2.0 * self.sigma_hat),
                    )

                x_lower = root_scalar(
                    lower_objective_x,
                    bracket=lower_bracket,
                    method="brentq",
                    xtol=1e-6,
                ).root
                x_upper = root_scalar(
                    upper_objective_x,
                    bracket=upper_bracket,
                    method="brentq",
                    xtol=1e-6,
                ).root
            except RuntimeError:
                continue
            except ValueError:
                continue

            x_mle_list.append(x_hat)
            x_lower_list.append(x_lower)
            x_upper_list.append(x_upper)
            y_valid.append(float(y_target))
            previous_mle_prev = previous_mle
            previous_mle = float(x_hat)
            previous_lower_prev = previous_lower
            previous_lower = float(x_lower)
            previous_lower_theta = np.asarray(lower_seed[0], dtype=float)
            previous_upper_prev = previous_upper
            previous_upper = float(x_upper)
            previous_upper_theta = np.asarray(upper_seed[0], dtype=float)

        if not y_valid:
            raise RuntimeError("Profile-likelihood inverse calibration did not converge for any y-grid points.")

        return ProfileLikelihoodBand(
            y_grid=np.asarray(y_valid, dtype=float),
            x_mle=np.asarray(x_mle_list, dtype=float),
            x_lower=np.asarray(x_lower_list, dtype=float),
            x_upper=np.asarray(x_upper_list, dtype=float),
            alpha=1.0 - float(confidence),
            method="profile_likelihood",
        )


def profile_likelihood_prediction_band(
    fitting_result: FittingResult,
    x_values,
    y_values,
    x_grid,
    confidence: float = 0.95,
    n_y_points: int = 20,
    generate_inverse_map: bool = False,
) -> ProfileLikelihoodPredictionResult:
    """Build a profile-likelihood prediction band and inverse-calibration map."""
    engine = GeneralizedProfileLikelihoodPrediction(fitting_result, x_values, y_values)
    return engine.build(
        x_grid=x_grid,
        confidence=confidence,
        n_y_points=n_y_points,
        generate_inverse_map=generate_inverse_map,
    )
