"""Serializable standard-curve model used by inverse estimation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import t
from sympy import Symbol, lambdify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .models import FittingResult
from anasens.uncertainty.bands import ConfidenceBand, PredictionBand


@dataclass(slots=True)
class StandardCurveModel:
    """Picklable curve model with prediction edges and error propagation."""

    model_type: str
    parameter_names: list[str]
    parameter_values: np.ndarray
    conc_std_min: float
    conc_std_max: float
    covar: np.ndarray
    df: float
    s: float
    fit_x: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    fit_y: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    raw_x: np.ndarray | None = None
    raw_y: np.ndarray | None = None
    expression_text: str | None = None
    error_model_name: str | None = None
    error_model_parameters: dict[str, float] | None = None
    pi_method: str | None = None
    ci_point_x: np.ndarray | None = None
    ci_point_lower: np.ndarray | None = None
    ci_point_upper: np.ndarray | None = None
    pi_point_x: np.ndarray | None = None
    pi_point_lower: np.ndarray | None = None
    pi_point_upper: np.ndarray | None = None
    ci_coverage: float | None = None
    pi_coverage: float | None = None
    ci_interpolation: str = "linear"
    pi_interpolation: str = "linear"
    provenance: dict[str, object] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)
    conc_est_min: float | None = None
    conc_est_max: float | None = None
    y_est_min: float | None = None
    y_est_max: float | None = None
    lod: float | None = None
    loq: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Normalize numeric inputs and derive limits immediately."""
        self.parameter_values = np.asarray(self.parameter_values, dtype=float)
        self.fit_x = np.asarray(self.fit_x, dtype=float)
        self.fit_y = np.asarray(self.fit_y, dtype=float)
        self.raw_x = None if self.raw_x is None else np.asarray(self.raw_x, dtype=float)
        self.raw_y = None if self.raw_y is None else np.asarray(self.raw_y, dtype=float)
        self.covar = np.asarray(self.covar, dtype=float)

        if self.fit_x.size and self.fit_y.size and self.fit_x.shape != self.fit_y.shape:
            raise ValueError("fit_x and fit_y must have the same shape.")

        self.provenance = dict(self.provenance or {})
        self.extras = dict(self.extras or {})

        self._derive_limits()

    @property
    def model(self):
        """Return callable model evaluator for any x."""
        return lambda x_values: self.predict(x_values)

    @property
    def upper_pi(self):
        """Return callable upper PI evaluator for any x."""
        alpha = 1.0 - float(self.pi_coverage) if self.pi_coverage is not None else 0.05
        return lambda x_values: self.upper_pred_edge(x_values, alpha=alpha)

    @property
    def lower_pi(self):
        """Return callable lower PI evaluator for any x."""
        alpha = 1.0 - float(self.pi_coverage) if self.pi_coverage is not None else 0.05
        return lambda x_values: self.lower_pred_edge(x_values, alpha=alpha)

    def predict(self, x_values):
        """Evaluate the fitted curve at x values."""
        x_numeric = np.asarray(x_values, dtype=float)
        params = np.asarray(self.parameter_values, dtype=float)

        if self.model_type == "polynomial":
            return np.polyval(params, x_numeric)

        if self.model_type == "power":
            amplitude, exponent, offset = params
            return amplitude * np.power(x_numeric, exponent) + offset

        if self.model_type == "custom":
            evaluator = _build_custom_evaluator(self.expression_text or "x", tuple(self.parameter_names))
            return np.asarray(evaluator(x_numeric, *params), dtype=float)

        if self.model_type == "quantile":
            return sum(coefficient * np.power(x_numeric, idx) for idx, coefficient in enumerate(params))

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _interpolated_band_edge(
        self,
        x_values,
        which: str,
        point_x: np.ndarray | None,
        point_lower: np.ndarray | None,
        point_upper: np.ndarray | None,
    ) -> np.ndarray | float | None:
        """Evaluate a stored band edge by interpolation when point samples exist."""
        if point_x is None or point_lower is None or point_upper is None:
            return None

        x_points = np.asarray(point_x, dtype=float)
        lower_points = np.asarray(point_lower, dtype=float)
        upper_points = np.asarray(point_upper, dtype=float)

        if x_points.size < 2:
            return None

        order = np.argsort(x_points)
        x_sorted = x_points[order]
        lower_sorted = lower_points[order]
        upper_sorted = upper_points[order]

        unique_x, unique_indices = np.unique(x_sorted, return_index=True)
        lower_unique = lower_sorted[unique_indices]
        upper_unique = upper_sorted[unique_indices]

        if unique_x.size < 2:
            return None

        edge_values = lower_unique if which == "lower" else upper_unique
        interpolator = interp1d(
            unique_x,
            edge_values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        x_eval = np.asarray(x_values, dtype=float)
        values = np.asarray(interpolator(x_eval), dtype=float)
        if np.asarray(x_values).ndim == 0:
            return float(values.reshape(-1)[0])
        return values

    def errprop(self, x_values, covar_params=None):
        """Compute delta-method variance term for model prediction uncertainty."""
        covar = self.covar if covar_params is None else np.asarray(covar_params, dtype=float)
        x_array = np.atleast_1d(np.asarray(x_values, dtype=float))

        variances: list[float] = []
        for x_value in x_array:
            gradient = _numeric_gradient(self, float(x_value), eps=1e-6)
            variance = float(gradient.T @ covar @ gradient)
            variances.append(max(variance, 0.0))

        variances_array = np.asarray(variances, dtype=float)
        if np.asarray(x_values).ndim == 0:
            return float(variances_array[0])
        return variances_array

    def upper_pred_edge(self, x_values, alpha: float = 0.05):
        """Upper prediction edge using Student-t multiplier."""
        interpolated = self._interpolated_band_edge(
            x_values,
            which="upper",
            point_x=self.pi_point_x,
            point_lower=self.pi_point_lower,
            point_upper=self.pi_point_upper,
        )
        if interpolated is not None:
            return interpolated

        t_multiplier = t.ppf(1.0 - alpha / 2.0, max(self.df, 1.0))
        variance = self.errprop(x_values, self.covar)
        return self.predict(x_values) + t_multiplier * np.sqrt(np.asarray(variance, dtype=float) + self.s**2)

    def lower_pred_edge(self, x_values, alpha: float = 0.05):
        """Lower prediction edge using Student-t multiplier."""
        interpolated = self._interpolated_band_edge(
            x_values,
            which="lower",
            point_x=self.pi_point_x,
            point_lower=self.pi_point_lower,
            point_upper=self.pi_point_upper,
        )
        if interpolated is not None:
            return interpolated

        t_multiplier = t.ppf(1.0 - alpha / 2.0, max(self.df, 1.0))
        variance = self.errprop(x_values, self.covar)
        return self.predict(x_values) - t_multiplier * np.sqrt(np.asarray(variance, dtype=float) + self.s**2)

    def upper_conf_edge(self, x_values, alpha: float = 0.05):
        """Upper confidence edge for the fitted mean using Student-t multiplier."""
        interpolated = self._interpolated_band_edge(
            x_values,
            which="upper",
            point_x=self.ci_point_x,
            point_lower=self.ci_point_lower,
            point_upper=self.ci_point_upper,
        )
        if interpolated is not None:
            return interpolated

        t_multiplier = t.ppf(1.0 - alpha / 2.0, max(self.df, 1.0))
        variance = self.errprop(x_values, self.covar)
        return self.predict(x_values) + t_multiplier * np.sqrt(np.asarray(variance, dtype=float))

    def lower_conf_edge(self, x_values, alpha: float = 0.05):
        """Lower confidence edge for the fitted mean using Student-t multiplier."""
        interpolated = self._interpolated_band_edge(
            x_values,
            which="lower",
            point_x=self.ci_point_x,
            point_lower=self.ci_point_lower,
            point_upper=self.ci_point_upper,
        )
        if interpolated is not None:
            return interpolated

        t_multiplier = t.ppf(1.0 - alpha / 2.0, max(self.df, 1.0))
        variance = self.errprop(x_values, self.covar)
        return self.predict(x_values) - t_multiplier * np.sqrt(np.asarray(variance, dtype=float))

    def confidence_band(self, x_grid: np.ndarray | None = None, alpha: float | None = None) -> ConfidenceBand:
        """Build a confidence-band object from the stored curve and edges."""
        x_values = self._default_band_grid(x_grid)
        alpha_value = float(1.0 - self.ci_coverage) if self.ci_coverage is not None else 0.05
        if alpha is not None:
            alpha_value = float(alpha)

        return ConfidenceBand(
            x_grid=x_values,
            mean=self.predict(x_values),
            lower=self.lower_conf_edge(x_values, alpha=alpha_value),
            upper=self.upper_conf_edge(x_values, alpha=alpha_value),
            alpha=alpha_value,
            method=str(self.provenance.get("confidence_method", "stored")),
            interpolation=self.ci_interpolation,
            extrapolate=True,
        )

    def prediction_band(self, x_grid: np.ndarray | None = None, alpha: float | None = None) -> PredictionBand:
        """Build a prediction-band object from the stored curve and edges."""
        x_values = self._default_band_grid(x_grid)
        alpha_value = float(1.0 - self.pi_coverage) if self.pi_coverage is not None else 0.05
        if alpha is not None:
            alpha_value = float(alpha)

        return PredictionBand(
            x_grid=x_values,
            mean=self.predict(x_values),
            lower=self.lower_pred_edge(x_values, alpha=alpha_value),
            upper=self.upper_pred_edge(x_values, alpha=alpha_value),
            alpha=alpha_value,
            method=str(self.provenance.get("prediction_method", self.pi_method or "stored")),
            interpolation=self.pi_interpolation,
            extrapolate=True,
        )

    def inverse_band_curve(
        self,
        x_values: np.ndarray | None = None,
        band: str = "prediction",
        prediction_alpha: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Build inverse-calibration and resolution curves from a fitted band.

        Purpose:
            Sweep across x-values, evaluate the fitted mean curve at each x to get
            a synthetic observation y_target, then substitute that same y_target
            into the lower and upper prediction edge functions and
            solve those edge equations back for x.

        Inputs:
            x_values: Optional x-grid to evaluate. If omitted, the fitted domain
                [conc_std_min, conc_std_max] is sampled on 50 evenly spaced points.
            band: Must be "prediction"; retained only as a narrow guard so the
                caller cannot accidentally request a confidence-band inversion.
            prediction_alpha: Optional override for the prediction-band alpha.

        Output:
            A dictionary containing the valid x-grid points, the fitted y-targets,
            the inverted MLE x-values, the lower and upper band solutions, the
            relative resolution curve, and the selected prediction-band metadata.
        """
        # Select the x-grid to sweep. This is the outer loop used by the notebook
        # style inverse construction: x -> fitted y, then edge(x) = y is inverted.
        x_grid = (
            np.asarray(x_values, dtype=float)
            if x_values is not None
            else np.linspace(float(self.conc_std_min), float(self.conc_std_max), 50)
        )

        # Normalize the requested band name once. This helper is intentionally
        # prediction-only because inverse calibration is defined on observations,
        # not on confidence bands for the fitted mean.
        band_key = band.strip().lower()
        if band_key != "prediction":
            raise ValueError("band must be 'prediction'.")

        # Prediction-band alpha defaults to the stored coverage unless the
        # caller supplies an explicit override.
        alpha = 1.0 - float(self.pi_coverage) if self.pi_coverage is not None else 0.05
        if prediction_alpha is not None:
            alpha = float(prediction_alpha)

        def lower_edge(x_value):
            # Lower prediction edge evaluated at a trial x-value or x-grid.
            # The band helper is intentionally vectorized here because the
            # inverse construction samples the entire x-grid at once.
            return np.asarray(self.lower_pred_edge(x_value, alpha=alpha), dtype=float)

        def upper_edge(x_value):
            # Upper prediction edge evaluated at a trial x-value or x-grid.
            # Keeping this vectorized avoids scalar coercion errors when the
            # inverse helper evaluates the whole sweep in one call.
            return np.asarray(self.upper_pred_edge(x_value, alpha=alpha), dtype=float)

        # Sample the fitted curve and band edges on the same x-grid so the
        # inverse can be built by direct substitution instead of iterative root
        # search. This is the stable version of the notebook-style construction.
        x_valid = np.asarray(x_grid, dtype=float)
        y_target = np.asarray(self.predict(x_valid), dtype=float)
        x_mle = x_valid.copy()

        # Evaluate the prediction band edges directly on the x-grid.
        lower_samples = np.asarray(lower_edge(x_valid), dtype=float)
        upper_samples = np.asarray(upper_edge(x_valid), dtype=float)

        # Convert edge curves into x(y) form with interpolation. This is the
        # direct inversion step: the x-values where the band edges produce the
        # fitted y-value at each sweep point.
        lower_order = np.argsort(lower_samples)
        upper_order = np.argsort(upper_samples)
        # For an increasing forward curve, intersecting a fixed y_target with
        # the *upper* forward PI edge gives the *lower* inverse x-limit, while
        # intersecting with the *lower* forward PI edge gives the *upper*
        # inverse x-limit.
        x_from_lower_forward_edge = np.interp(
            y_target,
            lower_samples[lower_order],
            x_valid[lower_order],
            left=np.nan,
            right=np.nan,
        )
        x_from_upper_forward_edge = np.interp(
            y_target,
            upper_samples[upper_order],
            x_valid[upper_order],
            left=np.nan,
            right=np.nan,
        )

        # Keep only the sweep points where both inverse edge lookups are valid.
        valid_mask = np.isfinite(x_from_lower_forward_edge) & np.isfinite(x_from_upper_forward_edge)
        if not np.any(valid_mask):
            raise RuntimeError("Inverse band curve generation did not produce any valid points.")

        x_valid = x_valid[valid_mask]
        y_target = y_target[valid_mask]
        x_mle = x_mle[valid_mask]
        x_lower = x_from_upper_forward_edge[valid_mask]
        x_upper = x_from_lower_forward_edge[valid_mask]

        # Keep bounds ordered even if local curvature/noise causes inversion
        # artifacts at isolated points.
        x_lower, x_upper = np.minimum(x_lower, x_upper), np.maximum(x_lower, x_upper)

        # Resolution is the relative width of the inverse band around the MLE.
        resolution = (x_upper - x_lower) / np.maximum(np.abs(x_mle), 1e-12)

        return {
            "x_grid": x_valid,
            "y_grid": y_target,
            "x_mle": x_mle,
            "x_lower": x_lower,
            "x_upper": x_upper,
            "resolution": resolution,
            "band": band_key,
            "alpha": alpha,
        }

    def _interpolated_pred_edge(self, x_values, which: str):
        """Evaluate prediction edges from stored point-based PI arrays if available."""
        if self.pi_point_x is None or self.pi_point_lower is None or self.pi_point_upper is None:
            return None

        return self._interpolated_band_edge(
            x_values,
            which=which,
            point_x=self.pi_point_x,
            point_lower=self.pi_point_lower,
            point_upper=self.pi_point_upper,
        )

    def _default_band_grid(self, x_grid: np.ndarray | None) -> np.ndarray:
        """Select a plotting grid for band evaluation."""
        if x_grid is not None:
            return np.asarray(x_grid, dtype=float)

        if self.fit_x.size:
            return np.asarray(self.fit_x, dtype=float)

        return np.linspace(float(self.conc_std_min), float(self.conc_std_max), 300)

    def _derive_limits(self) -> None:
        """Compute estimation-domain limits as soon as the object is created."""
        if not (self.conc_est_min is not None and self.conc_est_max is not None):
            if self.fit_x.size:
                self.conc_est_min = float(np.min(self.fit_x)) if self.conc_est_min is None else float(self.conc_est_min)
                self.conc_est_max = float(np.max(self.fit_x)) if self.conc_est_max is None else float(self.conc_est_max)
            else:
                self.conc_est_min = float(self.conc_std_min) if self.conc_est_min is None else float(self.conc_est_min)
                self.conc_est_max = float(self.conc_std_max) if self.conc_est_max is None else float(self.conc_est_max)

        # Detection limits are model- and workflow-specific. They are stored if
        # available in provenance, otherwise left unset so the caller can decide
        # whether to derive or report them.
        self.lod = self.provenance.get("lod", self.lod)
        self.loq = self.provenance.get("loq", self.loq)

        # Estimation y-limits are derived from the concentration estimation
        # bounds by evaluating the fitted model at those two concentrations.
        if self.y_est_min is None and self.conc_est_min is not None:
            try:
                y_min = np.asarray(self.predict(float(self.conc_est_min)), dtype=float)
                self.y_est_min = float(y_min.reshape(-1)[0])
            except Exception:
                self.y_est_min = None
        elif self.y_est_min is not None:
            self.y_est_min = float(self.y_est_min)

        if self.y_est_max is None and self.conc_est_max is not None:
            try:
                y_max = np.asarray(self.predict(float(self.conc_est_max)), dtype=float)
                self.y_est_max = float(y_max.reshape(-1)[0])
            except Exception:
                self.y_est_max = None
        elif self.y_est_max is not None:
            self.y_est_max = float(self.y_est_max)

    def show_provenance(self, print_output: bool = True) -> str:
        """Format and optionally print the curve provenance summary."""
        covariance_text = np.array2string(np.asarray(self.covar, dtype=float), precision=6, separator=", ")
        lines = [
            f"Schema version: {self.schema_version}",
            f"Model type: {self.model_type}",
            f"Parameters ({len(self.parameter_values)}): {np.asarray(self.parameter_values, dtype=float).tolist()}",
            f"Fit domain: [{self.conc_std_min:.6g}, {self.conc_std_max:.6g}]",
            f"Estimation domain: [{self.conc_est_min}, {self.conc_est_max}]",
            f"LOD: {self.lod}",
            f"LOQ: {self.loq}",
            f"CI coverage: {self.ci_coverage}",
            f"PI coverage: {self.pi_coverage}",
            f"Error model: {self.error_model_name}",
            f"Error model parameters: {self.error_model_parameters}",
            f"Covariance matrix shape: {tuple(np.asarray(self.covar).shape)}",
            f"Covariance matrix: {covariance_text}",
            f"Provenance: {self.provenance}",
            f"Extras: {self.extras}",
            f"Fit x: {self._summarize_array(self.fit_x)}",
            f"Fit y: {self._summarize_array(self.fit_y)}",
        ]
        if self.raw_x is not None or self.raw_y is not None:
            lines.append(f"Raw x: {self._summarize_array(self.raw_x)}")
            lines.append(f"Raw y: {self._summarize_array(self.raw_y)}")

        report = "\n".join(lines)
        if print_output:
            print(report)
        return report

    def _summarize_array(self, values: np.ndarray | None) -> str:
        """Return a compact summary of an array for provenance output."""
        if values is None:
            return "None"
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return "[]"
        return f"n={array.size}, min={float(np.nanmin(array)):.6g}, max={float(np.nanmax(array)):.6g}"


def build_standard_curve_model(
    fitting_result: FittingResult,
    x_values,
    y_values,
    model_type: str,
    expression_text: str | None = None,
    error_model_name: str | None = None,
    error_model_parameters: dict[str, float] | None = None,
    pi_method: str | None = None,
    ci_point_x=None,
    ci_point_lower=None,
    ci_point_upper=None,
    pi_point_x=None,
    pi_point_lower=None,
    pi_point_upper=None,
    ci_coverage: float | None = None,
    pi_coverage: float | None = None,
    provenance: dict[str, object] | None = None,
    extras: dict[str, object] | None = None,
) -> StandardCurveModel:
    """Create a serializable standard-curve object from fit results and source data."""
    x_numeric = np.asarray(x_values, dtype=float)
    y_numeric = np.asarray(y_values, dtype=float)

    params = np.asarray(fitting_result.parameter_values, dtype=float)
    predictions = _evaluate_model(model_type, params, x_numeric, fitting_result.parameter_names, expression_text)
    residuals = y_numeric - predictions

    n_samples = len(x_numeric)
    n_params = len(params)
    df = float(max(n_samples - n_params, 1))
    sigma2 = float(np.sum(np.square(residuals)) / df)
    s = float(np.sqrt(max(sigma2, 0.0)))

    if fitting_result.covariance is not None:
        covar = np.asarray(fitting_result.covariance, dtype=float)
    else:
        jacobian = _numeric_jacobian(model_type, params, x_numeric, fitting_result.parameter_names, expression_text)
        fisher = jacobian.T @ jacobian
        covar = sigma2 * np.linalg.pinv(fisher)

    provenance_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_method": fitting_result.fit_method,
        "model_name": fitting_result.model_name,
        "error_model": error_model_name,
        "display_equation": fitting_result.display_equation,
        "source": "curve_builder",
    }
    if provenance:
        provenance_data.update(provenance)

    extras_data = dict(extras or {})
    if fitting_result.extras:
        extras_data.update(fitting_result.extras)

    return StandardCurveModel(
        model_type=model_type,
        parameter_names=list(fitting_result.parameter_names),
        parameter_values=params,
        conc_std_min=float(np.min(x_numeric)),
        conc_std_max=float(np.max(x_numeric)),
        covar=np.asarray(covar, dtype=float),
        df=df,
        s=s,
        fit_x=x_numeric,
        fit_y=y_numeric,
        raw_x=x_numeric,
        raw_y=y_numeric,
        expression_text=expression_text,
        error_model_name=error_model_name,
        error_model_parameters=error_model_parameters,
        pi_method=pi_method,
        ci_point_x=np.asarray(ci_point_x, dtype=float) if ci_point_x is not None else None,
        ci_point_lower=np.asarray(ci_point_lower, dtype=float) if ci_point_lower is not None else None,
        ci_point_upper=np.asarray(ci_point_upper, dtype=float) if ci_point_upper is not None else None,
        pi_point_x=np.asarray(pi_point_x, dtype=float) if pi_point_x is not None else None,
        pi_point_lower=np.asarray(pi_point_lower, dtype=float) if pi_point_lower is not None else None,
        pi_point_upper=np.asarray(pi_point_upper, dtype=float) if pi_point_upper is not None else None,
        ci_coverage=float(ci_coverage) if ci_coverage is not None else None,
        pi_coverage=float(pi_coverage) if pi_coverage is not None else None,
        provenance=provenance_data,
        extras=extras_data,
    )


def _evaluate_model(
    model_type: str,
    params: np.ndarray,
    x_values: np.ndarray,
    parameter_names: list[str],
    expression_text: str | None,
):
    if model_type == "polynomial":
        return np.polyval(params, x_values)

    if model_type == "power":
        amplitude, exponent, offset = params
        return amplitude * np.power(x_values, exponent) + offset

    if model_type == "custom":
        evaluator = _build_custom_evaluator(expression_text or "x", tuple(parameter_names))
        return np.asarray(evaluator(x_values, *params), dtype=float)

    if model_type == "quantile":
        return sum(coefficient * np.power(x_values, idx) for idx, coefficient in enumerate(params))

    raise ValueError(f"Unsupported model type: {model_type}")


def _numeric_jacobian(
    model_type: str,
    params: np.ndarray,
    x_values: np.ndarray,
    parameter_names: list[str],
    expression_text: str | None,
    eps: float = 1e-6,
) -> np.ndarray:
    n_points = len(x_values)
    n_params = len(params)
    jacobian = np.zeros((n_points, n_params), dtype=float)

    for index in range(n_params):
        delta = np.zeros_like(params)
        delta[index] = eps

        plus = _evaluate_model(model_type, params + delta, x_values, parameter_names, expression_text)
        minus = _evaluate_model(model_type, params - delta, x_values, parameter_names, expression_text)
        jacobian[:, index] = (plus - minus) / (2.0 * eps)

    return jacobian


def _numeric_gradient(curve: StandardCurveModel, x_value: float, eps: float = 1e-6) -> np.ndarray:
    params = np.asarray(curve.parameter_values, dtype=float)
    gradient = np.zeros_like(params)

    for index in range(len(params)):
        delta = np.zeros_like(params)
        delta[index] = eps

        original = curve.parameter_values
        curve.parameter_values = params + delta
        plus = float(np.asarray(curve.predict(np.array([x_value])))[0])

        curve.parameter_values = params - delta
        minus = float(np.asarray(curve.predict(np.array([x_value])))[0])

        curve.parameter_values = original
        gradient[index] = (plus - minus) / (2.0 * eps)

    return gradient


@lru_cache(maxsize=16)
def _build_custom_evaluator(expression_text: str, parameter_names: tuple[str, ...]):
    x_symbol = Symbol("x")
    parameter_symbols = [Symbol(name) for name in parameter_names]

    try:
        expression = parse_latex(expression_text)
    except Exception:
        normalized = expression_text.strip().replace("^", "**")
        transformations = standard_transformations + (convert_xor, implicit_multiplication_application)
        expression = parse_expr(normalized, transformations=transformations)

    return lambdify((x_symbol, *parameter_symbols), expression, modules="numpy")
