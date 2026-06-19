"""Serializable standard-curve model used by inverse estimation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
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
    expression_text: str | None = None

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

        raise ValueError(f"Unsupported model type: {self.model_type}")

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
        t_multiplier = t.ppf(1.0 - alpha / 2.0, max(self.df, 1.0))
        variance = self.errprop(x_values, self.covar)
        return self.predict(x_values) + t_multiplier * np.sqrt(np.asarray(variance, dtype=float) + self.s**2)

    def lower_pred_edge(self, x_values, alpha: float = 0.05):
        """Lower prediction edge using Student-t multiplier."""
        t_multiplier = t.ppf(1.0 - alpha / 2.0, max(self.df, 1.0))
        variance = self.errprop(x_values, self.covar)
        return self.predict(x_values) - t_multiplier * np.sqrt(np.asarray(variance, dtype=float) + self.s**2)


def build_standard_curve_model(
    fitting_result: FittingResult,
    x_values,
    y_values,
    model_type: str,
    expression_text: str | None = None,
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

    jacobian = _numeric_jacobian(model_type, params, x_numeric, fitting_result.parameter_names, expression_text)
    fisher = jacobian.T @ jacobian
    covar = sigma2 * np.linalg.pinv(fisher)

    return StandardCurveModel(
        model_type=model_type,
        parameter_names=list(fitting_result.parameter_names),
        parameter_values=params,
        conc_std_min=float(np.min(x_numeric)),
        conc_std_max=float(np.max(x_numeric)),
        covar=np.asarray(covar, dtype=float),
        df=df,
        s=s,
        expression_text=expression_text,
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
