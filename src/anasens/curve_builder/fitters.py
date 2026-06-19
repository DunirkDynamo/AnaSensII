"""Mathematical fitting services for Plotter Fitter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import re

import numpy as np
from scipy.optimize import curve_fit
from sympy import Symbol, diff, lambdify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.parsing.latex import parse_latex

from .models import FittingResult


@dataclass(slots=True)
class PreparedModel:
    """Represent a SciPy-compatible model ready for fitting.

    Attributes:
        name: Human-readable model name.
        parameter_names: Ordered parameter names expected by the callable.
        function: Model function with signature ``f(x, *params)``.
        display_equation: Equation string shown to the user.
        initial_guess: Optional initial parameter estimates.
    """

    name: str
    parameter_names: list[str]
    function: Callable[..., np.ndarray]
    display_equation: str
    initial_guess: tuple[float, ...] | None = None


@dataclass(slots=True)
class DerivativeFunctionRegistry:
    """Store generated derivative callables with stable names for iteration.

    Attributes:
        names: Ordered derivative function names for deterministic looping.
        functions_by_name: Mapping from derivative function name to callable.
        parameters_by_name: Mapping from derivative function name to parameter symbol.
    """

    names: list[str]
    functions_by_name: dict[str, Callable[..., np.ndarray]]
    parameters_by_name: dict[str, str]


class FitService:
    """Create and apply mathematical fit models to numeric data.

    This service contains no UI code. It prepares model callables, delegates
    optimization to SciPy, and returns plotting-ready fit results.
    """

    def fit_polynomial(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        order: int,
    ) -> FittingResult:
        """Fit a polynomial of the requested order.

        Args:
            x_values: One-dimensional x-axis values.
            y_values: One-dimensional y-axis values.
            order: Polynomial order, where 1 is linear.

        Returns:
            A fitting result containing optimized parameters and fitted curve.

        Raises:
            ValueError: If the order is less than 1.
        """
        if order < 1:
            raise ValueError("Polynomial order must be at least 1.")

        # Parameter names are generated deterministically so both the stats box
        # and downstream tests can rely on a stable ordering.
        parameter_names = [f"a{i}" for i in range(order, -1, -1)]

        # The nested function is generated from the requested order and kept
        # local to avoid exposing implementation details in the public API.
        def polynomial_function(x_data: np.ndarray, *parameters: float) -> np.ndarray:
            result = np.zeros_like(x_data, dtype=float)
            for exponent, coefficient in zip(range(order, -1, -1), parameters):
                result = result + coefficient * np.power(x_data, exponent)
            return result

        prepared_model = PreparedModel(
            name=f"Polynomial (order {order})",
            parameter_names=parameter_names,
            function=polynomial_function,
            display_equation=self._build_polynomial_equation(order),
            initial_guess=tuple(1.0 for _ in range(order + 1)),
        )
        return self._fit_model(x_values, y_values, prepared_model)

    def fit_power_law(self, x_values: np.ndarray, y_values: np.ndarray) -> FittingResult:
        """Fit the power-law model ``y = A * x^r + b``.

        Args:
            x_values: One-dimensional x-axis values.
            y_values: One-dimensional y-axis values.

        Returns:
            A fitting result containing optimized parameters and fitted curve.
        """

        # The power-law callable matches SciPy's expected signature and keeps
        # the model definition close to its parameter metadata.
        def power_law_function(x_data: np.ndarray, amplitude: float, exponent: float, offset: float) -> np.ndarray:
            return amplitude * np.power(x_data, exponent) + offset

        prepared_model = PreparedModel(
            name="Power law",
            parameter_names=["A", "r", "b"],
            function=power_law_function,
            display_equation="y = A * x^r + b",
            initial_guess=(1.0, 1.0, 0.0),
        )
        return self._fit_model(x_values, y_values, prepared_model)

    def fit_custom_latex(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        latex_expression: str,
        parameter_names: list[str] | None = None,
    ) -> FittingResult:
        """Fit a custom model defined by a LaTeX expression.

        Args:
            x_values: One-dimensional x-axis values.
            y_values: One-dimensional y-axis values.
            latex_expression: LaTeX equation body using ``x`` as the independent variable.
            parameter_names: Optional explicit parameter ordering from user input.

        Returns:
            A fitting result containing optimized parameters and fitted curve.

        Raises:
            ValueError: If the LaTeX expression is invalid or missing fit variables.
        """
        prepared_model = self._prepare_custom_model(latex_expression, parameter_names)
        return self._fit_model(x_values, y_values, prepared_model)

    def build_custom_model_and_derivatives(
        self,
        latex_expression: str,
        parameter_names: list[str],
    ) -> tuple[Callable[..., np.ndarray], dict[str, Callable[..., np.ndarray]]]:
        """Build callable model and first derivatives for a custom expression.

        Args:
            latex_expression: LaTeX equation body with ``x`` and fit parameters.
            parameter_names: Ordered parameter names provided by the user.

        Returns:
            Tuple of ``(model_function, derivative_functions)`` where each
            callable is SciPy-style and accepts ``(x, *params)``.
        """
        if not parameter_names:
            raise ValueError("At least one parameter name is required.")

        model_function, derivative_registry = self.build_custom_model_derivative_registry(
            latex_expression,
            parameter_names,
        )

        # Backward-compatible return shape keyed by parameter symbol.
        derivative_functions_by_parameter: dict[str, Callable[..., np.ndarray]] = {}
        for derivative_name in derivative_registry.names:
            parameter_name = derivative_registry.parameters_by_name[derivative_name]
            derivative_functions_by_parameter[parameter_name] = derivative_registry.functions_by_name[derivative_name]

        return model_function, derivative_functions_by_parameter

    def build_custom_model_derivative_registry(
        self,
        latex_expression: str,
        parameter_names: list[str],
    ) -> tuple[Callable[..., np.ndarray], DerivativeFunctionRegistry]:
        """Build a custom model and a named derivative-function registry.

        The registry exposes an ordered list of derivative function names so
        downstream code can iterate without knowing parameter count beforehand.
        """
        if not parameter_names:
            raise ValueError("At least one parameter name is required.")

        expression = self._parse_custom_expression(latex_expression)
        x_symbol = Symbol("x")
        parameter_symbols = [Symbol(name) for name in parameter_names]

        if x_symbol not in expression.free_symbols:
            raise ValueError("Custom LaTeX fit must include x as the independent variable.")

        model_lambda = lambdify((x_symbol, *parameter_symbols), expression, modules="numpy")

        def model_function(x_data: np.ndarray, *parameters: float) -> np.ndarray:
            return np.asarray(model_lambda(x_data, *parameters), dtype=float)

        derivative_names: list[str] = []
        derivative_functions_by_name: dict[str, Callable[..., np.ndarray]] = {}
        parameters_by_name: dict[str, str] = {}

        for parameter_name, parameter_symbol in zip(parameter_names, parameter_symbols, strict=False):
            derivative_name = f"d_model_d_{parameter_name}"
            derivative_expression = diff(expression, parameter_symbol)
            derivative_lambda = lambdify((x_symbol, *parameter_symbols), derivative_expression, modules="numpy")

            def derivative_function(x_data: np.ndarray, *parameters: float, _fn=derivative_lambda) -> np.ndarray:
                return np.asarray(_fn(x_data, *parameters), dtype=float)

            derivative_names.append(derivative_name)
            derivative_functions_by_name[derivative_name] = derivative_function
            parameters_by_name[derivative_name] = parameter_name

        derivative_registry = DerivativeFunctionRegistry(
            names=derivative_names,
            functions_by_name=derivative_functions_by_name,
            parameters_by_name=parameters_by_name,
        )
        return model_function, derivative_registry

    def _fit_model(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        prepared_model: PreparedModel,
    ) -> FittingResult:
        """Run SciPy curve fitting for a prepared model.

        Args:
            x_values: One-dimensional x-axis values.
            y_values: One-dimensional y-axis values.
            prepared_model: Model metadata and callable.

        Returns:
            A fitting result ready for visualization.
        """
        # Arrays are normalized to float for stable interaction with SciPy and
        # to ensure the renderer receives a consistent numeric dtype.
        x_numeric = np.asarray(x_values, dtype=float)
        y_numeric = np.asarray(y_values, dtype=float)

        optimized_parameters, _ = curve_fit(
            prepared_model.function,
            x_numeric,
            y_numeric,
            p0=prepared_model.initial_guess,
            maxfev=20000,
        )

        # The fitted curve is evaluated on a sorted dense grid so overlays look
        # smooth even when the raw data arrives in an unsorted order.
        fitted_x = np.linspace(np.min(x_numeric), np.max(x_numeric), 400)
        fitted_y = prepared_model.function(fitted_x, *optimized_parameters)

        return FittingResult(
            model_name=prepared_model.name,
            parameter_names=prepared_model.parameter_names,
            parameter_values=optimized_parameters,
            fitted_x=fitted_x,
            fitted_y=fitted_y,
            display_equation=prepared_model.display_equation,
            model_function=prepared_model.function,
        )

    def _prepare_custom_model(
        self,
        latex_expression: str,
        parameter_names: list[str] | None = None,
    ) -> PreparedModel:
        """Parse a LaTeX model and convert it to a SciPy-ready callable.

        Args:
            latex_expression: LaTeX equation body supplied by the user.

        Returns:
            A prepared model instance.

        Raises:
            ValueError: If the expression cannot be parsed or does not define fit variables.
        """
        try:
            expression = self._parse_custom_expression(latex_expression)
        except Exception as exc:  # pragma: no cover - parser error details vary by version.
            raise ValueError("Could not parse the LaTeX expression.") from exc

        # Free symbols are partitioned into the required independent variable x
        # and the remaining fit parameters, which are sorted for reproducibility.
        x_symbol = Symbol("x")
        free_symbols = expression.free_symbols
        discovered_parameter_symbols = sorted(
            [symbol for symbol in free_symbols if symbol != x_symbol],
            key=lambda symbol: symbol.name,
        )

        if x_symbol not in free_symbols:
            raise ValueError("Custom LaTeX fit must include x as the independent variable.")
        if not discovered_parameter_symbols:
            raise ValueError("Custom LaTeX fit must include at least one fit parameter.")

        if parameter_names:
            parameter_symbols = [Symbol(name) for name in parameter_names]
            missing_symbols = [symbol.name for symbol in parameter_symbols if symbol not in free_symbols]
            if missing_symbols:
                missing_list = ", ".join(missing_symbols)
                raise ValueError(f"Custom LaTeX expression is missing parameter(s): {missing_list}.")
        else:
            parameter_symbols = discovered_parameter_symbols

        lambda_function = lambdify((x_symbol, *parameter_symbols), expression, modules="numpy")

        # A wrapper is required because SciPy expects the independent variable
        # followed by a flat parameter list rather than a tuple of symbols.
        def custom_function(x_data: np.ndarray, *parameters: float) -> np.ndarray:
            return np.asarray(lambda_function(x_data, *parameters), dtype=float)

        return PreparedModel(
            name="Custom LaTeX",
            parameter_names=[symbol.name for symbol in parameter_symbols],
            function=custom_function,
            display_equation=f"y = {latex_expression}",
            initial_guess=tuple(1.0 for _ in parameter_symbols),
        )

    def _parse_custom_expression(self, expression_text: str):
        """Parse a custom fit expression from LaTeX or a LaTeX-like math string.

        The GUI preview accepts general math-style input, so we first try the
        LaTeX parser and then fall back to a normalized SymPy parser for common
        inputs such as ``A x^{b} + c x + d``.
        """
        try:
            return parse_latex(expression_text)
        except Exception:
            normalized_text = self._normalize_custom_expression(expression_text)
            transformations = standard_transformations + (
                convert_xor,
                implicit_multiplication_application,
            )
            return parse_expr(normalized_text, transformations=transformations)

    def _normalize_custom_expression(self, expression_text: str) -> str:
        """Convert a small LaTeX-like subset into a SymPy-friendly expression."""
        normalized_text = expression_text.strip()
        normalized_text = normalized_text.replace("\\left", "").replace("\\right", "")
        normalized_text = normalized_text.replace("{", "(").replace("}", ")")
        normalized_text = re.sub(r"\\\s*frac\s*\(([^()]*)\)\s*\(([^()]*)\)", r"(\1)/(\2)", normalized_text)
        normalized_text = re.sub(r"([A-Za-z0-9_\)])\s+([A-Za-z_(])", r"\1*\2", normalized_text)
        normalized_text = normalized_text.replace("^", "**")
        return normalized_text

    def _build_polynomial_equation(self, order: int) -> str:
        """Create a readable polynomial equation string.

        Args:
            order: Polynomial order.

        Returns:
            A display-ready equation string.
        """
        terms: list[str] = []
        for exponent in range(order, -1, -1):
            coefficient_name = f"a{exponent}"
            if exponent > 1:
                terms.append(f"{coefficient_name} * x^{exponent}")
            elif exponent == 1:
                terms.append(f"{coefficient_name} * x")
            else:
                terms.append(coefficient_name)
        return "y = " + " + ".join(terms)
