"""Shared data models for plotting and fitting operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class LoadedDataset:
    """Represent a successfully loaded tabular dataset.

    Attributes:
        source_path: Original file path that was loaded.
        frame: DataFrame version of the source data.
    """

    source_path: str
    frame: pd.DataFrame


@dataclass(slots=True)
class FittingResult:
    """Represent the result of a curve fitting operation.

    Attributes:
        model_name: Human-readable model label.
        parameter_names: Ordered parameter names corresponding to values.
        parameter_values: Optimized parameter values returned by SciPy.
        fitted_x: Dense x values used to render the fit curve.
        fitted_y: Evaluated y values on fitted_x.
        display_equation: Equation string displayed in the statistics panel.
        model_function: Callable form of the fitted model.
        gradient_function: Callable returning first derivatives with respect to parameters.
        hessian_function: Callable returning second derivatives with respect to parameters.
        third_derivative_function: Callable returning third derivatives with respect to parameters.
    """

    model_name: str
    parameter_names: list[str]
    parameter_values: np.ndarray
    fitted_x: np.ndarray
    fitted_y: np.ndarray
    display_equation: str
    model_function: Callable[..., np.ndarray]
    gradient_function: Callable[..., np.ndarray] | None = None
    hessian_function: Callable[..., np.ndarray] | None = None
    third_derivative_function: Callable[..., np.ndarray] | None = None
    covariance: np.ndarray | None = None
    residual_std: float | None = None
    dof: float | None = None
    fit_method: str | None = None
    extras: dict[str, object] | None = None
