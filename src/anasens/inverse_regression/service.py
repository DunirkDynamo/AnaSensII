"""Core inverse regression estimation service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t

from anasens.inverse_regression.models import EstimateRow, InverseEstimationResult, RegressionType


class InverseRegressionService:
    """Compute inverse concentration estimates from calibrated signal values."""

    def estimate_least_squares(
        self,
        measured_value: float,
        curve_object,
        covar_params,
        degrees_of_freedom: float,
        residual_std: float,
        solver_lims: tuple[float, float],
    ) -> np.ndarray:
        """Estimate lower, best, upper concentration using t-based prediction interval inversion."""

        t_multiplier = t.ppf(0.975, degrees_of_freedom)

        def predint_upper(x: np.ndarray) -> np.ndarray:
            return (
                curve_object.predict(x)
                + t_multiplier * np.sqrt(curve_object.errprop(x, covar_params) + residual_std**2)
                - measured_value
            )

        def predint_lower(x: np.ndarray) -> np.ndarray:
            return (
                curve_object.predict(x)
                - t_multiplier * np.sqrt(curve_object.errprop(x, covar_params) + residual_std**2)
                - measured_value
            )

        def best_model(x: np.ndarray) -> np.ndarray:
            return curve_object.predict(x) - measured_value

        initial_guess = float(np.mean(solver_lims))
        upper_conc = least_squares(predint_lower, x0=initial_guess, bounds=solver_lims).x[0]
        best_conc = least_squares(best_model, x0=initial_guess, bounds=solver_lims).x[0]
        lower_conc = least_squares(predint_upper, x0=initial_guess, bounds=solver_lims).x[0]

        return np.array([lower_conc, best_conc, upper_conc], dtype=float)

    def estimate_quantile(self, measured_value: float, curve_object, solver_lims: tuple[float, float]) -> np.ndarray:
        """Estimate lower, best, upper concentration by inverting quantile prediction edges."""

        def predint_upper(x: np.ndarray) -> np.ndarray:
            return curve_object.upper_pred_edge(x) - measured_value

        def predint_lower(x: np.ndarray) -> np.ndarray:
            return curve_object.lower_pred_edge(x) - measured_value

        def best_model(x: np.ndarray) -> np.ndarray:
            return curve_object.predict(x) - measured_value

        initial_guess = float(np.mean(solver_lims))
        upper_conc = least_squares(predint_lower, x0=initial_guess, bounds=solver_lims).x[0]
        best_conc = least_squares(best_model, x0=initial_guess, bounds=solver_lims).x[0]
        lower_conc = least_squares(predint_upper, x0=initial_guess, bounds=solver_lims).x[0]

        return np.array([lower_conc, best_conc, upper_conc], dtype=float)

    def run(
        self,
        voltammograms,
        metric_column: str,
        standard_curve,
        regression_type: RegressionType,
    ) -> InverseEstimationResult:
        """Run inverse regression over all measured values in input voltammogram calibration tables."""

        solver_lims = (float(standard_curve.conc_std_min), float(standard_curve.conc_std_max))
        rows: list[EstimateRow] = []

        for voltammogram in voltammograms:
            table = voltammogram.calibration_table
            if metric_column not in table.columns:
                continue

            measured_values = table[metric_column].to_numpy(dtype=float)
            for measured_value in measured_values:
                try:
                    if regression_type == "quantile":
                        low, estimate, high = self.estimate_quantile(measured_value, standard_curve, solver_lims)
                    else:
                        low, estimate, high = self.estimate_least_squares(
                            measured_value,
                            standard_curve,
                            standard_curve.covar,
                            float(standard_curve.df),
                            float(standard_curve.s),
                            solver_lims,
                        )
                    rows.append(
                        EstimateRow(
                            name=voltammogram.name,
                            measured_value=float(measured_value),
                            estimate=float(estimate),
                            low_limit=float(low),
                            upper_limit=float(high),
                        )
                    )
                except Exception as exc:
                    rows.append(
                        EstimateRow(
                            name=voltammogram.name,
                            measured_value=float(measured_value),
                            estimate=None,
                            low_limit=None,
                            upper_limit=None,
                            error=str(exc),
                        )
                    )

        return InverseEstimationResult(
            metric_column=metric_column,
            regression_type=regression_type,
            rows=rows,
        )

    def export_csv(self, result: InverseEstimationResult, destination: Path) -> Path:
        """Export inverse estimation result to CSV."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_dataframe().to_csv(destination, index=False)
        return destination
