"""Core inverse regression estimation — pure computation, no UI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import least_squares

from anasens.inverse_regression.models import (
    ConcentrationEstimate,
    InverseRegressionRequest,
    InverseRegressionResult,
    MeasurementEstimates,
    RegressionType,
)


def _solve_least_squares_estimate(
    measured_value: float,
    curve,
    solver_limits: tuple[float, float],
) -> ConcentrationEstimate:
    """Solve inverse problem using delta-method prediction intervals.

    The prediction interval is built from the curve's error propagation function,
    the residual standard deviation s, and the t-distribution with curve.df degrees
    of freedom.
    """
    t_crit = scipy.stats.t.ppf(0.975, curve.df)
    bounds = ([solver_limits[0]], [solver_limits[1]])
    x0 = [float(np.mean(solver_limits))]

    def best_model(x):
        return curve.predict(x[0]) - measured_value

    def upper_root(x):
        return (
            curve.predict(x[0])
            - scipy.stats.t.ppf(0.975, curve.df)
            * np.sqrt(curve.errprop(x[0], curve.covar) + curve.s ** 2)
            - measured_value
        )

    def lower_root(x):
        return (
            curve.predict(x[0])
            + t_crit
            * np.sqrt(curve.errprop(x[0], curve.covar) + curve.s ** 2)
            - measured_value
        )

    best = least_squares(best_model, x0=x0, bounds=bounds).x[0]
    upper = least_squares(lower_root, x0=x0, bounds=bounds).x[0]
    lower = least_squares(upper_root, x0=x0, bounds=bounds).x[0]

    return ConcentrationEstimate(
        measured_value=measured_value,
        best_estimate=best,
        lower_limit=lower,
        upper_limit=upper,
    )


def _solve_quantile_estimate(
    measured_value: float,
    curve,
    solver_limits: tuple[float, float],
) -> ConcentrationEstimate:
    """Solve inverse problem using quantile regression prediction bands."""
    bounds = ([solver_limits[0]], [solver_limits[1]])
    x0 = [float(np.mean(solver_limits))]

    def best_model(x):
        return curve.predict(x[0]) - measured_value

    def upper_root(x):
        return curve.lower_pred_edge(x[0]) - measured_value

    def lower_root(x):
        return curve.upper_pred_edge(x[0]) - measured_value

    best = least_squares(best_model, x0=x0, bounds=bounds).x[0]
    upper = least_squares(upper_root, x0=x0, bounds=bounds).x[0]
    lower = least_squares(lower_root, x0=x0, bounds=bounds).x[0]

    return ConcentrationEstimate(
        measured_value=measured_value,
        best_estimate=best,
        lower_limit=lower,
        upper_limit=upper,
    )


class InverseRegressionEstimator:
    """Apply inverse regression to a collection of measurement calibration tables.

    Decouples the mathematical inversion from loading, UI, and export concerns.
    """

    def estimate(
        self,
        measurements: list[dict],
        standard_curve,
        request: InverseRegressionRequest,
    ) -> InverseRegressionResult:
        """Estimate concentrations for all measurements.

        Args:
            measurements: List of dicts, each with a ``calibration_table`` DataFrame
                          and optional ``name`` key.
            standard_curve: Loaded standard curve object matching QuantileStandardCurve
                            or LeastSquaresStandardCurve protocol.
            request: Fully specified inverse regression request.

        Returns:
            InverseRegressionResult with a flat exportable DataFrame.
        """
        solver_limits = (float(standard_curve.conc_std_min), float(standard_curve.conc_std_max))
        all_measurement_estimates: list[MeasurementEstimates] = []

        for measurement in measurements:
            name = str(measurement.get("name", "Unnamed"))
            table: pd.DataFrame = measurement.get("calibration_table", pd.DataFrame())

            if request.calibration_metric not in table.columns:
                continue

            measurement_result = MeasurementEstimates(measurement_name=name)
            for measured_value in table[request.calibration_metric].dropna():
                try:
                    if request.regression_type == "quantile":
                        estimate = _solve_quantile_estimate(
                            float(measured_value), standard_curve, solver_limits
                        )
                    else:
                        estimate = _solve_least_squares_estimate(
                            float(measured_value), standard_curve, solver_limits
                        )
                    measurement_result.estimates.append(estimate)
                except Exception as exc:
                    measurement_result.errors.append(
                        {"measured": str(measured_value), "error": str(exc)}
                    )

            all_measurement_estimates.append(measurement_result)

        result_table = self._build_result_table(all_measurement_estimates)

        return InverseRegressionResult(
            result_table=result_table,
            regression_type=request.regression_type,
            calibration_metric=request.calibration_metric,
        )

    def _build_result_table(self, all_estimates: list[MeasurementEstimates]) -> pd.DataFrame:
        rows: list[dict] = []
        for measurement in all_estimates:
            for estimate in measurement.estimates:
                rows.append(
                    {
                        "Measurement": measurement.measurement_name,
                        "Measured Value": estimate.measured_value,
                        "Best Estimate": estimate.best_estimate,
                        "Lower Limit (95%)": estimate.lower_limit,
                        "Upper Limit (95%)": estimate.upper_limit,
                    }
                )
            for error in measurement.errors:
                rows.append(
                    {
                        "Measurement": measurement.measurement_name,
                        "Measured Value": error.get("measured", ""),
                        "Best Estimate": np.nan,
                        "Lower Limit (95%)": np.nan,
                        "Upper Limit (95%)": np.nan,
                        "Error": error.get("error", ""),
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "Measurement",
                    "Measured Value",
                    "Best Estimate",
                    "Lower Limit (95%)",
                    "Upper Limit (95%)",
                ]
            )
        return pd.DataFrame(rows)
