"""Core calibration logic — pure computation, no UI."""

from __future__ import annotations

import pandas as pd
import numpy as np

from anasens.calibrator.models import (
    BatchCalibrationResult,
    CalibrationMethod,
    CalibrationRequest,
    CalibrationResult,
    MeasurementObject,
)


class DataTableCalibrator:
    """Calibrate a single extracted-parameters DataFrame.

    Matches each analyte row to exactly one control row using the Label column
    and any caller-specified match-filter columns, then applies the requested
    arithmetic calibration method to the peak_heights column.
    """

    def calibrate(self, dataframe: pd.DataFrame, request: CalibrationRequest) -> CalibrationResult:
        """Pair analyte and control rows and compute calibration values.

        Args:
            dataframe: Extracted-parameters table containing at least Label and peak_heights.
            request: Fully specified calibration request.

        Returns:
            CalibrationResult with a result table containing one row per matched pair.

        Raises:
            ValueError: If required columns are absent or labels are not found.
        """
        self._validate(dataframe, request)

        analyte_rows = dataframe[dataframe["Label"] == request.analyte_label]
        control_rows = dataframe[dataframe["Label"] == request.control_label]

        records: list[dict] = []
        for _, analyte_row in analyte_rows.iterrows():
            match_mask = pd.Series([True] * len(control_rows), index=control_rows.index)
            for column in request.match_filters:
                match_mask &= control_rows[column] == analyte_row[column]

            matched = control_rows[match_mask]
            if len(matched) != 1:
                calibrated_value = np.nan
                control_height = np.nan
            else:
                control_height = float(matched.iloc[0]["peak_heights"])
                analyte_height = float(analyte_row["peak_heights"])
                calibrated_value = self._apply_method(analyte_height, control_height, request.method)

            record: dict = {
                "Analyte": request.analyte_label,
                "Control": request.control_label,
                f"peak_heights_{request.analyte_label}": float(analyte_row["peak_heights"]),
                f"peak_heights_{request.control_label}": control_height,
                request.method: calibrated_value,
            }
            for column in request.match_filters:
                record[column] = analyte_row[column]
            if "Timestamp" in analyte_row.index:
                record["Timestamp"] = analyte_row["Timestamp"]
            records.append(record)

        result_table = pd.DataFrame(records)
        result_table = self._order_columns(result_table, request)

        return CalibrationResult(
            result_table=result_table,
            method=request.method,
            analyte_label=request.analyte_label,
            control_label=request.control_label,
            source_mode="data_table",
        )

    def _validate(self, dataframe: pd.DataFrame, request: CalibrationRequest) -> None:
        required = {"Label", "peak_heights"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        if request.analyte_label not in dataframe["Label"].values:
            raise ValueError(f"Analyte label '{request.analyte_label}' not found in Label column.")
        if request.control_label not in dataframe["Label"].values:
            raise ValueError(f"Control label '{request.control_label}' not found in Label column.")
        for column in request.match_filters:
            if column not in dataframe.columns:
                raise ValueError(f"Match filter column '{column}' not found in DataFrame.")

    def _apply_method(self, analyte: float, control: float, method: CalibrationMethod) -> float:
        if method == "subcal":
            return analyte - control
        if method == "divcal":
            return analyte / control if control != 0 else np.nan
        raise ValueError(f"Unsupported calibration method: {method}")

    def _order_columns(self, table: pd.DataFrame, request: CalibrationRequest) -> pd.DataFrame:
        preferred_order = (
            request.match_filters
            + ["Timestamp", "Analyte", "Control",
               f"peak_heights_{request.analyte_label}",
               f"peak_heights_{request.control_label}",
               request.method]
        )
        ordered = [col for col in preferred_order if col in table.columns]
        remaining = [col for col in table.columns if col not in ordered]
        return table[ordered + remaining]


class MeasurementObjectCalibrator:
    """Calibrate a collection of MeasurementObject instances.

    Each measurement is calibrated independently using its own extracted_parameters
    table. Results are also stacked into a single pooled output table.
    """

    def __init__(self) -> None:
        self._single_calibrator = DataTableCalibrator()

    def calibrate(
        self,
        measurements: list[MeasurementObject],
        request: CalibrationRequest,
    ) -> BatchCalibrationResult:
        """Calibrate each measurement and return a combined result.

        Args:
            measurements: List of loaded measurement objects.
            request: Fully specified calibration request.

        Returns:
            BatchCalibrationResult with per-measurement tables and a stacked result.

        Raises:
            ValueError: If no measurements could be calibrated.
        """
        per_measurement: dict[str, pd.DataFrame] = {}
        stacked_rows: list[pd.DataFrame] = []

        for measurement in measurements:
            try:
                result = self._single_calibrator.calibrate(measurement.extracted_parameters, request)
                table = result.result_table.copy()
                table.insert(0, "Measurement", measurement.measurement_name)
                measurement.calibration_table = table
                per_measurement[measurement.measurement_name] = table
                stacked_rows.append(table)
            except ValueError:
                continue

        if not stacked_rows:
            raise ValueError("No measurements could be calibrated with the provided request.")

        stacked_table = pd.concat(stacked_rows, ignore_index=True)
        stacked_result = CalibrationResult(
            result_table=stacked_table,
            method=request.method,
            analyte_label=request.analyte_label,
            control_label=request.control_label,
            source_mode="measurement_objects",
        )

        return BatchCalibrationResult(
            stacked_result=stacked_result,
            per_measurement_tables=per_measurement,
        )
