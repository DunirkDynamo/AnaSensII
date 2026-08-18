"""Inverse Estimation tab for native inverse regression workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QPlainTextEdit,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from anasens.curve_builder.models import FittingResult
from anasens.curve_builder.plotter import PlotRenderer
from anasens.inverse_regression.data_loader import InverseDataLoader
from anasens.inverse_regression.models import InverseEstimationResult
from anasens.inverse_regression.service import InverseRegressionService


class InverseEstimationTab(QWidget):
    """UI layer for loading calibration inputs and running inverse regression."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data_loader = InverseDataLoader()
        self.service = InverseRegressionService()

        self._voltammograms = []
        self._loaded_table: pd.DataFrame | None = None
        self._modified_table: pd.DataFrame | None = None
        self._standard_curve = None
        self._curve_preview_dialog: QDialog | None = None
        self._curve_preview_renderer: PlotRenderer | None = None
        self._curve_summary_widget: QPlainTextEdit | None = None
        self._latest_result: InverseEstimationResult | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Inverse Estimation")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        inputs_card = QFrame(self)
        inputs_card.setObjectName("card")
        inputs_layout = QGridLayout(inputs_card)
        inputs_layout.setContentsMargins(16, 16, 16, 16)
        inputs_layout.setHorizontalSpacing(10)
        inputs_layout.setVerticalSpacing(10)

        inputs_layout.addWidget(QLabel("Voltammogram folder"), 0, 0)
        self.voltammogram_folder_edit = QLineEdit(self)
        self.voltammogram_folder_edit.setReadOnly(True)
        inputs_layout.addWidget(self.voltammogram_folder_edit, 0, 1)
        folder_button = QPushButton("Browse")
        folder_button.clicked.connect(self._browse_voltammogram_folder)
        inputs_layout.addWidget(folder_button, 0, 2)

        inputs_layout.addWidget(QLabel("Load from table"), 1, 0)
        self.table_path_edit = QLineEdit(self)
        self.table_path_edit.setReadOnly(True)
        inputs_layout.addWidget(self.table_path_edit, 1, 1)
        table_button = QPushButton("Browse")
        table_button.clicked.connect(self._browse_data_table)
        inputs_layout.addWidget(table_button, 1, 2)

        inputs_layout.addWidget(QLabel("Table metric column"), 2, 0)
        self.table_metric_combo = QComboBox(self)
        self.table_metric_combo.setEnabled(False)
        inputs_layout.addWidget(self.table_metric_combo, 2, 1, 1, 2)

        inputs_layout.addWidget(QLabel("Measurement metric column"), 3, 0)
        self.metric_combo = QComboBox(self)
        self.metric_combo.setEnabled(False)
        inputs_layout.addWidget(self.metric_combo, 3, 1, 1, 2)

        inputs_layout.addWidget(QLabel("Standard curve file"), 4, 0)
        self.standard_curve_edit = QLineEdit(self)
        self.standard_curve_edit.setReadOnly(True)
        inputs_layout.addWidget(self.standard_curve_edit, 4, 1)
        standard_curve_button = QPushButton("Browse")
        standard_curve_button.clicked.connect(self._browse_standard_curve)
        inputs_layout.addWidget(standard_curve_button, 4, 2)

        inputs_layout.addWidget(QLabel("Regression type"), 5, 0)
        self.regression_combo = QComboBox(self)
        self.regression_combo.addItems(["Quantile Regression", "Least Squares Regression"])
        inputs_layout.addWidget(self.regression_combo, 5, 1, 1, 2)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Run Inverse Estimation")
        self.run_button.clicked.connect(self._run_estimation)
        self.run_button.setEnabled(False)
        action_row.addWidget(self.run_button)

        self.run_table_button = QPushButton("Run Table Inversion")
        self.run_table_button.clicked.connect(self._run_table_inversion)
        self.run_table_button.setEnabled(False)
        action_row.addWidget(self.run_table_button)

        self.export_button = QPushButton("Export Results CSV")
        self.export_button.clicked.connect(self._export_results)
        self.export_button.setEnabled(False)
        action_row.addWidget(self.export_button)
        action_row.addStretch(1)

        inputs_layout.addLayout(action_row, 6, 0, 1, 3)
        layout.addWidget(inputs_card)

        self.status_label = QLabel("Load voltammograms and standard curve to begin.")
        self.status_label.setObjectName("subtleBody")
        layout.addWidget(self.status_label)

        self._data_table_section = QFrame(self)
        self._data_table_section.setObjectName("card")
        data_table_layout = QVBoxLayout(self._data_table_section)
        data_table_layout.setContentsMargins(12, 12, 12, 12)
        data_table_layout.setSpacing(8)

        data_table_title = QLabel("Loaded data table preview")
        data_table_title.setObjectName("sectionTitle")
        data_table_layout.addWidget(data_table_title)

        self._data_table_preview = QTableWidget(self._data_table_section)
        self._data_table_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._data_table_preview.setAlternatingRowColors(True)
        self._data_table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._data_table_preview.verticalHeader().setVisible(True)
        data_table_layout.addWidget(self._data_table_preview, 1)
        self._data_table_section.setVisible(False)
        layout.addWidget(self._data_table_section, 1)

        self._modified_table_section = QFrame(self)
        self._modified_table_section.setObjectName("card")
        modified_table_layout = QVBoxLayout(self._modified_table_section)
        modified_table_layout.setContentsMargins(12, 12, 12, 12)
        modified_table_layout.setSpacing(8)

        modified_table_title = QLabel("Inverse-estimated data table preview")
        modified_table_title.setObjectName("sectionTitle")
        modified_table_layout.addWidget(modified_table_title)

        self._modified_table_preview = QTableWidget(self._modified_table_section)
        self._modified_table_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._modified_table_preview.setAlternatingRowColors(True)
        self._modified_table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._modified_table_preview.verticalHeader().setVisible(True)
        modified_table_layout.addWidget(self._modified_table_preview, 1)

        self._export_modified_table_button = QPushButton("Export Modified Table")
        self._export_modified_table_button.setEnabled(False)
        self._export_modified_table_button.clicked.connect(self._export_modified_table)
        modified_table_layout.addWidget(self._export_modified_table_button)

        self._modified_table_section.setVisible(False)
        layout.addWidget(self._modified_table_section, 1)

        self.results_table = QTableWidget(self)
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(
            ["Name", "Measured Value", "Estimate", "Low Limit", "Upper Limit", "Error"]
        )
        self.results_table.setAlternatingRowColors(True)
        layout.addWidget(self.results_table, 1)

    def _browse_voltammogram_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Voltammogram Pickles")
        if not folder:
            return

        try:
            voltammograms = self.data_loader.load_voltammograms_from_folder(Path(folder))
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return

        self._voltammograms = voltammograms
        self.voltammogram_folder_edit.setText(folder)

        columns = list(voltammograms[0].calibration_table.columns)
        self.metric_combo.clear()
        self.metric_combo.addItems(columns)
        self.metric_combo.setEnabled(True)
        self.status_label.setText(f"Loaded {len(voltammograms)} voltammogram object(s).")
        self._refresh_run_enabled_state()

    def _browse_data_table(self) -> None:
        """Load a tabular CSV or Excel file and preview the first rows in a scrollable table."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data Table",
            "",
            "Data tables (*.csv *.xls *.xlsx);;CSV (*.csv);;Excel (*.xls *.xlsx);;All files (*.*)",
        )
        if not file_path:
            return

        source = Path(file_path)
        try:
            if source.suffix.lower() == ".csv":
                frame = pd.read_csv(source)
            elif source.suffix.lower() in {".xls", ".xlsx"}:
                frame = pd.read_excel(source)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return

        if frame.empty:
            QMessageBox.warning(self, "Empty table", "The selected table is empty.")
            return

        self._loaded_table = frame.copy()
        self.table_path_edit.setText(file_path)
        self._populate_data_table_preview(frame)
        self._data_table_section.setVisible(True)
        self._populate_table_column_selector(frame)
        self.status_label.setText(f"Loaded data table: {source.name} | rows={len(frame)} | columns={len(frame.columns)}")
        self._refresh_table_run_enabled_state()

    def _populate_data_table_preview(self, frame: pd.DataFrame) -> None:
        """Show the first 10 rows of a loaded table in a scrollable preview widget."""
        preview = frame.head(10)
        self._data_table_preview.clear()
        self._data_table_preview.setRowCount(len(preview.index))
        self._data_table_preview.setColumnCount(len(preview.columns))
        self._data_table_preview.setHorizontalHeaderLabels([str(column) for column in preview.columns])

        for row_index, (_, row) in enumerate(preview.iterrows()):
            for col_index, value in enumerate(row):
                text = "" if pd.isna(value) else str(value)
                self._data_table_preview.setItem(row_index, col_index, QTableWidgetItem(text))

        self._data_table_preview.resizeColumnsToContents()

    def _populate_table_column_selector(self, frame: pd.DataFrame) -> None:
        """Populate the column dropdown used for inverse estimation on a loaded table."""
        self.table_metric_combo.clear()
        self.table_metric_combo.addItems([str(column) for column in frame.columns.tolist()])
        self.table_metric_combo.setEnabled(self.table_metric_combo.count() > 0)

    def _browse_standard_curve(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Standard Curve Pickle",
            "",
            "Pickle files (*.pkl *.pickle);;All files (*.*)",
        )
        if not file_path:
            return

        try:
            self._standard_curve = self.data_loader.load_standard_curve(Path(file_path))
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return

        self.standard_curve_edit.setText(file_path)
        self.status_label.setText("Standard curve loaded.")
        self._show_standard_curve_preview()
        self._refresh_run_enabled_state()
        self._refresh_table_run_enabled_state()

    def _refresh_run_enabled_state(self) -> None:
        ready = bool(self._voltammograms) and self._standard_curve is not None and self.metric_combo.count() > 0
        self.run_button.setEnabled(ready)

    def _refresh_table_run_enabled_state(self) -> None:
        ready = self._loaded_table is not None and self._standard_curve is not None and self.table_metric_combo.count() > 0
        self.run_table_button.setEnabled(ready)

    def _selected_regression_type(self) -> str:
        return "quantile" if "Quantile" in self.regression_combo.currentText() else "least_squares"

    def _run_estimation(self) -> None:
        if not self._voltammograms or self._standard_curve is None:
            QMessageBox.warning(self, "Missing Inputs", "Load both voltammograms and a standard curve first.")
            return

        metric_column = self.metric_combo.currentText().strip()
        if not metric_column:
            QMessageBox.warning(self, "Missing Metric", "Select a metric column.")
            return

        try:
            result = self.service.run(
                self._voltammograms,
                metric_column=metric_column,
                standard_curve=self._standard_curve,
                regression_type=self._selected_regression_type(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Run Failed", str(exc))
            return

        self._latest_result = result
        self._populate_results_table(result)
        self.export_button.setEnabled(True)
        self.status_label.setText(f"Completed {len(result.rows)} inverse estimate row(s).")

    def _run_table_inversion(self) -> None:
        """Run inverse estimation on the loaded data table using the selected numeric column."""
        if self._loaded_table is None or self._standard_curve is None:
            QMessageBox.warning(self, "Missing Inputs", "Load a data table and standard curve first.")
            return

        column_name = self.table_metric_combo.currentText().strip()
        if not column_name:
            QMessageBox.warning(self, "Missing Column", "Select a table column.")
            return

        if column_name not in self._loaded_table.columns:
            QMessageBox.warning(self, "Missing Column", "The selected table column is unavailable.")
            return

        numeric_series = pd.to_numeric(self._loaded_table[column_name], errors="coerce")
        if numeric_series.notna().sum() == 0:
            QMessageBox.warning(self, "Non-numeric Column", "The selected column must contain numeric values.")
            return

        regression_type = self._selected_regression_type()
        solver_lims = (float(self._standard_curve.conc_std_min), float(self._standard_curve.conc_std_max))

        lower_limits: list[float] = []
        estimates: list[float] = []
        upper_limits: list[float] = []

        for measured_value in numeric_series:
            if pd.isna(measured_value):
                lower_limits.append(np.nan)
                estimates.append(np.nan)
                upper_limits.append(np.nan)
                continue

            try:
                if regression_type == "quantile":
                    low, estimate, high = self.service.estimate_quantile(float(measured_value), self._standard_curve, solver_lims)
                else:
                    low, estimate, high = self.service.estimate_least_squares(
                        float(measured_value),
                        self._standard_curve,
                        self._standard_curve.covar,
                        float(self._standard_curve.df),
                        float(self._standard_curve.s),
                        solver_lims,
                    )
                lower_limits.append(float(low))
                estimates.append(float(estimate))
                upper_limits.append(float(high))
            except Exception:
                lower_limits.append(np.nan)
                estimates.append(np.nan)
                upper_limits.append(np.nan)

        modified_table = self._loaded_table.copy()
        modified_table["Lower Limit"] = lower_limits
        modified_table["Estimate"] = estimates
        modified_table["Upper Limit"] = upper_limits

        self._modified_table = modified_table
        self._populate_modified_table_preview(modified_table)
        self._modified_table_section.setVisible(True)
        self._export_modified_table_button.setEnabled(True)
        self.status_label.setText(f"Completed table inversion for column '{column_name}'.")

    def _populate_modified_table_preview(self, frame: pd.DataFrame) -> None:
        """Show the first 10 rows of the inverse-estimated table in a scrollable preview widget."""
        preview = frame.head(10)
        self._modified_table_preview.clear()
        self._modified_table_preview.setRowCount(len(preview.index))
        self._modified_table_preview.setColumnCount(len(preview.columns))
        self._modified_table_preview.setHorizontalHeaderLabels([str(column) for column in preview.columns])

        for row_index, (_, row) in enumerate(preview.iterrows()):
            for col_index, value in enumerate(row):
                text = "" if pd.isna(value) else str(value)
                self._modified_table_preview.setItem(row_index, col_index, QTableWidgetItem(text))

        self._modified_table_preview.resizeColumnsToContents()

    def _populate_results_table(self, result: InverseEstimationResult) -> None:
        frame = result.to_dataframe()
        self.results_table.setRowCount(len(frame))

        for row_index, (_, row) in enumerate(frame.iterrows()):
            self.results_table.setItem(row_index, 0, QTableWidgetItem(str(row["Name"])))
            self.results_table.setItem(row_index, 1, QTableWidgetItem(self._format_number(row["Measured Value"])))
            self.results_table.setItem(row_index, 2, QTableWidgetItem(self._format_number(row["Estimate"])))
            self.results_table.setItem(row_index, 3, QTableWidgetItem(self._format_number(row["Low Limit"])))
            self.results_table.setItem(row_index, 4, QTableWidgetItem(self._format_number(row["Upper Limit"])))
            self.results_table.setItem(row_index, 5, QTableWidgetItem("" if row["Error"] is None else str(row["Error"])))

        self.results_table.resizeColumnsToContents()

    def _format_number(self, value) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.6g}"
        except Exception:
            return str(value)

    def _export_results(self) -> None:
        if self._latest_result is None:
            QMessageBox.warning(self, "No Results", "Run inverse estimation before exporting.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Inverse Estimation Results",
            "inverse_estimation_results.csv",
            "CSV files (*.csv)",
        )
        if not file_path:
            return

        try:
            destination = self.service.export_csv(self._latest_result, Path(file_path))
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return

        QMessageBox.information(self, "Export Complete", f"Saved results to:\n{destination}")

    def _export_modified_table(self) -> None:
        """Export the modified data table containing inverse-estimation columns."""
        if self._modified_table is None:
            QMessageBox.warning(self, "No Modified Table", "Run table inversion before exporting.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Modified Data Table",
            "inverse_estimated_table.csv",
            "CSV files (*.csv);;Excel files (*.xlsx)",
        )
        if not file_path:
            return

        destination = Path(file_path)
        try:
            if destination.suffix.lower() == ".xlsx":
                self._modified_table.to_excel(destination, index=False)
            else:
                if destination.suffix == "":
                    destination = destination.with_suffix(".csv")
                self._modified_table.to_csv(destination, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return

        QMessageBox.information(self, "Export Complete", f"Saved modified table to:\n{destination}")

    def _show_standard_curve_preview(self) -> None:
        """Open a preview dialog showing the loaded curve, bands, and residual diagnostics."""
        if self._standard_curve is None:
            return

        fit_x = np.asarray(getattr(self._standard_curve, "fit_x", np.asarray([], dtype=float)), dtype=float)
        fit_y = np.asarray(getattr(self._standard_curve, "fit_y", np.asarray([], dtype=float)), dtype=float)
        if fit_x.size == 0 or fit_y.size == 0:
            return

        residuals = fit_y - np.asarray(self._standard_curve.predict(fit_x), dtype=float)

        dialog = QDialog(self)
        dialog.setWindowTitle("Loaded Standard Curve Preview")
        dialog.resize(1500, 900)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        scroll_area = QScrollArea(dialog)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll_area, 1)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        renderer_host = QWidget(content)
        content_layout.addWidget(renderer_host, 7)
        summary_box = QPlainTextEdit(content)
        summary_box.setReadOnly(True)
        summary_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        summary_box.setMinimumHeight(190)
        summary_box.setPlaceholderText("Curve provenance and model summary")
        content_layout.addWidget(summary_box, 3)
        scroll_area.setWidget(content)

        renderer = PlotRenderer(renderer_host)
        renderer.plot_data(fit_x, fit_y, "x", "y")

        faux_result = FittingResult(
            model_name=str(getattr(self._standard_curve, "model_type", "standard curve")),
            parameter_names=list(getattr(self._standard_curve, "parameter_names", [])),
            parameter_values=np.asarray(getattr(self._standard_curve, "parameter_values", np.asarray([], dtype=float)), dtype=float),
            fitted_x=fit_x,
            fitted_y=np.asarray(self._standard_curve.predict(fit_x), dtype=float),
            display_equation=str(getattr(self._standard_curve, "provenance", {}).get("display_equation", "Loaded standard curve")),
            model_function=lambda x_values, *params: self._standard_curve.predict(x_values),
            covariance=np.asarray(getattr(self._standard_curve, "covar", np.asarray([])), dtype=float),
            residual_std=float(getattr(self._standard_curve, "s", 0.0) or 0.0),
            dof=float(getattr(self._standard_curve, "df", 1.0) or 1.0),
            fit_method=str(getattr(self._standard_curve, "provenance", {}).get("fit_method", "loaded")),
            extras={"residuals": residuals},
        )

        try:
            renderer.overlay_fit(faux_result)
        except Exception:
            pass

        try:
            confidence_band = self._standard_curve.confidence_band(fit_x)
        except Exception:
            confidence_band = None
        try:
            prediction_band = self._standard_curve.prediction_band(fit_x)
        except Exception:
            prediction_band = None

        if confidence_band is not None or prediction_band is not None:
            renderer.overlay_bands(confidence_band, prediction_band)

        self._overlay_estimation_guides(renderer)

        # Reproduce the bottom estimate-limits/resolution plot from curve builder.
        try:
            band_data = self._standard_curve.inverse_band_curve(x_values=fit_x, band="prediction")
            renderer.plot_estimate_limits_band(band_data)
        except Exception:
            pass

        try:
            renderer.plot_residuals(
                fit_x,
                residuals,
                "x",
                error_model_name=getattr(self._standard_curve, "error_model_name", None),
                error_model_parameters=getattr(self._standard_curve, "error_model_parameters", None),
            )
        except Exception:
            pass

        summary_box.setPlainText(self._build_curve_summary_text())

        self._curve_preview_dialog = dialog
        self._curve_preview_renderer = renderer
        self._curve_summary_widget = summary_box
        dialog.show()

    def _overlay_estimation_guides(self, renderer: PlotRenderer) -> None:
        """Overlay estimation/measurement range guides on the preview fit axis."""
        if self._standard_curve is None:
            return

        conc_est_min = getattr(self._standard_curve, "conc_est_min", None)
        conc_est_max = getattr(self._standard_curve, "conc_est_max", None)
        y_est_min = getattr(self._standard_curve, "y_est_min", None)
        y_est_max = getattr(self._standard_curve, "y_est_max", None)

        if y_est_min is None and conc_est_min is not None:
            try:
                y_est_min = float(np.asarray(self._standard_curve.predict(float(conc_est_min)), dtype=float).reshape(-1)[0])
            except Exception:
                y_est_min = None
        if y_est_max is None and conc_est_max is not None:
            try:
                y_est_max = float(np.asarray(self._standard_curve.predict(float(conc_est_max)), dtype=float).reshape(-1)[0])
            except Exception:
                y_est_max = None

        if conc_est_min is None or conc_est_max is None or y_est_min is None or y_est_max is None:
            return

        axis = renderer.axis
        guide_color = "#22d3ee"
        y_bottom, _ = axis.get_ylim()

        # Horizontal dashed guides: measurement range in signal space.
        axis.hlines(
            [float(y_est_min), float(y_est_max)],
            xmin=float(min(conc_est_min, conc_est_max)),
            xmax=float(max(conc_est_min, conc_est_max)),
            colors=guide_color,
            linestyles="--",
            linewidth=1.8,
            alpha=0.95,
            zorder=4,
        )

        # Vertical dotted guides: estimation range in concentration space.
        axis.vlines(
            [float(conc_est_min), float(conc_est_max)],
            ymin=float(y_bottom),
            ymax=[float(y_est_max), float(y_est_max)],
            colors=guide_color,
            linestyles=":",
            linewidth=1.8,
            alpha=0.95,
            zorder=4,
        )

        handles, labels = axis.get_legend_handles_labels()
        if "Measurement Range" not in labels:
            handles.append(Line2D([0], [0], color=guide_color, linestyle="--", linewidth=1.8))
            labels.append("Measurement Range")
        if "Estimation Range" not in labels:
            handles.append(Line2D([0], [0], color=guide_color, linestyle=":", linewidth=1.8))
            labels.append("Estimation Range")

        axis.legend(handles, labels, loc="best", frameon=True, fontsize=8.8)
        renderer.canvas.draw_idle()

    def _build_curve_summary_text(self) -> str:
        """Build a human-readable summary of the loaded curve, provenance, and covariance."""
        if self._standard_curve is None:
            return "No curve loaded."

        provenance_text = ""
        if hasattr(self._standard_curve, "show_provenance"):
            provenance_text = self._standard_curve.show_provenance(print_output=False)
        else:
            provenance_text = str(getattr(self._standard_curve, "provenance", {}))

        covariance = np.asarray(getattr(self._standard_curve, "covar", np.asarray([], dtype=float)), dtype=float)
        covariance_lines = [
            "",
            "Covariance Matrix:",
            np.array2string(covariance, precision=6, separator=", "),
        ] if covariance.size else ["", "Covariance Matrix: unavailable"]

        conc_est_min = getattr(self._standard_curve, "conc_est_min", None)
        conc_est_max = getattr(self._standard_curve, "conc_est_max", None)

        y_est_min = getattr(self._standard_curve, "y_est_min", None)
        y_est_max = getattr(self._standard_curve, "y_est_max", None)

        if y_est_min is None and conc_est_min is not None:
            try:
                y_est_min = float(np.asarray(self._standard_curve.predict(float(conc_est_min)), dtype=float).reshape(-1)[0])
            except Exception:
                y_est_min = None
        if y_est_max is None and conc_est_max is not None:
            try:
                y_est_max = float(np.asarray(self._standard_curve.predict(float(conc_est_max)), dtype=float).reshape(-1)[0])
            except Exception:
                y_est_max = None

        conc_min_text = "None" if conc_est_min is None else f"{float(conc_est_min):.6g}"
        conc_max_text = "None" if conc_est_max is None else f"{float(conc_est_max):.6g}"
        y_min_text = "None" if y_est_min is None else f"{float(y_est_min):.6g}"
        y_max_text = "None" if y_est_max is None else f"{float(y_est_max):.6g}"

        return "\n".join([
            "Curve Summary",
            "===============",
            f"Model type: {getattr(self._standard_curve, 'model_type', 'unknown')}",
            f"Parameter names: {list(getattr(self._standard_curve, 'parameter_names', []))}",
            f"Parameter values: {np.asarray(getattr(self._standard_curve, 'parameter_values', np.asarray([], dtype=float)), dtype=float).tolist()}",
            *covariance_lines,
            f"Degrees of freedom: {getattr(self._standard_curve, 'df', None)}",
            f"Fit domain: [{getattr(self._standard_curve, 'conc_std_min', None)}, {getattr(self._standard_curve, 'conc_std_max', None)}]",
            f"Estimation domain: [{conc_min_text}, {conc_max_text}]",
            f"conc_est_min: {conc_min_text}    y_est_min: {y_min_text}",
            f"conc_est_max: {conc_max_text}    y_est_max: {y_max_text}",
            "",
            "Provenance",
            "----------",
            provenance_text,
        ])
