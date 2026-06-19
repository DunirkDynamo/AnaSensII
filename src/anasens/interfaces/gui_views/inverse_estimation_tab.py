"""Inverse Estimation tab for native inverse regression workflow."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

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
        self._standard_curve = None
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

        inputs_layout.addWidget(QLabel("Metric column"), 1, 0)
        self.metric_combo = QComboBox(self)
        self.metric_combo.setEnabled(False)
        inputs_layout.addWidget(self.metric_combo, 1, 1, 1, 2)

        inputs_layout.addWidget(QLabel("Standard curve file"), 2, 0)
        self.standard_curve_edit = QLineEdit(self)
        self.standard_curve_edit.setReadOnly(True)
        inputs_layout.addWidget(self.standard_curve_edit, 2, 1)
        standard_curve_button = QPushButton("Browse")
        standard_curve_button.clicked.connect(self._browse_standard_curve)
        inputs_layout.addWidget(standard_curve_button, 2, 2)

        inputs_layout.addWidget(QLabel("Regression type"), 3, 0)
        self.regression_combo = QComboBox(self)
        self.regression_combo.addItems(["Quantile Regression", "Least Squares Regression"])
        inputs_layout.addWidget(self.regression_combo, 3, 1, 1, 2)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Run Inverse Estimation")
        self.run_button.clicked.connect(self._run_estimation)
        self.run_button.setEnabled(False)
        action_row.addWidget(self.run_button)

        self.export_button = QPushButton("Export Results CSV")
        self.export_button.clicked.connect(self._export_results)
        self.export_button.setEnabled(False)
        action_row.addWidget(self.export_button)
        action_row.addStretch(1)

        inputs_layout.addLayout(action_row, 4, 0, 1, 3)
        layout.addWidget(inputs_card)

        self.status_label = QLabel("Load voltammograms and standard curve to begin.")
        self.status_label.setObjectName("subtleBody")
        layout.addWidget(self.status_label)

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
        self._refresh_run_enabled_state()

    def _refresh_run_enabled_state(self) -> None:
        ready = bool(self._voltammograms) and self._standard_curve is not None and self.metric_combo.count() > 0
        self.run_button.setEnabled(ready)

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
