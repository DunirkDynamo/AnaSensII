"""Inverse Estimation tab for the AnaSens2 Qt GUI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from anasens.inverse_regression.data_loader import load_measurement_pickles, load_standard_curve
from anasens.inverse_regression.estimator import InverseRegressionEstimator
from anasens.inverse_regression.models import InverseRegressionRequest


class InverseRegressionTab(QWidget):
    """Estimate analyte concentrations via inverse regression against a standard curve."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._measurements: list[dict] = []
        self._standard_curve = None
        self._result_df: pd.DataFrame | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        title = QLabel("Inverse Estimation")
        title.setObjectName("sectionTitle")
        root_layout.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ── Left panel: controls ──────────────────────────────────────────────
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(320)
        controls_scroll.setMaximumWidth(400)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(8, 8, 8, 8)

        # Step 1: load voltammograms
        step1_card = self._make_card()
        step1_layout = QVBoxLayout(step1_card)
        step1_layout.addWidget(self._section_label("1. Load calibrated measurements"))
        load_vgrams_btn = QPushButton("Browse folder")
        load_vgrams_btn.clicked.connect(self._load_measurements)
        step1_layout.addWidget(load_vgrams_btn)
        self._vgrams_status = QLabel("No measurements loaded")
        self._vgrams_status.setWordWrap(True)
        step1_layout.addWidget(self._vgrams_status)
        controls_layout.addWidget(step1_card)

        # Step 2: select calibration metric
        step2_card = self._make_card()
        step2_layout = QVBoxLayout(step2_card)
        step2_layout.addWidget(self._section_label("2. Calibration metric column"))
        self._metric_combo = QComboBox()
        step2_layout.addWidget(self._metric_combo)
        controls_layout.addWidget(step2_card)

        # Step 3: load standard curve
        step3_card = self._make_card()
        step3_layout = QVBoxLayout(step3_card)
        step3_layout.addWidget(self._section_label("3. Load standard curve"))
        load_curve_btn = QPushButton("Browse file")
        load_curve_btn.clicked.connect(self._load_standard_curve)
        step3_layout.addWidget(load_curve_btn)
        self._curve_status = QLabel("No standard curve loaded")
        self._curve_status.setWordWrap(True)
        step3_layout.addWidget(self._curve_status)
        controls_layout.addWidget(step3_card)

        # Step 4: regression type
        step4_card = self._make_card()
        step4_layout = QVBoxLayout(step4_card)
        step4_layout.addWidget(self._section_label("4. Regression type"))
        self._regression_combo = QComboBox()
        self._regression_combo.addItems([
            "Quantile Regression",
            "Least Squares Regression",
        ])
        step4_layout.addWidget(self._regression_combo)
        controls_layout.addWidget(step4_card)

        # Actions
        run_btn = QPushButton("Run Estimation")
        run_btn.clicked.connect(self._run_estimation)
        controls_layout.addWidget(run_btn)

        export_btn = QPushButton("Export Results to CSV")
        export_btn.clicked.connect(self._export_results)
        controls_layout.addWidget(export_btn)

        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls_widget)
        splitter.addWidget(controls_scroll)

        # ── Right panel: results + log ────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        results_label = QLabel("Concentration estimates")
        results_label.setObjectName("sectionTitle")
        right_layout.addWidget(results_label)

        self._result_table = QTableWidget()
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        right_layout.addWidget(self._result_table, 2)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Status messages appear here.")
        self._log.setMaximumHeight(140)
        right_layout.addWidget(self._log)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, 1)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_card(self) -> QFrame:
        card = QFrame(self)
        card.setObjectName("card")
        return card

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _log_message(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _load_measurements(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder containing calibrated measurement pickles"
        )
        if not folder:
            return
        try:
            self._measurements = load_measurement_pickles(Path(folder))
            self._populate_metric_combo()
            count = len(self._measurements)
            self._vgrams_status.setText(f"Loaded {count} measurements")
            self._log_message(f"Loaded {count} measurement pickles from: {folder}")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _populate_metric_combo(self) -> None:
        self._metric_combo.clear()
        if not self._measurements:
            return
        table = self._measurements[0].get("calibration_table", pd.DataFrame())
        if isinstance(table, pd.DataFrame):
            self._metric_combo.addItems(table.columns.tolist())

    def _load_standard_curve(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select standard curve pickle", "",
            "Pickle files (*.pkl *.pickle);;All files (*.*)"
        )
        if not path:
            return
        try:
            self._standard_curve = load_standard_curve(Path(path))
            self._curve_status.setText(f"Loaded: {Path(path).name}")
            self._log_message(f"Standard curve loaded from: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _run_estimation(self) -> None:
        if not self._measurements:
            QMessageBox.warning(self, "No data", "Load calibrated measurements first.")
            return
        if self._standard_curve is None:
            QMessageBox.warning(self, "No curve", "Load a standard curve file first.")
            return
        metric = self._metric_combo.currentText()
        if not metric:
            QMessageBox.warning(self, "No metric", "Select a calibration metric column.")
            return

        regression_text = self._regression_combo.currentText()
        regression_type = "quantile" if "Quantile" in regression_text else "least_squares"
        request = InverseRegressionRequest(
            calibration_metric=metric,
            regression_type=regression_type,
        )

        try:
            result = InverseRegressionEstimator().estimate(
                self._measurements, self._standard_curve, request
            )
            self._result_df = result.result_table
            self._show_result(result.result_table)
            rows = len(result.result_table)
            self._log_message(
                f"Estimation complete: {rows} estimates | method={regression_type} | metric={metric}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Estimation failed", str(exc))

    def _show_result(self, table: pd.DataFrame) -> None:
        self._result_table.clear()
        self._result_table.setRowCount(len(table))
        self._result_table.setColumnCount(len(table.columns))
        self._result_table.setHorizontalHeaderLabels(table.columns.tolist())
        for row_idx, (_, row) in enumerate(table.iterrows()):
            for col_idx, value in enumerate(row):
                if isinstance(value, float):
                    text = f"{value:.5g}" if not pd.isna(value) else "—"
                else:
                    text = str(value) if not (isinstance(value, float) and pd.isna(value)) else "—"
                self._result_table.setItem(row_idx, col_idx, QTableWidgetItem(text))

    def _export_results(self) -> None:
        if self._result_df is None or self._result_df.empty:
            QMessageBox.warning(self, "Nothing to export", "Run estimation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "inverse_regression_results.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            self._result_df.to_csv(path, index=False)
            self._log_message(f"Exported to: {path}")
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
