"""Inverse Estimation tab for the AnaSens2 Qt GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QDialog,
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

        self._show_plots_btn = QPushButton("Show inverse-band plots")
        self._show_plots_btn.clicked.connect(self._show_inverse_band_plots)
        self._show_plots_btn.setEnabled(False)
        controls_layout.addWidget(self._show_plots_btn)

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
            self._show_plots_btn.setEnabled(True)
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
            self._show_plots_btn.setEnabled(True)
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

    def _show_inverse_band_plots(self) -> None:
        """Show the inverse calibration and resolution plots in a scrollable dialog.

        Purpose:
            Render the prediction-band inverse curves in a wide viewport with a
            horizontal scrollbar so the figure can remain publication-sized.

        Inputs:
            None. The method uses the currently loaded standard curve.

        Output:
            None. A non-modal dialog is created and shown.
        """
        if self._standard_curve is None:
            QMessageBox.warning(self, "No curve", "Load a standard curve file first.")
            return

        try:
            # Use prediction-band inversion because the inverse calibration plot
            # is defined on observation uncertainty, not the fitted mean CI.
            band_data = self._standard_curve.inverse_band_curve(band="prediction")
        except Exception as exc:
            QMessageBox.critical(self, "Plot failed", str(exc))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Inverse Calibration and Resolution")
        dialog.resize(1280, 760)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Use a wide figure so the dialog can scroll horizontally instead of
        # compressing the content into a cramped viewport.
        figure = Figure(figsize=(18.0, 6.2), dpi=150, facecolor="#0c0820")
        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumWidth(1750)
        canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(canvas)

        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        plot_scroll.setFrameShape(QFrame.Shape.NoFrame)
        plot_scroll.setWidget(plot_host)
        layout.addWidget(plot_scroll)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        x_grid = band_data["x_grid"]
        y_grid = band_data["y_grid"]
        x_mle = band_data["x_mle"]
        x_lower = band_data["x_lower"]
        x_upper = band_data["x_upper"]
        resolution = band_data["resolution"]
        valid = np.isfinite(x_mle) & np.isfinite(x_lower) & np.isfinite(x_upper) & np.isfinite(resolution)

        figure.clear()
        ax1 = figure.add_subplot(111)
        ax2 = ax1.twinx()

        for axis in (ax1, ax2):
            axis.set_facecolor("#f8f7fb")
            axis.tick_params(axis="both", which="major", labelsize=10, length=6, width=1.1, direction="out")
            axis.tick_params(axis="both", which="minor", labelsize=9, length=3, width=0.9, direction="out")
            for spine in axis.spines.values():
                spine.set_linewidth(1.1)
                spine.set_color("#1c1c1c")

        ax1.plot(y_grid, x_mle, color="#111111", linewidth=2.4, label=r"MLE point estimate $\hat{x}_0(y_0)$")
        ax1.plot(y_grid, x_lower, color="#2f3b4c", linestyle="--", linewidth=1.9, label=r"Confidence Interval Limits")
        ax1.plot(y_grid, x_upper, color="#2f3b4c", linestyle="--", linewidth=1.9, label="_nolegend_")
        ax1.fill_between(y_grid, x_lower, x_upper, color="#1f77b4", alpha=0.14, label="Prediction uncertainty band")
        ax1.set_title("Inverse Calibration and Resolution", fontsize=15, pad=12, fontweight="semibold")
        ax1.set_xlabel(r"Hypothetical Measurement Observation ($y_0$)", fontsize=13, labelpad=8)
        ax1.set_ylabel(r"Inferred Parameter Estimate ($x_0$)", fontsize=13, labelpad=8)
        ax1.minorticks_on()
        ax1.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18)
        ax1.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.12)

        ax2.plot(
            x_mle[valid],
            resolution[valid],
            color="darkmagenta",
            linewidth=2.8,
            label=r"Relative width: $\frac{x_0^{\mathrm{upper}} - x_0^{\mathrm{lower}}}{\hat{x}_0}$",
        )
        ax2.set_ylabel("Resolution", fontsize=13, labelpad=8)
        ax2.minorticks_on()
        ax2.grid(False)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper left",
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            fontsize=10,
            borderpad=0.7,
        )

        figure.tight_layout(pad=1.4)
        canvas.draw_idle()

        dialog.show()
        self._plot_dialog = dialog
