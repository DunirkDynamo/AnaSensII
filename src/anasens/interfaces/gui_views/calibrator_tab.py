"""Calibrator tab for the AnaSens2 Qt GUI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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

from anasens.calibrator.calibrator import DataTableCalibrator, MeasurementObjectCalibrator
from anasens.calibrator.data_loader import (
    load_data_table,
    load_measurement_objects,
    save_measurement_objects,
)
from anasens.calibrator.models import (
    CalibrationRequest,
    MeasurementObject,
)


class CalibratorTab(QWidget):
    """Calibrate extracted peak-height parameters against a co-measured control."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataframe: pd.DataFrame | None = None
        self._measurements: list[MeasurementObject] = []
        self._mode: str = "data_table"
        self._match_filters: list[str] = []
        self._last_result: pd.DataFrame | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        title = QLabel("Calibrator")
        title.setObjectName("sectionTitle")
        root_layout.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ── Left panel: controls ──────────────────────────────────────────────
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(340)
        controls_scroll.setMaximumWidth(420)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(8, 8, 8, 8)

        # Mode
        mode_card = self._make_card()
        mode_layout = QVBoxLayout(mode_card)
        mode_layout.addWidget(self._section_label("Data source"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["CSV data table", "Measurement objects (pickles)"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        self._load_button = QPushButton("Load data")
        self._load_button.clicked.connect(self._load_data)
        mode_layout.addWidget(self._load_button)
        self._load_status = QLabel("No data loaded")
        self._load_status.setWordWrap(True)
        mode_layout.addWidget(self._load_status)
        controls_layout.addWidget(mode_card)

        # Calibration method
        method_card = self._make_card()
        method_layout = QVBoxLayout(method_card)
        method_layout.addWidget(self._section_label("Calibration method"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["subcal (subtraction)", "divcal (division)"])
        method_layout.addWidget(self._method_combo)
        controls_layout.addWidget(method_card)

        # Analyte / Control labels
        labels_card = self._make_card()
        labels_layout = QVBoxLayout(labels_card)
        labels_layout.addWidget(self._section_label("Grouping"))

        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Filter column"))
        self._group_column_combo = QComboBox()
        self._group_column_combo.currentIndexChanged.connect(self._on_group_column_changed)
        group_row.addWidget(self._group_column_combo, 1)
        labels_layout.addLayout(group_row)

        analyte_row = QHBoxLayout()
        analyte_row.addWidget(QLabel("Analyte"))
        self._analyte_combo = QComboBox()
        self._analyte_combo.setMinimumWidth(120)
        analyte_row.addWidget(self._analyte_combo, 1)
        labels_layout.addLayout(analyte_row)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Control"))
        self._control_combo = QComboBox()
        self._control_combo.setMinimumWidth(120)
        control_row.addWidget(self._control_combo, 1)
        labels_layout.addLayout(control_row)
        controls_layout.addWidget(labels_card)

        # Match filters
        filters_card = self._make_card()
        filters_layout = QVBoxLayout(filters_card)
        filters_layout.addWidget(self._section_label("Match filters (optional)"))
        self._filter_list_widget = QWidget()
        self._filter_list_layout = QVBoxLayout(self._filter_list_widget)
        self._filter_list_layout.setContentsMargins(0, 0, 0, 0)
        self._filter_list_layout.setSpacing(4)
        filters_layout.addWidget(self._filter_list_widget)

        add_filter_row = QHBoxLayout()
        self._filter_column_combo = QComboBox()
        add_filter_row.addWidget(self._filter_column_combo, 1)
        add_btn = QPushButton("Add filter")
        add_btn.clicked.connect(self._add_filter)
        add_filter_row.addWidget(add_btn)
        filters_layout.addLayout(add_filter_row)

        clear_filters_btn = QPushButton("Clear all filters")
        clear_filters_btn.clicked.connect(self._clear_filters)
        filters_layout.addWidget(clear_filters_btn)
        controls_layout.addWidget(filters_card)

        # Action buttons
        run_btn = QPushButton("Run Calibration")
        run_btn.clicked.connect(self._run_calibration)
        controls_layout.addWidget(run_btn)

        self._finalize_btn = QPushButton("Finalize && Save to Pickles")
        self._finalize_btn.setEnabled(False)
        self._finalize_btn.clicked.connect(self._finalize_and_save)
        controls_layout.addWidget(self._finalize_btn)

        export_btn = QPushButton("Export Result to CSV")
        export_btn.clicked.connect(self._export_result)
        controls_layout.addWidget(export_btn)

        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls_widget)
        splitter.addWidget(controls_scroll)

        # ── Right panel: preview + log ────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        preview_label = QLabel("Result preview")
        preview_label.setObjectName("sectionTitle")
        right_layout.addWidget(preview_label)

        self._result_table = QTableWidget()
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
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

    def _on_mode_changed(self) -> None:
        self._mode = "data_table" if self._mode_combo.currentIndex() == 0 else "measurement_objects"
        self._finalize_btn.setEnabled(False)

    def _load_data(self) -> None:
        if self._mode == "data_table":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select CSV data table", "", "CSV files (*.csv)"
            )
            if not path:
                return
            try:
                self._dataframe = load_data_table(Path(path))
                self._populate_group_columns(self._dataframe)
                self._populate_filter_columns(self._dataframe)
                self._load_status.setText(f"Loaded: {Path(path).name}")
                self._log_message(f"Loaded CSV: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Load failed", str(exc))
        else:
            folder = QFileDialog.getExistingDirectory(
                self, "Select parent directory with measurement subfolders"
            )
            if not folder:
                return
            try:
                self._measurements = load_measurement_objects(Path(folder))
                first_df = self._measurements[0].extracted_parameters
                self._populate_group_columns(first_df)
                self._populate_filter_columns(first_df)
                count = len(self._measurements)
                self._load_status.setText(f"Loaded {count} measurement objects")
                self._log_message(f"Loaded {count} measurement objects from: {folder}")
            except Exception as exc:
                QMessageBox.critical(self, "Load failed", str(exc))

    def _populate_group_columns(self, dataframe: pd.DataFrame) -> None:
        current = self._group_column_combo.currentText()
        candidate_columns = [column for column in dataframe.columns if dataframe[column].dropna().nunique() > 0]
        self._group_column_combo.blockSignals(True)
        self._group_column_combo.clear()
        self._group_column_combo.addItems(candidate_columns)
        if current and current in candidate_columns:
            self._group_column_combo.setCurrentText(current)
        elif "Label" in candidate_columns:
            self._group_column_combo.setCurrentText("Label")
        self._group_column_combo.blockSignals(False)
        self._on_group_column_changed()

    def _on_group_column_changed(self) -> None:
        self._refresh_group_values()
        dataframe = self._active_dataframe()
        if dataframe is not None:
            self._populate_filter_columns(dataframe)
        self._match_filters = [
            column for column in self._match_filters if column != self._group_column_combo.currentText()
        ]
        self._rebuild_filter_chips()

    def _refresh_group_values(self) -> None:
        group_column = self._group_column_combo.currentText()
        values = self._group_column_values(group_column)
        for combo in (self._analyte_combo, self._control_combo):
            combo.clear()
            combo.addItems(values)

    def _active_dataframe(self) -> pd.DataFrame | None:
        if self._mode == "data_table":
            return self._dataframe
        if self._measurements:
            return self._measurements[0].extracted_parameters
        return None

    def _all_dataframes(self) -> list[pd.DataFrame]:
        if self._mode == "data_table":
            return [self._dataframe] if self._dataframe is not None else []
        return [measurement.extracted_parameters for measurement in self._measurements]

    def _group_column_values(self, column: str) -> list[str]:
        values: set[str] = set()
        for dataframe in self._all_dataframes():
            if column not in dataframe.columns:
                continue
            values.update(self._unique_column_values(dataframe, column))
        return sorted(values)

    @staticmethod
    def _unique_column_values(dataframe: pd.DataFrame | None, column: str) -> list[str]:
        if dataframe is None or not column or column not in dataframe.columns:
            return []
        series = dataframe[column].dropna()
        return sorted({str(value) for value in series.tolist()})

    def _populate_filter_columns(self, dataframe: pd.DataFrame) -> None:
        self._filter_column_combo.clear()
        group_column = self._group_column_combo.currentText()
        filter_columns = [column for column in dataframe.columns.tolist() if column != group_column]
        self._filter_column_combo.addItems(filter_columns)

    def _add_filter(self) -> None:
        column = self._filter_column_combo.currentText()
        if not column or column in self._match_filters:
            return
        self._match_filters.append(column)
        chip = QCheckBox(column)
        chip.setChecked(True)
        chip.stateChanged.connect(lambda _, col=column: self._remove_filter_by_name(col))
        self._filter_list_layout.addWidget(chip)
        self._log_message(f"Added match filter: {column}")

    def _remove_filter_by_name(self, column: str) -> None:
        if column in self._match_filters:
            self._match_filters.remove(column)
        for i in range(self._filter_list_layout.count()):
            widget = self._filter_list_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.text() == column:
                widget.deleteLater()
                break

    def _clear_filters(self) -> None:
        self._match_filters.clear()
        self._rebuild_filter_chips()

    def _rebuild_filter_chips(self) -> None:
        while self._filter_list_layout.count():
            widget = self._filter_list_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for column in self._match_filters:
            chip = QCheckBox(column)
            chip.setChecked(True)
            chip.stateChanged.connect(lambda _, col=column: self._remove_filter_by_name(col))
            self._filter_list_layout.addWidget(chip)

    def _build_request(self) -> CalibrationRequest:
        method_text = self._method_combo.currentText()
        method = "subcal" if "subcal" in method_text else "divcal"
        return CalibrationRequest(
            group_column=self._group_column_combo.currentText(),
            analyte_label=self._analyte_combo.currentText(),
            control_label=self._control_combo.currentText(),
            method=method,
            match_filters=list(self._match_filters),
        )

    def _run_calibration(self) -> None:
        request = self._build_request()
        if not request.group_column:
            QMessageBox.warning(self, "Missing selection", "Select a filter column first.")
            return
        if not request.analyte_label or not request.control_label:
            QMessageBox.warning(self, "Missing selection", "Select analyte and control values.")
            return

        try:
            if self._mode == "data_table":
                if self._dataframe is None:
                    QMessageBox.warning(self, "No data", "Load a CSV data table first.")
                    return
                result = DataTableCalibrator().calibrate(self._dataframe, request)
                self._last_result = result.result_table
                self._finalize_btn.setEnabled(False)
            else:
                if not self._measurements:
                    QMessageBox.warning(self, "No data", "Load measurement objects first.")
                    return
                batch = MeasurementObjectCalibrator().calibrate(self._measurements, request)
                self._last_result = batch.stacked_result.result_table
                self._measurements = [
                    m for m in self._measurements if m.calibration_table is not None
                ]
                self._finalize_btn.setEnabled(True)

            self._show_result(self._last_result)
            self._log_message(
                f"Calibration complete: {len(self._last_result)} rows | method={request.method}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Calibration failed", str(exc))

    def _show_result(self, table: pd.DataFrame) -> None:
        self._result_table.clear()
        self._result_table.setRowCount(len(table))
        self._result_table.setColumnCount(len(table.columns))
        self._result_table.setHorizontalHeaderLabels(table.columns.tolist())
        for row_idx, (_, row) in enumerate(table.iterrows()):
            for col_idx, value in enumerate(row):
                text = f"{value:.5g}" if isinstance(value, float) else str(value)
                self._result_table.setItem(row_idx, col_idx, QTableWidgetItem(text))

    def _finalize_and_save(self) -> None:
        if not self._measurements:
            QMessageBox.warning(self, "Nothing to save", "No measurement objects loaded.")
            return
        try:
            updated = save_measurement_objects(self._measurements)
            self._log_message(f"Saved calibration tables to {len(updated)} pickle files.")
            QMessageBox.information(self, "Saved", f"Updated {len(updated)} measurement pickles.")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _export_result(self) -> None:
        if self._last_result is None or self._last_result.empty:
            QMessageBox.warning(self, "Nothing to export", "Run calibration first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export calibration result", "calibration_result.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            self._last_result.to_csv(path, index=False)
            self._log_message(f"Exported to: {path}")
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
