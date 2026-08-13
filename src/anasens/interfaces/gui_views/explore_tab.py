"""Explore tab for loading tabular data and generating publication-quality plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from anasens.curve_builder.fitters import FitService


class ExploreTab(QWidget):
    """Interactive data exploration tab for table preview and quick plotting."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataframe: pd.DataFrame | None = None
        self._last_scatter_xy: tuple[np.ndarray, np.ndarray] | None = None
        self._last_scatter_columns: tuple[str, str] | None = None
        self._plot_data_subset: pd.DataFrame | None = None
        self._derived_base_table: pd.DataFrame | None = None
        self._derived_table: pd.DataFrame | None = None
        self._derived_keep_columns: list[str] = []
        self._table_filters: list[dict[str, object]] = []
        self._fit_service = FitService()
        self._fit_stats_dialog: QDialog | None = None
        self._pairplot_dialog: QDialog | None = None
        self._active_table_filters_dialog: QDialog | None = None
        self._gaussian_marginals_dialog: QDialog | None = None

        self._figure = Figure(figsize=(7.5, 5.0), dpi=130, facecolor="#ffffff")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axis = self._figure.add_subplot(111)

        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        title = QLabel("Explore")
        title.setObjectName("sectionTitle")
        root_layout.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(480)
        controls_scroll.setMaximumWidth(620)

        controls_host = QWidget()
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(8, 8, 8, 8)

        load_card = self._make_card()
        load_layout = QVBoxLayout(load_card)
        load_layout.addWidget(self._section_label("1. Load data table"))
        load_button = QPushButton("Load CSV / Excel")
        load_button.clicked.connect(self._load_table)
        load_layout.addWidget(load_button)
        self._load_status = QLabel("No table loaded")
        self._load_status.setWordWrap(True)
        load_layout.addWidget(self._load_status)

        self._create_new_table_button = QPushButton("Create New Table")
        self._create_new_table_button.setEnabled(False)
        self._create_new_table_button.clicked.connect(self._open_keep_columns_dialog)
        load_layout.addWidget(self._create_new_table_button)

        filter_title = QLabel("Row filter for new table")
        filter_title.setObjectName("subtitle")
        load_layout.addWidget(filter_title)

        filter_layout = QGridLayout()
        filter_layout.setHorizontalSpacing(8)
        filter_layout.setVerticalSpacing(6)
        filter_layout.addWidget(QLabel("Column"), 0, 0)
        self._table_filter_column_combo = QComboBox()
        self._table_filter_column_combo.setEnabled(False)
        filter_layout.addWidget(self._table_filter_column_combo, 0, 1)

        filter_layout.addWidget(QLabel("Filter type"), 1, 0)
        self._table_filter_type_combo = QComboBox()
        self._table_filter_type_combo.addItems(["equal to", "greater than", "less than"])
        self._table_filter_type_combo.setEnabled(False)
        filter_layout.addWidget(self._table_filter_type_combo, 1, 1)

        filter_layout.addWidget(QLabel("Value"), 2, 0)
        self._table_filter_value_edit = QLineEdit()
        self._table_filter_value_edit.setPlaceholderText("For equal to: comma-separated values")
        self._table_filter_value_edit.setEnabled(False)
        filter_layout.addWidget(self._table_filter_value_edit, 2, 1)
        load_layout.addLayout(filter_layout)

        filter_actions = QHBoxLayout()
        self._apply_filter_button = QPushButton("Confirm Filter")
        self._apply_filter_button.setEnabled(False)
        self._apply_filter_button.clicked.connect(self._apply_new_table_filter)
        filter_actions.addWidget(self._apply_filter_button)

        self._show_filters_button = QPushButton("Show all filters")
        self._show_filters_button.setEnabled(False)
        self._show_filters_button.clicked.connect(self._show_active_table_filters)
        filter_actions.addWidget(self._show_filters_button)
        load_layout.addLayout(filter_actions)
        controls_layout.addWidget(load_card)

        plot_card = self._make_card()
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.addWidget(self._section_label("2. Choose plot type"))

        self._dataset_row = QWidget()
        dataset_row_layout = QHBoxLayout(self._dataset_row)
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        dataset_row_layout.setSpacing(8)
        self._dataset_label = QLabel("Dataset:")
        self._dataset_combo = QComboBox()
        self._dataset_combo.addItems(["Original data", "Filtered data"])
        self._dataset_combo.currentIndexChanged.connect(self._on_plot_dataset_changed)
        dataset_row_layout.addWidget(self._dataset_label)
        dataset_row_layout.addWidget(self._dataset_combo, 1)
        self._dataset_row.setVisible(False)
        plot_layout.addWidget(self._dataset_row)

        self._plot_type_combo = QComboBox()
        self._plot_type_combo.addItems(["Scatterplot", "Histogram", "Boxplot", "Pairplot", "Hexbin"])
        self._plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)
        self._plot_type_combo.setEnabled(False)
        plot_layout.addWidget(self._plot_type_combo)

        self._scatter_controls = QWidget()
        scatter_layout = QGridLayout(self._scatter_controls)
        scatter_layout.setContentsMargins(0, 0, 0, 0)
        scatter_layout.setHorizontalSpacing(8)
        scatter_layout.setVerticalSpacing(8)
        scatter_layout.addWidget(QLabel("X column"), 0, 0)
        self._scatter_x_combo = QComboBox()
        scatter_layout.addWidget(self._scatter_x_combo, 0, 1)
        scatter_layout.addWidget(QLabel("Y column"), 1, 0)
        self._scatter_y_combo = QComboBox()
        scatter_layout.addWidget(self._scatter_y_combo, 1, 1)
        scatter_layout.addWidget(QLabel("Hue/category (optional)"), 2, 0)
        self._scatter_hue_combo = QComboBox()
        scatter_layout.addWidget(self._scatter_hue_combo, 2, 1)

        self._fit_model_label = QLabel("Fit model")
        scatter_layout.addWidget(self._fit_model_label, 3, 0)
        self._fit_model_combo = QComboBox()
        self._fit_model_combo.addItems(["Polynomial", "Power", "Custom"])
        self._fit_model_combo.currentIndexChanged.connect(self._on_fit_model_changed)
        scatter_layout.addWidget(self._fit_model_combo, 3, 1)

        self._fit_poly_order_label = QLabel("Polynomial order")
        scatter_layout.addWidget(self._fit_poly_order_label, 4, 0)
        self._fit_poly_order_spin = QSpinBox()
        self._fit_poly_order_spin.setMinimum(1)
        self._fit_poly_order_spin.setMaximum(8)
        self._fit_poly_order_spin.setValue(2)
        scatter_layout.addWidget(self._fit_poly_order_spin, 4, 1)

        self._fit_custom_expression_label = QLabel("Custom expression")
        scatter_layout.addWidget(self._fit_custom_expression_label, 5, 0)
        self._fit_custom_expression_edit = QLineEdit()
        self._fit_custom_expression_edit.setPlaceholderText("Example: a*x^2 + b*x + c")
        self._fit_custom_expression_edit.setText("a*x + b")
        scatter_layout.addWidget(self._fit_custom_expression_edit, 5, 1)

        self._fit_custom_params_label = QLabel("Custom parameters")
        scatter_layout.addWidget(self._fit_custom_params_label, 6, 0)
        self._fit_custom_params_edit = QLineEdit()
        self._fit_custom_params_edit.setPlaceholderText("Comma-separated, e.g. a,b,c")
        self._fit_custom_params_edit.setText("a,b")
        scatter_layout.addWidget(self._fit_custom_params_edit, 6, 1)
        plot_layout.addWidget(self._scatter_controls)

        self._hist_controls = QWidget()
        hist_layout = QGridLayout(self._hist_controls)
        hist_layout.setContentsMargins(0, 0, 0, 0)
        hist_layout.setHorizontalSpacing(8)
        hist_layout.setVerticalSpacing(8)
        hist_layout.addWidget(QLabel("Value column"), 0, 0)
        self._hist_value_combo = QComboBox()
        hist_layout.addWidget(self._hist_value_combo, 0, 1)
        plot_layout.addWidget(self._hist_controls)

        self._box_controls = QWidget()
        box_layout = QGridLayout(self._box_controls)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setHorizontalSpacing(8)
        box_layout.setVerticalSpacing(8)
        box_layout.addWidget(QLabel("Category/hue column"), 0, 0)
        self._box_hue_combo = QComboBox()
        box_layout.addWidget(self._box_hue_combo, 0, 1)
        box_layout.addWidget(QLabel("Value column"), 1, 0)
        self._box_value_combo = QComboBox()
        box_layout.addWidget(self._box_value_combo, 1, 1)
        plot_layout.addWidget(self._box_controls)

        self._pairplot_controls = QWidget()
        pairplot_layout = QVBoxLayout(self._pairplot_controls)
        pairplot_layout.setContentsMargins(0, 0, 0, 0)
        pairplot_layout.setSpacing(6)
        pairplot_layout.addWidget(QLabel("Select columns (multi-select)"))
        self._pairplot_columns_list = QListWidget()
        self._pairplot_columns_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._pairplot_columns_list.setMinimumHeight(150)
        pairplot_layout.addWidget(self._pairplot_columns_list)
        pairplot_layout.addWidget(QLabel("Use Ctrl/Cmd-click or Shift-click to select many columns."))
        plot_layout.addWidget(self._pairplot_controls)

        self._hexbin_controls = QWidget()
        hexbin_layout = QGridLayout(self._hexbin_controls)
        hexbin_layout.setContentsMargins(0, 0, 0, 0)
        hexbin_layout.setHorizontalSpacing(8)
        hexbin_layout.setVerticalSpacing(8)
        hexbin_layout.addWidget(QLabel("X column"), 0, 0)
        self._hexbin_x_combo = QComboBox()
        hexbin_layout.addWidget(self._hexbin_x_combo, 0, 1)
        hexbin_layout.addWidget(QLabel("Y column"), 1, 0)
        self._hexbin_y_combo = QComboBox()
        hexbin_layout.addWidget(self._hexbin_y_combo, 1, 1)
        plot_layout.addWidget(self._hexbin_controls)

        actions_grid = QGridLayout()
        actions_grid.setHorizontalSpacing(8)
        actions_grid.setVerticalSpacing(8)
        self._generate_plot_button = QPushButton("Generate plot")
        self._generate_plot_button.clicked.connect(self._generate_plot)
        self._generate_plot_button.setEnabled(False)
        actions_grid.addWidget(self._generate_plot_button, 0, 0)

        self._clear_plot_button = QPushButton("New/Clear")
        self._clear_plot_button.clicked.connect(self._clear_plot_state)
        self._clear_plot_button.setEnabled(False)
        actions_grid.addWidget(self._clear_plot_button, 0, 1)

        self._add_contours_button = QPushButton("Add contours")
        self._add_contours_button.clicked.connect(self._add_scatter_contours)
        self._add_contours_button.setEnabled(False)
        actions_grid.addWidget(self._add_contours_button, 0, 2)

        self._add_fit_button = QPushButton("Add fit")
        self._add_fit_button.clicked.connect(self._add_scatter_fit)
        self._add_fit_button.setEnabled(False)
        self._add_fit_button.setVisible(False)
        actions_grid.addWidget(self._add_fit_button, 1, 0)

        self._export_plot_button = QPushButton("Export plot image")
        self._export_plot_button.clicked.connect(self._export_plot)
        self._export_plot_button.setEnabled(False)
        actions_grid.addWidget(self._export_plot_button, 1, 1)

        self._export_data_button = QPushButton("Export data")
        self._export_data_button.clicked.connect(self._export_data)
        self._export_data_button.setEnabled(False)
        actions_grid.addWidget(self._export_data_button, 1, 2)

        plot_layout.addLayout(actions_grid)
        controls_layout.addWidget(plot_card)

        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls_host)
        splitter.addWidget(controls_scroll)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        plot_label = QLabel("Plot preview")
        plot_label.setObjectName("sectionTitle")
        right_layout.addWidget(plot_label)

        plot_card_right = self._make_card()
        plot_card_right_layout = QVBoxLayout(plot_card_right)
        plot_card_right_layout.setContentsMargins(8, 8, 8, 8)
        self._canvas.setMinimumSize(660, 440)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_card_right_layout.addWidget(self._canvas)
        right_layout.addWidget(plot_card_right, 2)

        table_label = QLabel("Loaded data preview (first 10 rows)")
        table_label.setObjectName("sectionTitle")
        right_layout.addWidget(table_label)

        self._table_preview = QTableWidget()
        self._table_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table_preview.verticalHeader().setVisible(True)
        self._table_preview.setAlternatingRowColors(True)
        right_layout.addWidget(self._table_preview, 2)

        self._derived_table_section = QWidget()
        derived_layout = QVBoxLayout(self._derived_table_section)
        derived_layout.setContentsMargins(0, 0, 0, 0)
        derived_layout.setSpacing(6)

        self._derived_table_label = QLabel("New table preview (first 10 rows)")
        self._derived_table_label.setObjectName("sectionTitle")
        derived_layout.addWidget(self._derived_table_label)

        self._derived_table_preview = QTableWidget()
        self._derived_table_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._derived_table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._derived_table_preview.verticalHeader().setVisible(True)
        self._derived_table_preview.setAlternatingRowColors(True)
        derived_layout.addWidget(self._derived_table_preview, 1)

        derived_actions = QHBoxLayout()
        derived_actions.addStretch(1)
        self._export_new_table_button = QPushButton("Export New Table")
        self._export_new_table_button.setEnabled(False)
        self._export_new_table_button.clicked.connect(self._export_new_table)
        derived_actions.addWidget(self._export_new_table_button)
        derived_layout.addLayout(derived_actions)

        self._derived_table_section.setVisible(False)
        right_layout.addWidget(self._derived_table_section, 2)

        self._box_stats_section = QWidget()
        box_stats_layout = QVBoxLayout(self._box_stats_section)
        box_stats_layout.setContentsMargins(0, 0, 0, 0)
        box_stats_layout.setSpacing(6)

        self._box_stats_label = QLabel("Boxplot statistics")
        self._box_stats_label.setObjectName("sectionTitle")
        box_stats_layout.addWidget(self._box_stats_label)

        self._box_stats_text = QPlainTextEdit()
        self._box_stats_text.setReadOnly(True)
        self._box_stats_text.setPlaceholderText("Boxplot summary appears here after generating a boxplot.")
        self._box_stats_text.setMinimumHeight(180)
        box_stats_layout.addWidget(self._box_stats_text, 1)
        self._box_stats_section.setVisible(False)
        right_layout.addWidget(self._box_stats_section, 1)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Status messages appear here.")
        self._log.setMaximumHeight(120)
        right_layout.addWidget(self._log)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([560, 980])
        root_layout.addWidget(splitter, 1)

        self._apply_publication_style()
        self._on_plot_type_changed()
        self._on_fit_model_changed()

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

    def _load_table(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select data table",
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
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        if frame.empty:
            QMessageBox.warning(self, "Empty table", "The selected table is empty.")
            return

        self._dataframe = frame.copy()
        self._load_status.setText(f"Loaded: {source.name} | rows={len(frame)} | columns={len(frame.columns)}")
        self._populate_column_selectors(frame)
        self._populate_table_preview(frame)

        self._plot_type_combo.setEnabled(True)
        self._generate_plot_button.setEnabled(True)
        self._clear_plot_button.setEnabled(True)
        self._create_new_table_button.setEnabled(True)
        self._add_contours_button.setEnabled(False)
        self._export_plot_button.setEnabled(False)
        self._export_data_button.setEnabled(False)
        self._box_stats_text.clear()
        self._plot_data_subset = None
        self._derived_base_table = None
        self._derived_table = None
        self._derived_keep_columns = []
        self._table_filters = []
        self._table_filter_column_combo.clear()
        self._table_filter_column_combo.setEnabled(False)
        self._table_filter_type_combo.setEnabled(False)
        self._table_filter_value_edit.clear()
        self._table_filter_value_edit.setEnabled(False)
        self._apply_filter_button.setEnabled(False)
        self._show_filters_button.setEnabled(False)
        self._derived_table_preview.clear()
        self._derived_table_preview.setRowCount(0)
        self._derived_table_preview.setColumnCount(0)
        self._derived_table_section.setVisible(False)
        self._export_new_table_button.setEnabled(False)
        self._dataset_row.setVisible(False)
        self._dataset_combo.setCurrentText("Original data")
        self._dataset_combo.setEnabled(False)
        self._box_stats_text.clear()
        self._box_stats_section.setVisible(False)
        self._update_scatter_fit_controls_visibility()
        self._log_message(f"Loaded table from {source}")

    def _populate_column_selectors(self, frame: pd.DataFrame) -> None:
        columns = [str(column) for column in frame.columns.tolist()]

        for combo in (
            self._scatter_x_combo,
            self._scatter_y_combo,
            self._hist_value_combo,
            self._box_hue_combo,
            self._box_value_combo,
            self._hexbin_x_combo,
            self._hexbin_y_combo,
        ):
            combo.clear()
            combo.addItems(columns)

        self._scatter_hue_combo.clear()
        self._scatter_hue_combo.addItem("(None)")
        self._scatter_hue_combo.addItems(columns)

        self._pairplot_columns_list.clear()
        for column in columns:
            self._pairplot_columns_list.addItem(QListWidgetItem(column))

    def _populate_table_preview(self, frame: pd.DataFrame) -> None:
        preview = frame.head(10)
        self._table_preview.clear()
        self._table_preview.setRowCount(len(preview.index))
        self._table_preview.setColumnCount(len(preview.columns))
        self._table_preview.setHorizontalHeaderLabels([str(column) for column in preview.columns])

        for row_index, (_, row) in enumerate(preview.iterrows()):
            for col_index, value in enumerate(row):
                text = "" if pd.isna(value) else str(value)
                self._table_preview.setItem(row_index, col_index, QTableWidgetItem(text))

        self._table_preview.resizeColumnsToContents()

    def _populate_derived_table_preview(self, frame: pd.DataFrame | None) -> None:
        """Display first 10 rows of derived/new table."""
        self._derived_table_preview.clear()
        if frame is None or frame.empty:
            self._derived_table_preview.setRowCount(0)
            self._derived_table_preview.setColumnCount(0)
            return

        preview = frame.head(10)
        self._derived_table_preview.setRowCount(len(preview.index))
        self._derived_table_preview.setColumnCount(len(preview.columns))
        self._derived_table_preview.setHorizontalHeaderLabels([str(column) for column in preview.columns])

        for row_index, (_, row) in enumerate(preview.iterrows()):
            for col_index, value in enumerate(row):
                text = "" if pd.isna(value) else str(value)
                self._derived_table_preview.setItem(row_index, col_index, QTableWidgetItem(text))

        self._derived_table_preview.resizeColumnsToContents()

    def _open_keep_columns_dialog(self) -> None:
        """Open dialog to choose kept columns for building a derived table."""
        if self._dataframe is None:
            QMessageBox.warning(self, "No data", "Load a data table first.")
            return

        columns = [str(column) for column in self._dataframe.columns.tolist()]
        dialog = QDialog(self)
        dialog.setWindowTitle("Keep Columns")
        dialog.resize(520, 620)
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Select columns to keep in the new table")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        check_host = QWidget()
        check_layout = QVBoxLayout(check_host)
        check_layout.setContentsMargins(6, 6, 6, 6)
        check_layout.setSpacing(4)

        checkboxes: list[QCheckBox] = []
        for column_name in columns:
            checkbox = QCheckBox(column_name)
            checkbox.setChecked(True)
            checkboxes.append(checkbox)
            check_layout.addWidget(checkbox)
        check_layout.addStretch(1)

        scroll.setWidget(check_host)
        layout.addWidget(scroll, 1)

        action_row = QHBoxLayout()
        select_all_button = QPushButton("Select all")
        unselect_all_button = QPushButton("Unselect all")
        confirm_button = QPushButton("Confirm")

        def set_all(checked: bool) -> None:
            for box in checkboxes:
                box.setChecked(checked)

        select_all_button.clicked.connect(lambda: set_all(True))
        unselect_all_button.clicked.connect(lambda: set_all(False))

        def confirm_selection() -> None:
            selected = [box.text() for box in checkboxes if box.isChecked()]
            if not selected:
                QMessageBox.warning(dialog, "No columns selected", "Select at least one column.")
                return
            self._derived_keep_columns = selected
            self._derived_base_table = self._dataframe[selected].copy()
            self._table_filters = []
            self._refresh_derived_table_from_filters()

            self._table_filter_column_combo.clear()
            self._table_filter_column_combo.addItems(selected)
            self._table_filter_column_combo.setEnabled(True)
            self._table_filter_type_combo.setEnabled(True)
            self._table_filter_value_edit.setEnabled(True)
            self._apply_filter_button.setEnabled(True)
            self._show_filters_button.setEnabled(False)
            dialog.accept()

        confirm_button.clicked.connect(confirm_selection)

        action_row.addWidget(select_all_button)
        action_row.addWidget(unselect_all_button)
        action_row.addStretch(1)
        action_row.addWidget(confirm_button)
        layout.addLayout(action_row)

        dialog.exec()

    def _refresh_derived_table_from_filters(self) -> None:
        """Recompute derived table by applying all active filters in order."""
        if self._derived_base_table is None:
            self._derived_table = None
            self._populate_derived_table_preview(None)
            self._derived_table_section.setVisible(False)
            self._export_new_table_button.setEnabled(False)
            return

        frame = self._derived_base_table.copy()
        for filter_spec in self._table_filters:
            frame = self._apply_single_filter(frame, filter_spec)

        self._derived_table = frame
        self._populate_derived_table_preview(frame)
        self._derived_table_section.setVisible(True)
        self._export_new_table_button.setEnabled(True)
        self._dataset_row.setVisible(True)
        self._dataset_combo.setEnabled(True)
        self._show_filters_button.setEnabled(bool(self._table_filters))
        if self._dataset_combo.currentText().strip().lower() == "filtered data":
            self._on_plot_dataset_changed()

    def _on_plot_dataset_changed(self) -> None:
        """Refresh plotting selectors after switching plot dataset source."""
        frame = self._get_selected_plot_dataset()
        if frame is None:
            return

        self._populate_column_selectors(frame)
        self._last_scatter_xy = None
        self._last_scatter_columns = None
        self._add_contours_button.setEnabled(False)
        self._add_fit_button.setEnabled(False)
        self._update_scatter_fit_controls_visibility()

    def _get_selected_plot_dataset(self) -> pd.DataFrame | None:
        """Return the currently selected dataset for plotting controls and plot generation."""
        selected = self._dataset_combo.currentText().strip().lower()
        if selected == "filtered data" and self._derived_table is not None:
            return self._derived_table
        return self._dataframe

    def _apply_single_filter(self, frame: pd.DataFrame, filter_spec: dict[str, object]) -> pd.DataFrame:
        """Apply one filter spec to a frame and return filtered rows."""
        column_name = str(filter_spec.get("column", ""))
        filter_type = str(filter_spec.get("type", ""))
        raw_value = filter_spec.get("value", "")

        if column_name not in frame.columns:
            return frame

        if filter_type == "equal to":
            if isinstance(raw_value, list):
                target_values = [str(value).strip() for value in raw_value if str(value).strip()]
            else:
                target_values = [str(raw_value).strip()]
            if not target_values:
                return frame

            series = frame[column_name]
            numeric_targets: list[float] = []
            for value in target_values:
                try:
                    numeric_targets.append(float(value))
                except Exception:
                    pass

            mask = series.astype(str).isin(target_values)
            if numeric_targets:
                numeric_series = pd.to_numeric(series, errors="coerce")
                mask = mask | numeric_series.isin(numeric_targets)
            return frame[mask]

        numeric_series = pd.to_numeric(frame[column_name], errors="coerce")
        try:
            scalar = float(raw_value)
        except Exception:
            return frame.iloc[0:0]

        if filter_type == "greater than":
            return frame[numeric_series > scalar]
        if filter_type == "less than":
            return frame[numeric_series < scalar]

        return frame

    def _apply_new_table_filter(self) -> None:
        """Append a new row filter to the derived table workflow."""
        if self._derived_base_table is None:
            QMessageBox.warning(self, "No new table", "Create a new table first using Keep Columns.")
            return

        column_name = self._table_filter_column_combo.currentText().strip()
        filter_type = self._table_filter_type_combo.currentText().strip().lower()
        raw_value_text = self._table_filter_value_edit.text().strip()

        if not column_name:
            QMessageBox.warning(self, "Missing column", "Select a column to filter.")
            return
        if not raw_value_text:
            QMessageBox.warning(self, "Missing value", "Enter a filter value.")
            return

        if filter_type == "equal to":
            value_payload = [value.strip() for value in raw_value_text.split(",") if value.strip()]
            if not value_payload:
                QMessageBox.warning(self, "Missing values", "Provide one or more comma-separated values.")
                return
        else:
            value_payload = raw_value_text
            try:
                float(raw_value_text)
            except Exception:
                QMessageBox.warning(self, "Invalid numeric value", "Greater/less filters require a numeric value.")
                return

        filter_spec = {
            "column": column_name,
            "type": filter_type,
            "value": value_payload,
        }
        self._table_filters.append(filter_spec)
        self._refresh_derived_table_from_filters()

        self._table_filter_value_edit.clear()
        self._table_filter_type_combo.setCurrentText("equal to")
        self._log_message(f"Applied table filter: {column_name} {filter_type} {value_payload}")

    def _show_active_table_filters(self) -> None:
        """Open popup listing active derived-table filters with delete controls."""
        if not self._table_filters:
            QMessageBox.information(self, "No filters", "No active filters to show.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Active Table Filters")
        dialog.resize(860, 460)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        table = QTableWidget(dialog)
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Select", "Column", "Type", "Value"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setRowCount(len(self._table_filters))

        def render_table() -> None:
            table.setRowCount(len(self._table_filters))
            for row_index, filter_spec in enumerate(self._table_filters):
                checkbox_item = QTableWidgetItem("")
                checkbox_item.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
                checkbox_item.setCheckState(Qt.CheckState.Unchecked)
                table.setItem(row_index, 0, checkbox_item)
                table.setItem(row_index, 1, QTableWidgetItem(str(filter_spec.get("column", ""))))
                table.setItem(row_index, 2, QTableWidgetItem(str(filter_spec.get("type", ""))))
                value_repr = filter_spec.get("value", "")
                if isinstance(value_repr, list):
                    value_text = ", ".join(str(value) for value in value_repr)
                else:
                    value_text = str(value_repr)
                table.setItem(row_index, 3, QTableWidgetItem(value_text))

        render_table()
        layout.addWidget(table, 1)

        buttons_row = QHBoxLayout()
        select_all_button = QPushButton("Select all")
        unselect_all_button = QPushButton("Unselect all")
        delete_selected_button = QPushButton("Delete selected")

        def set_all(state: Qt.CheckState) -> None:
            for row_index in range(table.rowCount()):
                item = table.item(row_index, 0)
                if item is not None:
                    item.setCheckState(state)

        def delete_selected() -> None:
            selected_rows: list[int] = []
            for row_index in range(table.rowCount()):
                item = table.item(row_index, 0)
                if item is not None and item.checkState() == Qt.CheckState.Checked:
                    selected_rows.append(row_index)

            if not selected_rows:
                return

            for row_index in sorted(selected_rows, reverse=True):
                if 0 <= row_index < len(self._table_filters):
                    del self._table_filters[row_index]

            self._refresh_derived_table_from_filters()
            if not self._table_filters:
                dialog.accept()
                return
            render_table()

        select_all_button.clicked.connect(lambda: set_all(Qt.CheckState.Checked))
        unselect_all_button.clicked.connect(lambda: set_all(Qt.CheckState.Unchecked))
        delete_selected_button.clicked.connect(delete_selected)

        buttons_row.addWidget(select_all_button)
        buttons_row.addWidget(unselect_all_button)
        buttons_row.addStretch(1)
        buttons_row.addWidget(delete_selected_button)
        layout.addLayout(buttons_row)

        self._active_table_filters_dialog = dialog
        dialog.show()

    def _on_plot_type_changed(self) -> None:
        plot_type = self._plot_type_combo.currentText().strip().lower()
        is_scatter = plot_type == "scatterplot"
        is_boxplot = plot_type == "boxplot"
        self._scatter_controls.setVisible(plot_type == "scatterplot")
        self._hist_controls.setVisible(plot_type == "histogram")
        self._box_controls.setVisible(plot_type == "boxplot")
        self._pairplot_controls.setVisible(plot_type == "pairplot")
        self._hexbin_controls.setVisible(plot_type == "hexbin")
        self._add_contours_button.setVisible(is_scatter)
        self._add_fit_button.setVisible(is_scatter)
        self._add_contours_button.setEnabled(is_scatter and self._last_scatter_xy is not None)
        self._add_fit_button.setEnabled(is_scatter and self._last_scatter_xy is not None)
        self._box_stats_section.setVisible(is_boxplot and bool(self._box_stats_text.toPlainText().strip()))
        self._update_scatter_fit_controls_visibility()

    def _generate_plot(self) -> None:
        if self._dataframe is None:
            QMessageBox.warning(self, "No data", "Load a data table first.")
            return

        plot_type = self._plot_type_combo.currentText().strip().lower()
        try:
            if plot_type == "scatterplot":
                self._plot_scatter()
            elif plot_type == "histogram":
                self._plot_histogram()
            elif plot_type == "boxplot":
                self._plot_boxplot()
            elif plot_type == "pairplot":
                self._plot_pairplot()
            elif plot_type == "hexbin":
                self._plot_hexbin()
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        except Exception as exc:
            QMessageBox.critical(self, "Plot failed", str(exc))
            return

        self._export_plot_button.setEnabled(True)
        self._export_data_button.setEnabled(self._plot_data_subset is not None and not self._plot_data_subset.empty)
        plot_type = self._plot_type_combo.currentText().strip().lower()
        has_scatter_data = plot_type == "scatterplot" and self._last_scatter_xy is not None
        self._add_contours_button.setEnabled(has_scatter_data)
        self._add_fit_button.setEnabled(has_scatter_data)
        self._update_scatter_fit_controls_visibility()
        if plot_type != "pairplot":
            self._canvas.draw_idle()

    def _plot_scatter(self) -> None:
        frame = self._get_selected_plot_dataset()
        if frame is None:
            return

        x_column = self._scatter_x_combo.currentText()
        y_column = self._scatter_y_combo.currentText()
        hue_column = self._scatter_hue_combo.currentText()

        if not x_column or not y_column:
            raise ValueError("Select both x and y columns for a scatter plot.")

        frame = frame.copy()
        plot_frame = frame[[x_column, y_column]].copy()
        plot_frame[x_column] = pd.to_numeric(plot_frame[x_column], errors="coerce")
        plot_frame[y_column] = pd.to_numeric(plot_frame[y_column], errors="coerce")

        if hue_column and hue_column != "(None)":
            plot_frame[hue_column] = frame[hue_column].astype("string").fillna("NA")

        plot_frame = plot_frame.dropna(subset=[x_column, y_column])
        if plot_frame.empty:
            raise ValueError("Scatterplot requires numeric x and y values.")

        used_columns = [x_column, y_column]
        if hue_column and hue_column != "(None)":
            used_columns.append(hue_column)
        self._plot_data_subset = plot_frame[used_columns].copy()

        self._last_scatter_xy = (
            plot_frame[x_column].to_numpy(dtype=float),
            plot_frame[y_column].to_numpy(dtype=float),
        )
        self._last_scatter_columns = (x_column, y_column)
        self._reset_axis(title=f"Scatterplot: {y_column} vs {x_column}", x_label=x_column, y_label=y_column)

        if hue_column and hue_column != "(None)":
            categories = plot_frame[hue_column].astype("string").fillna("NA")
            unique_categories = categories.unique().tolist()
            colors = self._categorical_colors(len(unique_categories))

            for color, category in zip(colors, unique_categories, strict=False):
                mask = categories == category
                self._axis.scatter(
                    plot_frame.loc[mask, x_column],
                    plot_frame.loc[mask, y_column],
                    s=42,
                    alpha=0.85,
                    edgecolors="#0f172a",
                    linewidths=0.5,
                    color=color,
                    label=str(category),
                )
            self._axis.legend(loc="best", frameon=True, fontsize=9)
        else:
            self._axis.scatter(
                plot_frame[x_column],
                plot_frame[y_column],
                s=44,
                alpha=0.85,
                edgecolors="#0f172a",
                linewidths=0.5,
                color="#1f77b4",
            )

        self._box_stats_text.clear()
        self._box_stats_section.setVisible(False)
        self._log_message(f"Generated scatter plot with x={x_column}, y={y_column}, hue={hue_column}")

    def _plot_histogram(self) -> None:
        frame = self._get_selected_plot_dataset()
        if frame is None:
            return

        value_column = self._hist_value_combo.currentText()
        if not value_column:
            raise ValueError("Select a value column for histogram.")

        values = pd.to_numeric(frame[value_column], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError("Selected histogram column does not contain numeric values.")

        hist_subset = pd.DataFrame({value_column: pd.to_numeric(frame[value_column], errors="coerce")})
        hist_subset = hist_subset.dropna(subset=[value_column])
        self._plot_data_subset = hist_subset

        self._reset_axis(title=f"Histogram: {value_column}", x_label=value_column, y_label="Count")
        bins = min(max(int(np.sqrt(values.size)), 8), 50)
        self._axis.hist(values, bins=bins, color="#2563eb", edgecolor="#0f172a", linewidth=0.8, alpha=0.88)

        self._last_scatter_xy = None
        self._last_scatter_columns = None
        self._box_stats_text.clear()
        self._box_stats_section.setVisible(False)
        self._add_fit_button.setEnabled(False)
        self._add_contours_button.setEnabled(False)
        self._log_message(f"Generated histogram for {value_column}")

    def _plot_boxplot(self) -> None:
        frame = self._get_selected_plot_dataset()
        if frame is None:
            return

        category_column = self._box_hue_combo.currentText()
        value_column = self._box_value_combo.currentText()

        if not category_column or not value_column:
            raise ValueError("Select both category and value columns for boxplot.")

        frame = frame[[category_column, value_column]].copy()
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frame = frame.dropna(subset=[value_column])
        frame[category_column] = frame[category_column].astype("string").fillna("NA")
        if frame.empty:
            raise ValueError("Boxplot requires at least one numeric value.")

        self._plot_data_subset = frame[[category_column, value_column]].copy()

        groups = [group[value_column].to_numpy(dtype=float) for _, group in frame.groupby(category_column)]
        labels = [str(name) for name, _ in frame.groupby(category_column)]
        colors = self._categorical_colors(len(groups))

        self._reset_axis(title=f"Boxplot: {value_column} by {category_column}", x_label=category_column, y_label=value_column)
        boxplot = self._axis.boxplot(
            groups,
            labels=labels,
            patch_artist=True,
            medianprops={"color": "#111827", "linewidth": 1.6},
            whiskerprops={"color": "#334155", "linewidth": 1.1},
            capprops={"color": "#334155", "linewidth": 1.1},
            boxprops={"edgecolor": "#334155", "linewidth": 1.1},
        )
        for patch, color in zip(boxplot["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)

        self._axis.tick_params(axis="x", rotation=25)
        self._last_scatter_xy = None
        self._last_scatter_columns = None
        self._box_stats_text.setPlainText(self._boxplot_statistics_text(frame, category_column, value_column))
        self._box_stats_section.setVisible(True)
        self._add_fit_button.setEnabled(False)
        self._add_contours_button.setEnabled(False)
        self._log_message(f"Generated boxplot for value={value_column} grouped by {category_column}")

    def _on_fit_model_changed(self) -> None:
        """Update fit-parameter controls for the selected scatter fit model."""
        fit_model = self._fit_model_combo.currentText().strip().lower()
        is_poly = fit_model == "polynomial"
        is_custom = fit_model == "custom"
        should_show_fit_controls = self._should_show_fit_model_controls()

        self._fit_model_label.setVisible(should_show_fit_controls)
        self._fit_model_combo.setVisible(should_show_fit_controls)

        self._fit_poly_order_label.setVisible(should_show_fit_controls and is_poly)
        self._fit_poly_order_spin.setVisible(should_show_fit_controls and is_poly)
        self._fit_custom_expression_label.setVisible(should_show_fit_controls and is_custom)
        self._fit_custom_expression_edit.setVisible(should_show_fit_controls and is_custom)
        self._fit_custom_params_label.setVisible(should_show_fit_controls and is_custom)
        self._fit_custom_params_edit.setVisible(should_show_fit_controls and is_custom)

    def _should_show_fit_model_controls(self) -> bool:
        """Return true only after a scatter plot has been generated with usable data."""
        plot_type = self._plot_type_combo.currentText().strip().lower()
        return plot_type == "scatterplot" and self._last_scatter_xy is not None

    def _update_scatter_fit_controls_visibility(self) -> None:
        """Refresh visibility for fit model and model-specific parameter inputs."""
        self._on_fit_model_changed()

    def _add_scatter_fit(self) -> None:
        """Fit the currently shown scatter data with least-squares and overlay the fit line."""
        if self._last_scatter_xy is None:
            QMessageBox.warning(self, "No scatter data", "Generate a scatterplot first.")
            return

        x_values, y_values = self._last_scatter_xy
        if x_values.size < 3 or y_values.size < 3:
            QMessageBox.warning(self, "Insufficient data", "At least 3 points are needed to fit a model.")
            return

        fit_model = self._fit_model_combo.currentText().strip().lower()
        try:
            if fit_model == "polynomial":
                order = int(self._fit_poly_order_spin.value())
                fit_result = self._fit_service.fit_polynomial(x_values, y_values, order=order)
            elif fit_model == "power":
                fit_result = self._fit_service.fit_power_law(x_values, y_values)
            elif fit_model == "custom":
                expression = self._fit_custom_expression_edit.text().strip()
                parameter_text = self._fit_custom_params_edit.text().strip()
                parameter_names = [name.strip() for name in parameter_text.split(",") if name.strip()]
                fit_result = self._fit_service.fit_custom_latex(
                    x_values,
                    y_values,
                    expression,
                    parameter_names=parameter_names or None,
                )
            else:
                raise ValueError(f"Unsupported fit model: {fit_model}")
        except Exception as exc:
            QMessageBox.critical(self, "Fit failed", str(exc))
            return

        self._axis.plot(
            fit_result.fitted_x,
            fit_result.fitted_y,
            color="#dc2626",
            linewidth=2.2,
            label=f"Least-squares fit ({fit_result.model_name})",
            zorder=5,
        )
        self._axis.legend(loc="best", frameon=True, fontsize=9)
        self._canvas.draw()

        self._show_fit_statistics_window(fit_result)
        self._log_message(f"Added least-squares fit using {fit_result.model_name}")

    def _show_fit_statistics_window(self, fit_result) -> None:
        """Open a fit-statistics popup containing parameters and covariance entries."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Fit Statistics")
        dialog.resize(760, 560)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        stats_text = QPlainTextEdit(dialog)
        stats_text.setReadOnly(True)
        stats_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        stats_text.setPlainText(self._build_fit_statistics_text(fit_result))
        layout.addWidget(stats_text, 1)

        self._fit_stats_dialog = dialog
        dialog.show()

    def _build_fit_statistics_text(self, fit_result) -> str:
        """Build a readable statistics block for the latest least-squares fit."""
        lines = [
            "Least-Squares Fit Statistics",
            "",
            f"Model: {fit_result.model_name}",
            f"Equation: {fit_result.display_equation}",
            f"Residual std: {fit_result.residual_std}",
            f"Degrees of freedom: {fit_result.dof}",
            "",
            "Optimal Parameters:",
        ]

        for name, value in zip(fit_result.parameter_names, fit_result.parameter_values, strict=False):
            lines.append(f"  {name} = {float(value):.10g}")

        covariance = np.asarray(fit_result.covariance, dtype=float) if fit_result.covariance is not None else np.asarray([])
        lines.append("")
        if covariance.size == 0:
            lines.append("Covariance matrix: unavailable")
            return "\n".join(lines)

        lines.append(f"Covariance matrix shape: {tuple(covariance.shape)}")
        lines.append("Covariance matrix entries:")
        for row_index in range(covariance.shape[0]):
            entries = ", ".join(f"{float(entry):.10g}" for entry in covariance[row_index])
            lines.append(f"  row {row_index}: [{entries}]")

        return "\n".join(lines)

    def _boxplot_statistics_text(self, frame: pd.DataFrame, category_column: str, value_column: str) -> str:
        """Build grouped descriptive statistics for the current boxplot data."""
        lines = [
            f"Statistics for {value_column} grouped by {category_column}",
            "",
        ]

        grouped = frame.groupby(category_column, dropna=False)
        for group_name, group in grouped:
            values = group[value_column].to_numpy(dtype=float)
            if values.size == 0:
                continue
            quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
            lines.extend(
                [
                    f"Group: {group_name}",
                    f"  n = {values.size}",
                    f"  mean = {float(np.mean(values)):.6g}",
                    f"  median = {float(np.median(values)):.6g}",
                    f"  std = {float(np.std(values, ddof=1)):.6g}" if values.size > 1 else "  std = NA",
                    f"  min = {float(np.min(values)):.6g}",
                    f"  q05 = {float(quantiles[0]):.6g}",
                    f"  q25 = {float(quantiles[1]):.6g}",
                    f"  q50 = {float(quantiles[2]):.6g}",
                    f"  q75 = {float(quantiles[3]):.6g}",
                    f"  q95 = {float(quantiles[4]):.6g}",
                    f"  max = {float(np.max(values)):.6g}",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _add_scatter_contours(self) -> None:
        """Fit and overlay a 2D Gaussian contour (1σ and 2σ ellipses) on scatter data."""
        if self._last_scatter_xy is None:
            QMessageBox.warning(self, "No scatter data", "Generate a scatterplot first.")
            return

        x_values, y_values = self._last_scatter_xy
        if x_values.size < 3 or y_values.size < 3:
            QMessageBox.warning(self, "Insufficient data", "At least 3 points are needed for contour fitting.")
            return

        covariance = np.cov(np.vstack([x_values, y_values]), bias=False)
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            QMessageBox.warning(self, "Contour failed", "Could not compute a valid covariance matrix.")
            return

        determinant = float(np.linalg.det(covariance))
        if determinant <= 1e-18:
            QMessageBox.warning(self, "Contour failed", "Covariance matrix is singular; contours are unavailable.")
            return

        mean_vector = np.array([float(np.mean(x_values)), float(np.mean(y_values))], dtype=float)
        inverse_covariance = np.linalg.inv(covariance)

        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        x_pad = max((x_max - x_min) * 0.1, 1e-9)
        y_pad = max((y_max - y_min) * 0.1, 1e-9)

        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 140)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 140)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)

        stacked = np.stack([grid_x - mean_vector[0], grid_y - mean_vector[1]], axis=-1)
        mahal_sq = np.einsum("...i,ij,...j->...", stacked, inverse_covariance, stacked)
        normalizer = 1.0 / (2.0 * np.pi * np.sqrt(determinant))
        density = normalizer * np.exp(-0.5 * mahal_sq)

        # For a 2D Gaussian, constant-Mahalanobis-distance ellipses correspond
        # to density levels exp(-k^2/2), where k is sigma-equivalent radius.
        sigma1_level = normalizer * np.exp(-0.5 * (1.0 ** 2))
        sigma2_level = normalizer * np.exp(-0.5 * (2.0 ** 2))

        contour_set = self._axis.contour(
            grid_x,
            grid_y,
            density,
            levels=sorted([sigma2_level, sigma1_level]),
            colors=["#f97316", "#dc2626"],
            linewidths=[1.6, 1.8],
            linestyles=["--", "-"],
        )
        # Some Matplotlib versions do not expose QuadContourSet.collections,
        # so legend entries are attached to explicit ellipse artists below.

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle_deg = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))

        for sigma, color in ((1.0, "#dc2626"), (2.0, "#f97316")):
            width = 2.0 * sigma * np.sqrt(max(float(eigenvalues[0]), 0.0))
            height = 2.0 * sigma * np.sqrt(max(float(eigenvalues[1]), 0.0))
            ellipse = Ellipse(
                xy=(mean_vector[0], mean_vector[1]),
                width=width,
                height=height,
                angle=angle_deg,
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.85,
                linestyle=":" if sigma == 2.0 else "-.",
                label=f"{int(sigma)}σ Gaussian contour",
            )
            self._axis.add_patch(ellipse)

        self._axis.scatter(
            [mean_vector[0]],
            [mean_vector[1]],
            marker="x",
            s=90,
            linewidths=1.6,
            color="#111827",
            label="Gaussian mean",
            zorder=6,
        )
        self._axis.legend(loc="best", frameon=True, fontsize=9)
        self._canvas.draw_idle()

        self._show_gaussian_marginals_view(
            x_values=x_values,
            y_values=y_values,
            mean_vector=mean_vector,
            covariance=covariance,
        )

        x_name, y_name = self._last_scatter_columns or ("x", "y")
        self._log_message(f"Added 2D Gaussian 1σ/2σ contours for scatter ({x_name}, {y_name})")

    def _show_gaussian_marginals_view(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        mean_vector: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """Show a 2x2 layout with scatter+ellipses and the X/Y marginal histograms with Gaussian fits."""
        x_name, y_name = self._last_scatter_columns or ("X", "Y")
        mu_x = float(mean_vector[0])
        mu_y = float(mean_vector[1])
        sigma_x = float(np.sqrt(max(float(covariance[0, 0]), 0.0)))
        sigma_y = float(np.sqrt(max(float(covariance[1, 1]), 0.0)))

        def normal_pdf(grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
            if sigma <= 0.0 or not np.isfinite(sigma):
                return np.zeros_like(grid, dtype=float)
            z_values = (grid - mu) / sigma
            return np.exp(-0.5 * z_values * z_values) / (sigma * np.sqrt(2.0 * np.pi))

        def covariance_ellipse(cov_matrix: np.ndarray, n_sigma: float) -> tuple[float, float, float]:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            width = 2.0 * n_sigma * np.sqrt(max(float(eigenvalues[0]), 0.0))
            height = 2.0 * n_sigma * np.sqrt(max(float(eigenvalues[1]), 0.0))
            angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
            return width, height, angle

        figure = Figure(figsize=(9.0, 7.2), dpi=135, facecolor="#ffffff")
        grid_spec = GridSpec(
            2,
            2,
            figure=figure,
            width_ratios=[4.0, 1.25],
            height_ratios=[1.25, 4.0],
            hspace=0.05,
            wspace=0.05,
        )

        ax_scatter = figure.add_subplot(grid_spec[1, 0])
        ax_hist_x = figure.add_subplot(grid_spec[0, 0], sharex=ax_scatter)
        ax_hist_y = figure.add_subplot(grid_spec[1, 1], sharey=ax_scatter)
        ax_corner = figure.add_subplot(grid_spec[0, 1])
        ax_corner.axis("off")

        for axis in (ax_scatter, ax_hist_x, ax_hist_y):
            axis.set_facecolor("#ffffff")
            axis.grid(True, color="#e6ecf2", linewidth=0.7, alpha=0.9)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        ax_scatter.scatter(
            x_values,
            y_values,
            s=18,
            alpha=0.35,
            edgecolors="none",
            color="#2563eb",
        )

        for n_sigma, linestyle, color in ((1.0, "-", "#dc2626"), (2.0, "--", "#f97316")):
            width, height, angle = covariance_ellipse(covariance, n_sigma)
            ellipse = Ellipse(
                xy=(mu_x, mu_y),
                width=width,
                height=height,
                angle=angle,
                fill=False,
                linestyle=linestyle,
                linewidth=2.0,
                edgecolor=color,
                alpha=0.95,
            )
            ax_scatter.add_patch(ellipse)

        ax_scatter.scatter(
            [mu_x],
            [mu_y],
            s=86,
            marker="+",
            linewidths=2.0,
            color="#111827",
            zorder=6,
        )

        bins_x = min(max(int(np.sqrt(max(x_values.size, 1))), 12), 45)
        bins_y = min(max(int(np.sqrt(max(y_values.size, 1))), 12), 45)

        ax_hist_x.hist(
            x_values,
            bins=bins_x,
            density=True,
            alpha=0.35,
            edgecolor="none",
            color="#2563eb",
        )
        x_grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 500)
        ax_hist_x.plot(x_grid, normal_pdf(x_grid, mu_x, sigma_x), linewidth=2.0, color="#dc2626")
        ax_hist_x.axvline(mu_x, linestyle="--", linewidth=1.2, color="#111827")

        ax_hist_y.hist(
            y_values,
            bins=bins_y,
            density=True,
            orientation="horizontal",
            alpha=0.35,
            edgecolor="none",
            color="#2563eb",
        )
        y_grid = np.linspace(float(np.min(y_values)), float(np.max(y_values)), 500)
        ax_hist_y.plot(normal_pdf(y_grid, mu_y, sigma_y), y_grid, linewidth=2.0, color="#dc2626")
        ax_hist_y.axhline(mu_y, linestyle="--", linewidth=1.2, color="#111827")

        ax_hist_x.tick_params(axis="x", labelbottom=False)
        ax_hist_y.tick_params(axis="y", labelleft=False)

        ax_scatter.set_xlabel(x_name)
        ax_scatter.set_ylabel(y_name)
        ax_hist_x.set_ylabel("Density")
        ax_hist_y.set_xlabel("Density")

        dialog = QDialog(self)
        dialog.setWindowTitle("Gaussian Contours and Marginals")
        dialog.resize(1100, 840)
        dialog.setModal(False)

        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(10, 10, 10, 10)
        dialog_layout.setSpacing(8)

        canvas = FigureCanvasQTAgg(figure)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        dialog_layout.addWidget(canvas, 1)

        self._gaussian_marginals_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        canvas.draw()

    def _plot_pairplot(self) -> None:
        """Render a scrollable pairplot with histogram diagonal, scatter upper, and correlation lower."""
        frame = self._get_selected_plot_dataset()
        if frame is None:
            raise ValueError("Load a data table first.")

        selected_columns = [item.text() for item in self._pairplot_columns_list.selectedItems()]
        if len(selected_columns) < 2:
            raise ValueError("Select at least two columns for pairplot.")

        numeric_frame = frame[selected_columns].apply(pd.to_numeric, errors="coerce")
        usable_columns = [column for column in selected_columns if numeric_frame[column].notna().sum() > 1]
        if len(usable_columns) < 2:
            raise ValueError("At least two selected columns must contain numeric values.")

        pair_frame = numeric_frame[usable_columns]
        self._plot_data_subset = pair_frame.dropna(how="all").copy()
        n_columns = len(usable_columns)

        figure_size = max(8.0, 2.7 * n_columns)
        pair_figure = Figure(figsize=(figure_size, figure_size), dpi=145, facecolor="#ffffff")
        axes = pair_figure.subplots(n_columns, n_columns)
        if n_columns == 2:
            # Matplotlib returns a 2x2 ndarray for n=2, but this keeps typing explicit.
            axes = np.asarray(axes)

        cmap = colormaps.get_cmap("coolwarm")

        for row in range(n_columns):
            for col in range(n_columns):
                axis = axes[row, col]
                x_name = usable_columns[col]
                y_name = usable_columns[row]

                x_series = pair_frame[x_name]
                y_series = pair_frame[y_name]

                axis.set_facecolor("#ffffff")
                axis.grid(True, color="#e6ecf2", linewidth=0.65, alpha=0.95)
                axis.tick_params(axis="both", labelsize=7)

                if row == col:
                    values = x_series.dropna().to_numpy(dtype=float)
                    bins = min(max(int(np.sqrt(max(values.size, 1))), 8), 35)
                    axis.hist(values, bins=bins, color="#3b82f6", edgecolor="#0f172a", linewidth=0.6, alpha=0.88)
                elif row < col:
                    mask = x_series.notna() & y_series.notna()
                    axis.scatter(
                        x_series[mask].to_numpy(dtype=float),
                        y_series[mask].to_numpy(dtype=float),
                        s=10,
                        alpha=0.72,
                        color="#2563eb",
                        edgecolors="none",
                    )
                else:
                    mask = x_series.notna() & y_series.notna()
                    x_values = x_series[mask].to_numpy(dtype=float)
                    y_values = y_series[mask].to_numpy(dtype=float)
                    if x_values.size > 1 and y_values.size > 1:
                        corr = float(np.corrcoef(x_values, y_values)[0, 1])
                    else:
                        corr = float("nan")

                    if np.isfinite(corr):
                        color = cmap((corr + 1.0) / 2.0)
                        axis.set_facecolor(color)
                        text_value = f"r = {corr:.3f}"
                    else:
                        axis.set_facecolor("#f3f4f6")
                        text_value = "r = NA"

                    axis.text(
                        0.5,
                        0.5,
                        text_value,
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="#111827",
                    )

                if row < n_columns - 1:
                    axis.set_xticklabels([])
                else:
                    axis.set_xlabel(x_name, fontsize=8, fontweight="semibold")

                if col > 0:
                    axis.set_yticklabels([])
                else:
                    axis.set_ylabel(y_name, fontsize=8, fontweight="semibold")

        pair_figure.subplots_adjust(left=0.07, right=0.985, top=0.985, bottom=0.07, wspace=0.12, hspace=0.12)

        dialog = QDialog(self)
        dialog.setWindowTitle("Pairplot")
        dialog.resize(1260, 860)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)

        canvas = FigureCanvasQTAgg(pair_figure)
        pixel_size = int(max(1200, 390 * n_columns))
        canvas.setMinimumSize(pixel_size, pixel_size)
        canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        host_layout.addWidget(canvas)

        scroll.setWidget(host)
        layout.addWidget(scroll, 1)

        self._pairplot_dialog = dialog
        dialog.show()
        canvas.draw_idle()

        self._log_message(f"Generated pairplot for columns: {', '.join(usable_columns)}")

    def _plot_hexbin(self) -> None:
        """Render a hexbin density plot (2D histogram-like visualization)."""
        frame = self._get_selected_plot_dataset()
        if frame is None:
            return

        x_column = self._hexbin_x_combo.currentText()
        y_column = self._hexbin_y_combo.currentText()
        if not x_column or not y_column:
            raise ValueError("Select both x and y columns for a hexbin plot.")

        plot_frame = frame[[x_column, y_column]].copy()
        plot_frame[x_column] = pd.to_numeric(plot_frame[x_column], errors="coerce")
        plot_frame[y_column] = pd.to_numeric(plot_frame[y_column], errors="coerce")
        plot_frame = plot_frame.dropna(subset=[x_column, y_column])

        if plot_frame.empty:
            raise ValueError("Hexbin plot requires numeric x and y values.")

        self._plot_data_subset = plot_frame[[x_column, y_column]].copy()

        x_values = plot_frame[x_column].to_numpy(dtype=float)
        y_values = plot_frame[y_column].to_numpy(dtype=float)
        self._reset_axis(title=f"Hexbin: {y_column} vs {x_column}", x_label=x_column, y_label=y_column)

        hexbin = self._axis.hexbin(
            x_values,
            y_values,
            gridsize=35,
            cmap="viridis",
            mincnt=1,
            linewidths=0.2,
        )
        colorbar = self._figure.colorbar(hexbin, ax=self._axis, pad=0.012)
        colorbar.set_label("Count", fontsize=9)

        self._last_scatter_xy = None
        self._last_scatter_columns = None
        self._box_stats_text.clear()
        self._box_stats_section.setVisible(False)
        self._add_fit_button.setEnabled(False)
        self._add_contours_button.setEnabled(False)
        self._log_message(f"Generated hexbin plot with x={x_column}, y={y_column}")

    def _apply_publication_style(self) -> None:
        self._figure.set_facecolor("#ffffff")
        self._axis.set_facecolor("#ffffff")
        self._axis.grid(True, axis="both", color="#dbe2ea", linewidth=0.8, alpha=0.9)
        self._axis.spines["top"].set_visible(False)
        self._axis.spines["right"].set_visible(False)
        self._axis.spines["left"].set_linewidth(1.1)
        self._axis.spines["bottom"].set_linewidth(1.1)
        self._axis.tick_params(axis="both", labelsize=10)

    def _reset_axis(self, title: str, x_label: str, y_label: str) -> None:
        self._axis.clear()
        self._apply_publication_style()
        self._axis.set_title(title, fontsize=13, fontweight="bold", pad=10)
        self._axis.set_xlabel(x_label, fontsize=11, fontweight="semibold")
        self._axis.set_ylabel(y_label, fontsize=11, fontweight="semibold")

    def _categorical_colors(self, count: int) -> list[str]:
        if count <= 0:
            return []
        cmap = colormaps.get_cmap("tab20")
        return [cmap(index / max(count - 1, 1)) for index in range(count)]

    def _export_plot(self) -> None:
        default_name = "explore_plot.png"
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export plot image",
            default_name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not destination:
            return

        try:
            self._figure.savefig(destination, dpi=350, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        self._log_message(f"Exported plot image to {destination}")
        QMessageBox.information(self, "Export complete", f"Saved plot to:\n{destination}")

    def _clear_plot_state(self) -> None:
        """Reset plotting space and plot-configuration selections to defaults."""
        self._axis.clear()
        self._apply_publication_style()
        self._axis.set_title("Plot preview", fontsize=13, fontweight="bold", pad=10)
        self._axis.set_xlabel("")
        self._axis.set_ylabel("")
        self._canvas.draw_idle()

        self._last_scatter_xy = None
        self._last_scatter_columns = None
        self._box_stats_text.clear()
        self._box_stats_section.setVisible(False)
        self._add_contours_button.setEnabled(False)
        self._add_fit_button.setEnabled(False)
        self._export_plot_button.setEnabled(False)
        self._export_data_button.setEnabled(False)
        self._plot_data_subset = None

        self._plot_type_combo.setCurrentText("Scatterplot")

        if self._scatter_x_combo.count() > 0:
            self._scatter_x_combo.setCurrentIndex(0)
        if self._scatter_y_combo.count() > 0:
            self._scatter_y_combo.setCurrentIndex(0)
        if self._scatter_hue_combo.count() > 0:
            self._scatter_hue_combo.setCurrentIndex(0)

        if self._hist_value_combo.count() > 0:
            self._hist_value_combo.setCurrentIndex(0)

        if self._box_hue_combo.count() > 0:
            self._box_hue_combo.setCurrentIndex(0)
        if self._box_value_combo.count() > 0:
            self._box_value_combo.setCurrentIndex(0)

        if self._hexbin_x_combo.count() > 0:
            self._hexbin_x_combo.setCurrentIndex(0)
        if self._hexbin_y_combo.count() > 0:
            self._hexbin_y_combo.setCurrentIndex(0)

        self._pairplot_columns_list.clearSelection()

        self._fit_model_combo.setCurrentText("Polynomial")
        self._fit_poly_order_spin.setValue(2)
        self._fit_custom_expression_edit.setText("a*x + b")
        self._fit_custom_params_edit.setText("a,b")
        self._update_scatter_fit_controls_visibility()

        self._log_message("Plot area and selections reset to defaults.")

    def _export_data(self) -> None:
        """Export the current plot's used data rows to CSV or Excel."""
        if self._plot_data_subset is None or self._plot_data_subset.empty:
            QMessageBox.warning(self, "No plot data", "Generate a plot before exporting data.")
            return

        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export plot data",
            "plot_data_subset.csv",
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not destination:
            return

        try:
            output_path = Path(destination)
            if output_path.suffix.lower() == ".xlsx":
                self._plot_data_subset.to_excel(output_path, index=False)
            else:
                if output_path.suffix == "":
                    output_path = output_path.with_suffix(".csv")
                self._plot_data_subset.to_csv(output_path, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        self._log_message(f"Exported plot data subset to {output_path}")
        QMessageBox.information(self, "Export complete", f"Saved data to:\n{output_path}")

    def _export_new_table(self) -> None:
        """Export the current derived/new table to CSV or Excel."""
        if self._derived_table is None:
            QMessageBox.warning(self, "No new table", "Create a new table before exporting.")
            return

        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export new table",
            "new_table.csv",
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not destination:
            return

        try:
            output_path = Path(destination)
            if output_path.suffix.lower() == ".xlsx":
                self._derived_table.to_excel(output_path, index=False)
            else:
                if output_path.suffix == "":
                    output_path = output_path.with_suffix(".csv")
                self._derived_table.to_csv(output_path, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        self._log_message(f"Exported new table to {output_path}")
        QMessageBox.information(self, "Export complete", f"Saved new table to:\n{output_path}")
