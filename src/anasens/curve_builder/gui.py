"""PySide6 GUI for Plotter Fitter."""

from __future__ import annotations

import io
from itertools import combinations_with_replacement
from pathlib import Path
import pickle
import sys

import pandas as pd
from matplotlib.figure import Figure
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .data_loader import DataLoader
from .band_service import BandComputationResult, BandService
from .fitters import FitService
from .models import FittingResult, LoadedDataset
from .plotter import PlotRenderer
from .standard_curve import build_standard_curve_model


class SynthwaveBackdrop(QWidget):
    """Paint a retro synthwave background behind the application layout.

    The backdrop is purely decorative and exists to give the Qt application a
    strong visual identity without mixing presentation concerns into plot logic.
    """

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming convention.
        """Draw the synthwave horizon, glow, and perspective grid.

        Args:
            event: Qt paint event supplied by the framework.
        """
        del event

        # Core canvas geometry is derived once so every decorative element uses
        # the same horizon and perspective reference points.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        width = self.width()
        height = self.height()
        horizon_y = int(height * 0.46)
        vanishing_x = width / 2

        # The base sky uses layered gradients to transition from deep indigo to
        # warmer magenta tones near the horizon line.
        sky_gradient = QLinearGradient(0, 0, 0, height)
        sky_gradient.setColorAt(0.0, QColor("#050816"))
        sky_gradient.setColorAt(0.38, QColor("#1a1038"))
        sky_gradient.setColorAt(0.7, QColor("#2a124f"))
        sky_gradient.setColorAt(1.0, QColor("#120321"))
        painter.fillRect(self.rect(), sky_gradient)

        # A radial glow and stylized sun establish the synthwave focal point at
        # the horizon without interfering with foreground controls.
        glow_gradient = QRadialGradient(QPointF(vanishing_x, horizon_y - 42), width * 0.32)
        glow_gradient.setColorAt(0.0, QColor(255, 111, 214, 165))
        glow_gradient.setColorAt(0.55, QColor(123, 92, 255, 90))
        glow_gradient.setColorAt(1.0, QColor(5, 8, 22, 0))
        painter.fillRect(self.rect(), glow_gradient)

        sun_gradient = QRadialGradient(QPointF(vanishing_x, horizon_y - 18), width * 0.1)
        sun_gradient.setColorAt(0.0, QColor("#ffe27a"))
        sun_gradient.setColorAt(0.5, QColor("#ff76d8"))
        sun_gradient.setColorAt(1.0, QColor(255, 118, 216, 0))
        painter.setBrush(sun_gradient)
        painter.setPen(QPen(QColor(255, 210, 245, 150), 1.5))
        painter.drawEllipse(QPointF(vanishing_x, horizon_y - 18), width * 0.1, width * 0.1)

        # A faint star field adds depth while keeping the background stable and
        # deterministic instead of using random per-frame decoration.
        painter.setPen(QPen(QColor(210, 236, 255, 110), 1.2))
        star_points = [
            QPointF(width * 0.12, height * 0.12),
            QPointF(width * 0.22, height * 0.2),
            QPointF(width * 0.82, height * 0.16),
            QPointF(width * 0.68, height * 0.1),
            QPointF(width * 0.9, height * 0.24),
            QPointF(width * 0.38, height * 0.08),
            QPointF(width * 0.56, height * 0.19),
        ]
        for point in star_points:
            painter.drawPoint(point)

        # The ground grid uses a shared horizon line and perspective rays to
        # evoke the classic synthwave landscape beneath the application cards.
        horizon_pen = QPen(QColor("#48e3ff"), 2.0)
        painter.setPen(horizon_pen)
        painter.drawLine(0, horizon_y, width, horizon_y)

        grid_pen = QPen(QColor(255, 90, 200, 90), 1.0)
        painter.setPen(grid_pen)
        for step in range(1, 10):
            y_position = horizon_y + int((height - horizon_y) * (step / 10) ** 1.7)
            painter.drawLine(0, y_position, width, y_position)

        for index in range(-8, 9):
            x_position = vanishing_x + index * width * 0.08
            painter.drawLine(QPointF(vanishing_x, horizon_y), QPointF(x_position, height))

        # A subtle vignette keeps the viewer's attention centered on the main
        # interface region rather than on the outer corners of the window.
        vignette_rect = QRectF(0, 0, width, height)
        vignette_gradient = QRadialGradient(vignette_rect.center(), width * 0.68)
        vignette_gradient.setColorAt(0.72, QColor(0, 0, 0, 0))
        vignette_gradient.setColorAt(1.0, QColor(0, 0, 0, 110))
        painter.fillRect(self.rect(), vignette_gradient)


class PlotterFitterGUI(QMainWindow):
    """Coordinate the desktop interface for plotting and curve fitting.

    The GUI delegates file loading, numerical fitting, and plot rendering to
    dedicated collaborators so the window class remains an orchestration layer.
    """

    def __init__(self) -> None:
        """Build the application window and its child widgets."""
        super().__init__()

        # Service objects are created once so UI callbacks can stay lightweight
        # and focused on moving data between the interface and domain logic.
        self.data_loader = DataLoader()
        self.fit_service = FitService()
        self.band_service = BandService()
        self.loaded_dataset: LoadedDataset | None = None
        self.current_plot_result: FittingResult | None = None
        self._current_bands: BandComputationResult | None = None
        self._active_filters: list[tuple[str, str]] = []
        self._last_fit_x = None
        self._last_fit_y = None
        self._last_fit_type: str | None = None
        self._last_custom_expression: str | None = None
        self._method_plot_entries: list[dict[str, object]] = []
        self._current_method_plot_index: int = 0

        # Core window metadata is established before widget construction so the
        # layout code can assume the main window shell already exists.
        self.setWindowTitle("Plotter Fitter")
        self.resize(1320, 840)
        self.setMinimumSize(1020, 660)

        # Default control values are initialized centrally so the UI state is
        # predictable across fresh launches and future resets.
        self.default_fit_type = "polynomial"
        self.default_polynomial_order = "2"
        self.default_custom_latex = r"a x^{2} + b x + c"
        self.default_loaded_file_text = "No file loaded"

        self._configure_style()
        self._build_layout()

    def run(self) -> None:
        """Start the Qt event loop."""
        application = QApplication.instance()
        owns_application = application is None
        if application is None:
            application = QApplication(sys.argv)

        self.show()
        if owns_application:
            application.exec()

    def _configure_style(self) -> None:
        """Configure Qt widget styling for the application."""
        # The stylesheet centralizes color, spacing, and typography so the Qt
        # interface carries a consistent synthwave look across all widgets.
        self.setStyleSheet(
            """
            QMainWindow {
                background: #050816;
            }
            QWidget {
                color: #f5e9ff;
                font-family: "Trebuchet MS";
                font-size: 6.5pt;
                selection-background-color: #ff5ac8;
                selection-color: #090212;
            }
            QFrame#card {
                background: rgba(11, 8, 27, 0.84);
                border: 1px solid rgba(72, 227, 255, 0.65);
                border-radius: 18px;
            }
            QFrame#accentCard {
                background: rgba(15, 10, 34, 0.92);
                border: 1px solid rgba(255, 90, 200, 0.72);
                border-radius: 18px;
            }
            QLabel#title {
                color: #ff5ac8;
                font-family: "Segoe UI Black";
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#subtitle {
                color: #8fe8ff;
                font-size: 7pt;
                font-weight: 600;
                letter-spacing: 0px;
            }
            QLabel#sectionTitle {
                color: #48e3ff;
                font-family: "Segoe UI Semibold";
                font-size: 9pt;
                font-weight: 700;
            }
            QLabel#fileValue {
                color: #ffd6f6;
                font-weight: 600;
            }
            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #ff5ac8,
                    stop: 1 #7c5cff
                );
                color: #fff7ff;
                border: 1px solid rgba(72, 227, 255, 0.55);
                border-radius: 8px;
                padding: 3px 7px;
                font-weight: 600;
                text-transform: uppercase;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #ff79d6,
                    stop: 1 #8b6dff
                );
            }
            QPushButton:pressed {
                background: #5a35cc;
            }
            QPushButton:disabled {
                background: #54456e;
                color: #d3c3ea;
                border: 1px solid rgba(211, 195, 234, 0.25);
            }
            QComboBox, QLineEdit, QTextEdit {
                background: rgba(9, 8, 30, 0.9);
                border: 1px solid rgba(255, 90, 200, 0.45);
                border-radius: 8px;
                padding: 2px 5px;
                color: #f8f1ff;
            }
            QCheckBox {
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 11px;
                height: 11px;
            }
            QComboBox:hover, QLineEdit:hover, QTextEdit:hover {
                border: 1px solid rgba(72, 227, 255, 0.7);
            }
            QComboBox:focus, QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #48e3ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: #140a2f;
                color: #f8f1ff;
                border: 1px solid rgba(72, 227, 255, 0.55);
                selection-background-color: #ff5ac8;
                selection-color: #090212;
            }
            QTextEdit {
                font-family: "Consolas";
                color: #dffcff;
                line-height: 1.35;
            }
            QTextEdit#statsReadout {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(10, 8, 29, 0.95),
                    stop: 1 rgba(27, 10, 56, 0.94)
                );
                border: 1px solid rgba(72, 227, 255, 0.7);
                border-radius: 10px;
                padding: 7px;
                font-size: 5.6pt;
                selection-background-color: #48e3ff;
                selection-color: #090212;
            }
            QMessageBox {
                background: #120321;
            }
            """
        )

    def _build_layout(self) -> None:
        """Create and arrange all interface widgets."""
        # The top-level layout splits the window into a control region and a
        # content region so the plot and statistics remain visible together.
        central_widget = SynthwaveBackdrop(self)
        self.setCentralWidget(central_widget)

        outer_layout = QGridLayout(central_widget)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setHorizontalSpacing(8)
        outer_layout.setVerticalSpacing(8)
        outer_layout.setColumnStretch(0, 4)
        outer_layout.setColumnStretch(1, 3)
        outer_layout.setRowStretch(0, 1)

        left_panel = QWidget(central_widget)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        plot_frame = self._create_card_frame()
        plot_frame.setObjectName("card")
        plot_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(7, 7, 7, 7)
        plot_layout.setSpacing(5)

        method_nav_widget = QWidget(plot_frame)
        method_nav_layout = QHBoxLayout(method_nav_widget)
        method_nav_layout.setContentsMargins(0, 0, 0, 0)
        method_nav_layout.setSpacing(5)

        self.prev_method_plot_button = QPushButton("Prev")
        self.prev_method_plot_button.setEnabled(False)
        self.prev_method_plot_button.clicked.connect(self._handle_prev_method_plot)
        self.prev_method_plot_button.hide()

        self.next_method_plot_button = QPushButton("Next")
        self.next_method_plot_button.setEnabled(False)
        self.next_method_plot_button.clicked.connect(self._handle_next_method_plot)
        self.next_method_plot_button.hide()

        self.method_plot_label = QLabel("Method view: none")
        self.method_plot_label.setObjectName("subtitle")
        self.method_plot_label.hide()

        method_nav_layout.addWidget(self.prev_method_plot_button)
        method_nav_layout.addWidget(self.next_method_plot_button)
        method_nav_layout.addWidget(self.method_plot_label, 1)
        plot_layout.addWidget(method_nav_widget)

        stats_frame = self._create_card_frame()
        stats_frame.setObjectName("accentCard")
        stats_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        control_frame = self._create_card_frame()
        control_frame.setMinimumWidth(420)

        right_panel = QWidget(central_widget)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        left_layout.addWidget(plot_frame, 3)
        left_layout.addWidget(stats_frame, 2)
        right_layout.addWidget(control_frame, 1)

        outer_layout.addWidget(left_panel, 0, 0)
        outer_layout.addWidget(right_panel, 0, 1)

        self._build_controls(control_frame)
        self._build_statistics_panel(stats_frame)
        self.plot_renderer = PlotRenderer(plot_frame)

    def _create_card_frame(self) -> QFrame:
        """Create a shared card-style frame used across the interface.

        Returns:
            A configured frame with a vertical layout.
        """
        card_frame = QFrame(self)
        card_frame.setObjectName("card")
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(9)

        # Drop-shadow effects help the cards feel like luminous panels floating
        # above the painted synthwave landscape.
        shadow_effect = QGraphicsDropShadowEffect(card_frame)
        shadow_effect.setBlurRadius(28)
        shadow_effect.setOffset(0, 10)
        shadow_effect.setColor(QColor(255, 90, 200, 70))
        card_frame.setGraphicsEffect(shadow_effect)
        return card_frame

    def _build_controls(self, parent: QFrame) -> None:
        """Create the file, axis, fit, and plot controls.

        Args:
            parent: Parent frame for the control content.
        """
        parent_layout = parent.layout()

        # The upper control strip groups file and axis selection above the plot
        # so the workflow reads naturally from loading to plotting to fitting.
        top_controls = QWidget(parent)
        top_layout = QVBoxLayout(top_controls)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        parent_layout.addWidget(top_controls)

        file_label = QLabel("Loaded file:")
        file_label.setObjectName("sectionTitle")
        self.loaded_file_label = QLabel(self.default_loaded_file_text)
        self.loaded_file_label.setObjectName("fileValue")
        self.load_button = QPushButton("Load data")
        self.load_button.setMinimumWidth(86)
        self.load_button.clicked.connect(self._handle_load_data)

        load_section = self._create_card_frame()
        load_section.setGraphicsEffect(None)
        load_section.setObjectName("accentCard")
        load_section_layout = load_section.layout()
        load_section_layout.setContentsMargins(10, 10, 10, 10)
        load_section_layout.setSpacing(6)

        load_section_title = QLabel("Load data")
        load_section_title.setObjectName("sectionTitle")
        load_section_layout.addWidget(load_section_title)

        load_grid = QGridLayout()
        load_grid.setHorizontalSpacing(6)
        load_grid.setVerticalSpacing(4)
        load_grid.setColumnStretch(1, 1)
        load_grid.addWidget(file_label, 0, 0)
        load_grid.addWidget(self.loaded_file_label, 0, 1)
        load_grid.addWidget(self.load_button, 0, 2)
        load_section_layout.addLayout(load_grid)

        x_axis_label = QLabel("X-axis")
        y_axis_label = QLabel("Y-axis")
        filter_label = QLabel("Filter")
        self.x_axis_combobox = QComboBox(parent)
        self.y_axis_combobox = QComboBox(parent)
        self.filter_column_combobox = QComboBox(parent)
        self.filter_value_entry = QLineEdit(parent)
        self.add_filter_button = QPushButton("Add")
        self.confirm_filter_button = QPushButton("Confirm")
        self.clear_filters_button = QPushButton("Clear")
        self.show_active_filters_button = QPushButton("Show active filters")
        self.x_axis_combobox.setEditable(False)
        self.y_axis_combobox.setEditable(False)
        self.filter_column_combobox.setEditable(False)
        self.x_axis_combobox.setEnabled(False)
        self.y_axis_combobox.setEnabled(False)
        self.filter_column_combobox.setEnabled(False)
        self.filter_value_entry.setEnabled(False)
        self.add_filter_button.setEnabled(False)
        self.confirm_filter_button.setEnabled(False)
        self.clear_filters_button.setEnabled(False)
        self.x_axis_combobox.addItem("Load data to choose a column")
        self.y_axis_combobox.addItem("Load data to choose a column")
        self.filter_column_combobox.addItem("Choose filter column")
        self.filter_value_entry.setPlaceholderText("Exact value to keep")
        self.add_filter_button.clicked.connect(self._handle_add_filter)
        self.confirm_filter_button.clicked.connect(self._handle_confirm_filter)
        self.clear_filters_button.clicked.connect(self._handle_clear_filters)
        self.show_active_filters_button.clicked.connect(self._handle_show_active_filters)
        self.plot_button = QPushButton("Confirm plot")
        self.plot_button.setEnabled(False)
        self.plot_button.clicked.connect(self._handle_plot_data)

        axes_section = self._create_card_frame()
        axes_section.setGraphicsEffect(None)
        axes_section.setObjectName("accentCard")
        axes_section_layout = axes_section.layout()
        axes_section_layout.setContentsMargins(10, 10, 10, 10)
        axes_section_layout.setSpacing(6)

        axes_section_title = QLabel("Axes")
        axes_section_title.setObjectName("sectionTitle")
        axes_section_layout.addWidget(axes_section_title)

        axes_grid = QGridLayout()
        axes_grid.setHorizontalSpacing(6)
        axes_grid.setVerticalSpacing(4)
        axes_grid.setColumnStretch(1, 1)
        axes_grid.addWidget(x_axis_label, 0, 0)
        axes_grid.addWidget(self.x_axis_combobox, 0, 1)
        axes_grid.addWidget(y_axis_label, 1, 0)
        axes_grid.addWidget(self.y_axis_combobox, 1, 1)
        axes_grid.addWidget(self.plot_button, 2, 0, 1, 2)
        axes_section_layout.addLayout(axes_grid)

        filter_widget = QWidget(parent)
        filter_layout = QGridLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setHorizontalSpacing(5)
        filter_layout.setVerticalSpacing(5)
        filter_layout.setColumnStretch(1, 1)
        filter_layout.setColumnStretch(3, 1)
        filter_layout.addWidget(filter_label, 0, 0)
        filter_layout.addWidget(self.filter_column_combobox, 0, 1)
        filter_layout.addWidget(self.filter_value_entry, 0, 2, 1, 2)
        filter_layout.addWidget(self.add_filter_button, 1, 0)
        filter_layout.addWidget(self.confirm_filter_button, 1, 1)
        filter_layout.addWidget(self.clear_filters_button, 1, 2)
        filter_layout.addWidget(self.show_active_filters_button, 1, 3)

        filter_section = self._create_card_frame()
        filter_section.setGraphicsEffect(None)
        filter_section.setObjectName("accentCard")
        filter_section_layout = filter_section.layout()
        filter_section_layout.setContentsMargins(10, 10, 10, 10)
        filter_section_layout.setSpacing(6)

        filter_section_title = QLabel("Filter")
        filter_section_title.setObjectName("sectionTitle")
        filter_section_layout.addWidget(filter_section_title)
        filter_section_layout.addWidget(filter_widget)

        top_layout.addWidget(load_section)
        top_layout.addWidget(axes_section)
        top_layout.addWidget(filter_section)

        # Fit controls sit below the plot because they operate on the currently
        # displayed data rather than on file-loading or column selection.
        fit_frame = QWidget(parent)
        fit_layout = QVBoxLayout(fit_frame)
        fit_layout.setContentsMargins(0, 0, 0, 0)
        fit_layout.setSpacing(8)
        parent_layout.addWidget(fit_frame)

        fit_label = QLabel("Fit type")
        self.polynomial_label = QLabel("Polynomial order")
        custom_label = QLabel("Custom LaTeX")

        self.fit_combobox = QComboBox(parent)
        self.fit_combobox.addItems(["polynomial", "power", "custom", "quantile"])
        self.fit_combobox.setCurrentText(self.default_fit_type)
        self.fit_combobox.currentTextChanged.connect(self._handle_fit_type_change)

        self.polynomial_entry = QLineEdit(self.default_polynomial_order, parent)
        self.custom_entry = QLineEdit(self.default_custom_latex, parent)
        # Use a lightweight pop-up preview for custom function confirmation.
        self.latex_preview_dialog = self._create_latex_preview_dialog(parent)

        # Add parameter entry for fit parameters (comma-separated LaTeX)
        self.param_label = QLabel("Fit parameters (comma-separated)", parent)
        self.param_entry = QLineEdit(parent)
        self.param_entry.setPlaceholderText("e.g. a, b, c")
        self.param_entry.setEnabled(False)
        self.param_entry.setMaximumWidth(190)

        model_input_width = self.param_entry.maximumWidth()
        self.fit_combobox.setMaximumWidth(model_input_width)
        self.polynomial_entry.setMaximumWidth(model_input_width)
        self.custom_entry.setMaximumWidth(model_input_width)

        self.interval_methods_label = QLabel("Joint CI/PI methods", parent)
        self.interval_methods_label.setObjectName("sectionTitle")
        self.interval_methods_widget = QWidget(parent)
        self.interval_methods_layout = QGridLayout(self.interval_methods_widget)
        self.interval_methods_layout.setContentsMargins(0, 0, 0, 0)
        self.interval_methods_layout.setHorizontalSpacing(10)
        self.interval_methods_layout.setVerticalSpacing(6)

        self.interval_exact_checkbox = QRadioButton("Exact analytical", parent)
        self.interval_delta_checkbox = QRadioButton("Delta", parent)
        self.interval_delta_checkbox.setChecked(True)
        self.interval_monte_carlo_checkbox = QRadioButton("Monte Carlo", parent)
        self.interval_profile_checkbox = QRadioButton("Profile likelihood", parent)
        self.interval_quantile_checkbox = QRadioButton("Quantile regression", parent)

        self.interval_method_group = QButtonGroup(parent)
        self.interval_method_group.setExclusive(True)
        self.interval_method_group.addButton(self.interval_exact_checkbox)
        self.interval_method_group.addButton(self.interval_delta_checkbox)
        self.interval_method_group.addButton(self.interval_monte_carlo_checkbox)
        self.interval_method_group.addButton(self.interval_profile_checkbox)
        self.interval_method_group.addButton(self.interval_quantile_checkbox)
        self.interval_exact_checkbox.toggled.connect(lambda _: self._update_fit_input_visibility())
        self.interval_delta_checkbox.toggled.connect(lambda _: self._update_fit_input_visibility())
        self.interval_monte_carlo_checkbox.toggled.connect(lambda _: self._update_fit_input_visibility())
        self.interval_profile_checkbox.toggled.connect(lambda _: self._update_fit_input_visibility())
        self.interval_quantile_checkbox.toggled.connect(lambda _: self._update_fit_input_visibility())

        self.interval_methods_layout.addWidget(self.interval_exact_checkbox, 0, 0)
        self.interval_methods_layout.addWidget(self.interval_delta_checkbox, 0, 1)
        self.interval_methods_layout.addWidget(self.interval_monte_carlo_checkbox, 1, 0)
        self.interval_methods_layout.addWidget(self.interval_profile_checkbox, 1, 1)
        self.interval_methods_layout.addWidget(self.interval_quantile_checkbox, 2, 0)
        self.interval_methods_layout.setColumnStretch(2, 1)

        self.confidence_coverage_label = QLabel("CI coverage (%)", parent)
        self.confidence_coverage_entry = QLineEdit("95", parent)
        self.confidence_coverage_entry.setMaximumWidth(70)

        self.prediction_coverage_label = QLabel("PI coverage (%)", parent)
        self.prediction_coverage_entry = QLineEdit("95", parent)
        self.prediction_coverage_entry.setMaximumWidth(70)

        self.error_model_label = QLabel("Error model", parent)
        self.error_model_combobox = QComboBox(parent)
        self.error_model_combobox.addItems(
            [
                "gaussian",
                "laplace",
                "logistic",
                "student_t",
                "hyperbolic_secant",
                "skew_normal",
            ]
        )
        self.error_model_combobox.currentTextChanged.connect(lambda _: self._update_fit_input_visibility())
        self.error_model_na_label = QLabel("NA", parent)
        self.error_model_na_label.setVisible(False)

        self.delta_order_label = QLabel("Delta order", parent)
        self.delta_order_label.setObjectName("subtitle")
        self.delta_order_combobox = QComboBox(parent)
        self.delta_order_combobox.addItem("Order 1 (first-order)", 1)
        self.delta_order_combobox.addItem("Order 2 (higher-order)", 2)
        self.delta_order_combobox.setCurrentIndex(0)
        self.delta_order_combobox.currentIndexChanged.connect(self._handle_delta_order_changed)

        self.monte_carlo_mode_label = QLabel("Monte Carlo mode", parent)
        self.monte_carlo_mode_label.setObjectName("subtitle")
        self.monte_carlo_mode_combobox = QComboBox(parent)
        self.monte_carlo_mode_combobox.addItem("Parametric (error model)", "parametric")
        self.monte_carlo_mode_combobox.addItem("Residual bootstrap", "residual_bootstrap")
        self.monte_carlo_mode_label.setVisible(False)
        self.monte_carlo_mode_combobox.setVisible(False)

        self.confirm_preview_button = QPushButton("Confirm function")
        self.confirm_preview_button.clicked.connect(self._handle_confirm_custom_preview)

        self.apply_fit_button = QPushButton("Apply fit")
        self.apply_fit_button.setEnabled(False)
        self.apply_fit_button.clicked.connect(self._handle_apply_fit)

        self.export_model_button = QPushButton("Export model")
        self.export_model_button.setEnabled(False)
        self.export_model_button.clicked.connect(self._handle_export_model)

        model_section = self._create_card_frame()
        model_section.setGraphicsEffect(None)
        model_section.setObjectName("accentCard")
        model_section_layout = model_section.layout()
        model_section_layout.setContentsMargins(10, 10, 10, 10)
        model_section_layout.setSpacing(6)

        model_section_title = QLabel("Model")
        model_section_title.setObjectName("sectionTitle")
        model_section_layout.addWidget(model_section_title)

        model_grid = QGridLayout()
        model_grid.setHorizontalSpacing(6)
        model_grid.setVerticalSpacing(6)
        model_grid.setColumnStretch(1, 1)
        model_grid.setColumnStretch(3, 1)
        model_grid.addWidget(fit_label, 0, 0)
        model_grid.addWidget(self.fit_combobox, 0, 1)
        model_grid.addWidget(self.polynomial_label, 1, 0)
        model_grid.addWidget(self.polynomial_entry, 1, 1)
        model_grid.addWidget(custom_label, 2, 0)
        model_grid.addWidget(self.custom_entry, 2, 1, 1, 3)
        model_grid.addWidget(self.confirm_preview_button, 2, 4)
        model_grid.addWidget(self.param_label, 3, 0)
        model_grid.addWidget(self.param_entry, 3, 1, 1, 2)
        model_section_layout.addLayout(model_grid)

        interval_section = self._create_card_frame()
        interval_section.setGraphicsEffect(None)
        interval_section.setObjectName("accentCard")
        interval_section_layout = interval_section.layout()
        interval_section_layout.setContentsMargins(10, 10, 10, 10)
        interval_section_layout.setSpacing(6)
        interval_section_layout.addWidget(self.interval_methods_label)
        interval_section_layout.addWidget(self.interval_methods_widget)

        interval_row = QWidget(parent)
        interval_row_layout = QGridLayout(interval_row)
        interval_row_layout.setContentsMargins(0, 0, 0, 0)
        interval_row_layout.setHorizontalSpacing(6)
        interval_row_layout.setVerticalSpacing(4)
        interval_row_layout.setColumnStretch(5, 1)
        interval_row_layout.addWidget(self.delta_order_label, 0, 0)
        interval_row_layout.addWidget(self.delta_order_combobox, 0, 1)
        interval_row_layout.addWidget(self.error_model_label, 1, 0)
        interval_row_layout.addWidget(self.error_model_combobox, 1, 1)
        interval_row_layout.addWidget(self.error_model_na_label, 1, 1)
        interval_row_layout.addWidget(self.monte_carlo_mode_label, 2, 0)
        interval_row_layout.addWidget(self.monte_carlo_mode_combobox, 2, 1)
        interval_row_layout.addWidget(self.confidence_coverage_label, 3, 0)
        interval_row_layout.addWidget(self.confidence_coverage_entry, 3, 1)
        interval_row_layout.addWidget(self.prediction_coverage_label, 3, 2)
        interval_row_layout.addWidget(self.prediction_coverage_entry, 3, 3)
        interval_section_layout.addWidget(interval_row)

        actions_section = QWidget(parent)
        actions_layout = QHBoxLayout(actions_section)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)
        actions_layout.addWidget(self.apply_fit_button)
        actions_layout.addWidget(self.export_model_button)
        actions_layout.addStretch(1)

        fit_scroll = QScrollArea(parent)
        fit_scroll.setWidgetResizable(True)
        fit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        fit_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        fit_scroll.setFrameShape(QFrame.Shape.NoFrame)

        fit_scroll_content = QWidget(fit_scroll)
        fit_scroll_layout = QVBoxLayout(fit_scroll_content)
        fit_scroll_layout.setContentsMargins(0, 0, 0, 0)
        fit_scroll_layout.setSpacing(8)
        fit_scroll_layout.addWidget(model_section)
        fit_scroll_layout.addWidget(interval_section)
        fit_scroll_layout.addWidget(actions_section)
        fit_scroll_layout.addStretch(1)

        fit_scroll.setWidget(fit_scroll_content)
        fit_layout.addWidget(fit_scroll)

        top_controls.setMaximumWidth(760)
        fit_frame.setMaximumWidth(760)

        self._update_fit_input_visibility()

    def _create_latex_preview_dialog(self, parent: QWidget) -> QDialog:
        """Create a lightweight custom-function preview dialog."""
        dialog = QDialog(parent)
        dialog.setWindowTitle("Custom equation preview")
        dialog.setModal(False)
        dialog.resize(980, 760)

        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(16, 16, 16, 16)
        dialog_layout.setSpacing(12)

        self.latex_preview_label = QLabel(dialog)
        self.latex_preview_label.setText("Press 'Confirm function' to render preview.")
        self.latex_preview_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.latex_preview_label.setWordWrap(True)
        self.latex_preview_label.setObjectName("statsReadout")

        preview_scroll = QScrollArea(dialog)
        preview_scroll.setWidgetResizable(False)
        preview_scroll.setMinimumHeight(640)
        preview_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        preview_scroll.setWidget(self.latex_preview_label)
        preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)

        dialog_layout.addWidget(preview_scroll)
        dialog_layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        return dialog

    def _handle_confirm_custom_preview(self) -> None:
        """Render and show the custom function preview dialog."""
        if self.fit_combobox.currentText() != "custom":
            return
        self._update_latex_preview(self.custom_entry.text(), delta_order=int(self.delta_order_combobox.currentData()))
        self.latex_preview_dialog.show()
        self.latex_preview_dialog.raise_()
        self.latex_preview_dialog.activateWindow()

    def _handle_delta_order_changed(self) -> None:
        """Refresh the custom-function preview when the Delta order changes."""
        if self.fit_combobox.currentText() != "custom":
            return
        if not hasattr(self, "latex_preview_dialog"):
            return
        if self.latex_preview_dialog.isVisible():
            self._update_latex_preview(self.custom_entry.text(), delta_order=int(self.delta_order_combobox.currentData()))

    def _update_latex_preview(self, latex_code: str, delta_order: int = 1) -> None:
        """Render custom function/derivatives from one parsed expression."""
        param_names = self._parse_fit_parameters()
        parse_error: str | None = None
        gradient_lines: list[str] = []
        hessian_lines: list[str] = []
        third_derivative_lines: list[str] = []

        if latex_code.strip():
            try:
                from sympy import diff, latex as sympy_latex, symbols

                expression = self.fit_service._parse_custom_expression(latex_code)
                function_latex = sympy_latex(expression)
                parameter_symbols = [symbols(name) for name in param_names]

                if parameter_symbols:
                    gradient_actual = [sympy_latex(diff(expression, symbol)) for symbol in parameter_symbols]
                    gradient_lines = ["Gradient"]
                    for name, derivative in zip(param_names, gradient_actual, strict=False):
                        gradient_lines.append(f"\\frac{{\\partial f}}{{\\partial {name}}} = {derivative}")

                    if delta_order >= 2:
                        hessian_lines = ["Hessian"]
                        for symbol_i, name_i in zip(parameter_symbols, param_names, strict=False):
                            for symbol_j, name_j in zip(parameter_symbols, param_names, strict=False):
                                actual_entry = sympy_latex(diff(expression, symbol_i, symbol_j))
                                hessian_lines.append(
                                    f"\\frac{{\\partial^2 f}}{{\\partial {name_i}\\partial {name_j}}} = {actual_entry}"
                                )

                        for index_triplet in combinations_with_replacement(range(len(parameter_symbols)), 3):
                            symbol_triplet = [parameter_symbols[index] for index in index_triplet]
                            derivative_latex = sympy_latex(diff(expression, *symbol_triplet))

                            denominator_terms: list[str] = []
                            for index in sorted(set(index_triplet)):
                                multiplicity = index_triplet.count(index)
                                parameter_name = param_names[index]
                                if multiplicity == 1:
                                    denominator_terms.append(f"\\partial {parameter_name}")
                                else:
                                    denominator_terms.append(f"\\partial {parameter_name}^{multiplicity}")

                            derivative_label = f"\\frac{{\\partial^3 f}}{{{''.join(denominator_terms)}}}"
                            third_derivative_lines.append(f"{derivative_label} = {derivative_latex}")
                else:
                    gradient_lines = ["Enter comma-separated parameters to show derivatives."]
            except Exception as exc:
                function_latex = "\\text{(parse failed)}"
                parse_error = str(exc)
                gradient_lines = [f"\\text{{Parse error: {exc}}}"]
        else:
            function_latex = "\\text{(empty)}"

        self._render_math_preview(
            function_latex,
            gradient_lines,
            hessian_lines,
            third_derivative_lines,
            parse_error,
            delta_order,
        )

    def _render_math_preview(
        self,
        function_latex: str,
        gradient_lines: list[str],
        hessian_lines: list[str],
        third_derivative_entries: list[str],
        parse_error: str | None,
        delta_order: int,
    ) -> None:
        """Render function/derivatives to an image using Matplotlib MathText."""
        row_count = 2 + len(gradient_lines)
        if delta_order >= 2:
            row_count += len(hessian_lines) + len(third_derivative_entries)
        if parse_error:
            row_count += 1

        figure = Figure(figsize=(11, max(4.0, 0.24 * row_count)), dpi=120)
        axis = figure.add_subplot(111)
        axis.axis("off")

        y = 0.96
        line_step = 0.055

        axis.text(0.01, y, "Function:", color="#f5e9ff", fontsize=8, transform=axis.transAxes)
        y -= line_step
        axis.text(0.03, y, f"${function_latex}$", color="#f5e9ff", fontsize=8.5, transform=axis.transAxes)

        y -= line_step * 1.2
        axis.text(0.01, y, "Gradient:", color="#f5e9ff", fontsize=8, transform=axis.transAxes)

        for line in gradient_lines:
            y -= line_step
            if line == "Gradient":
                axis.text(0.03, y, line, color="#f5e9ff", fontsize=7.8, transform=axis.transAxes)
            else:
                axis.text(0.03, y, f"${line}$", color="#f5e9ff", fontsize=7.6, transform=axis.transAxes)

        if delta_order >= 2:
            y -= line_step * 1.2
            axis.text(0.01, y, "Hessian:", color="#f5e9ff", fontsize=8, transform=axis.transAxes)

            for line in hessian_lines:
                y -= line_step
                if line == "Hessian":
                    axis.text(0.03, y, line, color="#f5e9ff", fontsize=7.8, transform=axis.transAxes)
                else:
                    axis.text(0.03, y, f"${line}$", color="#f5e9ff", fontsize=6.8, transform=axis.transAxes)

            y -= line_step * 1.2
            axis.text(0.01, y, "Third derivatives (unique symmetric terms):", color="#f5e9ff", fontsize=8, transform=axis.transAxes)

            if third_derivative_entries:
                for line in third_derivative_entries:
                    y -= line_step
                    axis.text(0.03, y, f"${line}$", color="#f5e9ff", fontsize=6.8, transform=axis.transAxes)
            else:
                y -= line_step
                axis.text(
                    0.03,
                    y,
                    "No third derivatives available until parameters are entered.",
                    color="#dffcff",
                    fontsize=7,
                    transform=axis.transAxes,
                )

        if parse_error:
            y -= line_step * 1.2
            axis.text(
                0.01,
                y,
                f"Parse error: {parse_error}",
                color="#ffd6f6",
                fontsize=7,
                transform=axis.transAxes,
            )

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight", facecolor="#1a1038", edgecolor="#1a1038")
        image = QImage.fromData(buffer.getvalue(), "PNG")
        self.latex_preview_label.setPixmap(QPixmap.fromImage(image))

    def _parse_fit_parameters(self) -> list[str]:
        """Parse comma-separated parameter names entered in LaTeX form."""
        return [parameter.strip() for parameter in self.param_entry.text().split(",") if parameter.strip()]

    def _make_curve_fit_function(self, latex_code: str, param_names: list[str]):
        """Build a callable f(x, *params) from a LaTeX expression."""
        from sympy import lambdify, symbols
        from sympy.parsing.latex import parse_latex

        try:
            expression = parse_latex(latex_code)
            x_symbol = symbols("x")
            parameter_symbols = symbols(",".join(param_names)) if param_names else tuple()
            if not isinstance(parameter_symbols, tuple):
                parameter_symbols = (parameter_symbols,)
            compiled = lambdify((x_symbol, *parameter_symbols), expression, modules="numpy")
        except Exception as exc:
            raise ValueError(f"Invalid custom model expression: {exc}") from exc

        def model_function(x_values, *parameter_values):
            return compiled(x_values, *parameter_values)

        return model_function

    def _build_statistics_panel(self, parent: QFrame) -> None:
        """Create the dedicated statistics display frame.

        Args:
            parent: Parent frame for statistics content.
        """
        parent_layout = parent.layout()

        # The statistics text box is read-only and separate from the plot so fit
        # details remain visible while the user experiments with other options.
        title_label = QLabel("Fit Statistics")
        title_label.setObjectName("sectionTitle")
        parent_layout.addWidget(title_label)

        self.stats_text = QTextEdit(parent)
        self.stats_text.setObjectName("statsReadout")
        self.stats_text.setReadOnly(True)
        self.stats_text.setAcceptRichText(False)
        self.stats_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        parent_layout.addWidget(self.stats_text, 1)
        self._set_statistics_text("SYSTEM READY\n\nLoad data, choose columns, and apply a fit to populate the readout.")

    def _handle_load_data(self) -> None:
        """Open a file dialog and load a tabular dataset."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a data file",
            "",
            "Supported data (*.csv *.xls *.xlsx *.json *.pkl *.pickle);;CSV (*.csv);;Excel (*.xls *.xlsx);;JSON (*.json);;Pickled DataFrame (*.pkl *.pickle);;All files (*.*)",
        )
        if not file_path:
            return

        try:
            self.loaded_dataset = self.data_loader.load(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        # Column names populate both dropdowns because the user can freely pick
        # any available heading for either plotted axis.
        column_names = list(self.loaded_dataset.frame.columns)
        self._set_combobox_items(self.x_axis_combobox, column_names)
        self._set_combobox_items(self.y_axis_combobox, column_names)
        self._set_combobox_items(self.filter_column_combobox, ["Choose filter column", *column_names])
        self._active_filters = []
        self._refresh_active_filters_label()
        if not column_names:
            self.x_axis_combobox.setEnabled(False)
            self.y_axis_combobox.setEnabled(False)
            self.filter_column_combobox.setEnabled(False)
            self.filter_value_entry.setEnabled(False)
            self.add_filter_button.setEnabled(False)
            self.confirm_filter_button.setEnabled(False)
            self.clear_filters_button.setEnabled(False)
            self.plot_button.setEnabled(False)
            self.apply_fit_button.setEnabled(False)
            QMessageBox.warning(self, "No columns", "The loaded dataset has no columns to select.")
            return

        self.x_axis_combobox.setEnabled(True)
        self.y_axis_combobox.setEnabled(True)
        self.filter_column_combobox.setEnabled(False)
        self.filter_value_entry.setEnabled(False)
        self.add_filter_button.setEnabled(True)
        self.confirm_filter_button.setEnabled(False)
        self.clear_filters_button.setEnabled(False)
        self.plot_button.setEnabled(True)
        self.apply_fit_button.setEnabled(True)
        self.x_axis_combobox.setCurrentText(column_names[0])
        self.y_axis_combobox.setCurrentText(column_names[1] if len(column_names) > 1 else column_names[0])
        self.filter_column_combobox.setCurrentText("Choose filter column")
        self.filter_value_entry.clear()
        self.loaded_file_label.setText(self.loaded_dataset.source_path)
        self._set_statistics_text(
            "DATA LINK ESTABLISHED\n\nDataset loaded. Select x/y columns, adjust fit settings, then confirm the plot and apply a fit."
        )

    def _handle_plot_data(self) -> None:
        """Validate selections and display the chosen data columns."""
        if self.loaded_dataset is None:
            QMessageBox.warning(self, "No data", "Load a dataset before plotting.")
            return

        try:
            x_values, y_values = self._extract_numeric_series(
                self.loaded_dataset.frame,
                self.x_axis_combobox.currentText(),
                self.y_axis_combobox.currentText(),
                self._active_filters,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Plot failed", str(exc))
            return

        self.plot_renderer.plot_data(
            x_values,
            y_values,
            self.x_axis_combobox.currentText(),
            self.y_axis_combobox.currentText(),
        )
        self.current_plot_result = None
        self.export_model_button.setEnabled(False)
        self._set_statistics_text("TRACE RENDERED\n\nData plotted. Choose a fit type and apply a fit.")

    def _handle_apply_fit(self) -> None:
        """Compute a fit for the currently selected data columns."""
        if self.loaded_dataset is None:
            QMessageBox.warning(self, "No data", "Load a dataset before fitting.")
            return

        fit_type = self.fit_combobox.currentText()
        selected_methods = self._selected_joint_interval_methods()
        if fit_type != "quantile" and not selected_methods:
            QMessageBox.warning(
                self,
                "No interval method",
                "Select at least one interval method before applying the fit.",
            )
            return

        selected_prediction_methods = ["quantile_regression"] if fit_type == "quantile" else list(selected_methods)
        selected_confidence_methods = [] if fit_type == "quantile" else list(selected_methods)
        if not selected_prediction_methods:
            QMessageBox.warning(
                self,
                "No prediction method",
                "Select at least one prediction method before applying the fit.",
            )
            return

        delta_order = int(self.delta_order_combobox.currentData())
        try:
            confidence_alpha = self._coverage_to_alpha(self.confidence_coverage_entry.text(), "CI coverage")
            prediction_alpha = self._coverage_to_alpha(self.prediction_coverage_entry.text(), "PI coverage")
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid coverage", str(exc))
            return

        try:
            x_values, y_values = self._extract_numeric_series(
                self.loaded_dataset.frame,
                self.x_axis_combobox.currentText(),
                self.y_axis_combobox.currentText(),
                self._active_filters,
            )
            fitting_result = self._fit_selected_model(x_values, y_values)
        except Exception as exc:
            QMessageBox.critical(self, "Fit failed", str(exc))
            return

        x_numeric = x_values.to_numpy(dtype=float)
        y_numeric = y_values.to_numpy(dtype=float)

        self.plot_renderer.plot_data(
            x_values,
            y_values,
            self.x_axis_combobox.currentText(),
            self.y_axis_combobox.currentText(),
        )
        self.plot_renderer.overlay_fit(fitting_result)
        bands: BandComputationResult | None = None
        try:
            bands = self.band_service.compute_confidence_bands(
                fitting_result=fitting_result,
                x_values=x_numeric,
                y_values=y_numeric,
                methods=selected_confidence_methods,
                prediction_methods=selected_prediction_methods,
                delta_order=delta_order,
                confidence_alpha=confidence_alpha,
                prediction_alpha=prediction_alpha,
                error_model_name=self.error_model_combobox.currentText(),
                monte_carlo_mode=self._selected_monte_carlo_mode(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Band computation failed", str(exc))
            return

        self.plot_renderer.overlay_prediction_bands(bands.prediction_bands)
        self.plot_renderer.overlay_confidence_bands(bands.confidence_bands)
        for profile_map in bands.profile_prediction_maps.values():
            self.plot_renderer.overlay_profile_prediction_map(profile_map)

        self.current_plot_result = fitting_result
        self._current_bands = bands
        self._last_fit_x = x_numeric
        self._last_fit_y = y_numeric
        self._last_fit_type = self.fit_combobox.currentText()
        self._last_custom_expression = self.custom_entry.text().strip() if self._last_fit_type == "custom" else None
        self.export_model_button.setEnabled(True)
        self._display_fitting_result(fitting_result, bands=bands)
        self._prompt_save_curve_models(
            fitting_result=fitting_result,
            x_values=x_numeric,
            y_values=y_numeric,
            method_keys=[
                *list(bands.prediction_bands.keys()),
                *list(bands.profile_prediction_maps.keys()),
            ],
            bands=bands,
        )

    def _handle_export_model(self) -> None:
        """Export fitted standard-curve artifacts (plot, model pickle, details)."""
        if self.current_plot_result is None or self._last_fit_x is None or self._last_fit_y is None:
            QMessageBox.warning(self, "No fit", "Apply a fit before exporting a model.")
            return

        parent_dir = QFileDialog.getExistingDirectory(self, "Select parent directory for export")
        if not parent_dir:
            return

        folder_name, folder_ok = QInputDialog.getText(self, "Export folder", "Enter export folder name:")
        if not folder_ok or not folder_name.strip():
            return

        model_name, model_ok = QInputDialog.getText(self, "Model name", "Enter model name:")
        if not model_ok or not model_name.strip():
            return

        export_dir = Path(parent_dir) / folder_name.strip()
        export_dir.mkdir(parents=True, exist_ok=True)

        model_type = self._last_fit_type or "polynomial"
        curve_object = build_standard_curve_model(
            self.current_plot_result,
            self._last_fit_x,
            self._last_fit_y,
            model_type=model_type,
            expression_text=self._last_custom_expression,
            error_model_name=self.error_model_combobox.currentText(),
            error_model_parameters=self._error_model_parameters_from_sigma(
                self._current_bands.sigma if self._current_bands is not None else None
            ),
            ci_coverage=(1.0 - self._current_bands.confidence_alpha) if self._current_bands is not None else None,
            pi_coverage=(1.0 - self._current_bands.prediction_alpha) if self._current_bands is not None else None,
        )

        plot_path = export_dir / f"{model_name.strip()}-plot.png"
        model_path = export_dir / f"{model_name.strip()}_model.pkl"
        details_path = export_dir / f"{model_name.strip()}_details.txt"

        self.plot_renderer.figure.savefig(plot_path, dpi=300, bbox_inches="tight")
        with model_path.open("wb") as file_handle:
            pickle.dump(curve_object, file_handle)
        details_path.write_text(self.stats_text.toPlainText(), encoding="utf-8")

        QMessageBox.information(
            self,
            "Export complete",
            f"Exported artifacts to:\n{export_dir}\n\n"
            f"- {plot_path.name}\n"
            f"- {model_path.name}\n"
            f"- {details_path.name}",
        )

    def _prompt_save_curve_models(
        self,
        fitting_result: FittingResult,
        x_values,
        y_values,
        method_keys: list[str],
        bands: BandComputationResult | None,
    ) -> None:
        """Prompt the user to save each computed curve model in sequence."""
        if not method_keys:
            method_keys = ["fit"]

        saved_files: list[str] = []
        model_type = self._last_fit_type or self.fit_combobox.currentText()
        error_model_name = self.error_model_combobox.currentText()
        sigma = bands.sigma if bands is not None else None
        error_model_parameters = self._error_model_parameters_from_sigma(sigma)

        for method_key in method_keys:
            method_label = self._prediction_method_display_name(method_key)
            response = QMessageBox.question(
                self,
                "Save model",
                f"Save curve model for {method_label}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if response != QMessageBox.StandardButton.Yes:
                continue

            suggested_name = f"{model_type}_{method_key}".replace(" ", "_")
            model_name, model_ok = QInputDialog.getText(
                self,
                "Curve name",
                f"Enter a name for {method_label} curve:",
                text=suggested_name,
            )
            if not model_ok or not model_name.strip():
                continue

            destination_dir = QFileDialog.getExistingDirectory(
                self,
                f"Select folder to save {method_label} curve",
            )
            if not destination_dir:
                continue

            pi_point_x, pi_point_lower, pi_point_upper = self._pi_points_for_method(method_key, bands)

            curve_object = build_standard_curve_model(
                fitting_result,
                x_values,
                y_values,
                model_type=model_type,
                expression_text=self._last_custom_expression,
                error_model_name=error_model_name,
                error_model_parameters=error_model_parameters,
                pi_method=method_key,
                pi_point_x=pi_point_x,
                pi_point_lower=pi_point_lower,
                pi_point_upper=pi_point_upper,
                ci_coverage=(1.0 - bands.confidence_alpha) if bands is not None else None,
                pi_coverage=(1.0 - bands.prediction_alpha) if bands is not None else None,
            )

            model_path = Path(destination_dir) / f"{model_name.strip()}_model.pkl"
            with model_path.open("wb") as file_handle:
                pickle.dump(curve_object, file_handle)
            saved_files.append(str(model_path))

        if saved_files:
            QMessageBox.information(
                self,
                "Models saved",
                "Saved model files:\n\n" + "\n".join(saved_files),
            )

    def _pi_points_for_method(
        self,
        method_key: str,
        bands: BandComputationResult | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return point-based PI arrays for persistence/interpolation."""
        if bands is None:
            return None, None, None

        prediction_band = bands.prediction_bands.get(method_key)
        if prediction_band is not None:
            return (
                np.asarray(prediction_band.x_grid, dtype=float),
                np.asarray(prediction_band.lower_values, dtype=float),
                np.asarray(prediction_band.upper_values, dtype=float),
            )

        profile_map = bands.profile_prediction_maps.get(method_key)
        if profile_map is not None:
            x_mle = np.asarray(profile_map["x_mle"], dtype=float)
            y_grid = np.asarray(profile_map["y_grid"], dtype=float)
            x_lower = np.asarray(profile_map["x_lower"], dtype=float)
            x_upper = np.asarray(profile_map["x_upper"], dtype=float)

            if x_mle.size < 2 or y_grid.size < 2:
                return None, None, None

            # Approximate y-bounds as functions of x for persistence.
            lower_order = np.argsort(x_lower)
            upper_order = np.argsort(x_upper)

            x_lower_sorted = x_lower[lower_order]
            y_lower_sorted = y_grid[lower_order]
            x_upper_sorted = x_upper[upper_order]
            y_upper_sorted = y_grid[upper_order]

            x_lower_unique, idx_lower = np.unique(x_lower_sorted, return_index=True)
            y_lower_unique = y_lower_sorted[idx_lower]
            x_upper_unique, idx_upper = np.unique(x_upper_sorted, return_index=True)
            y_upper_unique = y_upper_sorted[idx_upper]

            if x_lower_unique.size < 2 or x_upper_unique.size < 2:
                return None, None, None

            x_min = float(max(np.min(x_lower_unique), np.min(x_upper_unique)))
            x_max = float(min(np.max(x_lower_unique), np.max(x_upper_unique)))
            if x_max <= x_min:
                return None, None, None

            x_common = np.linspace(x_min, x_max, 300)
            lower_interp = np.interp(x_common, x_lower_unique, y_lower_unique)
            upper_interp = np.interp(x_common, x_upper_unique, y_upper_unique)
            return x_common, lower_interp, upper_interp

        return None, None, None

    def _handle_prev_method_plot(self) -> None:
        """Navigate to the previous method plot entry if available."""
        if not self._method_plot_entries:
            return
        self._current_method_plot_index = max(0, self._current_method_plot_index - 1)
        self._refresh_method_plot_nav_state()

    def _handle_next_method_plot(self) -> None:
        """Navigate to the next method plot entry if available."""
        if not self._method_plot_entries:
            return
        max_index = len(self._method_plot_entries) - 1
        self._current_method_plot_index = min(max_index, self._current_method_plot_index + 1)
        self._refresh_method_plot_nav_state()

    def _refresh_method_plot_nav_state(self) -> None:
        """Update method-navigation labels and enabled states."""
        if not self._method_plot_entries:
            self.prev_method_plot_button.setEnabled(False)
            self.next_method_plot_button.setEnabled(False)
            self.method_plot_label.setText("Method view: none")
            return

        max_index = len(self._method_plot_entries) - 1
        self.prev_method_plot_button.setEnabled(self._current_method_plot_index > 0)
        self.next_method_plot_button.setEnabled(self._current_method_plot_index < max_index)

        label = self._method_plot_entries[self._current_method_plot_index].get("label", "method")
        self.method_plot_label.setText(
            f"Method view: {label} ({self._current_method_plot_index + 1}/{len(self._method_plot_entries)})"
        )

    def _error_model_parameters_from_sigma(self, sigma: float | None) -> dict[str, float] | None:
        """Return persisted error-model parameters derived from residual sigma."""
        if sigma is None:
            return None

        sigma_value = float(max(sigma, 1e-12))
        error_model_name = self.error_model_combobox.currentText().strip().lower()

        if error_model_name == "gaussian":
            return {"sigma": sigma_value}
        if error_model_name == "laplace":
            return {"b": sigma_value / (2.0 ** 0.5)}
        if error_model_name == "logistic":
            return {"s": sigma_value * (3.0 ** 0.5) / np.pi}
        if error_model_name == "student_t":
            return {"df": 5.0, "scale": sigma_value * ((5.0 - 2.0) / 5.0) ** 0.5}
        if error_model_name == "hyperbolic_secant":
            return {"scale": sigma_value}
        if error_model_name == "skew_normal":
            return {"a": 4.0, "scale": sigma_value}
        return None

    def _fit_selected_model(self, x_values, y_values) -> FittingResult:
        """Dispatch to the requested fitting method.

        Args:
            x_values: Numeric x-axis values.
            y_values: Numeric y-axis values.

        Returns:
            The computed fitting result.
        """
        fit_type = self.fit_combobox.currentText()

        # Each fit type is dispatched explicitly so parameter prompts remain
        # understandable and model-specific validation stays local.
        if fit_type == "polynomial":
            polynomial_order = int(self.polynomial_entry.text())
            return self.fit_service.fit_polynomial(x_values, y_values, polynomial_order)
        if fit_type == "power":
            return self.fit_service.fit_power_law(x_values, y_values)
        if fit_type == "custom":
            latex_expression = self.custom_entry.text().strip()
            parameter_names = self._parse_fit_parameters()
            return self.fit_service.fit_custom_latex(
                x_values,
                y_values,
                latex_expression,
                parameter_names=parameter_names or None,
            )
        if fit_type == "quantile":
            polynomial_order = int(self.polynomial_entry.text())
            coverage = 1.0 - self._coverage_to_alpha(self.prediction_coverage_entry.text(), "PI coverage")
            return self.fit_service.fit_quantile_polynomial(
                x_values,
                y_values,
                polynomial_order,
                coverage=coverage,
            )

        raise ValueError(f"Unsupported fit type: {fit_type}")

    def _display_fitting_result(
        self,
        fitting_result: FittingResult,
        bands: BandComputationResult | None = None,
    ) -> None:
        """Render fitting metadata into the statistics panel.

        Args:
            fitting_result: The computed fitting result.
        """
        # The statistics box shows the model equation followed by a stable list
        # of optimized parameters that mirrors the internal parameter ordering.
        lines = [
            "FIT LOCKED",
            "",
            f"Model: {fitting_result.model_name}",
            f"Equation: {fitting_result.display_equation}",
            "",
            "Parameters:",
        ]
        for parameter_name, parameter_value in zip(
            fitting_result.parameter_names,
            fitting_result.parameter_values,
            strict=False,
        ):
            lines.append(f"  {parameter_name} = {parameter_value:.6g}")

        if bands is not None:
            lines.extend(["", "Confidence bands:"])

            if bands.confidence_bands:
                for method_key, confidence_band in bands.confidence_bands.items():
                    display_name = self._confidence_method_display_name(method_key)
                    if method_key.strip().lower().startswith("delta"):
                        display_name = f"{display_name} (order {bands.delta_order})"
                    lines.append(f"  {display_name}: {(1.0 - confidence_band.alpha) * 100:.1f}%")
            else:
                lines.append("  None computed.")

            if bands.confidence_errors:
                lines.append("  Errors:")
                for method_key, error in bands.confidence_errors.items():
                    lines.append(f"    {self._confidence_method_display_name(method_key)}: {error}")

            lines.extend(
                [
                    "",
                    "Prediction intervals:",
                    f"  Coverage: {(1.0 - bands.prediction_alpha) * 100:.1f}%",
                    (
                        "  Error model: NA"
                        if (fitting_result.fit_method or "").strip().lower() == "quantile_regression"
                        else f"  Error model: {self.error_model_combobox.currentText()}"
                    ),
                    f"  Residual sigma: {bands.sigma:.6g}",
                    f"  Degrees of freedom: {bands.dof:.0f}",
                ]
            )

            if bands.prediction_bands:
                lines.append("  Methods:")
                for method_key, prediction_band in bands.prediction_bands.items():
                    lines.append(
                        f"    {self._prediction_method_display_name(method_key)}: {(1.0 - prediction_band.alpha) * 100:.1f}%"
                    )
            else:
                lines.append("  Methods: none computed")

            if bands.profile_prediction_maps:
                for method_key, profile_map in bands.profile_prediction_maps.items():
                    lines.append(
                        f"  {self._prediction_method_display_name(method_key)} map points: {len(profile_map['y_grid'])}"
                    )

            if bands.prediction_errors:
                lines.append("  Errors:")
                for method_key, error in bands.prediction_errors.items():
                    lines.append(f"    {self._prediction_method_display_name(method_key)}: {error}")

        self._set_statistics_text("\n".join(lines))

    def _selected_joint_interval_methods(self) -> list[str]:
        """Return the joint interval methods selected by the user."""
        if self.interval_exact_checkbox.isChecked():
            return ["exact_analytic"]
        if self.interval_delta_checkbox.isChecked():
            return ["delta"]
        if self.interval_monte_carlo_checkbox.isChecked():
            return ["monte_carlo"]
        if self.interval_profile_checkbox.isChecked():
            return ["profile_likelihood"]
        if self.interval_quantile_checkbox.isChecked():
            return ["quantile_regression"]

        return []

    def _selected_monte_carlo_mode(self) -> str:
        """Return the Monte Carlo mode selected by the user."""
        selected_mode = self.monte_carlo_mode_combobox.currentData()
        if isinstance(selected_mode, str) and selected_mode:
            return selected_mode
        return "parametric"

    def _coverage_to_alpha(self, coverage_text: str, label: str) -> float:
        """Convert coverage percent text (e.g. 95) into alpha (e.g. 0.05)."""
        try:
            coverage = float(coverage_text)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric (e.g. 95).") from exc

        if not (0.0 < coverage < 100.0):
            raise ValueError(f"{label} must be between 0 and 100 (exclusive).")

        return 1.0 - (coverage / 100.0)

    def _confidence_method_display_name(self, method_key: str) -> str:
        """Return a readable label for a confidence method key."""
        normalized_key = method_key.strip().lower()
        if normalized_key in {"exact_analytic", "gaussian_analytic"}:
            return "Exact Analytical"
        if normalized_key.startswith("delta"):
            return "Delta Method"
        if normalized_key == "monte_carlo":
            return "Monte Carlo"
        if normalized_key == "profile_likelihood":
            return "Profile Likelihood"
        return method_key.replace("_", " ").title()

    def _prediction_method_display_name(self, method_key: str) -> str:
        """Return a readable label for a prediction method key."""
        normalized_key = method_key.strip().lower()
        if normalized_key in {"exact_analytic", "gaussian_analytic"}:
            return "Exact Analytical"
        if normalized_key.startswith("delta"):
            return "Delta Method"
        if normalized_key == "monte_carlo":
            return "Monte Carlo"
        if normalized_key == "profile_likelihood_modified":
            return "Profile Likelihood (Modified)"
        if normalized_key == "quantile_regression":
            return "Quantile Regression"
        return method_key.replace("_", " ").title()

    def _extract_numeric_series(
        self,
        frame: pd.DataFrame,
        x_column: str,
        y_column: str,
        filters: list[tuple[str, str]] | None = None,
    ):
        """Extract paired numeric data from the selected DataFrame columns.

        Args:
            frame: Source DataFrame.
            x_column: Selected x-axis column name.
            y_column: Selected y-axis column name.
            filters: Optional list of exact-match filters to apply in sequence.

        Returns:
            A tuple of numeric pandas Series objects with matching valid rows.

        Raises:
            ValueError: If selections are missing or the resulting data is empty.
        """
        if not x_column or not y_column:
            raise ValueError("Select both x-axis and y-axis columns.")

        filtered_frame = frame
        if filters:
            for filter_column, filter_value in filters:
                filtered_frame = self._filter_frame_by_value(filtered_frame, filter_column, filter_value)
                if filtered_frame.empty:
                    raise ValueError("No rows matched the confirmed filters.")

        # Conversion is performed on just the selected columns, and rows with
        # invalid values are dropped in pairs so the plotted data stays aligned.
        numeric_frame = filtered_frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if numeric_frame.empty:
            raise ValueError("The selected columns do not contain usable numeric data.")

        return numeric_frame[x_column], numeric_frame[y_column]

    def _handle_add_filter(self) -> None:
        """Enable the pending filter controls so the user can define one more filter."""
        if self.loaded_dataset is None:
            QMessageBox.warning(self, "No data", "Load a dataset before adding filters.")
            return

        self.filter_column_combobox.setEnabled(True)
        self.filter_value_entry.setEnabled(True)
        self.confirm_filter_button.setEnabled(True)
        self.add_filter_button.setEnabled(False)
        self.filter_column_combobox.setCurrentText("Choose filter column")
        self.filter_value_entry.clear()
        self.filter_value_entry.setFocus()

    def _handle_confirm_filter(self) -> None:
        """Commit the pending filter to the active filter list."""
        filter_column = self.filter_column_combobox.currentText().strip()
        filter_value = self.filter_value_entry.text().strip()

        if not filter_column or filter_column == "Choose filter column":
            QMessageBox.warning(self, "Missing filter column", "Choose a filter column before confirming.")
            return
        if not filter_value:
            QMessageBox.warning(self, "Missing filter value", "Enter a filter value before confirming.")
            return

        self._active_filters.append((filter_column, filter_value))
        self._refresh_active_filters_label()
        self.filter_column_combobox.setEnabled(False)
        self.filter_value_entry.setEnabled(False)
        self.confirm_filter_button.setEnabled(False)
        self.add_filter_button.setEnabled(True)
        self.clear_filters_button.setEnabled(True)
        self.filter_column_combobox.setCurrentText("Choose filter column")
        self.filter_value_entry.clear()

    def _refresh_active_filters_label(self) -> None:
        """Update the active-filter controls."""
        self.clear_filters_button.setEnabled(bool(self._active_filters))
        self.show_active_filters_button.setEnabled(True)

    def _handle_show_active_filters(self) -> None:
        """Open a pop-up window showing the active filters in two columns."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Active filters")
        dialog.resize(640, 360)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        table = QTableWidget(dialog)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Filter variable", "Value"])
        table.setRowCount(max(len(self._active_filters), 1))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setAlternatingRowColors(True)

        if not self._active_filters:
            table.setItem(0, 0, QTableWidgetItem("No active filters"))
            table.setItem(0, 1, QTableWidgetItem(""))
        else:
            for row_index, (column, value) in enumerate(self._active_filters):
                table.setItem(row_index, 0, QTableWidgetItem(column))
                table.setItem(row_index, 1, QTableWidgetItem(value))

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.close)

        layout.addWidget(table)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.setModal(False)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._active_filters_dialog = dialog

    def _remove_filter(self, index: int) -> None:
        """Remove one confirmed filter by index and refresh the visible list."""
        if index < 0 or index >= len(self._active_filters):
            return

        del self._active_filters[index]
        self._refresh_active_filters_label()

    def _handle_clear_filters(self) -> None:
        """Remove all confirmed filters and reset pending filter controls."""
        self._active_filters = []
        self._refresh_active_filters_label()
        self.filter_column_combobox.setCurrentText("Choose filter column")
        self.filter_value_entry.clear()
        self.filter_column_combobox.setEnabled(False)
        self.filter_value_entry.setEnabled(False)
        self.confirm_filter_button.setEnabled(False)
        self.add_filter_button.setEnabled(self.loaded_dataset is not None)

    def _filter_frame_by_value(self, frame: pd.DataFrame, filter_column: str, filter_value: str) -> pd.DataFrame:
        """Return rows whose filter column matches the requested value."""
        series = frame[filter_column]

        if pd.api.types.is_numeric_dtype(series):
            try:
                numeric_value = float(filter_value)
                return frame[series.astype(float) == numeric_value]
            except ValueError:
                pass

        if pd.api.types.is_bool_dtype(series):
            normalized = filter_value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return frame[series == True]  # noqa: E712
            if normalized in {"false", "0", "no", "n", "off"}:
                return frame[series == False]  # noqa: E712

        normalized_series = series.astype(str).str.strip()
        return frame[normalized_series == filter_value]

    def _handle_fit_type_change(self, value: str) -> None:
        """Update fit inputs and clear model-specific fields on model changes."""
        # Keep only the active model's input fields populated.
        if value == "polynomial":
            self.custom_entry.clear()
            self.param_entry.clear()
            self._update_latex_preview("")
        elif value == "custom":
            self.polynomial_entry.clear()
            self.custom_entry.clear()
            self.param_entry.clear()
            self._update_latex_preview("")
        elif value == "quantile":
            self.custom_entry.clear()
            self.param_entry.clear()
            self._update_latex_preview("")
        else:  # power
            self.polynomial_entry.clear()
            self.custom_entry.clear()
            self.param_entry.clear()
            self._update_latex_preview("")

        self._update_fit_input_visibility()

    def _update_fit_input_visibility(self) -> None:
        """Show or hide fit-specific parameter input widgets."""
        fit_type = self.fit_combobox.currentText()
        uses_polynomial_order = fit_type in {"polynomial", "quantile"}
        self.polynomial_label.setVisible(uses_polynomial_order)
        self.polynomial_entry.setVisible(uses_polynomial_order)
        self.polynomial_entry.setEnabled(uses_polynomial_order)
        self.custom_entry.setEnabled(fit_type == "custom")
        self.confirm_preview_button.setVisible(fit_type == "custom")
        self.confirm_preview_button.setEnabled(fit_type == "custom")
        self.param_label.setEnabled(fit_type == "custom")
        self.param_entry.setEnabled(fit_type == "custom")

        quantile_mode = fit_type == "quantile"
        self.interval_methods_label.setEnabled(not quantile_mode)
        self.interval_methods_widget.setEnabled(not quantile_mode)
        self.confidence_coverage_label.setEnabled(not quantile_mode)
        self.confidence_coverage_entry.setEnabled(not quantile_mode)

        self.error_model_label.setEnabled(not quantile_mode)
        self.error_model_combobox.setVisible(not quantile_mode)
        self.error_model_combobox.setEnabled(not quantile_mode)
        self.error_model_na_label.setVisible(quantile_mode)

        delta_selected = self.interval_delta_checkbox.isChecked() and not quantile_mode
        self.delta_order_label.setEnabled(delta_selected)
        self.delta_order_combobox.setEnabled(delta_selected)

        monte_carlo_selected = self.interval_monte_carlo_checkbox.isChecked() and not quantile_mode
        self.monte_carlo_mode_label.setVisible(monte_carlo_selected)
        self.monte_carlo_mode_combobox.setVisible(monte_carlo_selected)
        self.monte_carlo_mode_label.setEnabled(monte_carlo_selected)
        self.monte_carlo_mode_combobox.setEnabled(monte_carlo_selected)

        supports_exact = fit_type == "polynomial" and self.error_model_combobox.currentText().strip().lower() == "gaussian"
        self.interval_exact_checkbox.setEnabled((not quantile_mode) and supports_exact)
        if not supports_exact:
            self.interval_exact_checkbox.setChecked(False)

        self.interval_quantile_checkbox.setEnabled(quantile_mode)

        if quantile_mode:
            self.interval_exact_checkbox.setChecked(False)
            self.interval_delta_checkbox.setChecked(False)
            self.interval_monte_carlo_checkbox.setChecked(False)
            self.interval_profile_checkbox.setChecked(False)
            self.interval_quantile_checkbox.setChecked(True)
        else:
            self.interval_quantile_checkbox.setChecked(False)
            if not self._selected_joint_interval_methods():
                self.interval_delta_checkbox.setChecked(True)

        if fit_type == "custom":
            self._update_latex_preview(self.custom_entry.text())

    def _set_statistics_text(self, message: str) -> None:
        """Replace the statistics panel text.

        Args:
            message: Text to display.
        """
        self.stats_text.setPlainText(message)

    def _set_combobox_items(self, combobox: QComboBox, items: list[str]) -> None:
        """Replace all items in a combobox.

        Args:
            combobox: Combobox to update.
            items: Items that should be displayed.
        """
        combobox.clear()
        combobox.addItems(items)
