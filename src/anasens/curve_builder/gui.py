"""PySide6 GUI for Plotter Fitter."""

from __future__ import annotations

import io
from pathlib import Path
import pickle
import sys

import pandas as pd
from matplotlib.figure import Figure
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .data_loader import DataLoader
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
        self.loaded_dataset: LoadedDataset | None = None
        self.current_plot_result: FittingResult | None = None
        self._active_filters: list[tuple[str, str]] = []
        self._last_fit_x = None
        self._last_fit_y = None
        self._last_fit_type: str | None = None
        self._last_custom_expression: str | None = None

        # Core window metadata is established before widget construction so the
        # layout code can assume the main window shell already exists.
        self.setWindowTitle("Plotter Fitter")
        self.resize(1320, 840)
        self.setMinimumSize(1120, 720)

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
                font-size: 11pt;
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
                font-size: 28px;
                font-weight: 700;
                letter-spacing: 2px;
            }
            QLabel#subtitle {
                color: #8fe8ff;
                font-size: 11pt;
                font-weight: 600;
                letter-spacing: 1px;
            }
            QLabel#sectionTitle {
                color: #48e3ff;
                font-family: "Segoe UI Semibold";
                font-size: 14pt;
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
                border-radius: 10px;
                padding: 10px 16px;
                font-weight: 700;
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
                border-radius: 10px;
                padding: 8px;
                color: #f8f1ff;
            }
            QComboBox:hover, QLineEdit:hover, QTextEdit:hover {
                border: 1px solid rgba(72, 227, 255, 0.7);
            }
            QComboBox:focus, QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #48e3ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
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
                border-radius: 12px;
                padding: 12px;
                font-size: 9pt;
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
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setHorizontalSpacing(14)
        outer_layout.setVerticalSpacing(12)
        outer_layout.setColumnStretch(0, 4)
        outer_layout.setColumnStretch(1, 2)
        outer_layout.setRowStretch(1, 1)

        title_label = QLabel("Plotter Fitter")
        title_label.setObjectName("title")
        outer_layout.addWidget(title_label, 0, 0, 1, 2)

        control_frame = self._create_card_frame()
        stats_frame = self._create_card_frame()
        stats_frame.setObjectName("accentCard")
        stats_frame.setMinimumWidth(380)
        outer_layout.addWidget(control_frame, 2, 0)
        outer_layout.addWidget(stats_frame, 2, 1)

        self._build_controls(control_frame)
        self._build_statistics_panel(stats_frame)

    def _create_card_frame(self) -> QFrame:
        """Create a shared card-style frame used across the interface.

        Returns:
            A configured frame with a vertical layout.
        """
        card_frame = QFrame(self)
        card_frame.setObjectName("card")
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(14)

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
        top_layout = QGridLayout(top_controls)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setHorizontalSpacing(12)
        top_layout.setVerticalSpacing(12)
        top_layout.setColumnStretch(1, 1)
        parent_layout.addWidget(top_controls)

        file_label = QLabel("Loaded file:")
        file_label.setObjectName("sectionTitle")
        self.loaded_file_label = QLabel(self.default_loaded_file_text)
        self.loaded_file_label.setObjectName("fileValue")
        self.load_button = QPushButton("Load data")
        self.load_button.clicked.connect(self._handle_load_data)

        top_layout.addWidget(file_label, 0, 0)
        top_layout.addWidget(self.loaded_file_label, 0, 1)
        top_layout.addWidget(self.load_button, 0, 2)

        x_axis_label = QLabel("X-axis")
        y_axis_label = QLabel("Y-axis")
        filter_label = QLabel("Filter")
        self.x_axis_combobox = QComboBox(parent)
        self.y_axis_combobox = QComboBox(parent)
        self.filter_column_combobox = QComboBox(parent)
        self.filter_value_entry = QLineEdit(parent)
        self.add_filter_button = QPushButton("Add filter")
        self.confirm_filter_button = QPushButton("Confirm filter")
        self.clear_filters_button = QPushButton("Clear filters")
        self.active_filters_list = QListWidget(parent)
        self.active_filters_list.setMinimumHeight(76)
        self.active_filters_list.setMaximumHeight(160)
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
        self.plot_button = QPushButton("Confirm plot")
        self.plot_button.setEnabled(False)
        self.plot_button.clicked.connect(self._handle_plot_data)

        top_layout.addWidget(x_axis_label, 1, 0)
        top_layout.addWidget(self.x_axis_combobox, 1, 1)
        top_layout.addWidget(y_axis_label, 1, 2)
        top_layout.addWidget(self.y_axis_combobox, 1, 3)
        top_layout.addWidget(self.plot_button, 1, 4)
        top_layout.addWidget(filter_label, 2, 0)
        top_layout.addWidget(self.filter_column_combobox, 2, 1)
        top_layout.addWidget(self.filter_value_entry, 2, 2, 1, 2)
        top_layout.addWidget(self.add_filter_button, 2, 4)
        top_layout.addWidget(self.confirm_filter_button, 2, 5)
        top_layout.addWidget(self.clear_filters_button, 2, 6)
        top_layout.addWidget(self.active_filters_list, 3, 0, 1, 6)

        # The plot frame is isolated from the statistics frame so fit metadata
        # never competes with the visualization for screen space.
        plot_frame = QFrame(parent)
        plot_frame.setObjectName("card")
        plot_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        parent_layout.addWidget(plot_frame, 1)
        self.plot_renderer = PlotRenderer(plot_frame)

        # Fit controls sit below the plot because they operate on the currently
        # displayed data rather than on file-loading or column selection.
        fit_frame = QWidget(parent)
        fit_layout = QGridLayout(fit_frame)
        fit_layout.setContentsMargins(0, 0, 0, 0)
        fit_layout.setHorizontalSpacing(12)
        fit_layout.setVerticalSpacing(12)
        fit_layout.setColumnStretch(3, 1)
        parent_layout.addWidget(fit_frame)

        fit_label = QLabel("Fit type")
        self.polynomial_label = QLabel("Polynomial order")
        custom_label = QLabel("Custom LaTeX")

        self.fit_combobox = QComboBox(parent)
        self.fit_combobox.addItems(["polynomial", "power", "custom"])
        self.fit_combobox.setCurrentText(self.default_fit_type)
        self.fit_combobox.currentTextChanged.connect(self._handle_fit_type_change)

        self.polynomial_entry = QLineEdit(self.default_polynomial_order, parent)
        self.custom_entry = QLineEdit(self.default_custom_latex, parent)
        # Use a lightweight plain-text preview for custom function confirmation.
        self.latex_preview = self._create_latex_preview_widget(parent)

        # Add parameter entry for fit parameters (comma-separated LaTeX)
        self.param_label = QLabel("Fit parameters (LaTeX, comma-separated)", parent)
        self.param_entry = QLineEdit(parent)
        self.param_entry.setPlaceholderText("e.g. a, b, c")
        self.param_entry.setEnabled(False)
        self.preview_mode_label = QLabel(parent)
        self.preview_mode_label.setObjectName("subtitle")
        self.preview_mode_label.setText(self._build_preview_mode_text())
        self.preview_mode_label.setEnabled(False)

        self.confirm_preview_button = QPushButton("Confirm function")
        self.confirm_preview_button.clicked.connect(self._handle_confirm_custom_preview)

        self.apply_fit_button = QPushButton("Apply fit")
        self.apply_fit_button.setEnabled(False)
        self.apply_fit_button.clicked.connect(self._handle_apply_fit)

        self.export_model_button = QPushButton("Export model")
        self.export_model_button.setEnabled(False)
        self.export_model_button.clicked.connect(self._handle_export_model)

        fit_layout.addWidget(fit_label, 0, 0)
        fit_layout.addWidget(self.fit_combobox, 0, 1)
        fit_layout.addWidget(self.polynomial_label, 0, 2)
        fit_layout.addWidget(self.polynomial_entry, 0, 3)
        fit_layout.addWidget(self.apply_fit_button, 0, 4)
        fit_layout.addWidget(self.export_model_button, 0, 5)
        fit_layout.addWidget(custom_label, 1, 0)
        fit_layout.addWidget(self.custom_entry, 1, 1, 1, 3)
        fit_layout.addWidget(self.confirm_preview_button, 1, 4)
        fit_layout.addWidget(self.param_label, 2, 0)
        fit_layout.addWidget(self.param_entry, 2, 1, 1, 4)
        fit_layout.addWidget(self.latex_preview, 3, 0, 1, 5)
        fit_layout.addWidget(self.preview_mode_label, 4, 0, 1, 5)

        self._update_fit_input_visibility()

    def _create_latex_preview_widget(self, parent: QWidget):
        """Create a lightweight custom-function preview widget."""
        self.latex_preview_label = QLabel(parent)
        self.latex_preview_label.setText("Press 'Confirm function' to render preview.")
        self.latex_preview_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.latex_preview_label.setWordWrap(True)
        self.latex_preview_label.setObjectName("statsReadout")

        preview_scroll = QScrollArea(parent)
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setMinimumHeight(180)
        preview_scroll.setMaximumHeight(340)
        preview_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        preview_scroll.setWidget(self.latex_preview_label)
        preview_scroll.setVisible(False)
        return preview_scroll

    def _build_preview_mode_text(self) -> str:
        """Return a status string that indicates the active preview mode."""
        return "Preview mode: Lightweight rendered math (Matplotlib MathText)"

    def _handle_confirm_custom_preview(self) -> None:
        """Render custom function preview on explicit user confirmation."""
        if self.fit_combobox.currentText() != "custom":
            return
        self._update_latex_preview(self.custom_entry.text())

    def _update_latex_preview(self, latex_code: str) -> None:
        """Render custom function/derivatives from one parsed expression."""
        param_names = self._parse_fit_parameters()
        derivatives: dict[str, str] = {}
        parse_error: str | None = None

        if latex_code.strip():
            try:
                from sympy import diff, latex as sympy_latex, symbols

                expression = self.fit_service._parse_custom_expression(latex_code)
                function_latex = sympy_latex(expression)

                for name in param_names:
                    symbol = symbols(name)
                    derivatives[name] = sympy_latex(diff(expression, symbol))
            except Exception as exc:
                function_latex = "\\text{(parse failed)}"
                parse_error = str(exc)
                derivatives = {name: f"ERROR: parse error: {exc}" for name in param_names}
        else:
            function_latex = "\\text{(empty)}"

        derivative_latex_lines: list[tuple[str, str]] = []
        for name in param_names:
            derivative_value = derivatives.get(name, "")
            if derivative_value.startswith("ERROR:"):
                derivative_latex_lines.append((name, derivative_value[6:].strip()))
            else:
                derivative_latex_lines.append((name, derivative_value))

        self._render_math_preview(function_latex, derivative_latex_lines, parse_error, not param_names)

        self.latex_preview.setVisible(self.fit_combobox.currentText() == "custom")

    def _render_math_preview(
        self,
        function_latex: str,
        derivative_entries: list[tuple[str, str]],
        parse_error: str | None,
        show_parameter_hint: bool,
    ) -> None:
        """Render function/derivatives to an image using Matplotlib MathText."""
        row_count = 3 + len(derivative_entries)
        if parse_error:
            row_count += 1
        if show_parameter_hint:
            row_count += 1

        figure = Figure(figsize=(10, max(2.4, 0.5 * row_count)), dpi=120)
        axis = figure.add_subplot(111)
        axis.axis("off")

        y = 0.96
        line_step = 0.11

        axis.text(0.01, y, "Function:", color="#f5e9ff", fontsize=11, transform=axis.transAxes)
        y -= line_step
        axis.text(0.03, y, f"${function_latex}$", color="#f5e9ff", fontsize=12, transform=axis.transAxes)

        y -= line_step * 1.2
        axis.text(0.01, y, "Derivatives:", color="#f5e9ff", fontsize=11, transform=axis.transAxes)

        if show_parameter_hint:
            y -= line_step
            axis.text(
                0.03,
                y,
                "Enter comma-separated parameters to generate derivatives.",
                color="#dffcff",
                fontsize=10,
                transform=axis.transAxes,
            )

        for parameter_name, derivative_value in derivative_entries:
            y -= line_step
            if derivative_value.startswith("parse error:") or derivative_value.startswith("error:"):
                axis.text(
                    0.03,
                    y,
                    f"d/d{parameter_name}: {derivative_value}",
                    color="#ffd6f6",
                    fontsize=10,
                    transform=axis.transAxes,
                )
            else:
                derivative_latex = f"\\frac{{\\partial}}{{\\partial {parameter_name}}}({function_latex}) = {derivative_value}"
                axis.text(0.03, y, f"${derivative_latex}$", color="#f5e9ff", fontsize=11, transform=axis.transAxes)

        if parse_error:
            y -= line_step * 1.2
            axis.text(
                0.01,
                y,
                f"Parse error: {parse_error}",
                color="#ffd6f6",
                fontsize=10,
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

        readout_label = QLabel("Parameter Readout")
        readout_label.setObjectName("subtitle")
        parent_layout.addWidget(readout_label)

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
            "DATA LINK ESTABLISHED\n\nDataset loaded. Select x/y columns, add any filters you want, then confirm the plot."
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

        self.plot_renderer.plot_data(
            x_values,
            y_values,
            self.x_axis_combobox.currentText(),
            self.y_axis_combobox.currentText(),
        )
        self.plot_renderer.overlay_fit(fitting_result)
        self.current_plot_result = fitting_result
        self._last_fit_x = x_values.to_numpy(dtype=float)
        self._last_fit_y = y_values.to_numpy(dtype=float)
        self._last_fit_type = self.fit_combobox.currentText()
        self._last_custom_expression = self.custom_entry.text().strip() if self._last_fit_type == "custom" else None
        self.export_model_button.setEnabled(True)
        self._display_fitting_result(fitting_result)

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

        raise ValueError(f"Unsupported fit type: {fit_type}")

    def _display_fitting_result(self, fitting_result: FittingResult) -> None:
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
        self._set_statistics_text("\n".join(lines))

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
        """Render confirmed filters as removable rows."""
        self.active_filters_list.clear()
        self.clear_filters_button.setEnabled(bool(self._active_filters))

        if not self._active_filters:
            placeholder_item = QListWidgetItem("No active filters")
            placeholder_item.setFlags(placeholder_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.active_filters_list.addItem(placeholder_item)
            return

        for index, (column, value) in enumerate(self._active_filters):
            item = QListWidgetItem(self.active_filters_list)
            row_widget = QWidget(self.active_filters_list)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 4, 8, 4)
            row_layout.setSpacing(8)

            row_label = QLabel(f"{column} = {value}", row_widget)
            remove_button = QPushButton("Remove", row_widget)
            remove_button.clicked.connect(lambda _checked=False, filter_index=index: self._remove_filter(filter_index))

            row_layout.addWidget(row_label, 1)
            row_layout.addWidget(remove_button)

            item.setSizeHint(row_widget.sizeHint())
            self.active_filters_list.addItem(item)
            self.active_filters_list.setItemWidget(item, row_widget)

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
        else:  # power
            self.polynomial_entry.clear()
            self.custom_entry.clear()
            self.param_entry.clear()
            self._update_latex_preview("")

        self._update_fit_input_visibility()

    def _update_fit_input_visibility(self) -> None:
        """Show or hide fit-specific parameter input widgets and LaTeX preview."""
        fit_type = self.fit_combobox.currentText()
        self.polynomial_label.setVisible(fit_type == "polynomial")
        self.polynomial_entry.setVisible(fit_type == "polynomial")
        self.polynomial_entry.setEnabled(fit_type == "polynomial")
        self.custom_entry.setEnabled(fit_type == "custom")
        self.confirm_preview_button.setVisible(fit_type == "custom")
        self.confirm_preview_button.setEnabled(fit_type == "custom")
        self.latex_preview.setVisible(fit_type == "custom")
        self.param_label.setEnabled(fit_type == "custom")
        self.param_entry.setEnabled(fit_type == "custom")
        self.preview_mode_label.setVisible(fit_type == "custom")
        self.preview_mode_label.setEnabled(fit_type == "custom")
        self.preview_mode_label.setText(self._build_preview_mode_text())
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
