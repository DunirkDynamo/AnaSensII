"""Qt GUI launcher for AnaSens2."""

from __future__ import annotations

import sys

from PySide6.QtCore import QPointF, QRectF, QProcess
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from anasens.interfaces.gui_views.curve_builder_tab import CurveBuilderTab
from anasens.interfaces.gui_views.inverse_estimation_tab import InverseEstimationTab
from anasens.interfaces.gui_views.pipeline_tab import PipelineRunRequest, PipelineTab
from anasens.interfaces.gui_views.calibrator_tab import CalibratorTab


class SynthwaveBackdrop(QWidget):
    """Paint a retro synthwave backdrop behind the app shell."""

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming convention.
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        width = self.width()
        height = self.height()
        horizon_y = int(height * 0.45)
        vanishing_x = width / 2

        sky_gradient = QLinearGradient(0, 0, 0, height)
        sky_gradient.setColorAt(0.0, QColor("#050816"))
        sky_gradient.setColorAt(0.4, QColor("#1a1038"))
        sky_gradient.setColorAt(0.72, QColor("#2a124f"))
        sky_gradient.setColorAt(1.0, QColor("#120321"))
        painter.fillRect(self.rect(), sky_gradient)

        glow_gradient = QRadialGradient(QPointF(vanishing_x, horizon_y - 48), width * 0.34)
        glow_gradient.setColorAt(0.0, QColor(255, 111, 214, 150))
        glow_gradient.setColorAt(0.55, QColor(123, 92, 255, 85))
        glow_gradient.setColorAt(1.0, QColor(5, 8, 22, 0))
        painter.fillRect(self.rect(), glow_gradient)

        sun_gradient = QRadialGradient(QPointF(vanishing_x, horizon_y - 18), width * 0.1)
        sun_gradient.setColorAt(0.0, QColor("#ffe27a"))
        sun_gradient.setColorAt(0.45, QColor("#ff76d8"))
        sun_gradient.setColorAt(1.0, QColor(255, 118, 216, 0))
        painter.setBrush(sun_gradient)
        painter.setPen(QPen(QColor(255, 210, 245, 140), 1.4))
        painter.drawEllipse(QPointF(vanishing_x, horizon_y - 18), width * 0.1, width * 0.1)

        painter.setPen(QPen(QColor(210, 236, 255, 100), 1.2))
        for point in [
            QPointF(width * 0.12, height * 0.12),
            QPointF(width * 0.22, height * 0.2),
            QPointF(width * 0.82, height * 0.16),
            QPointF(width * 0.68, height * 0.1),
            QPointF(width * 0.9, height * 0.24),
            QPointF(width * 0.38, height * 0.08),
            QPointF(width * 0.56, height * 0.19),
        ]:
            painter.drawPoint(point)

        painter.setPen(QPen(QColor("#48e3ff"), 2.0))
        painter.drawLine(0, horizon_y, width, horizon_y)

        painter.setPen(QPen(QColor(255, 90, 200, 90), 1.0))
        for step in range(1, 10):
            y_position = horizon_y + int((height - horizon_y) * (step / 10) ** 1.7)
            painter.drawLine(0, y_position, width, y_position)

        for index in range(-8, 9):
            x_position = vanishing_x + index * width * 0.08
            painter.drawLine(QPointF(vanishing_x, horizon_y), QPointF(x_position, height))

        vignette_gradient = QRadialGradient(QRectF(0, 0, width, height).center(), width * 0.68)
        vignette_gradient.setColorAt(0.72, QColor(0, 0, 0, 0))
        vignette_gradient.setColorAt(1.0, QColor(0, 0, 0, 110))
        painter.fillRect(self.rect(), vignette_gradient)


class AnaSensWindow(QMainWindow):
    """Main Qt shell for the AnaSens2 application."""

    def __init__(self) -> None:
        super().__init__()
        self._process: QProcess | None = None

        self.setWindowTitle("AnaSens2")
        self.resize(1420, 900)
        self.setMinimumSize(1200, 780)

        self._configure_style()
        self._build_ui()

    def _configure_style(self) -> None:
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
            QLabel#subtleBody {
                color: #d7cdf4;
            }
            QTabWidget::pane {
                border: none;
                margin-top: 4px;
            }
            QTabBar::tab {
                background: rgba(12, 9, 31, 0.9);
                color: #d7cdf4;
                border: 1px solid rgba(72, 227, 255, 0.45);
                border-bottom: none;
                padding: 10px 18px;
                margin-right: 6px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
            QTabBar::tab:selected {
                color: #fff7ff;
                background: rgba(255, 90, 200, 0.28);
                border-color: rgba(255, 90, 200, 0.8);
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
            QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox {
                background: rgba(9, 8, 30, 0.9);
                border: 1px solid rgba(255, 90, 200, 0.45);
                border-radius: 10px;
                padding: 8px;
                color: #f8f1ff;
            }
            QComboBox:hover, QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover, QSpinBox:hover {
                border: 1px solid rgba(72, 227, 255, 0.7);
            }
            QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus {
                border: 1px solid #48e3ff;
            }
            """
        )

    def _build_ui(self) -> None:
        backdrop = SynthwaveBackdrop(self)
        self.setCentralWidget(backdrop)

        outer_layout = QVBoxLayout(backdrop)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setSpacing(14)

        header_card = QFrame(backdrop)
        header_card.setObjectName("card")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("AnaSens2")
        title.setObjectName("title")
        header_layout.addWidget(title)

        shadow = QGraphicsDropShadowEffect(header_card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(255, 90, 200, 70))
        header_card.setGraphicsEffect(shadow)
        outer_layout.addWidget(header_card)

        tab_card = QFrame(backdrop)
        tab_card.setObjectName("card")
        tab_layout = QVBoxLayout(tab_card)
        tab_layout.setContentsMargins(12, 12, 12, 12)
        tab_layout.setSpacing(10)

        self.tabs = QTabWidget(tab_card)
        self.pipeline_tab = PipelineTab(self.tabs)
        self.curve_tab = CurveBuilderTab(self.tabs)
        self.inverse_tab = InverseEstimationTab(self.tabs)
        self.calibrator_tab = CalibratorTab(self.tabs)

        self.tabs.addTab(self.pipeline_tab, "Pipeline")
        self.tabs.addTab(self.curve_tab, "Curve Builder")
        self.tabs.addTab(self.inverse_tab, "Inverse Estimation")
        self.tabs.addTab(self.calibrator_tab, "Calibrator")

        tab_layout.addWidget(self.tabs)
        outer_layout.addWidget(tab_card, 1)

        self.pipeline_tab.runRequested.connect(self._launch_native_pipeline)

    def _launch_native_pipeline(self, request: PipelineRunRequest) -> None:
        if not request.input_folder:
            self.pipeline_tab.append_log("Select an input folder before running the pipeline.")
            return

        self.pipeline_tab.append_log(
            "Launching native pipeline in a separate process so the Qt shell stays responsive..."
        )
        self.pipeline_tab.append_log(
            self._format_pipeline_request_log(request)
        )

        if self._process is not None:
            self._process.deleteLater()

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_process_output)
        self._process.readyReadStandardError.connect(self._read_process_output)
        self._process.finished.connect(lambda *_: self._process_finished())

        args = [
            "-m",
            "anasens.interfaces.pipeline_launcher",
            "--input-folder",
            request.input_folder,
            "--analysis-mode",
            request.analysis_mode,
            "--cut-start-points",
            str(request.cut_start_points),
            "--cut-end-points",
            str(request.cut_end_points),
            "--smooth-window-length",
            str(request.smooth_window_length),
            "--smooth-polyorder",
            str(request.smooth_polyorder),
            "--baseline-algo",
            str(request.baseline_algo),
            "--baseline-snip-expected-fwhm",
            str(request.baseline_snip_expected_fwhm),
            "--baseline-snip-window-size",
            str(request.baseline_snip_window_size),
            "--baseline-moving-average-window-size",
            str(request.baseline_moving_average_window_size),
            "--baseline-moving-average-max-iter",
            str(request.baseline_moving_average_max_iter),
            "--baseline-moving-average-forward"
            if request.baseline_moving_average_forward
            else "--no-baseline-moving-average-forward",
            "--pairplot" if request.pairplot else "--no-pairplot",
            "--drift-2d" if request.drift_2d else "--no-drift-2d",
            "--drift-1d" if request.drift_1d else "--no-drift-1d",
            "--overlay-scans" if request.overlay_scans else "--no-overlay-scans",
        ]
        self._process.start(sys.executable, args)
        self.pipeline_tab.set_running_state(True)

    def _format_pipeline_request_log(self, request: PipelineRunRequest) -> str:
        if request.baseline_algo == "moving_average":
            baseline_summary = "baseline(moving_average window/forward/max_iter)={}/{}/{}".format(
                request.baseline_moving_average_window_size,
                request.baseline_moving_average_forward,
                request.baseline_moving_average_max_iter,
            )
        else:
            baseline_summary = "baseline(snip window/FWHM)={}/{}".format(
                request.baseline_snip_window_size,
                request.baseline_snip_expected_fwhm,
            )

        return "Requested mode={} | cut(start/end)={}/{} | smooth(window/polyorder)={}/{} | {} | folder={}".format(
            request.analysis_mode,
            request.cut_start_points,
            request.cut_end_points,
            request.smooth_window_length,
            request.smooth_polyorder,
            baseline_summary,
            request.input_folder or "[not set]",
        )

    def _read_process_output(self) -> None:
        if self._process is None:
            return
        output = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if output:
            self.pipeline_tab.append_log(output)

    def _process_finished(self) -> None:
        self.pipeline_tab.set_running_state(False)
        self.pipeline_tab.append_log("Native pipeline process finished.")


def launch_gui() -> None:
    """Launch the Qt GUI application."""
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv)

    window = AnaSensWindow()
    window.show()
    if owns_application:
        application.exec()
