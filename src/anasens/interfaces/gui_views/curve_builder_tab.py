"""Curve Builder tab embedding the Plotter-Fitter UI."""

from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from anasens.curve_builder.gui import PlotterFitterGUI


class CurveBuilderTab(QWidget):
    """Embed Plotter-Fitter in the AnaSens tabbed shell."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._plotter_window: PlotterFitterGUI | None = None
        self._embedded_widget: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._loading_label = QLabel("Curve Builder is loading when opened.", self)
        self._loading_label.setWordWrap(True)
        layout.addWidget(self._loading_label)

    def showEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt naming convention.
        super().showEvent(event)
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Construct and embed the heavy Plotter-Fitter UI on first use."""
        if self._plotter_window is not None:
            return

        layout = self.layout()
        if layout is None:
            return

        self._plotter_window = PlotterFitterGUI()
        embedded_widget = self._plotter_window.takeCentralWidget()
        if embedded_widget is None:
            return

        self._embedded_widget = embedded_widget
        self._loading_label.hide()
        layout.addWidget(embedded_widget)
