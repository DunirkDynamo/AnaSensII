"""Curve Builder tab embedding the Plotter-Fitter UI."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from anasens.curve_builder.gui import PlotterFitterGUI


class CurveBuilderTab(QWidget):
    """Embed Plotter-Fitter in the AnaSens tabbed shell."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Reuse the existing Plotter-Fitter window implementation by hosting
        # its central widget inside this tab.
        self._plotter_window = PlotterFitterGUI()
        embedded_widget = self._plotter_window.takeCentralWidget()
        if embedded_widget is not None:
            layout.addWidget(embedded_widget)
