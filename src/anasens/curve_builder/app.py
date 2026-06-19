"""Application entry points for Plotter Fitter."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .gui import PlotterFitterGUI


class PlotterFitterApplication:
    """Create and run the desktop application.

    This class is intentionally small so the CLI entry point only needs
    to depend on one orchestration object.
    """

    def run(self) -> None:
        """Start the GUI event loop."""
        application = QApplication.instance()
        owns_application = application is None
        if application is None:
            application = QApplication(sys.argv)

        gui = PlotterFitterGUI()
        gui.show()

        if owns_application:
            application.exec()
