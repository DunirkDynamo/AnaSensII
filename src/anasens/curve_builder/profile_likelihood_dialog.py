"""Popup dialog for profile-likelihood inverse-calibration plots."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from anasens.uncertainty.bands import ProfileLikelihoodBand


class ProfileLikelihoodDialog(QDialog):
    """Display inverse-calibration and resolution plots in a popup window."""

    def __init__(self, profile_map: ProfileLikelihoodBand, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Profile Likelihood Inverse Calibration")
        self.resize(1380, 720)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.figure = Figure(figsize=(12.8, 5.3), dpi=110, facecolor="#0c0820")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self._render(profile_map)

    def _render(self, profile_map: ProfileLikelihoodBand) -> None:
        y_grid = np.asarray(profile_map.y_grid, dtype=float)
        x_mle = np.asarray(profile_map.x_mle, dtype=float)
        x_lower = np.asarray(profile_map.x_lower, dtype=float)
        x_upper = np.asarray(profile_map.x_upper, dtype=float)

        resolution = np.full_like(x_mle, np.nan, dtype=float)
        valid = np.isfinite(x_mle) & np.isfinite(x_lower) & np.isfinite(x_upper)
        resolution[valid] = (x_upper[valid] - x_lower[valid]) / x_mle[valid]

        self.figure.clear()
        ax1, ax2 = self.figure.subplots(1, 2)

        ax1.plot(y_grid, x_mle, color="#111111", linewidth=2.2, label=r"MLE Point Estimate $\hat{x}_0(y_0)$")
        ax1.plot(y_grid, x_lower, color="#1f77b4", linestyle="--", linewidth=1.6, label=r"Profile CI Lower ($x_0^{\mathrm{lower}}$)")
        ax1.plot(y_grid, x_upper, color="#d62728", linestyle="--", linewidth=1.6, label=r"Profile CI Upper ($x_0^{\mathrm{upper}}$)")
        ax1.fill_between(y_grid, x_lower, x_upper, color="#1f77b4", alpha=0.16, label="Calibration Uncertainty Band")
        ax1.set_title("Inverse Calibration Function", fontsize=13, pad=10)
        ax1.set_xlabel(r"Hypothetical Measurement Observation ($y_0$)", fontsize=11)
        ax1.set_ylabel(r"Inferred Parameter Estimate ($x_0$)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.55)
        ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

        ax2.plot(x_mle[valid], resolution[valid], color="darkmagenta", linewidth=2.6, label=r"Relative Width: $\frac{x_0^{\mathrm{upper}} - x_0^{\mathrm{lower}}}{\hat{x}_0}$")
        ax2.set_title("Resolution Curve", fontsize=13, pad=10)
        ax2.set_xlabel("Estimate", fontsize=11)
        ax2.set_ylabel("Resolution", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.55)
        ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

        self.figure.tight_layout()
        self.canvas.draw_idle()
