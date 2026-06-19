"""Plot rendering helpers for Plotter Fitter."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .models import FittingResult


class PlotRenderer:
    """Render styled plots in a PySide6 widget.

    The renderer owns matplotlib objects and presentation styling only. It does
    not compute fits or transform user input.
    """

    def __init__(self, parent: QWidget) -> None:
        """Create a renderer and attach its canvas to a Qt parent widget.

        Args:
            parent: Widget that will contain the matplotlib canvas.
        """
        # The figure and axis are configured once and then reused to avoid
        # recreating Qt widgets on every plotting interaction.
        self.figure = Figure(figsize=(7.0, 5.2), dpi=110, facecolor="#0c0820")
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)

        # The parent widget owns the layout so the plot canvas can stretch with
        # the main window while the renderer stays focused on presentation.
        parent_layout = parent.layout()
        if parent_layout is None:
            parent_layout = QVBoxLayout(parent)
            parent_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(self.canvas)

        self._apply_base_style()

    def plot_data(self, x_values, y_values, x_label: str, y_label: str) -> None:
        """Render the selected x and y columns.

        Args:
            x_values: Numeric x-axis values.
            y_values: Numeric y-axis values.
            x_label: Label displayed on the x-axis.
            y_label: Label displayed on the y-axis.
        """
        # The axis is reset for a fresh plot while preserving the shared styling.
        self.axis.clear()
        self._apply_base_style()

        # Scatter styling is chosen to feel more deliberate than matplotlib's
        # defaults while remaining clear on light backgrounds.
        # The plot backdrop uses the current data range to place a stylized
        # synthwave horizon and glow behind the actual data marks.
        x_numeric = np.asarray(x_values, dtype=float)
        y_numeric = np.asarray(y_values, dtype=float)
        x_min = float(np.min(x_numeric))
        x_max = float(np.max(x_numeric))
        y_min = float(np.min(y_numeric))
        y_max = float(np.max(y_numeric))
        self._draw_backdrop(x_min, x_max, y_min, y_max)

        self.axis.scatter(
            x_values,
            y_values,
            s=42,
            color="#48e3ff",
            edgecolors="#ffd3f7",
            linewidths=1.0,
            alpha=0.98,
            zorder=3,
            label="Data",
        )

        self.axis.set_xlim(x_min, x_max)
        self.axis.set_ylim(y_min, y_max)

        self.axis.set_xlabel(x_label, fontsize=12, fontweight="bold", color="#f8f1ff")
        self.axis.set_ylabel(y_label, fontsize=12, fontweight="bold", color="#f8f1ff")
        self._style_legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def overlay_fit(self, fitting_result: FittingResult) -> None:
        """Overlay a fitted curve on the existing plot.

        Args:
            fitting_result: Result object containing the fitted curve data.
        """
        self.axis.plot(
            fitting_result.fitted_x,
            fitting_result.fitted_y,
            color="#ff5ac8",
            linewidth=3.0,
            zorder=4,
            label=fitting_result.model_name,
        )
        self._style_legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_backdrop(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """Paint a synthwave-style backdrop inside the plotting area.

        Args:
            x_min: Minimum x value in the displayed data.
            x_max: Maximum x value in the displayed data.
            y_min: Minimum y value in the displayed data.
            y_max: Maximum y value in the displayed data.
        """
        # The ranges are expanded slightly so the decorative backdrop has room
        # to breathe around the data without clipping points at the border.
        x_padding = max((x_max - x_min) * 0.05, 1e-9)
        y_padding = max((y_max - y_min) * 0.08, 1e-9)
        x_left = x_min - x_padding
        x_right = x_max + x_padding
        y_bottom = y_min - y_padding
        y_top = y_max + y_padding

        # A vertical gradient anchors the dark sky and warmer horizon tones
        # directly inside the chart so the plot feels integrated with the UI.
        gradient_height = 240
        gradient = np.linspace(0.0, 1.0, gradient_height)[:, None]
        top_color = np.array([12, 8, 32, 255], dtype=float) / 255.0
        bottom_color = np.array([55, 15, 91, 255], dtype=float) / 255.0
        gradient_rgba = top_color * (1.0 - gradient)[:, :, None] + bottom_color * gradient[:, :, None]
        self.axis.imshow(
            gradient_rgba,
            extent=(x_left, x_right, y_bottom, y_top),
            aspect="auto",
            interpolation="bicubic",
            zorder=0,
        )

        # The horizon glow is represented with stacked translucent spans and a
        # neon line to evoke a sunset without masking the data.
        horizon_y = y_bottom + (y_top - y_bottom) * 0.34
        self.axis.axhspan(horizon_y, y_top, color="#20114a", alpha=0.22, zorder=0.2)
        self.axis.axhspan(horizon_y - (y_top - y_bottom) * 0.08, horizon_y, color="#ff5ac8", alpha=0.12, zorder=0.25)
        self.axis.axhline(horizon_y, color="#ff9de2", linewidth=1.6, alpha=0.55, zorder=0.35)

        # Perspective grid lines below the horizon reinforce the retro-futurist
        # look while staying subtle enough to avoid fighting the actual axes.
        grid_color = "#8b6dff"
        for step in np.linspace(horizon_y - (y_top - y_bottom) * 0.02, y_bottom, 7):
            self.axis.axhline(step, color=grid_color, linewidth=0.7, alpha=0.16, zorder=0.3)
        for fraction in np.linspace(0.1, 0.9, 9):
            x_position = x_left + (x_right - x_left) * fraction
            self.axis.plot(
                [(x_left + x_right) / 2, x_position],
                [horizon_y, y_bottom],
                color="#48e3ff",
                linewidth=0.6,
                alpha=0.12,
                zorder=0.3,
            )

        self.axis.set_xlim(x_left, x_right)
        self.axis.set_ylim(y_bottom, y_top)

    def _style_legend(self) -> None:
        """Apply a neon card treatment to the matplotlib legend."""
        legend = self.axis.legend(frameon=True, loc="best", title="Trace")
        if legend is None:
            return

        # The legend is styled like a compact synthwave control card so it
        # visually belongs with the rest of the PySide6 interface.
        legend.get_frame().set_facecolor("#120a2a")
        legend.get_frame().set_edgecolor("#48e3ff")
        legend.get_frame().set_linewidth(1.1)
        legend.get_frame().set_alpha(0.82)
        title = legend.get_title()
        title.set_color("#ff9de2")
        title.set_fontweight("bold")
        for text in legend.get_texts():
            text.set_color("#f8f1ff")

    def _apply_base_style(self) -> None:
        """Apply consistent plot styling to the managed axis."""
        # Spine, tick, and grid styling are centralized so every redraw uses the
        # same synthwave visual language requested for the application.
        self.axis.set_facecolor("#140a2f")
        for spine in self.axis.spines.values():
            spine.set_color("#f8f1ff")
            spine.set_linewidth(2.2)

        self.axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=6,
            width=1.4,
            color="#f8f1ff",
            labelcolor="#f8f1ff",
            labelsize=10,
        )
        self.axis.tick_params(axis="x", colors="#48e3ff")
        self.axis.tick_params(axis="y", colors="#ff9de2")
        self.axis.grid(False)
        self.axis.yaxis.grid(True, color="#48e3ff", linewidth=1.1, linestyle="-", alpha=0.34)
        self.axis.xaxis.grid(True, color="#ff5ac8", linewidth=0.8, linestyle="--", alpha=0.30)
