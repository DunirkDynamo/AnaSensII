"""Plot rendering helpers for Plotter Fitter."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import patheffects as pe
from PySide6.QtWidgets import QVBoxLayout, QWidget

from anasens.uncertainty.bands import ConfidenceBand, PredictionBand

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

        # Shared color tokens keep confidence/prediction overlays aligned with
        # the rest of the curve-builder synthwave palette.
        self._colors = {
            "data_face": "#48e3ff",
            "data_edge": "#ffd3f7",
            "fit": "#ff5ac8",
            "fit_glow": "#ff9de2",
            "confidence_fill": "#3ed6ff",
            "confidence_edge": "#9eeeff",
            "prediction_fill": "#8b6dff",
            "prediction_edge": "#cdbbff",
            "text": "#f8f1ff",
        }

        self._confidence_styles = {
            "delta": {
                "fill": "#3ed6ff",
                "edge": "#9eeeff",
                "linestyle": "-",
                "alpha": 0.18,
            },
            "monte_carlo": {
                "fill": "#ff5ac8",
                "edge": "#ffd3f7",
                "linestyle": "--",
                "alpha": 0.14,
            },
            "profile_likelihood": {
                "fill": "#ffe27a",
                "edge": "#fff0b3",
                "linestyle": ":",
                "alpha": 0.13,
            },
        }

        self._prediction_styles = {
            "exact_analytic": {
                "fill": "#8b6dff",
                "edge": "#cdbbff",
                "linestyle": "--",
                "alpha": 0.16,
            },
            "gaussian_analytic": {
                "fill": "#8b6dff",
                "edge": "#cdbbff",
                "linestyle": "--",
                "alpha": 0.16,
            },
            "delta": {
                "fill": "#2ed4aa",
                "edge": "#b8ffe8",
                "linestyle": "-.",
                "alpha": 0.14,
            },
            "monte_carlo": {
                "fill": "#ff7a59",
                "edge": "#ffd3c4",
                "linestyle": ":",
                "alpha": 0.12,
            },
            "quantile_regression": {
                "fill": "#83d968",
                "edge": "#dff7cf",
                "linestyle": "-",
                "alpha": 0.10,
            },
        }

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
            color=self._colors["data_face"],
            edgecolors=self._colors["data_edge"],
            linewidths=1.0,
            alpha=0.98,
            zorder=3,
            label="Data",
        )

        self.axis.set_xlim(x_min, x_max)
        self.axis.set_ylim(y_min, y_max)

        self.axis.set_xlabel(x_label, fontsize=12, fontweight="bold", color=self._colors["text"])
        self.axis.set_ylabel(y_label, fontsize=12, fontweight="bold", color=self._colors["text"])
        self._style_legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def overlay_fit(self, fitting_result: FittingResult) -> None:
        """Overlay a fitted curve on the existing plot.

        Args:
            fitting_result: Result object containing the fitted curve data.
        """
        fit_line, = self.axis.plot(
            fitting_result.fitted_x,
            fitting_result.fitted_y,
            color=self._colors["fit"],
            linewidth=3.0,
            zorder=4,
            label=fitting_result.model_name,
        )
        fit_line.set_path_effects(
            [
                pe.Stroke(linewidth=6.5, foreground=self._colors["fit_glow"], alpha=0.18),
                pe.Normal(),
            ]
        )
        self._style_legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def overlay_bands(
        self,
        confidence_band: ConfidenceBand | None,
        prediction_band: PredictionBand | None,
    ) -> None:
        """Overlay confidence and prediction bands on the active plot."""
        if prediction_band is not None:
            self.overlay_prediction_band(prediction_band)

        if confidence_band is not None:
            self.overlay_confidence_bands({confidence_band.method: confidence_band})

        self._style_legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def overlay_prediction_band(self, prediction_band: PredictionBand, method_key: str | None = None) -> None:
        """Overlay one prediction band on the active plot."""
        if prediction_band is not None:
            method = self._normalize_method_key(method_key or prediction_band.method)
            style = self._prediction_styles.get(
                method,
                {
                    "fill": self._colors["prediction_fill"],
                    "edge": self._colors["prediction_edge"],
                    "linestyle": "--",
                    "alpha": 0.16,
                },
            )
            self.axis.plot(
                prediction_band.x_grid,
                prediction_band.lower_values,
                color=style["edge"],
                linewidth=1.35,
                alpha=0.9,
                linestyle="--",
                zorder=2.4,
                label=f"{self._prediction_method_label(method)} ({int((1.0 - prediction_band.alpha) * 100)}% PI)",
            )
            self.axis.plot(
                prediction_band.x_grid,
                prediction_band.upper_values,
                color=style["edge"],
                linewidth=1.35,
                alpha=0.9,
                linestyle="--",
                zorder=2.4,
                label="_nolegend_",
            )

    def overlay_prediction_bands(self, prediction_bands: dict[str, PredictionBand]) -> None:
        """Overlay one or more prediction bands on the active plot."""
        for method_key, prediction_band in prediction_bands.items():
            self.overlay_prediction_band(prediction_band, method_key=method_key)

    def overlay_profile_prediction_map(self, profile_map: dict[str, np.ndarray]) -> None:
        """Overlay modified profile-likelihood PI map as x-bounds over y-grid."""
        y_grid = profile_map["y_grid"]
        x_mle = profile_map["x_mle"]
        x_lower = profile_map["x_lower"]
        x_upper = profile_map["x_upper"]

        self.axis.plot(
            x_mle,
            y_grid,
            color="#ffe27a",
            linewidth=1.6,
            alpha=0.9,
            label="Profile Likelihood Modified (MLE)",
            zorder=2.5,
        )
        self.axis.plot(
            x_lower,
            y_grid,
            color="#fff0b3",
            linewidth=1.1,
            linestyle="--",
            alpha=0.85,
            label="_nolegend_",
            zorder=2.45,
        )
        self.axis.plot(
            x_upper,
            y_grid,
            color="#fff0b3",
            linewidth=1.1,
            linestyle="--",
            alpha=0.85,
            label="Profile Likelihood Modified",
            zorder=2.45,
        )
        self.axis.fill_betweenx(
            y_grid,
            x_lower,
            x_upper,
            color="#ffe27a",
            alpha=0.06,
            zorder=2.35,
            label="_nolegend_",
        )

    def overlay_confidence_bands(self, confidence_bands: dict[str, ConfidenceBand]) -> None:
        """Overlay one or more confidence bands on the active plot."""
        for index, (method_key, confidence_band) in enumerate(confidence_bands.items()):
            style = self._confidence_styles.get(
                self._normalize_method_key(method_key),
                {
                    "fill": "#48e3ff",
                    "edge": "#f8f1ff",
                    "linestyle": "-.",
                    "alpha": 0.14,
                },
            )
            zorder = 2.0 + index * 0.08
            label = f"{self._confidence_method_label(method_key)} ({int((1.0 - confidence_band.alpha) * 100)}% CI)"

            self.axis.fill_between(
                confidence_band.x_grid,
                confidence_band.lower_values,
                confidence_band.upper_values,
                color=style["fill"],
                alpha=style["alpha"],
                zorder=zorder,
                label=label,
            )
            self.axis.plot(
                confidence_band.x_grid,
                confidence_band.lower_values,
                color=style["edge"],
                linewidth=1.2,
                alpha=0.8,
                linestyle=style["linestyle"],
                zorder=zorder + 0.1,
                label="_nolegend_",
            )
            self.axis.plot(
                confidence_band.x_grid,
                confidence_band.upper_values,
                color=style["edge"],
                linewidth=1.2,
                alpha=0.8,
                linestyle=style["linestyle"],
                zorder=zorder + 0.1,
                label="_nolegend_",
            )

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

    def _normalize_method_key(self, method_key: str) -> str:
        return method_key.strip().lower()

    def _confidence_method_label(self, method_key: str) -> str:
        normalized_key = self._normalize_method_key(method_key)
        if normalized_key.startswith("delta"):
            return "Delta Method"
        if normalized_key == "monte_carlo":
            return "Monte Carlo"
        if normalized_key == "profile_likelihood":
            return "Profile Likelihood"
        return method_key.replace("_", " ").title()

    def _prediction_method_label(self, method_key: str) -> str:
        normalized_key = self._normalize_method_key(method_key)
        if normalized_key in {"exact_analytic", "gaussian_analytic"}:
            return "Exact Analytical"
        if normalized_key == "delta":
            return "Delta Method"
        if normalized_key == "monte_carlo":
            return "Monte Carlo"
        if normalized_key == "quantile_regression":
            return "Quantile Regression"
        if normalized_key == "profile_likelihood_modified":
            return "Profile Likelihood Modified"
        return method_key.replace("_", " ").title()

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
