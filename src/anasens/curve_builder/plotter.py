"""Plot rendering helpers for Plotter Fitter."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import patheffects as pe
from scipy.stats import hypsecant, laplace, logistic, norm, skewnorm, t
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from anasens.uncertainty.bands import ConfidenceBand, PredictionBand, ProfileLikelihoodBand

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
        # recreating Qt widgets on every plotting interaction. The canvas is
        # intentionally tall so the fit view can occupy the full top portion of
        # the window while the additional diagnostics are reached by vertical
        # scrolling.
        self.figure = Figure(figsize=(11.2, 20.0), dpi=120, facecolor="#0c0820")
        grid = self.figure.add_gridspec(4, 1, height_ratios=(2.2, 2.2, 2.2, 2.2), hspace=0.36)
        self.axis = self.figure.add_subplot(grid[0, :])
        self.residual_axis = self.figure.add_subplot(grid[1, :], sharex=self.axis)
        self.histogram_axis = self.figure.add_subplot(grid[2, :])
        self.estimate_axis = self.figure.add_subplot(grid[3, :])
        self.estimate_axis_twin = self.estimate_axis.twinx()
        self._current_fitting_result: FittingResult | None = None
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(1900)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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

        # The parent widget owns the layout so the plot canvas can stretch with
        # the main window while the renderer stays focused on presentation.
        parent_layout = parent.layout()
        if parent_layout is None:
            parent_layout = QVBoxLayout(parent)
            parent_layout.setContentsMargins(0, 0, 0, 0)

        # Wrap the canvas in a scroll area so the main window can stay compact
        # while still exposing the wider plot figure in full.
        scroll_host = QWidget(parent)
        scroll_layout = QVBoxLayout(scroll_host)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(self.canvas)

        self.plot_scroll_area = QScrollArea(parent)
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.plot_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.plot_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.plot_scroll_area.setWidget(scroll_host)
        parent_layout.addWidget(self.plot_scroll_area)

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

        self._build_axes()

    def plot_data(self, x_values, y_values, x_label: str, y_label: str) -> None:
        """Render the selected x and y columns.

        Args:
            x_values: Numeric x-axis values.
            y_values: Numeric y-axis values.
            x_label: Label displayed on the x-axis.
            y_label: Label displayed on the y-axis.
        """
        # Rebuild the figure so each data selection starts from the standard
        # vertically stacked layout: fit, residuals, histogram, and estimate.
        self._build_axes()

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

        x_gap = 0.05 * (x_max - x_min)
        y_gap = 0.05 * (y_max - y_min)

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

        self.axis.set_xlim(x_min - x_gap, x_max + x_gap)
        self.axis.set_ylim(y_min - y_gap, y_max + y_gap)

        self.axis.set_xlabel(x_label, fontsize=7.9, fontweight="bold", color=self._colors["text"])
        self.axis.set_ylabel(y_label, fontsize=7.9, fontweight="bold", color=self._colors["text"])
        self._style_legend()

        self._render_residual_placeholders(x_label)
        self._render_estimate_limits_placeholder()
        self.canvas.draw_idle()

    def plot_residuals(
        self,
        x_values,
        residuals,
        x_label: str,
        error_model_name: str | None = None,
        error_model_parameters: dict[str, float] | None = None,
    ) -> None:
        """Render residual diagnostics below the main fit plot."""
        x_numeric = np.asarray(x_values, dtype=float)
        residual_numeric = np.asarray(residuals, dtype=float)

        self.residual_axis.clear()
        self.histogram_axis.clear()
        self._apply_base_style(self.residual_axis)
        self._apply_base_style(self.histogram_axis)

        self.residual_axis.scatter(
            x_numeric,
            residual_numeric,
            s=36,
            color=self._colors["data_face"],
            edgecolors=self._colors["data_edge"],
            linewidths=0.9,
            alpha=0.96,
            zorder=3,
        )
        self.residual_axis.axhline(0.0, color=self._colors["fit"], linewidth=1.4, linestyle="--", alpha=0.85, zorder=2)
        self.residual_axis.set_title("Residuals vs x", fontsize=7.1, color=self._colors["text"], pad=8)
        self.residual_axis.set_xlabel(x_label, fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.residual_axis.set_ylabel("Residual", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.residual_axis.grid(True, color="#48e3ff", linewidth=0.8, linestyle="--", alpha=0.20)

        bins = min(max(int(np.sqrt(residual_numeric.size)) or 8, 8), 24)
        _, edges, _ = self.histogram_axis.hist(
            residual_numeric,
            bins=bins,
            density=True,
            color=self._colors["fit"],
            edgecolor=self._colors["data_edge"],
            alpha=0.82,
            zorder=3,
        )
        self.histogram_axis.axvline(0.0, color=self._colors["data_face"], linewidth=1.4, linestyle="--", alpha=0.85, zorder=2)
        self.histogram_axis.set_title("Residual histogram", fontsize=7.1, color=self._colors["text"], pad=8)
        self.histogram_axis.set_xlabel("Residual", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.histogram_axis.set_ylabel("Density", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.histogram_axis.grid(True, color="#ff5ac8", linewidth=0.8, linestyle="--", alpha=0.18)

        if error_model_name and error_model_parameters:
            overlay_x = np.linspace(float(np.min(residual_numeric)), float(np.max(residual_numeric)), 400)
            overlay_pdf = self._error_model_pdf(overlay_x, error_model_name, error_model_parameters)
            if overlay_pdf is not None and edges.size > 1:
                self.histogram_axis.plot(
                    overlay_x,
                    overlay_pdf,
                    color="#ffffff",
                    linewidth=2.0,
                    linestyle="-",
                    zorder=4,
                    label=f"{error_model_name.strip().replace('_', ' ').title()} fit",
                )
                legend = self.histogram_axis.legend(
                    loc="upper right",
                    fontsize=8.5,
                    frameon=True,
                    facecolor="#130826",
                    framealpha=0.9,
                )
                for text in legend.get_texts():
                    text.set_color("white")

        self.canvas.draw_idle()

    def plot_estimate_limits_band(self, band_data: dict[str, np.ndarray]) -> None:
        """Render estimate limits and resolution against the MLE estimate axis.

        Args:
            band_data: Dictionary returned by StandardCurveModel.inverse_band_curve,
                containing x/y sweep points plus the inverted lower/upper edges.
        """
        x_mle = np.asarray(band_data["x_mle"], dtype=float)
        x_lower = np.asarray(band_data["x_lower"], dtype=float)
        x_upper = np.asarray(band_data["x_upper"], dtype=float)
        resolution = np.asarray(band_data["resolution"], dtype=float)
        valid_mask = np.isfinite(x_mle) & np.isfinite(x_lower) & np.isfinite(x_upper) & np.isfinite(resolution)

        self.estimate_axis.clear()
        self.estimate_axis_twin.clear()
        self._apply_base_style(self.estimate_axis)
        self._apply_base_style(self.estimate_axis_twin)

        self.estimate_axis_twin.spines["right"].set_color("#f8f1ff")
        self.estimate_axis_twin.spines["left"].set_visible(False)
        self.estimate_axis_twin.yaxis.set_label_position("right")
        self.estimate_axis_twin.yaxis.tick_right()
        self.estimate_axis_twin.tick_params(axis="y", colors="#ff9de2")
        self.estimate_axis_twin.grid(False)

        if np.any(valid_mask):
            x_estimate = x_mle[valid_mask]
            lower_values = x_lower[valid_mask]
            upper_values = x_upper[valid_mask]
            resolution_values = resolution[valid_mask]
            order = np.argsort(x_estimate)
            x_estimate = x_estimate[order]
            lower_values = lower_values[order]
            upper_values = upper_values[order]
            resolution_values = resolution_values[order]

            self.estimate_axis.plot(
                x_estimate,
                lower_values,
                color="#2f3b4c",
                linestyle="--",
                linewidth=1.9,
                label=r"Confidence Band Edge for $\hat{x}_0$",
                zorder=3.5,
            )
            self.estimate_axis.plot(
                x_estimate,
                upper_values,
                color="#2f3b4c",
                linestyle="--",
                linewidth=1.9,
                label="_nolegend_",
                zorder=3.5,
            )
            self.estimate_axis.fill_between(
                x_estimate,
                lower_values,
                upper_values,
                color="#1f77b4",
                alpha=0.14,
                label="Estimate limit band",
                zorder=2.8,
            )
            self.estimate_axis_twin.plot(
                x_estimate,
                resolution_values,
                color="darkmagenta",
                linewidth=2.8,
                label=r"Resolution: $\frac{x_0^{\mathrm{upper}} - x_0^{\mathrm{lower}}}{\hat{x}_0}$",
                zorder=5,
            )
            resolution_min = float(np.min(resolution_values))
            resolution_max = float(np.max(resolution_values))
            resolution_gap = max(0.05 * (resolution_max - resolution_min), 1e-9)
            self.estimate_axis_twin.set_ylim(resolution_min - resolution_gap, resolution_max + resolution_gap)

        self.estimate_axis.set_title("Estimate Limits and Resolution", fontsize=11.2, pad=12, fontweight="semibold", color=self._colors["text"])
        self.estimate_axis.set_xlabel(r"Hypothetical ($\hat{x}$, $y_{obs}$) Pairs", fontsize=9.8, labelpad=8, color=self._colors["text"])
        self.estimate_axis.set_ylabel(r"Limit on Estimate ($x_0$)", fontsize=9.8, labelpad=8, color=self._colors["text"])
        self.estimate_axis.minorticks_on()
        self.estimate_axis.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18)
        self.estimate_axis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.12)
        self.estimate_axis_twin.set_ylabel("Resolution", fontsize=9.8, labelpad=8, color=self._colors["text"])
        self.estimate_axis_twin.minorticks_on()

        handles_left, labels_left = self.estimate_axis.get_legend_handles_labels()
        handles_right, labels_right = self.estimate_axis_twin.get_legend_handles_labels()
        self.estimate_axis.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            loc="upper left",
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            fontsize=10,
            borderpad=0.7,
        )

        # The estimate panel is redrawn after fitting, so custom tick labels
        # are reapplied here to persist the (x, y) display on the bottom axis.
        if self._current_fitting_result is not None:
            self._set_fitted_tick_labels(self._current_fitting_result)

        self.canvas.draw_idle()

    def _error_model_pdf(
        self,
        residuals: np.ndarray,
        error_model_name: str,
        error_model_parameters: dict[str, float],
    ) -> np.ndarray | None:
        """Evaluate the selected residual density for histogram overlay."""
        model_key = error_model_name.strip().lower()
        params = {key: float(value) for key, value in error_model_parameters.items()}

        if model_key == "gaussian" and "sigma" in params:
            return norm.pdf(residuals, loc=0.0, scale=max(params["sigma"], 1e-12))
        if model_key == "laplace" and "b" in params:
            return laplace.pdf(residuals, loc=0.0, scale=max(params["b"], 1e-12))
        if model_key == "logistic" and "s" in params:
            return logistic.pdf(residuals, loc=0.0, scale=max(params["s"], 1e-12))
        if model_key == "student_t" and {"df", "scale"}.issubset(params):
            return t.pdf(residuals, df=max(params["df"], 1e-12), loc=0.0, scale=max(params["scale"], 1e-12))
        if model_key == "hyperbolic_secant" and "scale" in params:
            return hypsecant.pdf(residuals, loc=0.0, scale=max(params["scale"], 1e-12))
        if model_key == "skew_normal" and {"a", "scale"}.issubset(params):
            return skewnorm.pdf(residuals, a=params["a"], loc=0.0, scale=max(params["scale"], 1e-12))

        return None

    def _set_fitted_tick_labels(self, fitting_result: FittingResult) -> None:
        """Replace the bottom-plot x-axis tick labels with fitted-model x/y pairs."""
        tick_positions = self.estimate_axis.get_xticks()
        if tick_positions.size == 0:
            return

        x_ticks = np.asarray(tick_positions, dtype=float)
        try:
            y_values = np.asarray(fitting_result.model_function(x_ticks, *fitting_result.parameter_values), dtype=float)
        except Exception:
            y_values = np.full_like(x_ticks, np.nan, dtype=float)

        labels = []
        for x_value, y_value in zip(x_ticks, y_values):
            if np.isfinite(y_value):
                labels.append(f"({x_value:.3g}, {y_value:.3g})")
            else:
                labels.append(f"{x_value:.3g}")

        axis = self.estimate_axis
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(labels)
        axis.tick_params(axis="x", labelbottom=True)
        axis.tick_params(axis="x", labelrotation=35)
        for tick_label in axis.get_xticklabels():
            tick_label.set_rotation(35)
            tick_label.set_ha("right")
            tick_label.set_rotation_mode("anchor")

    def overlay_fit(self, fitting_result: FittingResult) -> None:
        """Overlay a fitted curve on the existing plot.

        Args:
            fitting_result: Result object containing the fitted curve data.
        """
        self._current_fitting_result = fitting_result
        fit_line, = self.axis.plot(
            fitting_result.fitted_x,
            fitting_result.fitted_y,
            color=self._colors["fit"],
            linewidth=2.2,
            zorder=4,
            label=fitting_result.model_name,
        )
        fit_line.set_path_effects(
            [
                pe.Stroke(linewidth=4.8, foreground=self._colors["fit_glow"], alpha=0.18),
                pe.Normal(),
            ]
        )
        self._set_fitted_tick_labels(fitting_result)
        self._style_legend()
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

    def overlay_profile_prediction_map(self, profile_map: ProfileLikelihoodBand) -> None:
        """Overlay modified profile-likelihood PI map as x-bounds over y-grid."""
        y_grid = profile_map.y_grid
        x_mle = profile_map.x_mle
        x_lower = profile_map.x_lower
        x_upper = profile_map.x_upper

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
        y_padding = max((y_max - y_min) * 0.05, 1e-9)
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

    def _build_axes(self) -> None:
        """Reset the figure to the fit, residual, histogram, and estimate layout."""
        self.figure.clear()
        grid = self.figure.add_gridspec(4, 1, height_ratios=(2.2, 2.2, 2.2, 2.2), hspace=0.36)
        self.axis = self.figure.add_subplot(grid[0, :])
        self.residual_axis = self.figure.add_subplot(grid[1, :], sharex=self.axis)
        self.histogram_axis = self.figure.add_subplot(grid[2, :])
        self.estimate_axis = self.figure.add_subplot(grid[3, :])
        self.estimate_axis_twin = self.estimate_axis.twinx()

        # Reserve more space at the right edge so the estimate panel's twin
        # y-axis label and tick labels are not clipped by the frame boundary.
        self.figure.subplots_adjust(left=0.075, right=0.915, top=0.955, bottom=0.075)

        self._apply_base_style(self.axis)
        self._apply_base_style(self.residual_axis)
        self._apply_base_style(self.histogram_axis)
        self._apply_base_style(self.estimate_axis)

    def _render_residual_placeholders(self, x_label: str) -> None:
        """Show residual panels before a fit has been applied."""
        for axis in (self.residual_axis, self.histogram_axis):
            axis.clear()
            self._apply_base_style(axis)

        self.residual_axis.set_title("Residuals vs x", fontsize=7.1, color=self._colors["text"], pad=8)
        self.residual_axis.set_xlabel(x_label, fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.residual_axis.set_ylabel("Residual", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.residual_axis.text(
            0.5,
            0.52,
            "Residual scatter appears after fitting",
            transform=self.residual_axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=self._colors["text"],
            alpha=0.58,
        )

        self.histogram_axis.set_title("Residual histogram", fontsize=7.1, color=self._colors["text"], pad=8)
        self.histogram_axis.set_xlabel("Residual", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.histogram_axis.set_ylabel("Density", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.histogram_axis.text(
            0.5,
            0.52,
            "Residual histogram appears after fitting",
            transform=self.histogram_axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=self._colors["text"],
            alpha=0.58,
        )

    def _render_estimate_limits_placeholder(self) -> None:
        """Show the estimate-limits panel placeholder before a fit has been applied."""
        self.estimate_axis.clear()
        self.estimate_axis_twin.clear()
        self._apply_base_style(self.estimate_axis)
        self._apply_base_style(self.estimate_axis_twin)
        self.estimate_axis_twin.spines["right"].set_color("#f8f1ff")
        self.estimate_axis_twin.spines["left"].set_visible(False)
        self.estimate_axis_twin.yaxis.set_label_position("right")
        self.estimate_axis_twin.yaxis.tick_right()
        self.estimate_axis_twin.grid(False)

        self.estimate_axis.set_title("Estimate Limits and Resolution", fontsize=7.1, color=self._colors["text"], pad=8)
        self.estimate_axis.set_xlabel(r"Hypothetical ($\hat{x}$, $y_{obs}$) Pairs", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.estimate_axis.set_ylabel(r"Limit on Estimate ($x_0$)", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.estimate_axis.text(
            0.5,
            0.52,
            "Estimate limit band appears after fitting",
            transform=self.estimate_axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=self._colors["text"],
            alpha=0.58,
        )

        self.estimate_axis_twin.set_ylabel("Resolution", fontsize=7.1, fontweight="bold", color=self._colors["text"])
        self.estimate_axis_twin.text(
            0.5,
            0.42,
            "Resolution curve shares the same x-axis",
            transform=self.estimate_axis_twin.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            color=self._colors["text"],
            alpha=0.52,
        )

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
        if normalized_key in {"profile_likelihood", "profile_likelihood_modified"}:
            return "Profile Likelihood"
        return method_key.replace("_", " ").title()

    def _apply_base_style(self, axis=None) -> None:
        """Apply consistent plot styling to the managed axis."""
        # Spine, tick, and grid styling are centralized so every redraw uses the
        # same synthwave visual language requested for the application.
        axis = self.axis if axis is None else axis
        axis.set_facecolor("#140a2f")
        for spine in axis.spines.values():
            spine.set_color("#f8f1ff")
            spine.set_linewidth(2.2)

        axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=6,
            width=1.4,
            color="#f8f1ff",
            labelcolor="#f8f1ff",
            labelsize=10,
        )
        axis.tick_params(axis="x", colors="#48e3ff")
        axis.tick_params(axis="y", colors="#ff9de2")
        axis.grid(False)
        axis.yaxis.grid(True, color="#48e3ff", linewidth=1.1, linestyle="-", alpha=0.34)
        axis.xaxis.grid(True, color="#ff5ac8", linewidth=0.8, linestyle="--", alpha=0.30)
        axis.title.set_color(self._colors["text"])
        axis.xaxis.label.set_color(self._colors["text"])
        axis.yaxis.label.set_color(self._colors["text"])
