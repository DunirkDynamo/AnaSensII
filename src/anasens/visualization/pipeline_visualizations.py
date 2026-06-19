"""Visualization generators for the native analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from anasens.domain import Measurement
from anasens.visualization.plots import save_plot


def render_pairplot(extracted_table, output_dir: Path) -> None:
    """Render pairplot for selected numeric columns with label hue."""
    numeric_columns = extracted_table.select_dtypes(include=["number"]).columns.tolist()
    selected = [name for name in ["peak_potentials", "peak_heights", "Timestamp"] if name in numeric_columns]
    if not selected:
        selected = numeric_columns[: min(4, len(numeric_columns))]

    if not selected:
        return

    hue_column = "Label" if "Label" in extracted_table.columns else None
    pairplot = sns.pairplot(extracted_table, vars=selected, hue=hue_column, corner=True)
    save_plot(pairplot.fig, output_dir, "extracted_params_pairplot.png")
    plt.close(pairplot.fig)


def render_peak_coordinate_density(extracted_table, output_dir: Path) -> None:
    """Render peak potential/current density scatter plot."""
    if "peak_potentials" not in extracted_table.columns or "peak_heights" not in extracted_table.columns:
        return

    fig, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=extracted_table, x="peak_potentials", y="peak_heights", hue="Label", ax=axis)
    axis.set_xlabel("Potential\n(V)")
    axis.set_ylabel("Current\n(nA)")
    axis.set_title("Peak-Coordinates Density")
    save_plot(fig, output_dir, "peak_coordinates_plot.png")
    plt.close(fig)


def render_2d_peak_drift(extracted_table, output_dir: Path) -> None:
    """Render timestamp-colored 2D peak drift plot."""
    required = {"peak_potentials", "peak_heights", "Timestamp"}
    if not required.issubset(extracted_table.columns):
        return

    fig, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        extracted_table["peak_potentials"],
        extracted_table["peak_heights"],
        c=extracted_table["Timestamp"],
        cmap="viridis",
        alpha=0.8,
    )
    axis.set_xlabel("Potential\n(V)")
    axis.set_ylabel("Current\n(nA)")
    axis.set_title("Time-Drift of Peak-Coordinates")
    fig.colorbar(scatter, ax=axis, label="Timestamp\n(s)")
    save_plot(fig, output_dir, "peak_coords_time_drift.png")
    plt.close(fig)


def render_1d_peak_drift(extracted_table, output_dir: Path) -> None:
    """Render dual-axis drift of peak coordinates vs time."""
    required = {"peak_potentials", "peak_heights", "Timestamp"}
    if not required.issubset(extracted_table.columns):
        return

    fig, axis1 = plt.subplots(figsize=(9, 6))
    axis2 = axis1.twinx()

    axis1.plot(extracted_table["Timestamp"], extracted_table["peak_potentials"], color="#5AA9E6", label="Potential")
    axis2.plot(extracted_table["Timestamp"], extracted_table["peak_heights"], color="#FF6392", label="Current")

    axis1.set_xlabel("Timestamp\n(s)")
    axis1.set_ylabel("Potential\n(V)", color="#5AA9E6")
    axis2.set_ylabel("Current\n(nA)", color="#FF6392")
    axis1.set_title("Peak Coordinates vs Time")
    save_plot(fig, output_dir, "peak_coords_dual_axis_vs_time.png")
    plt.close(fig)


def render_overlay_scans(measurement: Measurement, output_dir: Path) -> None:
    """Render overlayed processed scans grouped by label."""
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    for sweep in measurement.sweeps:
        if sweep.processed_frame is None or sweep.extracted_peak_parameters is None:
            continue

        frame = sweep.processed_frame
        if "V" not in frame.columns or "I_corrected" not in frame.columns:
            continue

        label = str(sweep.qualifiers.get("Label", "Unknown"))
        peaks = np.asarray(sweep.extracted_peak_parameters.get("peak_indices", []), dtype=int)
        grouped.setdefault(label, []).append(
            (
                frame["V"].to_numpy(),
                frame["I_corrected"].to_numpy(),
                peaks,
                np.asarray([float(sweep.qualifiers.get("Timestamp", 0.0))]),
            )
        )

    for label, traces in grouped.items():
        fig, axis = plt.subplots(figsize=(9, 6))
        for index, (x_values, y_values, peak_indices, timestamp) in enumerate(traces):
            line = axis.plot(x_values, y_values, alpha=0.75, label=f"t={timestamp[0]:.1f}s" if index < 10 else "_nolegend_")[0]
            if peak_indices.size > 0:
                valid = peak_indices[(peak_indices >= 0) & (peak_indices < len(x_values))]
                axis.scatter(x_values[valid], y_values[valid], color=line.get_color(), marker="x", s=24)

        axis.set_xlabel("Potential\n(V)")
        axis.set_ylabel("Current\n(nA)")
        axis.set_title(f"Overlayed Scans with Tagged Peaks\n(Label: {label})")
        axis.legend(loc="best")
        safe_label = label.replace(" ", "_")
        save_plot(fig, output_dir, f"{safe_label}_overlay_scans_tagged_peaks.png")
        plt.close(fig)
