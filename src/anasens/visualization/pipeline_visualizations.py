"""Visualization generators for the native analysis pipeline.

This module contains renderer-level plotting routines used by
``NativeAnalysisPipeline``. Each renderer accepts an output directory and
emits a deterministic filename so visual artifacts can be collected by
automation and compared across runs.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde, pearsonr
import seaborn as sns

from anasens.domain import Measurement
from anasens.visualization.plots import save_plot


def _make_pretty(axis, x_title: str | None = None, y_title: str | None = None, title: str | None = None) -> None:
    """Apply the legacy light-background plot styling used by the old scripts."""
    axis.set_facecolor("whitesmoke")
    axis.grid(True, color="#bdbdbd", linewidth=0.8, alpha=0.5)
    axis.set_axisbelow(True)

    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    axis.tick_params(axis="both", labelsize=10)
    if x_title is not None:
        axis.set_xlabel(x_title)
    if y_title is not None:
        axis.set_ylabel(y_title)
    if title is not None:
        axis.set_title(title)


def _add_correlation(x, y, **kwargs) -> None:
    """Annotate a Pearson correlation coefficient on the active axis."""
    axis = plt.gca()
    if len(x) < 2 or len(y) < 2:
        return

    r_value, _ = pearsonr(x, y)
    axis.annotate(
        f"r = {r_value:.2f}",
        xy=(0.5, 0.9),
        xycoords=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def _order_column(extracted_table) -> str | None:
    """Select the preferred x-axis ordering column for drift plots.

    :param extracted_table: Extracted parameter table.
    :type extracted_table: pd.DataFrame
    :return: Best available ordering column name.
    :rtype: str | None
    """
    if "Scan Order" in extracted_table.columns:
        return "Scan Order"
    if "Timestamp" in extracted_table.columns:
        return "Timestamp"
    return None


def _safe_name(value: str) -> str:
    """Convert free-form labels/titles into filename-safe tokens.

    :param value: Raw label or measurement name.
    :type value: str
    :return: Sanitized underscore-delimited name.
    :rtype: str
    """
    return "_".join(part for part in value.replace("/", " ").replace("\\", " ").split() if part) or "measurement"


def _measurement_title(measurement: Measurement) -> str:
    """Return the first non-empty measurement title found in sweep qualifiers.

    :param measurement: Measurement payload containing sweeps.
    :type measurement: Measurement
    :return: Human-readable title fallback chain.
    :rtype: str
    """
    for sweep in measurement.sweeps:
        title = str(sweep.qualifiers.get("MeasurementTitle", "")).strip()
        if title:
            return title
    return measurement.measurement_name


def render_pairplot(extracted_table, output_dir: Path, measurement_index: int = 0) -> None:
    """Render the extracted-parameter pairplot.

    :param extracted_table: Extracted parameter table.
    :type extracted_table: pd.DataFrame
    :param output_dir: Folder where the image is written.
    :type output_dir: Path
    :param measurement_index: Filename tag identifying measurement index.
    :type measurement_index: int
    :return: None
    :rtype: None
    """
    # Choose a compact, high-signal subset first so plots remain readable.
    numeric_columns = extracted_table.select_dtypes(include=["number"]).columns.tolist()
    selected = [name for name in ["peak_potentials", "peak_heights", "Scan Order", "Timestamp"] if name in numeric_columns]
    if not selected:
        selected = numeric_columns[: min(4, len(numeric_columns))]

    if not selected:
        return

    # Carry label groups into the pairplot when available for colored grouping.
    plot_data = extracted_table[selected].copy()
    hue_column = "Label" if "Label" in extracted_table.columns else None
    if hue_column is not None:
        plot_data[hue_column] = extracted_table[hue_column]

    # Grouped branch: custom PairGrid so we can mix KDE/scatter/correlation.
    if hue_column is not None:
        hue_categories = plot_data[hue_column].dropna().unique().tolist()
        palette = sns.color_palette(n_colors=max(len(hue_categories), 1))
        pair_grid = sns.PairGrid(plot_data, vars=selected, hue=hue_column, palette=palette)
        pair_grid.map_lower(sns.scatterplot, alpha=0.7, edgecolor="none")
        pair_grid.map_lower(sns.kdeplot, levels=5, alpha=0.3, linewidths=1)
        pair_grid.map_upper(sns.scatterplot, alpha=0.7, edgecolor="none")
        pair_grid.map_upper(_add_correlation)
        pair_grid.map_diag(sns.histplot, alpha=0.7, edgecolor="none", kde=True)
        pair_grid.fig.suptitle(f"Pairplot with {hue_column} as hue", y=1.02)
        pair_grid.add_legend(title=hue_column)
        pair_grid.fig.tight_layout()
        tagged_filename = f"{measurement_index}_extracted_params_pairplot.png"
        save_plot(pair_grid.fig, output_dir, tagged_filename)
        plt.close(pair_grid.fig)
        return

    # Fallback branch: plain pairplot when no grouping label is present.
    pairplot = sns.pairplot(plot_data, vars=selected)
    pairplot.fig.suptitle("Pairplot with extracted parameters", y=1.02)
    pairplot.fig.tight_layout()
    tagged_filename = f"{measurement_index}_extracted_params_pairplot.png"
    save_plot(pairplot.fig, output_dir, tagged_filename)
    plt.close(pairplot.fig)


def _grouped_peak_arrays(extracted_table):
    """Return label-grouped table slices with robust fallbacks.

    :param extracted_table: Extracted parameter table.
    :type extracted_table: pd.DataFrame
    :return: Sequence of ``(group_name, sub_table)`` tuples.
    :rtype: list[tuple[str, pd.DataFrame]]
    """
    label_column = "Label" if "Label" in extracted_table.columns else None
    if label_column is None:
        return [("Group 0", extracted_table)]

    groups = extracted_table[label_column].dropna().unique().tolist()
    if not groups:
        return [("Group 0", extracted_table)]

    return [(group, extracted_table[extracted_table[label_column] == group]) for group in groups]


def _create_kde_scatterplot(x, y, labels, colors=None, alpha: float = 0.7, figsize=(16.10, 10), point_size: float = 150):
    """Create grouped scatter + KDE contour plot.

    :param x: Peak potential values.
    :type x: np.ndarray | list[float]
    :param y: Peak height values.
    :type y: np.ndarray | list[float]
    :param labels: Group labels aligned with x/y arrays.
    :type labels: np.ndarray | list[str]
    :param colors: Optional color sequence.
    :type colors: list | None
    :param alpha: Marker alpha for scatter points.
    :type alpha: float
    :param figsize: Figure size in inches.
    :type figsize: tuple[float, float]
    :param point_size: Scatter marker size.
    :type point_size: float
    :return: Figure and axis containing rendered plot.
    :rtype: tuple[plt.Figure, plt.Axes]
    """
    # Normalize all incoming vectors to ndarray so masking/slicing is uniform.
    x = np.asarray(x)
    y = np.asarray(y)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 1:
        raise ValueError("Need at least one group")

    if colors is None:
        colors = sns.color_palette("bright", n_colors=len(unique_labels))

    fig, axis = plt.subplots(figsize=figsize)

    # Plot each label group independently so we can attach per-group contours.
    for index, group in enumerate(unique_labels):
        mask = labels == group
        x_vals = x[mask]
        y_vals = y[mask]

        axis.scatter(x_vals, y_vals, color=colors[index], label=group, alpha=alpha, s=point_size)

        # KDE overlays are best-effort; very small or singular groups may fail.
        if len(x_vals) > 1:
            try:
                kde = gaussian_kde(np.vstack([x_vals, y_vals]))
                x_buffer = 0.2 * (x_vals.max() - x_vals.min()) if x_vals.max() != x_vals.min() else 1e-6
                y_buffer = 0.2 * (y_vals.max() - y_vals.min()) if y_vals.max() != y_vals.min() else 1e-6
                xx, yy = np.mgrid[
                    x_vals.min() - x_buffer : x_vals.max() + x_buffer : 100j,
                    y_vals.min() - y_buffer : y_vals.max() + y_buffer : 100j,
                ]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = np.reshape(kde(positions).T, xx.shape)
                contours = axis.contour(xx, yy, density, levels=2, colors=[colors[index]], linewidths=1.5)
                axis.clabel(contours, inline=True, fontsize=8, fmt=lambda value: f"{value:.3f}")
            except Exception:
                pass

    axis.set_xlabel("Potential\n(V)")
    axis.set_ylabel("Current\n(nA)")
    axis.legend(title="Groups")
    axis.set_facecolor("whitesmoke")
    _make_pretty(axis)
    fig.tight_layout()
    return fig, axis


def render_peak_coordinate_density(extracted_table, output_dir: Path, measurement_index: int = 0) -> None:
    """Render grouped peak potential/current density view.

    :param extracted_table: Extracted parameter table.
    :type extracted_table: pd.DataFrame
    :param output_dir: Folder where the image is written.
    :type output_dir: Path
    :param measurement_index: Filename tag identifying measurement index.
    :type measurement_index: int
    :return: None
    :rtype: None
    """
    # Guard for partial tables where required numeric columns are absent.
    if "peak_potentials" not in extracted_table.columns or "peak_heights" not in extracted_table.columns:
        return

    grouped = _grouped_peak_arrays(extracted_table)
    x_values = extracted_table["peak_potentials"].to_numpy()
    y_values = extracted_table["peak_heights"].to_numpy()
    labels = extracted_table["Label"].to_numpy() if "Label" in extracted_table.columns else np.array(["Group 0"] * len(extracted_table))

    fig, axis = _create_kde_scatterplot(x_values, y_values, labels)
    _make_pretty(axis, x_title="Potential\n(V)", y_title="Current\n(nA)", title="Peak-Coordinates Density")
    tagged_filename = f"{measurement_index}_peak_coordinates_plot.png"
    save_plot(fig, output_dir, tagged_filename)
    plt.close(fig)


def render_2d_peak_drift(extracted_table, output_dir: Path, measurement_index: int = 0) -> None:
    """Render 2D drift plot of peak potential vs peak height colored by time.

    :param extracted_table: Extracted parameter table.
    :type extracted_table: pd.DataFrame
    :param output_dir: Folder where the image is written.
    :type output_dir: Path
    :param measurement_index: Filename tag identifying measurement index.
    :type measurement_index: int
    :return: None
    :rtype: None
    """
    # Require the core axes and time dimension; otherwise skip rendering.
    required = {"peak_potentials", "peak_heights", "Timestamp"}
    if not required.issubset(extracted_table.columns):
        return

    x_values = extracted_table["peak_potentials"].to_numpy()
    y_values = extracted_table["peak_heights"].to_numpy()
    timestamps = extracted_table["Timestamp"].to_numpy()
    labels = extracted_table["Label"].to_numpy() if "Label" in extracted_table.columns else np.array(["Group 0"] * len(extracted_table))

    unique_labels = np.unique(labels)
    color_maps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys"]

    fig, axis = plt.subplots(figsize=(16.18, 10))

    # Render each label on the same axis using a dedicated colormap ramp.
    for index, group in enumerate(unique_labels):
        mask = labels == group
        x_sub = x_values[mask]
        y_sub = y_values[mask]
        z_sub = timestamps[mask]

        cmap_name = color_maps[index % len(color_maps)]
        cmap = plt.get_cmap(cmap_name)
        truncated_cmap = LinearSegmentedColormap.from_list(
            f"truncated_{cmap_name}", cmap(np.linspace(0.1, 0.9, 256))
        )

        scatter = axis.scatter(
            x_sub,
            y_sub,
            c=z_sub,
            cmap=truncated_cmap,
            edgecolor="black",
            s=150,
            alpha=1.0,
            label=str(group),
        )
        fig.colorbar(scatter, ax=axis).set_label(f"{group}-Timestamp\n(s)" if len(unique_labels) > 1 else "Timestamp\n(s)")

    _make_pretty(axis, x_title="Potential\n(V)", y_title="Current\n(nA)", title="Time-Drift of Peak-Coordinates")
    tagged_filename = f"{measurement_index}_peak_coords_time_drift.png"
    save_plot(fig, output_dir, tagged_filename)
    plt.close(fig)


def render_1d_peak_drift(extracted_table, output_dir: Path, measurement_index: int = 0) -> None:
    """Render dual-axis 1D drift (current and potential) over time.

    :param extracted_table: Extracted parameter table.
    :type extracted_table: pd.DataFrame
    :param output_dir: Folder where the image is written.
    :type output_dir: Path
    :param measurement_index: Filename tag identifying measurement index.
    :type measurement_index: int
    :return: None
    :rtype: None
    """
    # Skip quickly when required numeric columns are unavailable.
    required = {"peak_potentials", "peak_heights", "Timestamp"}
    if not required.issubset(extracted_table.columns):
        return

    time_values = extracted_table["Timestamp"].to_numpy()
    potential_values = extracted_table["peak_potentials"].to_numpy()
    current_values = extracted_table["peak_heights"].to_numpy()
    labels = extracted_table["Label"].to_numpy() if "Label" in extracted_table.columns else np.array(["Group 0"] * len(extracted_table))

    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        return

    y_colors = ["navy", "darkgreen", "purple", "lime"]
    x_colors = ["crimson", "orange", "pink", "dodgerblue"]
    y_markers = ["^", "s", "o"]
    x_markers = ["D", "P", "X"]

    fig, all_axes = plt.subplots(len(unique_labels), 1, figsize=(16.18, 10 * len(unique_labels)), sharex=True)
    if len(unique_labels) == 1:
        all_axes = [all_axes]

    axes_list: list[tuple[plt.Axes, plt.Axes]] = []
    # One subplot per label, each with a twinx axis for potential/current.
    for index, (group, axis) in enumerate(zip(unique_labels, all_axes)):
        mask = labels == group
        t_group = time_values[mask]
        y_group = current_values[mask]
        x_group = potential_values[mask]

        axis1 = axis
        axis1.set_ylabel("Current\n(nA)", color=y_colors[index % len(y_colors)])
        y_plot = axis1.scatter(
            t_group,
            y_group,
            color=y_colors[index % len(y_colors)],
            marker=y_markers[index % len(y_markers)],
            label="Current",
            s=150,
        )
        axis1.tick_params(axis="y", labelcolor=y_colors[index % len(y_colors)])

        axis2 = axis1.twinx()
        axis2.set_ylabel("Potential\n(V)", color=x_colors[index % len(x_colors)])
        x_plot = axis2.scatter(
            t_group,
            x_group,
            color=x_colors[index % len(x_colors)],
            marker=x_markers[index % len(x_markers)],
            label="Potential",
            s=150,
        )
        axis2.tick_params(axis="y", labelcolor=x_colors[index % len(x_colors)])

        if len(unique_labels) > 1:
            axis1.text(0.02, 0.9, f"Group {group}", transform=axis1.transAxes, bbox=dict(facecolor="white", alpha=0.8))

        axis1.legend([y_plot, x_plot], [line.get_label() for line in [y_plot, x_plot]], bbox_to_anchor=(1, 1), loc="upper left")
        _make_pretty(axis1)
        axes_list.append((axis1, axis2))

    axes_list[-1][0].set_xlabel("Timestamp\n(s)")
    fig.suptitle("Individual Time-Drift of Current and Potential")
    fig.tight_layout()
    tagged_filename = f"{measurement_index}_peak_coords_dual_axis_vs_time.png"
    save_plot(fig, output_dir, tagged_filename)
    plt.close(fig)


def render_overlay_scans(measurement: Measurement, output_dir: Path, measurement_index: int = 0) -> None:
    """Render overlayed baseline-corrected scans with tagged peaks.

    :param measurement: Measurement containing processed sweeps.
    :type measurement: Measurement
    :param output_dir: Folder where the image is written.
    :type output_dir: Path
    :param measurement_index: Filename tag identifying measurement index.
    :type measurement_index: int
    :return: None
    :rtype: None
    """
    # Group traces by label so each figure focuses on one analyte/channel.
    grouped_traces: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]] = {}

    for sweep in measurement.sweeps:
        if sweep.processed_frame is None or sweep.extracted_peak_parameters is None:
            continue

        frame = sweep.processed_frame
        if "V" not in frame.columns or "I_corrected" not in frame.columns:
            continue

        label = str(sweep.qualifiers.get("Label", "Group 0"))
        peaks = np.asarray(sweep.extracted_peak_parameters.get("peak_indices", []), dtype=int)
        scan_order = float(sweep.qualifiers.get("Timestamp", sweep.qualifiers.get("ScanOrder", 0.0)))
        grouped_traces.setdefault(label, []).append((frame["V"].to_numpy(), frame["I_corrected"].to_numpy(), peaks, scan_order))

    if not grouped_traces:
        return

    # For each label group: sort by scan order, draw trace + peaks + colorbar.
    figures: list[plt.Figure] = []
    axes: list[plt.Axes] = []
    for label, traces in grouped_traces.items():
        traces.sort(key=lambda item: item[3])
        fig, axis = plt.subplots(figsize=(16.18, 10))
        cmap_range = (0.1, 1.0)
        cmap = plt.get_cmap("Greens")
        time_values = [trace[3] for trace in traces]
        norm_data = Normalize(vmin=min(time_values), vmax=max(time_values))

        for x_values, y_values, peak_indices, scan_order in traces:
            normalized_time = norm_data(scan_order)
            scaled_time = cmap_range[0] + (cmap_range[1] - cmap_range[0]) * normalized_time
            color = cmap(scaled_time)
            axis.scatter(x_values, y_values, color=color, label="_nolegend_", s=120)
            axis.plot(x_values, y_values, color=color)
            for peak_index in peak_indices:
                if 0 <= peak_index < len(x_values):
                    axis.scatter(x_values[peak_index], y_values[peak_index], marker="x", c="red", s=500, linewidths=2, label="_nolegend_")

        subset_cmap = ListedColormap(cmap(np.linspace(cmap_range[0], cmap_range[1], 256)))
        sm = ScalarMappable(norm=norm_data, cmap=subset_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axis)
        cbar.set_label("Time (arbitrary units)")

        axis.scatter([], [], marker="x", c="red", s=500, linewidths=2, label=f"{label} Peaks")
        axis.legend(loc="upper left", bbox_to_anchor=(0.05, -0.1), fontsize=16)
        _make_pretty(
            axis,
            x_title="Potential\n(V)",
            y_title="Current\n(nA)",
            title=f"Overlayed Scans with Time as Colour-Factor\n(Label : {label})",
        )
        tagged_filename = f"{measurement_index}_{_safe_name(label)}_overlay_scans_tagged_peaks.png"
        save_plot(fig, output_dir, tagged_filename)
        figures.append(fig)
        axes.append(axis)
        plt.close(fig)


def render_preprocessing_overview(measurement: Measurement, output_dir: Path, measurement_index: int = 0) -> None:
    """Render side-by-side preprocessing diagnostic plots.

    Left subplot overlays ``I_processing``, ``I_smoothed``, and the baseline
    estimate for every sweep in the measurement, coloured by scan order.
    Right subplot shows the baseline-subtracted ``I_corrected`` trace for the
    same sweeps.  A shared colorbar encodes scan-order (timestamp).

    :param measurement: Measurement with processed sweeps.
    :type measurement: Measurement
    :param output_dir: Folder where the image is written.
    :type output_dir: Path
    :param measurement_index: Filename tag identifying measurement index.
    :type measurement_index: int
    :return: None
    :rtype: None
    """
    # Collect sweeps that have been fully preprocessed.
    sweeps_data: list[tuple[object, float]] = []
    for sweep in measurement.sweeps:
        if sweep.processed_frame is None:
            continue
        frame = sweep.processed_frame
        required = {"V", "I_processing", "I_smoothed", "baseline", "I_corrected"}
        if not required.issubset(frame.columns):
            continue
        scan_order = float(sweep.qualifiers.get("Timestamp", sweep.qualifiers.get("ScanOrder", 0.0)))
        sweeps_data.append((frame, scan_order))

    if not sweeps_data:
        return

    # Sort chronologically so the colormap encodes actual scan progression.
    sweeps_data.sort(key=lambda item: item[1])
    time_values = [s[1] for s in sweeps_data]
    norm = Normalize(vmin=min(time_values), vmax=max(time_values))
    cmap = plt.get_cmap("viridis")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7), layout="constrained")

    for frame, scan_order in sweeps_data:
        color = cmap(norm(scan_order))
        v = frame["V"].to_numpy(dtype=float)
        # Raw sign-normalized trace: thin and semi-transparent for context.
        ax_left.plot(v, frame["I_processing"].to_numpy(dtype=float),
                     color=color, linewidth=0.7, alpha=0.35, linestyle="-")
        # Smoothed trace: solid, more prominent.
        ax_left.plot(v, frame["I_smoothed"].to_numpy(dtype=float),
                     color=color, linewidth=1.4, alpha=0.80, linestyle="-")
        # Baseline estimate: dashed so it is visually distinct from the signal.
        ax_left.plot(v, frame["baseline"].to_numpy(dtype=float),
                     color=color, linewidth=1.4, alpha=0.80, linestyle="--")
        # Baseline-subtracted result on the right panel.
        ax_right.plot(v, frame["I_corrected"].to_numpy(dtype=float),
                      color=color, linewidth=1.4, alpha=0.85, linestyle="-")

    # Legend uses proxy handles to show linestyle conventions without repeating
    # sweep-level colors, which are already encoded in the colorbar.
    legend_handles = [
        Line2D([0], [0], color="dimgray", linewidth=0.7, alpha=0.6,
               linestyle="-", label="I processing (sign-normalised)"),
        Line2D([0], [0], color="dimgray", linewidth=1.4,
               linestyle="-", label="I smoothed"),
        Line2D([0], [0], color="dimgray", linewidth=1.4,
               linestyle="--", label="Baseline estimate"),
    ]
    ax_left.legend(handles=legend_handles, loc="best", fontsize=10,
                   framealpha=0.92, edgecolor="#cccccc")

    # Shared colorbar encodes scan progression across both panels.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_left, ax_right], shrink=0.82, pad=0.02)
    cbar.set_label("Scan order (timestamp)", fontsize=11)

    _make_pretty(ax_left,
                 x_title="Potential (V)",
                 y_title="Current (A)",
                 title="Preprocessing: Raw / Smoothed / Baseline")
    _make_pretty(ax_right,
                 x_title="Potential (V)",
                 y_title="Current (A)",
                 title="Baseline-Subtracted Voltammogram")

    fig.suptitle(f"Preprocessing Overview — Measurement {measurement_index}",
                 fontsize=14, fontweight="bold")

    tagged_filename = f"{measurement_index}_preprocessing_overview.png"
    save_plot(fig, output_dir, tagged_filename)
    plt.close(fig)
