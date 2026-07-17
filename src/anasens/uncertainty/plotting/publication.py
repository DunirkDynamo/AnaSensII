"""
Publication-quality plotting.

This module provides a standard plotting routine for visualizing:

- experimental data,
- fitted models,
- confidence bands,
- prediction bands.

The resulting figures are intended to be suitable for publication with
minimal additional customization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def publication_plot(
    x_data=None,
    y_data=None,
    model=None,
    confidence_band=None,
    prediction_band=None,
    ax=None,
    figsize=(6.5, 4.5),
    dpi=300,
    title=None,
    xlabel="x",
    ylabel="y",
    data_label="Data",
    model_label="Model",
    confidence_label="95% Confidence Band",
    prediction_label="95% Prediction Band",
):
    """
    Create a publication-quality regression plot.

    Parameters
    ----------
    x_data : array-like, optional
        Experimental x values.

    y_data : array-like, optional
        Experimental y values.

    model : tuple, optional
        Tuple

            (x_model, y_model)

        describing the fitted model curve.

    confidence_band : ConfidenceBand, optional
        Confidence band object.

    prediction_band : PredictionBand, optional
        Prediction band object.

    ax : matplotlib.axes.Axes, optional
        Existing axes object.

    figsize : tuple, optional
        Figure size in inches.

    dpi : int, optional
        Figure resolution.

    title : str, optional
        Plot title.

    xlabel : str, optional
        X-axis label.

    ylabel : str, optional
        Y-axis label.

    data_label : str, optional
        Legend label for experimental data.

    model_label : str, optional
        Legend label for fitted model.

    confidence_label : str, optional
        Legend label for confidence band.

    prediction_label : str, optional
        Legend label for prediction band.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the completed figure.

    Notes
    -----
    Bands are drawn in the following order:

    1. Prediction band
    2. Confidence band
    3. Model
    4. Experimental data

    This ordering maximizes visibility.
    """

    # Create a new figure unless an existing axes object was supplied.
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ------------------------------------------------------------------
    # Draw prediction band.
    # ------------------------------------------------------------------

    if prediction_band is not None:
        ax.fill_between(
            prediction_band.x_grid,
            prediction_band.lower_values,
            prediction_band.upper_values,
            alpha=0.20,
            label=prediction_label,
            zorder=1,
        )

    # ------------------------------------------------------------------
    # Draw confidence band.
    # ------------------------------------------------------------------

    if confidence_band is not None:
        ax.fill_between(
            confidence_band.x_grid,
            confidence_band.lower_values,
            confidence_band.upper_values,
            alpha=0.35,
            label=confidence_label,
            zorder=2,
        )

    # ------------------------------------------------------------------
    # Draw fitted model.
    # ------------------------------------------------------------------

    if model is not None:
        x_model, y_model = model

        ax.plot(
            x_model,
            y_model,
            linewidth=2.0,
            label=model_label,
            zorder=3,
        )

    # ------------------------------------------------------------------
    # Draw experimental data.
    # ------------------------------------------------------------------

    if x_data is not None and y_data is not None:
        ax.scatter(
            x_data,
            y_data,
            s=35,
            label=data_label,
            zorder=4,
        )

    # ------------------------------------------------------------------
    # Final figure formatting.
    # ------------------------------------------------------------------

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=False)

    ax.minorticks_on()

    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
    )

    plt.tight_layout()

    return ax