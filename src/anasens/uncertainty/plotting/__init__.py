"""
Publication-quality plotting utilities.

This package contains visualization routines for displaying fitted models,
confidence bands, prediction bands, and experimental data.

Plotting functions are intentionally separated from the statistical
inference code. They do not perform model fitting or interval
construction—they simply visualize the results.
"""

from .publication import publication_plot

__all__ = [
    "publication_plot",
]