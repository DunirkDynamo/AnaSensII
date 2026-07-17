"""
Band representations for fitted-model and prediction uncertainty.

This package contains lightweight storage and evaluation objects for
interval bands computed elsewhere in the library.

Available band types
--------------------
ConfidenceBand
    Represents uncertainty in the fitted model mean only.

PredictionBand
    Represents uncertainty in future observations and includes both
    model uncertainty and observation noise.

Design principle
----------------
Band classes do not perform statistical computation.

They only store, interpolate, and evaluate interval curves that have
already been computed by confidence-interval or prediction-interval
methods.
"""

from .confidence_band import ConfidenceBand
from .prediction_band import PredictionBand

__all__ = [
    "ConfidenceBand",
    "PredictionBand",
]