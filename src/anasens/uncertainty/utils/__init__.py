"""
Shared utility functions.

This package contains low-level helper routines used by multiple parts of
the library.

Utilities should be:

- small,
- reusable,
- independent of statistical model classes,
- free of high-level inference logic.

Examples include parameter sampling, quantile calculations, and numerical
derivative routines.
"""

from .parameter_sampling import sample_parameters
from .quantiles import equal_tail_interval

__all__ = [
    "sample_parameters",
    "equal_tail_interval",
]