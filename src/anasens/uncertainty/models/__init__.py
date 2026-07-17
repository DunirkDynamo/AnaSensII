"""
mc_inference.models

This package defines deterministic statistical models of the form:

    y = f(x; θ)

The models in this package are purely functional:
they define only the expected value of the observed variable.

No stochastic assumptions are included here.
All probabilistic structure is introduced separately via:

    mc_inference.error_models

Design principle
----------------
- Models are deterministic mappings (x, θ) → y
- No likelihoods are defined here
- No randomness is included here
- All uncertainty is handled externally

This separation ensures that:
- the same model can be reused with different error distributions
- inference logic remains decoupled from functional form
"""

from .linear import LinearModel

__all__ = [
    "LinearModel",
]