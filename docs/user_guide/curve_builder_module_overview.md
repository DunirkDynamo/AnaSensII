# Curve Builder Module Overview

## Purpose

This document explains the Curve Builder module in AnaSens2 as it exists so far. It focuses on the current application structure, the GUI workflow, the fitting pipeline, the uncertainty-band integration, and the custom-function derivative machinery that now supports higher-order analysis.

The Curve Builder tab is the part of the application that lets a user:

- load a tabular dataset,
- choose x and y columns,
- optionally filter the data,
- select a model family,
- fit that model to the data,
- inspect the fitted curve together with uncertainty bands,
- and export the result for downstream use.

## Module Structure

The module is centered around four main pieces:

- [gui.py](../../src/anasens/curve_builder/gui.py): the PySide6 window, widgets, callbacks, and user-facing workflow.
- [fitters.py](../../src/anasens/curve_builder/fitters.py): model fitting, custom LaTeX parsing, and derivative construction.
- [band_service.py](../../src/anasens/curve_builder/band_service.py): confidence-band and prediction-band orchestration.
- [plotter.py](../../src/anasens/curve_builder/plotter.py): styled Matplotlib rendering for the scatter plot, fitted curve, and overlays.

Supporting data containers live in [models.py](../../src/anasens/curve_builder/models.py), while [standard_curve.py](../../src/anasens/curve_builder/standard_curve.py) packages the fitted result for export.

## Current GUI Workflow

The Curve Builder tab is organized around a simple sequence:

1. Load a dataset.
2. Select x and y columns.
3. Optionally apply exact-match filters.
4. Choose a fit type.
5. Apply the fit.
6. View the fitted curve and uncertainty overlays.
7. Export the plot, fitted model, and summary text.

The current fit types are:

- polynomial,
- power law,
- custom LaTeX / math-style expression.

### Data Selection

The data-loading layer populates the x-axis, y-axis, and filter-column dropdowns after a file is loaded. Before that, the selectors are disabled so the user cannot choose invalid inputs.

### Custom Functions

When the fit type is set to custom, the user enters a mathematical expression and a comma-separated list of parameter names. The module parses the expression with SymPy, builds a SciPy-ready callable, and now also constructs analytical derivatives.

The preview box shows the parsed function and derivative information in a scrollable panel. That preview is intentionally explicit:

- the function is rendered first,
- the gradient is shown as a vector form and then with actual derivative entries,
- the Hessian is shown as a matrix form and then with actual second derivatives,
- third derivatives are shown as a compact list of unique symmetric terms.

## Fitting Pipeline

The fitting path is handled by [FitService](../../src/anasens/curve_builder/fitters.py).

For built-in models, the service constructs a callable model and dispatches to SciPy curve fitting.
For custom models, it does more work:

- parse the expression,
- identify the independent variable `x`,
- resolve the parameter names,
- build a numerical model function,
- generate analytical first-, second-, and third-order derivatives when possible.

The fitted result is stored in a [FittingResult](../../src/anasens/curve_builder/models.py) object that carries:

- the model name,
- the optimized parameters,
- the fitted curve values,
- the display equation,
- the callable model,
- the covariance matrix,
- the residual scale,
- and optional derivative callables for higher-order uncertainty calculations.

## Uncertainty Integration

The Curve Builder no longer treats uncertainty as an afterthought. After a fit is computed, [BandService](../../src/anasens/curve_builder/band_service.py) builds the overlays that are rendered on top of the data and fitted curve.

### Confidence Bands

The model-confidence side currently supports selectable methods through checkboxes in the GUI:

- Delta Method,
- Monte Carlo,
- Profile likelihood.

The user can select multiple methods, and the plot will overlay them together using different colors and line styles.

Delta Method order is separately selectable through a dropdown. Order 1 is the standard first-order propagation. Order 2 is available for higher-order corrections and uses Hessian and third-derivative information when supplied.

### Prediction Bands

The prediction-band side is currently fixed to a Gaussian analytic prediction band.

This means the current Curve Builder implementation computes prediction uncertainty for new observations using the fitted location curve together with the residual variance estimate and the fitted parameter covariance.

There is not yet a user-facing prediction-interval method selector in the GUI. That is the next extension point.

## Plot Rendering

The rendering layer in [plotter.py](../../src/anasens/curve_builder/plotter.py) is deliberately presentation-only. It does not fit models or estimate uncertainty. Its job is to draw the outputs cleanly and consistently.

The plot currently includes:

- observed data points,
- fitted curve,
- confidence band overlays,
- prediction band overlay.

The styling is intentionally synthwave-inspired and consistent with the rest of the application. Different confidence methods use different colors and line styles so multiple bands can be read at a glance.

## Exported Output

After a fit is applied, the GUI can export:

- the rendered plot image,
- a pickled fitted model object,
- a text summary of the fit and uncertainty results.

The exported summary is built from the same statistics panel text that is shown in the GUI, so the on-screen results and the saved record stay aligned.

## Mathematical Summary

For observations $y_i$ and model $f(x;\theta)$, the current Curve Builder assumes the standard regression form

$$
y_i = f(x_i;\theta) + \varepsilon_i.
$$

For confidence bands, the target is the fitted mean $f(x;\hat\theta)$.
For prediction bands, the target is the future observation at the same $x$, which is centered on the fitted location and then widened by observation noise.

The key distinction is:

- confidence band: uncertainty in the fitted curve itself,
- prediction band: uncertainty in a future measurement.

## What Is Implemented Right Now

At this stage, the Curve Builder module supports:

- data loading and column selection,
- standard and custom fitting,
- analytical derivatives for custom expressions,
- multiple confidence-band methods via checkboxes,
- Delta Method order selection,
- a fixed Gaussian prediction band,
- scrollable derivative previews for custom functions,
- export of the fit and plot artifacts.

## What Comes Next

The next major extension is prediction-interval method selection in the GUI. The intended design is to follow the same checkbox-driven pattern used for confidence bands, while keeping the plot output readable and the module structure consistent.
