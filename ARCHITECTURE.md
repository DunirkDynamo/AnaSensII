# ARCHITECTURE

This document describes the AnaSens2 source package structure and the purpose of each major area.

## Package layout

```text
src/anasens/
  __init__.py              # Package exports and version
  version.py               # Single source of package version
  domain.py                # Core data models (SweepData, Measurement, PeakSet)
  config.py                # Runtime configuration dataclasses
  errors.py                # Package-specific exceptions

  io/
    csv_loader.py          # CSV sweep loading
    measurement_store.py   # Pickle-based persistence helpers
    serialization.py       # Generic save helpers (for example DataFrame to CSV)

  preprocess/
    cutoff.py              # Leading-point trimming
    smoothing.py           # Savitzky-Golay smoothing
    baseline.py            # Baseline subtraction wrappers

  features/
    peaks.py               # Peak finding and width metrics
    integration.py         # Simpson integration helpers
    tables.py              # Extracted-parameter table builders

  visualization/
    plots.py               # Plot persistence and shared plotting helpers

  pipeline/
    runner.py              # End-to-end orchestrator entrypoint
    workflows.py           # Named stage workflows

  curve_builder/
    gui.py                 # Integrated Plotter-Fitter Qt interface
    fitters.py             # Polynomial, power-law, and custom LaTeX fit services
    plotter.py             # Matplotlib rendering helpers for curve builder
    data_loader.py         # Tabular data loading utilities
    models.py              # LoadedDataset and FittingResult dataclasses

  inverse_regression/
    data_loader.py         # Pickle loading for calibrated voltammograms and standard curves
    service.py             # Inverse estimation engine (quantile and least-squares)
    models.py              # Inverse estimation result dataclasses

  interfaces/
    cli.py                 # Typer command-line app (`anasens`)
    gui_app.py             # PySide6 synthwave GUI shell with tabbed workflow entrypoints
    gui_views/             # Qt tab/view modules for pipeline and future tools
```

## Design intent

- Domain-first: data structures are explicit and typed.
- UI-independent core: math and processing live outside CLI/GUI layers.
- Staged migration: legacy scripts remain authoritative until parity tests pass.
- Documentation-ready: all public modules are autodoc-friendly.

## Scope update

- The new package currently focuses on the standard analysis pipeline only.
- Curve Builder is now integrated in-package by embedding Plotter-Fitter under `src/anasens/curve_builder`.
- Legacy standard-curve and inverse-regression scripts remain in `legacy/` as historical references during migration.
- The package execution surface is GUI-first through the `gui` command.
- The Qt GUI Pipeline tab launches the native object-oriented pipeline subprocess entrypoint.
- The Qt GUI Curve Builder tab hosts the integrated Plotter-Fitter interface with retained plotting and fitting functionality.
- The Qt GUI Inverse Estimation tab now hosts a native object-oriented inverse regression workflow backed by `src/anasens/inverse_regression`.

## Data flow summary

1. IO loads sweep CSV data into in-memory structures.
2. Preprocess transforms current traces (cutoff, smooth, baseline subtraction).
3. Features extracts peaks, widths, and integrated metrics.
4. Visualization and IO persist plots, tables, and serialized objects.
5. Interfaces expose this pipeline through CLI and GUI.
