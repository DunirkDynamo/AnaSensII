# Changelog

## Unreleased

- Added package GUI launcher with Run Full Pipeline action.
- Wired pipeline runner to execute legacy workflow through legacy/CSV_pipeline.py.
- Moved legacy scripts into top-level legacy folder:
	- legacy/CSV_pipeline.py
	- legacy/pipeline_functions.py
	- legacy/Calibrator.py
	- legacy/Standard Curve Creator.py
	- legacy/Inverse Regression.py
- Updated QUICK_START.md to document full analysis execution via GUI.
- Expanded QUICK_START.md with GUI-first usage guidance and developer usage patterns.
- Removed tests directory entirely.
- Removed new-package standard-curve and inverse-regression modules to refocus AnaSens2 on the analysis pipeline.
- Removed new-package fit-curve and inverse CLI entrypoints.
- Removed the empty new-package regression package and its GUI placeholder views.
- Updated QUICK_START.md and ARCHITECTURE.md to reflect pipeline-only scope.
- Declared Ekidna as an external dependency in pyproject so legacy imports resolve from the installed package.
- Added statsmodels to core dependencies because the installed Ekidna package imports it at runtime.
- Replaced the Tk launcher with a PySide6 synthwave GUI shell with Pipeline, Curve Builder, and Inverse Estimation tabs.
- Updated the Qt Pipeline tab to launch an internal subprocess entrypoint, removing dependence on a direct pipeline CLI command.
- Removed the direct pipeline CLI command surface to keep the product GUI-first.
- Removed non-essential subtitle text from the main AnaSens2 header card.
- Removed descriptive helper text from the Pipeline tab header and switched Input Folder and Visualization cards to a side-by-side layout to reduce vertical space.
- Integrated Plotter-Fitter into AnaSens under `src/anasens/curve_builder` to preserve existing curve-fitting functionality inside the main package.
- Replaced the Curve Builder placeholder tab with an embedded Plotter-Fitter interface in the main GUI.
- Added Plotter-Fitter runtime dependencies (`sympy`, `antlr4-python3-runtime`, `openpyxl`) to AnaSens project dependencies.
- Temporarily documented Plotter-Fitter dependencies under a `gui` extra during migration; this was later removed in favor of default dependencies.
- Updated package docs (`README.md`, `QUICK_START.md`, `ARCHITECTURE.md`) to describe the integrated Plotter-Fitter Curve Builder tab and its retained capabilities.
- Moved GUI runtime dependencies into default project dependencies and removed the separate `gui` extras requirement, so standard install supports the only GUI workflow.
- Implemented a native object-oriented pipeline (`NativeAnalysisPipeline`) with separated loader, preprocessing, extraction, tabulation, persistence, and visualization responsibilities.
- Added typed CSV measurement loader services in `src/anasens/io/measurement_loader.py` to decouple file IO from pipeline orchestration.
- Added native visualization generators with runtime toggles for pairplot, drift views, and overlay scans.
- Switched GUI pipeline subprocess execution to the native pipeline path (no legacy script execution for `anasens gui` pipeline runs).
- Added a native `src/anasens/inverse_regression` package with object-oriented data loading, estimation service, and typed result models.
- Replaced the Inverse Estimation placeholder with a functional Qt tab that replicates legacy inverse-regression intent (quantile/least-squares modes, estimates table, CSV export).
- Added Curve Builder model export in the packaged GUI (plot image, pickled standard-curve object, and fit details), enabling inverse-estimation input artifacts from the new package.
- Added robust GUI pipeline process logging so both stdout and stderr lines are shown in the Pipeline tab output panel.
- Refined pssession/json output folder strategy to use one source-level master folder with tagged filenames across shared Analysis subfolders.
- Added streaming batch execution for directory-based pssession/json runs (one source file at a time) to reduce peak memory usage.
- Removed redundant `I_original` from processed CSV exports and retained raw current in the existing `I` column.
- Corrected moving-average baseline semantics so `baseline` stores the baseline estimate and `I_corrected` is computed as `I_smoothed - baseline`.
- Added trailing-edge correction for moving-average baseline subtraction: points not covered by full windows are zeroed in `I_corrected`.
- Added preprocessing overview visualization (side-by-side raw/smoothed/baseline overlay and corrected trace) with GUI and CLI toggles.
- Fixed preprocessing overview layout warning by switching to constrained layout for multi-axis colorbar rendering.
- Removed Curve Builder subtitle text (`Neon plotting and curve fitting for tabular data`) per UI simplification.
- Updated executable build automation to trigger on version tags (`v*.*.*`) and attach Windows exe artifacts to GitHub Releases.
- Fixed executable build install path to use declared dependencies (`pip install -e .`) instead of removed `.[gui]` extras.

## 0.1.0

- Initial source-layout package scaffold under `src/anasens`
- Added CLI entrypoint scaffold
- Added inverse regression module preserving existing delta-method workflow
- Added Sphinx docs scaffold
- Added CI, docs, and executable-build workflow scaffolds
- Added `QUICK_START.md` and `ARCHITECTURE.md`
- Removed scaffold tests (tests to be added later)
