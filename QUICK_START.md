# QUICK_START

This guide gets AnaSens2 running quickly for both developers and users.

## 1) Prerequisites

- Python 3.10+
- Git
- Windows, macOS, or Linux

## 2) Clone the repository

```bash
git clone https://github.com/<your-org>/AnaSens2.git
cd AnaSens2
```

## 3) Install options

### Editable install (developer mode)

Use this when actively modifying package code.

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### Full install (standard user mode)

Use this when you only need to run the package.

```bash
python -m pip install --upgrade pip
python -m pip install .
```

## 4) Verify install

```bash
anasens --version
```

## 5) CLI usage
Use `anasens --help` to see available commands:

```bash
anasens --help
```

Typical output includes commands like:

- `gui` (opens the Qt application shell)

### Example A: launching full pipeline from GUI entrypoint

```bash
anasens gui
```

Then click **Run Full Pipeline** in the launcher window.

### Example B: checking command-level help

```bash
anasens gui --help
```

### Example C: common developer loop

```bash
python -m pip install -e .[dev]
anasens --version
anasens gui
```

If you modify package code while in editable mode, re-running `anasens gui` uses your latest source changes without reinstall.

## 6) Run a full analysis (quick summary)

The package provides a GUI-first workflow that runs the native object-oriented pipeline from the Qt shell.

### GUI: run full pipeline

```bash
anasens gui
```

What happens next:

- In the Pipeline tab, select analysis mode, data folder, visualization options, and cutoff.
- The pipeline runs smoothing, baseline subtraction, parameter extraction, tabulation, and plotting.
- Outputs are written into analysis subfolders for processed sweeps, extracted parameters, sweep info, and visualizations.

This opens the PySide6 synthwave shell with Pipeline, Curve Builder, and Inverse Estimation tabs. In the Pipeline tab, click Run Full Pipeline.

## 7) Curve Builder tab (Plotter-Fitter integration)

The Curve Builder tab now embeds Plotter-Fitter directly inside AnaSens.

What you can do in Curve Builder:

- Load tabular data (`.csv`, `.xls`, `.xlsx`, `.json`, `.pkl`, `.pickle`)
- Choose x/y columns and render the plot
- Apply polynomial, power-law, or custom LaTeX fits
- View model equation and fitted parameter readout

Launch path is still:

```bash
anasens gui
```

Then open the Curve Builder tab.

## 8) Inverse Estimation tab

The Inverse Estimation tab now runs native inverse regression workflows in-package.

What you can do in Inverse Estimation:

- Load a folder of calibrated voltammogram pickles (objects with `calibration_table`)
- Choose the calibrated metric column to invert
- Load a standard-curve pickle object
- Select regression mode (`Quantile Regression` or `Least Squares Regression`)
- Run inverse estimates and export results to CSV

## 9) Testing status

Automated tests are intentionally deferred at this stage of migration and are not yet committed.

When tests are added later, this guide will include the exact `pytest` commands.

## 10) Build docs locally

```bash
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in your browser.

## 11) Build executable locally (optional)

The GitHub workflow is the primary path for official executables, but local testing is possible after PyInstaller setup.

```bash
python -m pip install pyinstaller
pyinstaller --onefile --name anasens-gui src/anasens/interfaces/cli.py
```

## 12) GitHub executable build (tag-selected)

Use the `Build Executable` workflow in GitHub Actions.

- Open Actions -> Build Executable
- Click Run workflow
- Enter the release tag in `version_tag` (example: `v0.1.0`)
- Run workflow

The workflow checks out that tag and uploads platform-specific executable artifacts.
