"""Command-line entry point for Plotter Fitter."""

from .app import PlotterFitterApplication


def main() -> None:
    """Run the Plotter Fitter desktop application."""
    PlotterFitterApplication().run()


if __name__ == "__main__":
    main()
