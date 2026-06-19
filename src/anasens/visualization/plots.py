"""Basic plotting helper functions."""

from pathlib import Path

from matplotlib.figure import Figure


def save_plot(fig: Figure, save_path: Path, filename: str = "plot.png", dpi: int = 300) -> Path:
    """Save matplotlib figure and return output path."""
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / filename
    fig.savefig(out, dpi=dpi)
    return out
