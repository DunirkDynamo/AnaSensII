"""Persistence utilities for measurement objects."""

from pathlib import Path
import pickle
from typing import Any


def save_pickle(data: Any, path: Path) -> None:
    """Serialize object to pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    """Load object from pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)
