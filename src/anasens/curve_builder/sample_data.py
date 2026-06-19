"""Generate sample datasets for manual testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_dataframe(random_seed: int = 17, sample_count: int = 80) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create a noisy quadratic dataset for testing.

    Args:
        random_seed: Seed used for deterministic random noise.
        sample_count: Number of samples to generate.

    Returns:
        A tuple containing the generated DataFrame and the true model parameters.
    """
    # Ground-truth parameters are defined explicitly so they can be recorded and
    # later compared against fitted estimates during manual validation.
    truth = {
        "quadratic": 1.75,
        "linear": -2.4,
        "intercept": 4.2,
        "noise_std": 1.1,
    }

    # Inputs span a moderate range and receive additive Gaussian noise with a
    # constant standard deviation, matching the requested dataset behavior.
    random_generator = np.random.default_rng(random_seed)
    x_values = np.linspace(-6.0, 6.0, sample_count)
    noiseless_y = (
        truth["quadratic"] * np.power(x_values, 2)
        + truth["linear"] * x_values
        + truth["intercept"]
    )
    noisy_y = noiseless_y + random_generator.normal(0.0, truth["noise_std"], size=sample_count)

    frame = pd.DataFrame(
        {
            "x": x_values,
            "y": noisy_y,
            "y_true": noiseless_y,
            "noise": noisy_y - noiseless_y,
        }
    )
    return frame, truth


def write_sample_files(output_directory: str | Path) -> list[Path]:
    """Write the generated dataset to several supported file formats.

    Args:
        output_directory: Directory where sample data files will be written.

    Returns:
        A list of generated file paths.
    """
    # The output directory is created on demand so this utility can be run from
    # a clean checkout without requiring any prior manual setup.
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    frame, truth = generate_sample_dataframe()

    # Each file corresponds to one loader path supported by the package, which
    # makes the directory a compact end-to-end manual test bundle.
    csv_path = output_path / "quadratic_sample.csv"
    excel_path = output_path / "quadratic_sample.xlsx"
    json_path = output_path / "quadratic_sample.json"
    pickle_path = output_path / "quadratic_sample.pkl"
    truth_path = output_path / "quadratic_truth.txt"

    frame.to_csv(csv_path, index=False)
    frame.to_excel(excel_path, index=False)
    frame.to_json(json_path, orient="columns", indent=2)
    frame.to_pickle(pickle_path)
    truth_path.write_text(
        "\n".join(
            [
                "Quadratic ground truth parameters",
                f"quadratic = {truth['quadratic']}",
                f"linear = {truth['linear']}",
                f"intercept = {truth['intercept']}",
                f"noise_std = {truth['noise_std']}",
            ]
        ),
        encoding="utf-8",
    )

    return [csv_path, excel_path, json_path, pickle_path, truth_path]


def main() -> None:
    """Generate sample data in the repository's data directory."""
    write_sample_files(Path(__file__).resolve().parents[2] / "sample_data")


if __name__ == "__main__":
    main()
