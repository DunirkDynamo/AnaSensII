"""Internal subprocess entrypoint for GUI-triggered pipeline execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from anasens.config import PipelineConfig, PreprocessConfig, VisualizationConfig
from anasens.pipeline.runner import run_pipeline  # noqa: F401 – re-exported for subprocess entrypoint


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native analysis pipeline from GUI request.")
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--analysis-mode", choices=["single", "batch"], required=True)
    parser.add_argument("--cutoff-points", type=int, default=0)
    parser.add_argument("--cut-start-points", type=int, default=None)
    parser.add_argument("--cut-end-points", type=int, default=0)
    parser.add_argument("--smooth-window-length", type=int, default=17)
    parser.add_argument("--smooth-polyorder", type=int, default=3)
    parser.add_argument("--baseline-algo", choices=["snip", "moving_average"], default="snip")
    parser.add_argument("--baseline-snip-expected-fwhm", type=int, default=35)
    parser.add_argument("--baseline-snip-window-size", type=int, default=10)
    parser.add_argument("--baseline-moving-average-window-size", type=int, default=10)
    parser.add_argument("--baseline-moving-average-forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-moving-average-max-iter", type=int, default=1000)
    parser.add_argument("--pairplot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drift-2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drift-1d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay-scans", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cut_start = args.cut_start_points
    if cut_start is None:
        cut_start = args.cutoff_points

    config = PipelineConfig(
        input_path=Path(args.input_folder),
        analysis_mode=args.analysis_mode,
        preprocess=PreprocessConfig(
            cut_start_points=max(0, int(cut_start)),
            cut_end_points=max(0, int(args.cut_end_points)),
            smooth_window_length=max(3, int(args.smooth_window_length)),
            smooth_polyorder=max(1, int(args.smooth_polyorder)),
            baseline_algo=str(args.baseline_algo),
            baseline_snip_expected_fwhm=max(1, int(args.baseline_snip_expected_fwhm)),
            baseline_snip_window_size=max(1, int(args.baseline_snip_window_size)),
            baseline_moving_average_window_size=max(1, int(args.baseline_moving_average_window_size)),
            baseline_moving_average_forward=bool(args.baseline_moving_average_forward),
            baseline_moving_average_max_iter=max(1, int(args.baseline_moving_average_max_iter)),
        ),
        visualization=VisualizationConfig(
            pairplot=bool(args.pairplot),
            drift_2d=bool(args.drift_2d),
            drift_1d=bool(args.drift_1d),
            overlay_scans=bool(args.overlay_scans),
        ),
    )
    run_pipeline(config=config)


if __name__ == "__main__":
    main()
