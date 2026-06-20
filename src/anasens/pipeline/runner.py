"""Pipeline runner entrypoints.

This module provides a thin facade over
:class:`~anasens.pipeline.native_pipeline.NativeAnalysisPipeline` so
the subprocess entrypoint and any future programmatic callers have a
stable single-function interface.
"""

from __future__ import annotations

from anasens.config import PipelineConfig
from anasens.errors import ConfigurationError
from anasens.pipeline.native_pipeline import NativeAnalysisPipeline


def run_pipeline(config: PipelineConfig) -> None:
    """Run the end-to-end native analysis pipeline from a configuration object.

    This is the canonical entry point used by the GUI subprocess launcher
    (:mod:`anasens.interfaces.pipeline_launcher`) and any scripts that
    drive the pipeline programmatically.

    Args:
        config: Fully populated
            :class:`~anasens.config.PipelineConfig` describing the input
            source, analysis mode, preprocessing settings, and
            visualization flags.

    Raises:
        ConfigurationError: If ``config`` is ``None``.

    Example::

        from pathlib import Path
        from anasens.config import PipelineConfig
        from anasens.pipeline.runner import run_pipeline

        config = PipelineConfig(
            input_path=Path("/data/session.json"),
            input_source="json",
            analysis_mode="single",
        )
        run_pipeline(config)
    """
    # Defensive check: the pipeline cannot run without configuration.
    if config is None:
        raise ConfigurationError("run_pipeline requires a PipelineConfig.")

    NativeAnalysisPipeline().run_from_config(config)
