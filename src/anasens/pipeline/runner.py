"""Pipeline runner entrypoints."""

from __future__ import annotations

from anasens.config import PipelineConfig
from anasens.errors import ConfigurationError
from anasens.pipeline.native_pipeline import NativeAnalysisPipeline


def run_pipeline(config: PipelineConfig) -> None:
    """Run the native object-oriented end-to-end analysis pipeline."""
    if config is None:
        raise ConfigurationError("run_pipeline requires a PipelineConfig.")

    NativeAnalysisPipeline().run_from_config(config)
