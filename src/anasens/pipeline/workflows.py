"""Named workflows for native preprocessing, extraction, and visualization."""

from __future__ import annotations

from anasens.config import PipelineConfig
from anasens.domain import Measurement
from anasens.pipeline.native_pipeline import NativeAnalysisPipeline


def run_native_pipeline_workflow(config: PipelineConfig) -> list[Measurement]:
    """Execute the native object-oriented end-to-end pipeline workflow."""
    return NativeAnalysisPipeline().run_from_config(config)
