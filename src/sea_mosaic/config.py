"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Tunable parameters for a single run_pipeline invocation."""

    ransac_threshold: float = 3.0
    reference_index: int = 0
    output_dir: Path | None = None
