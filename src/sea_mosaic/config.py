"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PipelineConfig:
    """Tunable parameters for a single run_pipeline invocation.

    altitude_m / dfov_deg are required with no default -- they feed
    geo.projection.estimate_pixels_per_meter, and (like pixels_per_meter itself, see
    compose.py/CLAUDE.md) are camera/flight-specific numbers with no sensible universal
    default; omitting them fails at PipelineConfig construction time, not partway
    through run_pipeline. Everything else here is optional and defaults to None/the
    existing baseline behavior, matching build_pose_graph/compose_global_transforms's
    own optionality for the same parameters.
    """

    altitude_m: float
    dfov_deg: float
    ransac_threshold: float = 3.0
    reference_index: int = 0
    output_dir: Path | None = None
    gps_positions: dict[int, np.ndarray] | None = None
    gimbal_yaw: dict[int, float] | None = None
    yaw_anchor_weight: float | None = None
    pairs: list[tuple[int, int]] | None = None
    loops: list[list[int]] | None = None
