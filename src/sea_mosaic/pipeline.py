"""Pipeline entrypoint: estimate -> compose -> warp -> blend -> evaluate metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sea_mosaic.config import PipelineConfig
from sea_mosaic.geo.camera import CameraIntrinsics, CameraPose
from sea_mosaic.matcher import Matcher


def run_pipeline(
    image_paths: list[Path],
    matcher: Matcher,
    gps_positions: dict[int, np.ndarray] | None = None,
    camera_intrinsics: dict[int, CameraIntrinsics] | None = None,
    camera_poses: dict[int, CameraPose] | None = None,
    config: PipelineConfig | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run the full estimate -> compose -> warp -> blend pipeline and evaluate metrics.

    Returns (stitched_image, metrics_df), where metrics_df is the one-row DataFrame from
    metrics.evaluate_stitching_metrics.

    camera_intrinsics / camera_poses are reserved parameters for a future direct
    georeferencing integration (see sea_mosaic.geo.camera / sea_mosaic.geo.direct); this
    skeleton does not use them. metrics_df's 'method' column is populated from
    matcher.name.
    """
    ...
