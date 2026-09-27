"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class PipelineConfig:
    """Tunable parameters for a single run_pipeline invocation. Nothing is required.

    Global poses come from global_poses.estimate_global_poses (Stage A->D), which needs no
    flight altitude or field of view: pixels_per_meter is estimated from the data. The old
    altitude_m / dfov_deg / reference_index / gps_positions / gimbal_yaw /
    yaw_anchor_weight fields belonged to the removed joint pose-graph optimization and no
    longer exist (see CLAUDE.md).

    latlons: EXIF-sourced (lat, lon) per image index -- pass the output of
        gps_placement.load_exif_latlons, which already drops coordinates that exist only in
        DJI XMP. Without GPS no image can be placed and the run is reported as failed.
    heading_anchor_source: absolute heading anchors for Stage A -- "gps" (GPS track bearing,
        no DJI metadata), "gimbal" (GimbalYawDegree; requires gimbal_yaw_deg) or "none".
    gimbal_yaw_deg: GimbalYawDegree per image index in degrees (io_utils.load_gimbal_yaw).
        Used only, and required, with heading_anchor_source="gimbal"; giving it with any
        other source is an error, never silently ignored. Both checks live in
        estimate_global_poses, not here, so there is one place that enforces them.
    pairs: image pairs to match; None means consecutive-neighbour pairs.
    loops: image-index loops for metrics.compute_cycle_loop_error.
    """

    ransac_threshold: float = 3.0
    output_dir: Path | None = None
    latlons: dict[int, tuple[float, float]] | None = None
    heading_anchor_source: Literal["gps", "gimbal", "none"] = "gps"
    gimbal_yaw_deg: dict[int, float] | None = None
    pairs: list[tuple[int, int]] | None = None
    loops: list[list[int]] | None = None
