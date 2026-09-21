"""Stage 2 (compose): derive global image-to-mosaic transforms via pose-graph optimization.

Never chains pairwise homographies (CLAUDE.md architectural constraint 4) — delegates to
posegraph.build_pose_graph / posegraph.optimize_pose_graph.
"""

from __future__ import annotations

import numpy as np

from sea_mosaic.posegraph import build_pose_graph, optimize_pose_graph
from sea_mosaic.types import GlobalTransforms, PairResult


def compose_global_transforms(
    pair_results: list[PairResult],
    gps_positions: dict[int, np.ndarray] | None = None,
    reference_index: int = 0,
    *,
    pixels_per_meter: float,
    inlier_count_reference: float,
    gimbal_yaw: dict[int, float] | None = None,
    yaw_anchor_weight: float | None = None,
) -> GlobalTransforms:
    """Compose global image-to-mosaic transforms via GPS/yaw-anchored pose-graph optimization.

    Delegates to posegraph.build_pose_graph (pair_results, gps_positions,
    pixels_per_meter, inlier_count_reference, gimbal_yaw, yaw_anchor_weight) followed by
    posegraph.optimize_pose_graph (reference_index) — no pairwise-homography chaining
    happens here or in either delegate.

    pixels_per_meter and inlier_count_reference are required, no-default keyword-only
    parameters, passed straight through to build_pose_graph (see its docstring for why
    neither may silently default). estimate.default_inlier_count_reference is available
    for a caller (e.g. pipeline.py) that wants a reasonable inlier_count_reference
    computed from its own pair_results rather than picking one blindly; computing and
    passing it is the caller's responsibility, not this function's.

    gimbal_yaw and yaw_anchor_weight are optional and passed straight through to
    build_pose_graph unchanged (see its docstring for the same "gimbal_yaw is optional,
    but yaw_anchor_weight is required whenever gimbal_yaw is given" rule).
    """
    graph = build_pose_graph(
        pair_results,
        gps_positions,
        pixels_per_meter=pixels_per_meter,
        inlier_count_reference=inlier_count_reference,
        gimbal_yaw=gimbal_yaw,
        yaw_anchor_weight=yaw_anchor_weight,
    )
    return optimize_pose_graph(graph, reference_index=reference_index)
