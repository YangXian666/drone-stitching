"""Stage 2 (compose): derive global image-to-mosaic transforms via pose-graph optimization.

Never chains pairwise homographies (CLAUDE.md architectural constraint 4) — delegates to
posegraph.build_pose_graph / posegraph.optimize_pose_graph.
"""

from __future__ import annotations

import numpy as np

from sea_mosaic.types import GlobalTransforms, PairResult


def compose_global_transforms(
    pair_results: list[PairResult],
    gps_positions: dict[int, np.ndarray] | None = None,
    reference_index: int = 0,
) -> GlobalTransforms:
    """Compose global image-to-mosaic transforms via GPS-anchored pose-graph optimization."""
    ...
