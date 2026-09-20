"""Unit tests for sea_mosaic.compose.compose_global_transforms.

compose_global_transforms is pure wiring (build_pose_graph -> optimize_pose_graph); the
optimizer's own math is already covered in tests/test_posegraph.py, and build_pose_graph's
PairResult/GPS-position mapping is covered in tests/test_posegraph.py's build_pose_graph
tests. This file focuses on: given real PairResult objects (not a hand-built PoseGraph),
does the full compose_global_transforms path converge near ground truth, and does it
correctly require pixels_per_meter/inlier_count_reference with no silent default.
"""

from __future__ import annotations

import numpy as np
import pytest

from sea_mosaic.compose import compose_global_transforms
from sea_mosaic.types import PairResult

_PIXELS_PER_METER = 10.0
_INLIER_COUNT_REFERENCE = 100.0

# True positions in meters (matches tests/test_posegraph.py's pixel-space _TRUE_POSITIONS
# once multiplied by _PIXELS_PER_METER=10.0: 0, 100, 200, 300, 400).
_TRUE_POSITIONS_M = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0), 3: (30.0, 0.0), 4: (40.0, 0.0)}

# Anchor noise in meters (== tests/test_posegraph.py's pixel-space _ANCHOR_OFFSETS
# divided by _PIXELS_PER_METER, so the resulting pixel-space anchor noise is identical).
_ANCHOR_OFFSETS_M = {
    0: (0.03, -0.02),
    1: (-0.04, 0.05),
    2: (0.02, 0.04),
    3: (-0.05, -0.03),
    4: (0.04, 0.03),
}

# Edge translations in PIXELS (homographies are always pixel-space, regardless of the
# meter-space GPS anchors) — identical noise pattern to test_posegraph.py's clean case:
# true edge translation -100px + a small deterministic bias.
_EDGE_TRANSLATION_PX = {
    (0, 1): (-97.0, 1.0),
    (1, 2): (-97.0, -1.0),
    (2, 3): (-97.0, 1.0),
    (3, 4): (-97.0, -1.0),
}


def _homography(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _pair_result(src_index: int, dst_index: int) -> PairResult:
    tx, ty = _EDGE_TRANSLATION_PX[(src_index, dst_index)]
    match_count = 100
    return PairResult(
        src_index=src_index,
        dst_index=dst_index,
        src_points=np.zeros((match_count, 2)),
        dst_points=np.zeros((match_count, 2)),
        inlier_mask=np.ones(match_count, dtype=bool),  # inlier_count == match_count == 100
        homography=_homography(tx, ty),
    )


def _gps_positions_m() -> dict[int, np.ndarray]:
    return {
        index: np.array([x + _ANCHOR_OFFSETS_M[index][0], y + _ANCHOR_OFFSETS_M[index][1]])
        for index, (x, y) in _TRUE_POSITIONS_M.items()
    }


def test_compose_global_transforms_converges_near_truth_from_real_pair_results() -> None:
    pair_results = [_pair_result(src, dst) for src, dst in sorted(_EDGE_TRANSLATION_PX)]
    gps_positions = _gps_positions_m()

    result = compose_global_transforms(
        pair_results,
        gps_positions,
        reference_index=0,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
    )

    assert result.optimization_status == "converged"
    assert np.isfinite(result.residual_error)

    true_positions_px = {
        index: np.array([x, y]) * _PIXELS_PER_METER for index, (x, y) in _TRUE_POSITIONS_M.items()
    }
    errors = {
        index: float(np.linalg.norm(result.transforms[index][:2, 2] - true_positions_px[index]))
        for index in true_positions_px
    }
    assert all(error < 2.0 for error in errors.values())
    assert abs(errors[4] - errors[1]) < 1.0  # no drift accumulation along the chain


def test_compose_global_transforms_requires_pixels_per_meter_and_inlier_count_reference() -> None:
    pair_results = [_pair_result(0, 1)]
    gps_positions = _gps_positions_m()

    with pytest.raises(TypeError):
        compose_global_transforms(pair_results, gps_positions, inlier_count_reference=1.0)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        compose_global_transforms(pair_results, gps_positions, pixels_per_meter=1.0)  # type: ignore[call-arg]
