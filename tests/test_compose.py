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


def test_compose_global_transforms_requires_yaw_anchor_weight_when_gimbal_yaw_given() -> None:
    """compose_global_transforms must forward gimbal_yaw's required-weight check through
    to build_pose_graph, not swallow or ignore it."""
    pair_results = [_pair_result(0, 1)]
    gps_positions = _gps_positions_m()

    with pytest.raises(ValueError):
        compose_global_transforms(
            pair_results,
            gps_positions,
            reference_index=0,
            pixels_per_meter=_PIXELS_PER_METER,
            inlier_count_reference=_INLIER_COUNT_REFERENCE,
            gimbal_yaw={0: 0.0, 1: -40.0},
        )


def test_compose_global_transforms_forwards_gimbal_yaw_to_build_pose_graph() -> None:
    """Regression guard for a real oversight found in this project: compose_global_
    transforms's signature was never updated to accept/forward gimbal_yaw and
    yaw_anchor_weight when YawAnchor was added to build_pose_graph and
    optimize_pose_graph, so YawAnchor was silently unusable through the public wrapper
    (TypeError: unexpected keyword argument) despite both lower-level functions and
    their own tests being complete and green.

    This must actually move the result, not just avoid raising -- a corrupted-rotation
    edge overridden by a yaw anchor, exercised through the full compose_global_transforms
    path (mirrors tests/test_posegraph.py's equivalent hand-built-PoseGraph check, with
    the same hand-verified expected angle: weight=5.0 against a 90deg-corrupted edge with
    unit information converges to ~37deg, not exactly 40deg)."""
    corrupted_homography = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90deg rotation
    pair_result = PairResult(
        src_index=0,
        dst_index=1,
        src_points=np.zeros((10, 2)),
        dst_points=np.zeros((10, 2)),
        inlier_mask=np.ones(10, dtype=bool),
        homography=corrupted_homography,
    )
    true_rotation_deg = 40.0
    gimbal_yaw = {0: 0.0, 1: true_rotation_deg}  # relative_yaw s.t. theta_target = 40deg directly

    result_without = compose_global_transforms(
        [pair_result],
        gps_positions=None,
        reference_index=0,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=10.0,
    )
    result_with = compose_global_transforms(
        [pair_result],
        gps_positions=None,
        reference_index=0,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=10.0,
        gimbal_yaw=gimbal_yaw,
        yaw_anchor_weight=5.0,
    )

    def _angle_deg(result) -> float:
        pose = result.transforms[1]
        return float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))

    angle_without = _angle_deg(result_without)
    angle_with = _angle_deg(result_with)

    assert abs(angle_without - (-90.0)) < 1e-3  # unrescued baseline, same as the posegraph-level test
    assert abs(angle_with - true_rotation_deg) < 10.0
    assert abs(angle_with - true_rotation_deg) < 0.3 * abs(angle_without - true_rotation_deg)
