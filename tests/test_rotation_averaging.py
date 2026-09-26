"""Tests for sea_mosaic.rotation_averaging (Stage A: pure rotation averaging).

Stage A only consumes edges' relative rotations -- no GPS, no GimbalYawDegree (see
CLAUDE.md's 分階段架構決定). Every expected value here comes from an independent
derivation (hand calculation, a closed form, or an independently constructed camera
model), never from calling the function under test to compute its own expectation --
the lesson of the _yaw_target_vector sign bug recorded in CLAUDE.md.

Convention under test: theta_ab = theta_dst - theta_src, where a node's angle is the
rotation of its image-to-mosaic pose. For a pair's homography H (src pixels -> dst
pixels, i.e. H ~= inv(pose_dst) @ pose_src), theta_ab = -angle(H).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sea_mosaic.estimate import match_pair
from sea_mosaic.io_utils import load_gimbal_yaw
from sea_mosaic.matcher import MatchResult
from sea_mosaic.rotation_averaging import (
    RelativeRotation,
    RotationAveragingResult,
    average_rotations,
    relative_rotation_from_homography,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dji_smoke"


def _wrap(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2 * np.pi) - np.pi)


def _assert_angles_close(actual: float, expected: float, abs_tol: float = 1e-9) -> None:
    assert abs(_wrap(actual - expected)) < abs_tol, (np.degrees(actual), np.degrees(expected))


def _edge(src: int, dst: int, theta_deg: float, weight: float = 1.0) -> RelativeRotation:
    return RelativeRotation(
        src_index=src, dst_index=dst, theta_rad=float(np.radians(theta_deg)), weight=weight
    )


# ---------------------------------------------------------------------------
# average_rotations
# ---------------------------------------------------------------------------


def test_single_edge_recovers_relative_angle_with_smallest_index_as_gauge() -> None:
    result = average_rotations([_edge(0, 1, 30.0)])

    assert isinstance(result, RotationAveragingResult)
    _assert_angles_close(result.angles[0], 0.0)
    _assert_angles_close(result.angles[1], np.radians(30.0))


def test_noise_free_chain_across_plus_minus_180_is_recovered_exactly() -> None:
    # Serpentine-like: a 175 deg flip plus angles that straddle +-180 deg. The expected
    # values are simply the true angles relative to node 0.
    true_deg = {0: 0.0, 1: 170.0, 2: -175.0, 3: -5.0, 4: 179.0}
    edges = [_edge(a, a + 1, true_deg[a + 1] - true_deg[a]) for a in range(4)]

    result = average_rotations(edges)

    for index, angle_deg in true_deg.items():
        _assert_angles_close(result.angles[index], np.radians(angle_deg))


def test_gauge_uses_smallest_index_for_non_contiguous_indices() -> None:
    result = average_rotations([_edge(42, 7, -20.0), _edge(7, 3, 50.0)])

    # theta_7 = theta_42 - 20, theta_3 = theta_7 + 50; gauge: theta_3 = 0.
    _assert_angles_close(result.angles[3], 0.0)
    _assert_angles_close(result.angles[7], np.radians(-50.0))
    _assert_angles_close(result.angles[42], np.radians(-30.0))
    assert set(result.angles) == {3, 7, 42}


def test_result_is_invariant_to_edge_order_and_edge_direction() -> None:
    edges = [_edge(0, 1, 40.0), _edge(1, 2, -70.0), _edge(2, 3, 15.0), _edge(3, 0, 20.0, 2.0)]
    shuffled_and_reversed = [
        RelativeRotation(src_index=0, dst_index=3, theta_rad=float(np.radians(-20.0)), weight=2.0),
        _edge(2, 3, 15.0),
        RelativeRotation(src_index=2, dst_index=1, theta_rad=float(np.radians(70.0)), weight=1.0),
        _edge(0, 1, 40.0),
    ]

    a = average_rotations(edges)
    b = average_rotations(shuffled_and_reversed)

    for index in a.angles:
        _assert_angles_close(a.angles[index], b.angles[index], abs_tol=1e-9)


@pytest.mark.parametrize("n, loop_error_deg", [(3, 6.0), (5, 10.0)])
def test_equal_weight_loop_spreads_loop_error_evenly(n: int, loop_error_deg: float) -> None:
    # Closed form: gauge-transforming an equal-weight n-cycle with total loop error eps
    # gives a circulant measurement matrix whose constant mode is the solution, so every
    # edge ends up with residual exactly eps/n.
    true_deg = [37.0 * k for k in range(n)]
    edges = [_edge(k, (k + 1) % n, true_deg[(k + 1) % n] - true_deg[k]) for k in range(n)]
    corrupted = edges[-1]
    edges[-1] = RelativeRotation(
        src_index=corrupted.src_index,
        dst_index=corrupted.dst_index,
        theta_rad=corrupted.theta_rad + float(np.radians(loop_error_deg)),
        weight=1.0,
    )

    result = average_rotations(edges)

    for edge in edges:
        residual = _wrap(
            result.angles[edge.dst_index] - result.angles[edge.src_index] - edge.theta_rad
        )
        assert residual == pytest.approx(-np.radians(loop_error_deg) / n, abs=1e-9)


def test_parallel_edges_average_as_weighted_complex_mean() -> None:
    # Two nodes, two measurements: closed form arg(w1 e^{i th1} + w2 e^{i th2}).
    w1, th1, w2, th2 = 3.0, np.radians(10.0), 1.0, np.radians(50.0)
    expected = float(np.angle(w1 * np.exp(1j * th1) + w2 * np.exp(1j * th2)))

    result = average_rotations(
        [
            RelativeRotation(src_index=0, dst_index=1, theta_rad=float(th1), weight=w1),
            RelativeRotation(src_index=0, dst_index=1, theta_rad=float(th2), weight=w2),
        ]
    )

    _assert_angles_close(result.angles[1] - result.angles[0], expected)


def test_disconnected_components_are_labelled_and_gauged_separately() -> None:
    component_a = [_edge(0, 1, 30.0), _edge(1, 2, 30.0)]
    component_b = [_edge(5, 9, -45.0)]

    result = average_rotations(component_a + component_b)
    alone_a = average_rotations(component_a)
    alone_b = average_rotations(component_b)

    assert result.component_of[0] == result.component_of[1] == result.component_of[2]
    assert result.component_of[5] == result.component_of[9]
    assert result.component_of[0] != result.component_of[5]
    # Each component has its own gauge (its own smallest index at 0).
    _assert_angles_close(result.angles[0], 0.0)
    _assert_angles_close(result.angles[5], 0.0)
    for index, angle in {**alone_a.angles, **alone_b.angles}.items():
        _assert_angles_close(result.angles[index], angle)


def test_component_ids_are_ordered_by_smallest_node_index() -> None:
    result = average_rotations([_edge(8, 9, 10.0), _edge(1, 4, 10.0)])

    assert result.component_of[1] == result.component_of[4] == 0
    assert result.component_of[8] == result.component_of[9] == 1


@pytest.mark.parametrize(
    "bad_edge",
    [
        RelativeRotation(src_index=2, dst_index=2, theta_rad=0.1, weight=1.0),
        RelativeRotation(src_index=0, dst_index=1, theta_rad=float("nan"), weight=1.0),
        RelativeRotation(src_index=0, dst_index=1, theta_rad=float("inf"), weight=1.0),
        RelativeRotation(src_index=0, dst_index=1, theta_rad=0.1, weight=0.0),
        RelativeRotation(src_index=0, dst_index=1, theta_rad=0.1, weight=-1.0),
        RelativeRotation(src_index=0, dst_index=1, theta_rad=0.1, weight=float("nan")),
    ],
)
def test_invalid_edges_raise_value_error(bad_edge: RelativeRotation) -> None:
    with pytest.raises(ValueError):
        average_rotations([_edge(0, 1, 5.0), bad_edge])


def test_empty_edges_returns_empty_result() -> None:
    result = average_rotations([])

    assert result.angles == {}
    assert result.component_of == {}


def test_single_outlier_edge_corrupts_every_edge_in_its_loop() -> None:
    # Capability boundary (not a feature): the spectral relaxation is least-squares, not
    # robust. One edge wrong by 90 deg in an equal-weight 4-cycle drags every edge by
    # 90/4 = 22.5 deg (same closed form as the loop-error test) -- correct edges included.
    n, outlier_deg = 4, 90.0
    edges = [_edge(k, (k + 1) % n, 0.0) for k in range(n)]
    edges[0] = _edge(0, 1, outlier_deg)

    result = average_rotations(edges)

    for edge in edges[1:]:
        residual = _wrap(
            result.angles[edge.dst_index] - result.angles[edge.src_index] - edge.theta_rad
        )
        assert abs(np.degrees(residual)) == pytest.approx(outlier_deg / n, abs=1e-6)


# ---------------------------------------------------------------------------
# relative_rotation_from_homography
# ---------------------------------------------------------------------------

_IMAGE_SHAPE = (3040, 4056)  # (rows, cols), the DJI H20T wide camera


def _sim2(theta_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize(
    "theta_src_deg, theta_dst_deg, scale",
    [(0.0, 30.0, 1.0), (39.0, -23.2, 1.0), (170.0, -175.0, 1.0), (-60.0, 45.0, 1.3)],
)
def test_similarity_homography_gives_dst_minus_src_angle(
    theta_src_deg: float, theta_dst_deg: float, scale: float
) -> None:
    # Built independently from two node poses: H = inv(pose_dst) @ pose_src.
    pose_src = _sim2(theta_src_deg, 1.0, 120.0, -40.0)
    pose_dst = _sim2(theta_dst_deg, scale, 2500.0, 700.0)
    homography = np.linalg.inv(pose_dst) @ pose_src

    theta_ab = relative_rotation_from_homography(homography, _IMAGE_SHAPE)

    _assert_angles_close(theta_ab, np.radians(theta_dst_deg - theta_src_deg))


def _pinhole_plane_homography(
    yaw_src_deg: float,
    yaw_dst_deg: float,
    displacement_en_m: tuple[float, float],
    tilt_dst_deg: float,
    tilt_azimuth_deg: float,
) -> np.ndarray:
    """Exact plane-induced homography between two downward cameras over ground z=0.

    World frame is (East, North, Up); a camera with yaw psi (compass bearing, clockwise
    from north) has its image top pointing along psi. Camera 2 can carry an extra small
    tilt about a horizontal camera-frame axis at tilt_azimuth_deg.
    """
    rows, cols = _IMAGE_SHAPE
    f = np.hypot(cols, rows) / 2 / np.tan(np.radians(82.9 / 2))
    K = np.array([[f, 0.0, cols / 2], [0.0, f, rows / 2], [0.0, 0.0, 1.0]])

    def world_to_camera(yaw_deg: float, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        p = np.radians(yaw_deg)
        nadir = np.array(
            [[np.cos(p), -np.sin(p), 0.0], [-np.sin(p), -np.cos(p), 0.0], [0.0, 0.0, -1.0]]
        )
        tilt_vec = np.radians(tilt_deg) * np.array(
            [np.cos(np.radians(azimuth_deg)), np.sin(np.radians(azimuth_deg)), 0.0]
        )
        tilt, _ = cv2.Rodrigues(tilt_vec)
        return tilt @ nadir

    def ground_to_image(R: np.ndarray, centre: np.ndarray) -> np.ndarray:
        return K @ np.column_stack([R[:, 0], R[:, 1], -R @ centre])

    altitude = 100.0
    G_src = ground_to_image(world_to_camera(yaw_src_deg, 0.0, 0.0), np.array([0.0, 0.0, altitude]))
    G_dst = ground_to_image(
        world_to_camera(yaw_dst_deg, tilt_dst_deg, tilt_azimuth_deg),
        np.array([displacement_en_m[0], displacement_en_m[1], altitude]),
    )
    H = G_dst @ np.linalg.inv(G_src)
    return H / H[2, 2]


def _upper_left_polar_angle(homography: np.ndarray) -> float:
    # Reference for the biased alternative (what optimize_pose_graph implicitly uses).
    U, _, Vt = np.linalg.svd(homography[:2, :2])
    R = U @ Vt
    return float(-np.arctan2(R[1, 0], R[0, 0]))


def test_untilted_pinhole_homography_recovers_camera_yaw_difference_exactly() -> None:
    # Independent truth: the two cameras' yaws. Also pins the sign convention
    # theta_ab = psi_dst - psi_src for real (compass-bearing) camera yaws.
    H = _pinhole_plane_homography(39.0, -23.2, (-3.8, 14.2), 0.0, 0.0)

    theta_ab = relative_rotation_from_homography(H, _IMAGE_SHAPE)

    _assert_angles_close(theta_ab, np.radians(-23.2 - 39.0), abs_tol=1e-9)


def test_tilted_pinhole_homography_centre_jacobian_beats_upper_left_block() -> None:
    # Mechanism behind CLAUDE.md's 第二個座標系陷阱: with a perspective row, the angle
    # depends on where it's linearized. Measured on this model (2 deg tilt at 45 deg
    # azimuth, straight-line leg): centre-Jacobian error 0.095 deg vs H[:2,:2]-polar
    # error 0.876 deg.
    yaw = 70.8
    step = (13.5 * np.sin(np.radians(yaw)), 13.5 * np.cos(np.radians(yaw)))
    H = _pinhole_plane_homography(yaw, yaw, step, 2.0, 45.0)

    centre_error_deg = abs(np.degrees(_wrap(relative_rotation_from_homography(H, _IMAGE_SHAPE))))
    upper_left_error_deg = abs(np.degrees(_wrap(_upper_left_polar_angle(H))))

    assert centre_error_deg < 0.15
    assert upper_left_error_deg > 3 * centre_error_deg


def test_reflecting_homography_raises_value_error() -> None:
    mirror = np.diag([1.0, -1.0, 1.0])

    with pytest.raises(ValueError):
        relative_rotation_from_homography(mirror, _IMAGE_SHAPE)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_non_finite_homography_raises_value_error(bad_value: float) -> None:
    H = np.eye(3)
    H[0, 1] = bad_value

    with pytest.raises(ValueError):
        relative_rotation_from_homography(H, _IMAGE_SHAPE)


# ---------------------------------------------------------------------------
# End to end on real images, against an independently measured truth
# ---------------------------------------------------------------------------


class _SiftRatioMatcher:
    """Test-local SIFT + knnMatch + Lowe ratio 0.75 matcher (the configuration CLAUDE.md
    settled on for this data; the production SIFT matcher is not implemented yet)."""

    name = "sift_bf_ratio"

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        sift = cv2.SIFT_create()
        kp_a, des_a = sift.detectAndCompute(cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY), None)
        kp_b, des_b = sift.detectAndCompute(cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY), None)
        good = [
            m
            for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
            if m.distance < 0.75 * n.distance
        ]
        return MatchResult(
            src_points=np.float64([kp_a[m.queryIdx].pt for m in good]),
            dst_points=np.float64([kp_b[m.trainIdx].pt for m in good]),
        )


def test_real_pair_rotation_matches_gimbal_yaw_difference() -> None:
    # 0352 -> 0353 (the U-turn). Truth is the GimbalYawDegree difference (-62.2 deg), an
    # independent sensor reading never used by Stage A itself. Measured when this test was
    # written: -62.785 deg (error -0.585 deg, 40 inliers).
    path_src = FIXTURES / "DJI_20230127131426_0352_W.JPG"
    path_dst = FIXTURES / "DJI_20230127131429_0353_W.JPG"
    image_src, image_dst = cv2.imread(str(path_src)), cv2.imread(str(path_dst))
    pair = match_pair(_SiftRatioMatcher(), image_src, image_dst, 0, 1)

    theta_ab = relative_rotation_from_homography(pair.homography, image_src.shape)
    result = average_rotations(
        [RelativeRotation(src_index=0, dst_index=1, theta_rad=theta_ab, weight=1.0)]
    )

    gimbal_delta = np.radians(load_gimbal_yaw(path_dst) - load_gimbal_yaw(path_src))
    error_deg = np.degrees(_wrap(result.angles[1] - result.angles[0] - gimbal_delta))
    assert abs(error_deg) < 1.0
