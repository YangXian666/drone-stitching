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


def test_noise_free_long_chain_with_heterogeneous_weights_is_recovered_exactly() -> None:
    # Weights span the real inlier_count/median range seen on data/ (~0.01..11). On a long
    # chain, the plain measurement matrix W's principal eigenvector localizes around the
    # heaviest edges and decays exponentially away from them until entries underflow to
    # exactly 0 -- their phases become meaningless (found on the 52-image run: 23
    # consecutive line-B nodes came out at exactly 0.00 deg, then +-180 deg jumps). Noise
    # is not needed to trigger it; the expected values are simply the true angles.
    n = 50
    true_rad = np.radians(np.arange(n) * 7.3 % 360 - 180)
    weights = np.geomspace(0.01, 11.0, n - 1)
    edges = [
        RelativeRotation(
            src_index=k,
            dst_index=k + 1,
            theta_rad=float(true_rad[k + 1] - true_rad[k]),
            weight=float(weights[k]),
        )
        for k in range(n - 1)
    ]

    result = average_rotations(edges)

    for k in range(n):
        _assert_angles_close(result.angles[k], true_rad[k] - true_rad[0], abs_tol=1e-9)


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
    # gives a circulant connection Laplacian whose constant mode is the solution, so every
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
# Real images: consistency only (no external truth -- see CLAUDE.md's scope decision)
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


def _forward_reverse_theta_sum_deg(name_a: str, name_b: str) -> float:
    image_a = cv2.imread(str(FIXTURES / name_a))
    image_b = cv2.imread(str(FIXTURES / name_b))
    matcher = _SiftRatioMatcher()
    forward = match_pair(matcher, image_a, image_b, 0, 1)
    reverse = match_pair(matcher, image_b, image_a, 1, 0)
    theta_ab = relative_rotation_from_homography(forward.homography, image_a.shape)
    theta_ba = relative_rotation_from_homography(reverse.homography, image_b.shape)
    return float(np.degrees(_wrap(theta_ab + theta_ba)))


def test_real_pair_forward_and_reverse_rotations_are_consistent() -> None:
    # Consistency, not accuracy: matching the same real pair in both directions (two
    # independent SIFT/RANSAC fits) must give theta_ab ~= -theta_ba. No external truth is
    # used -- per CLAUDE.md's 最小可行版本範圍決定, formal tests may not use GimbalYawDegree
    # or any DJI XMP field as an expected value. 0357 -> 0358 is a straight-line pair
    # with ~2400 inliers; measured when written: sum -0.066 deg (linearization-point part
    # +0.015 deg, separate-fit part +0.081 deg).
    total = _forward_reverse_theta_sum_deg(
        "DJI_20230127131440_0357_W.JPG", "DJI_20230127131442_0358_W.JPG"
    )

    assert abs(total) < 0.5


def test_real_weak_pair_forward_reverse_repeatability() -> None:
    # Weak-edge repeatability, not accuracy. 0352 -> 0353 (the U-turn) has the lowest
    # inlier count in the whole 52-image dataset (40 forward / 57 reverse). Measured when
    # written: theta_ab + theta_ba = -0.819 deg, decomposed into
    #   +0.047 deg  linearization reference point (src centre vs dst centre), and
    #   +0.867 deg  two independent SIFT/RANSAC fits (the reverse fit's perspective row is
    #               ~6x smaller than the forward fit's).
    # This is the edge's natural uncertainty, not a bug in theta extraction -- hence the
    # looser 1.0 deg bound than the high-inlier consistency test above.
    total = _forward_reverse_theta_sum_deg(
        "DJI_20230127131426_0352_W.JPG", "DJI_20230127131429_0353_W.JPG"
    )

    assert abs(total) < 1.0


# ---------------------------------------------------------------------------
# Optional heading anchors (e.g. GimbalYawDegree), added through a ground node
# ---------------------------------------------------------------------------
# Per CLAUDE.md's 範圍調整 (2026-09-26): GimbalYawDegree may enter Stage A as an optional
# weak anchor. With anchors the result is in the absolute frame (a node's angle equals
# its anchor's frame, e.g. compass yaw in the north-up pixel frame); without anchors
# Stage A must behave exactly as before. All expected values below are synthetic.


def _chain(true_deg: list[float], bias_deg: float = 0.0) -> list[RelativeRotation]:
    return [_edge(k, k + 1, true_deg[k + 1] - true_deg[k] + bias_deg) for k in range(len(true_deg) - 1)]


def test_anchor_weight_constant_is_the_measured_value() -> None:
    from sea_mosaic.rotation_averaging import GIMBAL_YAW_ANCHOR_WEIGHT

    assert GIMBAL_YAW_ANCHOR_WEIGHT == 1.0  # CLAUDE.md: chosen from the real-data sweep


def test_exact_anchors_give_absolute_angles() -> None:
    true_deg = [70.8, 70.8, 39.0, -23.2, -54.6, -104.3]
    anchors = {k: float(np.radians(a)) for k, a in enumerate(true_deg)}

    result = average_rotations(_chain(true_deg), heading_anchors=anchors, anchor_weight=1.0)

    for k, a in enumerate(true_deg):
        _assert_angles_close(result.angles[k], np.radians(a), abs_tol=1e-9)
    assert set(result.angles) == set(range(6))  # the internal ground node is not reported


def test_anchors_wrap_across_plus_minus_180() -> None:
    true_deg = [179.0, -179.0, 175.0, -170.0]
    anchors = {k: float(np.radians(a)) for k, a in enumerate(true_deg)}

    result = average_rotations(_chain(true_deg), heading_anchors=anchors, anchor_weight=1.0)

    for k, a in enumerate(true_deg):
        _assert_angles_close(result.angles[k], np.radians(a), abs_tol=1e-9)


def test_biased_chain_with_unbiased_anchors_matches_small_angle_least_squares() -> None:
    # Every edge carries +0.4 deg (loop-consistent drift); anchors are unbiased. Independent
    # derivation: with e_k = theta_k - psi_k, minimize sum (e_{k+1} - e_k - b)^2 (edge weight
    # 1) + w * sum e_k^2 -- a linear least-squares problem solved directly below. The
    # spectral relaxation matches it to third order in the inconsistency (~3e-4 deg here).
    true_deg = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    n, b, w = len(true_deg), np.radians(0.4), 1.0
    anchors = {k: float(np.radians(a)) for k, a in enumerate(true_deg)}
    rows, rhs = [], []
    for k in range(n - 1):
        row = np.zeros(n)
        row[k], row[k + 1] = -1.0, 1.0
        rows.append(row)
        rhs.append(b)
    for k in range(n):
        row = np.zeros(n)
        row[k] = np.sqrt(w)
        rows.append(row)
        rhs.append(0.0)
    expected_error = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]

    result = average_rotations(_chain(true_deg, bias_deg=0.4), heading_anchors=anchors, anchor_weight=w)

    for k in range(n):
        actual_error_deg = np.degrees(_wrap(result.angles[k] - np.radians(true_deg[k])))
        assert actual_error_deg == pytest.approx(np.degrees(expected_error[k]), abs=1e-3)


def test_anchors_suppress_drift_compared_with_no_anchors() -> None:
    true_deg = [0.0] * 12
    edges = _chain(true_deg, bias_deg=0.4)

    free = average_rotations(edges)
    anchored = average_rotations(edges, heading_anchors={k: 0.0 for k in range(12)}, anchor_weight=1.0)

    # Without anchors the gauge is node 0, so drift accumulates to 11 * 0.4 = 4.4 deg.
    assert np.degrees(_wrap(free.angles[11] - free.angles[0])) == pytest.approx(4.4, abs=1e-6)
    worst = max(abs(np.degrees(_wrap(a))) for a in anchored.angles.values())
    assert worst < 1.0


def test_inconsistent_anchors_in_one_component_resolve_to_weighted_least_squares() -> None:
    # Several anchored nodes in one component, anchors with small independent errors n_k
    # (like per-image GimbalYawDegree sensor noise); edges exact. Explicit rule: the frame
    # is fixed by the ground node, and the orientation is the least-squares compromise of
    # all anchors -- no single anchor defines it. Three checks:
    # (1) equals the small-angle linear least squares
    #     min sum_edges w_e (e_j - e_i)^2 + w sum_k (e_k - n_k)^2, solved directly;
    # (2) exact invariant (edge gradients cancel pairwise): with equal anchor weights,
    #     mean_k e_k == mean_k n_k, whatever the edge weights;
    # (3) rigid limit (edge weight 1e6): every node is offset by mean_k n_k.
    true_deg = [70.8, 70.8, 70.8, 71.0, 70.5, 70.8]
    noise_deg = [0.3, -0.2, 0.5, -0.1, 0.0, 0.2]
    n, w = len(true_deg), 1.0
    anchors = {k: float(np.radians(true_deg[k] + noise_deg[k])) for k in range(n)}

    for edge_weight in (1.0, 1e6):
        edges = [
            RelativeRotation(k, k + 1, float(np.radians(true_deg[k + 1] - true_deg[k])), edge_weight)
            for k in range(n - 1)
        ]
        result = average_rotations(edges, heading_anchors=anchors, anchor_weight=w)
        errors_deg = np.array([np.degrees(_wrap(result.angles[k] - np.radians(true_deg[k]))) for k in range(n)])

        rows, rhs = [], []
        for k in range(n - 1):
            row = np.zeros(n)
            row[k], row[k + 1] = -np.sqrt(edge_weight), np.sqrt(edge_weight)
            rows.append(row)
            rhs.append(0.0)
        for k in range(n):
            row = np.zeros(n)
            row[k] = np.sqrt(w)
            rows.append(row)
            rhs.append(np.sqrt(w) * np.radians(noise_deg[k]))
        expected_deg = np.degrees(np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0])

        assert errors_deg == pytest.approx(expected_deg, abs=1e-3)  # (1)
        assert errors_deg.mean() == pytest.approx(np.mean(noise_deg), abs=1e-3)  # (2)
        if edge_weight == 1e6:
            assert errors_deg == pytest.approx([np.mean(noise_deg)] * n, abs=1e-3)  # (3)


def test_nodes_without_anchor_follow_edges_in_the_absolute_frame() -> None:
    true_deg = [30.0, 45.0, 60.0, 75.0]
    anchors = {0: float(np.radians(30.0))}  # only node 0 anchored

    result = average_rotations(_chain(true_deg), heading_anchors=anchors, anchor_weight=1.0)

    for k, a in enumerate(true_deg):
        _assert_angles_close(result.angles[k], np.radians(a), abs_tol=1e-9)


def test_anchors_join_otherwise_disconnected_components() -> None:
    edges = [_edge(0, 1, 20.0), _edge(5, 6, -10.0)]
    anchors = {0: float(np.radians(100.0)), 5: float(np.radians(-60.0))}

    result = average_rotations(edges, heading_anchors=anchors, anchor_weight=1.0)

    assert len(set(result.component_of.values())) == 1
    _assert_angles_close(result.angles[1], np.radians(120.0), abs_tol=1e-9)
    _assert_angles_close(result.angles[6], np.radians(-70.0), abs_tol=1e-9)


def test_anchor_only_node_takes_its_anchor_value() -> None:
    result = average_rotations([_edge(0, 1, 15.0)], heading_anchors={0: 0.2, 9: -1.1}, anchor_weight=1.0)

    _assert_angles_close(result.angles[9], -1.1, abs_tol=1e-9)
    _assert_angles_close(result.angles[1], 0.2 + np.radians(15.0), abs_tol=1e-9)


@pytest.mark.parametrize(
    "anchors, weight",
    [
        ({0: 0.1}, None),
        ({0: 0.1}, 0.0),
        ({0: 0.1}, -1.0),
        ({0: 0.1}, float("nan")),
        ({0: float("nan")}, 1.0),
        ({0: float("inf")}, 1.0),
    ],
)
def test_invalid_anchor_inputs_raise(anchors, weight) -> None:
    with pytest.raises(ValueError):
        average_rotations([_edge(0, 1, 5.0)], heading_anchors=anchors, anchor_weight=weight)


def test_no_anchors_is_bitwise_identical_to_before() -> None:
    # Values recorded (float.hex) from average_rotations before anchors existed.
    edges = [
        RelativeRotation(0, 1, 0.30, 1.0),
        RelativeRotation(1, 2, -0.52, 2.5),
        RelativeRotation(2, 3, 1.10, 0.4),
        RelativeRotation(3, 0, -0.95, 1.7),
        RelativeRotation(1, 3, 0.61, 0.9),
    ]
    recorded = {0: "0x0.0p+0", 1: "0x1.47e28ca1b8450p-2", 2: "-0x1.8e7b22560a730p-3", 3: "0x1.e050f4a2faa90p-1"}

    for result in (average_rotations(edges), average_rotations(edges, heading_anchors=None)):
        assert {k: float(v).hex() for k, v in sorted(result.angles.items())} == recorded
        assert result.component_of == {0: 0, 1: 0, 2: 0, 3: 0}
