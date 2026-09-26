"""Tests for sea_mosaic.refinement (Stage D: bounded refinement from Stage C's poses).

Stage D refines only positions (CLAUDE.md's Stage D 改為只精修位置): headings phi_i come
from Stage A+C unchanged, and the unknowns are each node's image-centre position m_i in the
north-up pixel frame plus one global kappa on the GPS targets; pose_i(x) = R(phi_i)(x - c_i)
+ m_i with scale 1. Residual terms: sampled RANSAC inlier correspondences
pose_i(x_src) - pose_j(x_dst) = (m_i - m_j) - d_k with d_k a known constant per point (a pure
relative-translation constraint, no homography linearization), and GPS m_i - kappa * p_i.
Bounds are guard rails; every hit is reported.

No GimbalYawDegree or DJI XMP field is used anywhere in Stage D;
expected values come from hand calculation, closed forms, or the synthetic pinhole
camera model. Where a test states a capability ratio, the value measured when the test
was written is recorded next to it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sea_mosaic.estimate import match_pair
from sea_mosaic.frame_alignment import AlignedPoses, HeadingEdge, align_to_gps_frame
from sea_mosaic.gps_placement import (
    GpsPlacement,
    PairDisplacement,
    estimate_pixels_per_meter,
    load_exif_latlons,
    place_by_gps,
)
from sea_mosaic.refinement import (
    EDGE_SIGMA_FLOOR_PX,
    GPS_SIGMA_M,
    K_POINTS_PER_EDGE,
    KAPPA_BOUND,
    MAX_POSITION_CHANGE_M,
    RefinementEdge,
    edge_from_pair_result,
    refine_poses,
    sample_correspondences,
)
from sea_mosaic.rotation_averaging import (
    RelativeRotation,
    average_rotations,
    relative_rotation_from_homography,
)
from synthetic_camera import (
    FOCAL_PX,
    IMAGE_CENTRE,
    IMAGE_SHAPE,
    SyntheticCamera,
    ground_to_image,
    latlon_from_en,
    plane_homography,
)
from test_gps_placement import FIXTURES, _SiftRatioMatcher, _strip_xmp

ALTITUDE = 100.0
PPM = FOCAL_PX / ALTITUDE  # exact ground scale of an untilted nadir pinhole camera


# ---------------------------------------------------------------------------
# Synthetic world helpers
# ---------------------------------------------------------------------------


def _wrap(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2 * np.pi) - np.pi)


def _line(n: int, yaw_deg: float = 70.8, step_m: float = 13.4, start=(0.0, 0.0)) -> list[SyntheticCamera]:
    course = np.radians(83.0 if yaw_deg == 70.8 else yaw_deg)
    return [
        SyntheticCamera(start[0] + k * step_m * np.sin(course), start[1] + k * step_m * np.cos(course), ALTITUDE, yaw_deg)
        for k in range(n)
    ]


def _true_pose(camera: SyntheticCamera, origin: SyntheticCamera) -> np.ndarray:
    """Image-to-frame pose of a camera: pixel -> ground (E, N) -> north-up pixels, origin at
    the origin camera's ground point. For untilted nadir cameras this is a pure rotation by
    the compass yaw with scale 1 (hand derivation, verified in the helper test below)."""
    to_frame = np.array(
        [[PPM, 0.0, -PPM * origin.east_m], [0.0, -PPM, PPM * origin.north_m], [0.0, 0.0, 1.0]]
    )
    pose = to_frame @ np.linalg.inv(ground_to_image(camera))
    return pose / pose[2, 2]


def _centre(camera: SyntheticCamera, origin: SyntheticCamera) -> np.ndarray:
    return PPM * np.array([camera.east_m - origin.east_m, -(camera.north_m - origin.north_m)])


def _correspondences(src: SyntheticCamera, dst: SyntheticCamera, spacing_m: float = 5.0):
    """Ground-grid points projected exactly into both cameras, kept where both see them."""
    rows, cols = IMAGE_SHAPE
    lo_e, hi_e = min(src.east_m, dst.east_m) - 80, max(src.east_m, dst.east_m) + 80
    lo_n, hi_n = min(src.north_m, dst.north_m) - 80, max(src.north_m, dst.north_m) + 80
    east, north = np.meshgrid(np.arange(lo_e, hi_e, spacing_m), np.arange(lo_n, hi_n, spacing_m))
    ground = np.stack([east.ravel(), north.ravel(), np.ones(east.size)])

    def project(camera):
        p = ground_to_image(camera) @ ground
        return (p[:2] / p[2]).T

    a, b = project(src), project(dst)
    inside = np.all((a >= 0) & (a < [cols, rows]) & (b >= 0) & (b < [cols, rows]), axis=1)
    return a[inside], b[inside]


def _rotation_about_centre(angle_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    R = np.array([[c, -s], [s, c]])
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = IMAGE_CENTRE - R @ IMAGE_CENTRE
    return T


def _edge(cameras, i, j, dst_rotation_deg: float = 0.0) -> RefinementEdge:
    """Exact edge; optionally rotate the dst points (and the homography, consistently)
    about the dst image centre to inject a rotation bias into this edge."""
    src_points, dst_points = _correspondences(cameras[i], cameras[j])
    H = plane_homography(cameras[i], cameras[j])
    if dst_rotation_deg:
        T = _rotation_about_centre(dst_rotation_deg)
        dst_points = (T[:2, :2] @ dst_points.T).T + T[:2, 2]
        H = T @ H
    return RefinementEdge(i, j, src_points, dst_points, H, IMAGE_SHAPE)


def _world(cameras: dict[int, SyntheticCamera], pairs, *, init_heading_offsets_deg=None,
           init_position_offsets_m=None, gps_scale=1.0, gps_offsets_m=None, edge_bias_deg=None):
    """Stage C output (poses = truth unless perturbed), Stage B placement (GPS targets),
    edges, and the true poses/centres."""
    origin = cameras[min(cameras)]
    truth = {k: _true_pose(c, origin) for k, c in cameras.items()}
    centres = {k: _centre(c, origin) for k, c in cameras.items()}
    headings = {k: np.radians(c.yaw_deg) for k, c in cameras.items()}

    init_poses, init_headings = {}, {}
    for k in cameras:
        phi = headings[k] + np.radians((init_heading_offsets_deg or {}).get(k, 0.0))
        m = centres[k] + PPM * np.asarray((init_position_offsets_m or {}).get(k, (0.0, 0.0)))
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        pose = np.eye(3)
        pose[:2, :2] = R
        pose[:2, 2] = m - R @ IMAGE_CENTRE
        init_poses[k], init_headings[k] = pose, _wrap(phi)
    aligned = AlignedPoses(
        poses=init_poses, headings_rad=init_headings, component_offsets_rad={0: 0.0},
        unaligned_components=set(), unlocated=set(), unoriented=set(),
    )
    targets = {
        k: gps_scale * centres[k] + PPM * np.asarray((gps_offsets_m or {}).get(k, (0.0, 0.0)))
        for k in cameras
    }
    placement = GpsPlacement(centres_px=targets, unlocated=set(), origin_index=min(cameras))
    edges = [_edge(cameras, i, j, (edge_bias_deg or {}).get((i, j), 0.0)) for i, j in pairs]
    return aligned, placement, edges, truth, centres, headings


SHAPES = {k: IMAGE_SHAPE for k in range(40)}


def _heading_errors_deg(result, headings) -> np.ndarray:
    return np.array([abs(np.degrees(_wrap(result.headings_rad[k] - headings[k]))) for k in headings])


def test_helper_true_pose_is_rotation_by_compass_yaw_with_unit_scale() -> None:
    cam = SyntheticCamera(10.0, 5.0, ALTITUDE, -104.3)
    pose = _true_pose(cam, SyntheticCamera(0.0, 0.0, ALTITUDE, 0.0))

    assert np.hypot(pose[0, 0], pose[1, 0]) == pytest.approx(1.0, abs=1e-9)
    assert np.degrees(np.arctan2(pose[1, 0], pose[0, 0])) == pytest.approx(-104.3, abs=1e-9)
    assert pose @ np.append(IMAGE_CENTRE, 1.0) == pytest.approx([10.0 * PPM, -5.0 * PPM, 1.0], abs=1e-6)


# ---------------------------------------------------------------------------
# D1 / D2 / D3 / D6: behaviour on synthetic data with known truth
# ---------------------------------------------------------------------------


def test_perfect_data_is_a_fixed_point() -> None:  # D1
    cameras = dict(enumerate(_line(4)))
    aligned, placement, edges, truth, centres, headings = _world(cameras, [(0, 1), (1, 2), (2, 3), (0, 2)])

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    assert result.status == "converged"
    assert result.kappa == pytest.approx(1.0, abs=1e-9)
    assert result.bound_hits.position == set()
    assert not result.bound_hits.kappa
    for k in cameras:
        assert result.poses[k] == pytest.approx(truth[k], abs=1e-6)
        assert result.centres_px[k] == pytest.approx(centres[k], abs=1e-6)
        assert _wrap(result.headings_rad[k] - headings[k]) == pytest.approx(0.0, abs=1e-8)


def test_headings_pass_through_bitwise_even_when_drifted_and_edges_biased() -> None:  # N1
    # Stage D never touches headings: whatever Stage C hands over (here with injected drift)
    # comes out bit-for-bit, and each pose's rotation block is exactly R(heading).
    cameras = dict(enumerate(_line(6)))
    pairs = [(k, k + 1) for k in range(5)]
    aligned, placement, edges, _, _, _ = _world(
        cameras, pairs, init_heading_offsets_deg={k: 0.5 * k for k in cameras},
        edge_bias_deg={p: 0.4 for p in pairs},
    )

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    for k in cameras:
        phi = aligned.headings_rad[k]
        assert result.headings_rad[k] == phi
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        assert np.array_equal(result.poses[k][:2, :2], R)


def test_positions_equal_independently_built_linear_least_squares() -> None:  # N2
    # With headings fixed, every correspondence gives (m_i - m_j) - d_k, d_k = R_j(x_d - c) -
    # R_i(x_s - c), and GPS gives m_i - kappa p_i: a linear problem in (m, kappa). Built and
    # solved here directly (lstsq) from the documented sigmas, with slightly wrong headings
    # and noisy GPS so the answer is not trivially the truth. Edges carry <= K points, so
    # every point is used and the sampling order does not matter.
    cameras = dict(enumerate(_line(3)))
    pairs = [(0, 1), (1, 2), (0, 2)]
    aligned, placement, full_edges, _, _, _ = _world(
        cameras, pairs, init_heading_offsets_deg={0: 0.3, 1: -0.2, 2: 0.1},
        gps_offsets_m={0: (0.4, -0.1), 1: (-0.3, 0.2), 2: (0.1, 0.3)},
    )
    edges = [RefinementEdge(e.src_index, e.dst_index, e.src_points[:6], e.dst_points[:6], e.homography, IMAGE_SHAPE) for e in full_edges]

    result = refine_poses(aligned, placement, edges, PPM, SHAPES, loss="linear")

    rot = {k: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for k, a in aligned.headings_rad.items()}
    col = {k: 2 * k for k in cameras}  # m_k at columns 2k, 2k+1; kappa last
    rows, rhs = [], []
    for e in edges:  # sigma_edge is the floor: exact synthetic points have ~0 reprojection error
        for xs, xd in zip(e.src_points, e.dst_points):
            d = rot[e.dst_index] @ (xd - IMAGE_CENTRE) - rot[e.src_index] @ (xs - IMAGE_CENTRE)
            for axis in range(2):
                row = np.zeros(7)
                row[col[e.src_index] + axis], row[col[e.dst_index] + axis] = 1.0, -1.0
                rows.append(row / EDGE_SIGMA_FLOOR_PX)
                rhs.append(d[axis] / EDGE_SIGMA_FLOOR_PX)
    sigma_gps = GPS_SIGMA_M * PPM
    for k in cameras:
        for axis in range(2):
            row = np.zeros(7)
            row[col[k] + axis], row[6] = 1.0, -placement.centres_px[k][axis]
            rows.append(row / sigma_gps)
            rhs.append(0.0)
    solution = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]

    assert result.kappa == pytest.approx(solution[6], abs=1e-9)
    for k in cameras:
        assert result.centres_px[k] == pytest.approx(solution[col[k] : col[k] + 2], abs=1e-6)


def test_uniform_gps_shift_moves_all_centres_by_that_shift() -> None:  # D3a
    cameras = dict(enumerate(_line(4)))
    shift_m = (3.0, -2.0)
    aligned, placement, edges, _, centres, headings = _world(
        cameras, [(0, 1), (1, 2), (2, 3)], gps_offsets_m={k: shift_m for k in range(4)}
    )

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    assert result.kappa == pytest.approx(1.0, abs=1e-9)
    for k in cameras:
        assert result.centres_px[k] == pytest.approx(centres[k] + PPM * np.array(shift_m), abs=1e-4)
        assert _wrap(result.headings_rad[k] - headings[k]) == pytest.approx(0.0, abs=1e-8)


def test_per_node_gps_noise_does_not_distort_local_geometry() -> None:  # D3b
    # +-0.37 m alternating GPS noise (zero centroid) vs exact edges and exact (fixed)
    # headings: the local geometry must follow the edges, not the GPS noise (~10 px).
    # Rigid-invariant checks (neighbour distances, turning angles) plus -- now that headings
    # are fixed and nothing can rotate the block -- the neighbour displacement vectors
    # themselves. Bounds 0.5 px / 0.01 deg.
    cameras = dict(enumerate(_line(6)))
    pairs = [(k, k + 1) for k in range(5)]
    noise = {k: ((0.37, -0.37) if k % 2 == 0 else (-0.37, 0.37)) for k in cameras}
    aligned, placement, edges, _, centres, _ = _world(cameras, pairs, gps_offsets_m=noise)

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    def turning_deg(points, a, b, c):
        u, v = points[b] - points[a], points[c] - points[b]
        return np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v))

    for i, j in pairs:
        refined = result.centres_px[j] - result.centres_px[i]
        assert abs(np.linalg.norm(refined) - np.linalg.norm(centres[j] - centres[i])) < 0.5
        assert np.linalg.norm(refined - (centres[j] - centres[i])) < 0.5
    for a in range(4):
        diff = turning_deg(result.centres_px, a, a + 1, a + 2) - turning_deg(centres, a, a + 1, a + 2)
        assert abs(diff) < 0.01


def test_kappa_absorbs_a_pixels_per_meter_error() -> None:  # D6
    # GPS targets computed with a ppm estimate 3% too high -> kappa = 1/1.03 exactly.
    cameras = dict(enumerate(_line(4)))
    aligned, placement, edges, truth, _, _ = _world(cameras, [(0, 1), (1, 2), (2, 3)], gps_scale=1.03)

    result = refine_poses(aligned, placement, edges, PPM * 1.03, SHAPES)

    assert result.kappa == pytest.approx(1.0 / 1.03, rel=1e-6)
    for k in cameras:
        assert result.poses[k] == pytest.approx(truth[k], abs=1e-4)


# ---------------------------------------------------------------------------
# D4: robust loss
# ---------------------------------------------------------------------------


def test_huber_loss_limits_damage_from_one_wrong_edge() -> None:  # D4
    # Line with skip edges (redundancy); edge 3->4's dst points rotated an extra 30 deg
    # about the dst centre (homography rotated consistently, so its own reprojection error
    # stays tiny). With headings fixed this edge becomes a gross relative-translation
    # outlier. Capability claim: worst position error with Huber <= 0.25 x with a linear
    # (plain least-squares) loss.
    cameras = dict(enumerate(_line(8)))
    pairs = [(k, k + 1) for k in range(7)] + [(k, k + 2) for k in range(6)]
    aligned, placement, edges, _, centres, _ = _world(cameras, pairs, edge_bias_deg={(3, 4): 30.0})

    def worst_position_error(result):
        return max(np.linalg.norm(result.centres_px[k] - centres[k]) for k in cameras)

    huber = worst_position_error(refine_poses(aligned, placement, edges, PPM, SHAPES))
    linear = worst_position_error(refine_poses(aligned, placement, edges, PPM, SHAPES, loss="linear"))

    assert huber <= 0.25 * linear


# ---------------------------------------------------------------------------
# D5: guard-rail bounds are reported
# ---------------------------------------------------------------------------


def test_kappa_bound_hit_is_reported() -> None:  # D5a
    # GPS 1.12x the edge geometry: kappa wants 1/1.12 = 0.893, below 1 - 0.077.
    cameras = dict(enumerate(_line(4)))
    aligned, placement, edges, _, _, _ = _world(cameras, [(0, 1), (1, 2), (2, 3)], gps_scale=1.12)

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    assert result.kappa == pytest.approx(1.0 - KAPPA_BOUND, abs=1e-9)
    assert result.bound_hits.kappa


def test_position_bound_hit_is_reported() -> None:  # D5c
    # Per-axis box guard: a 6 m x-offset the data wants undone hits the 5 m bound on x.
    cameras = dict(enumerate(_line(4)))
    aligned, placement, edges, _, _, _ = _world(
        cameras, [(0, 1), (1, 2), (2, 3)], init_position_offsets_m={1: (6.0, 0.0)}
    )

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    initial_x = (aligned.poses[1] @ np.append(IMAGE_CENTRE, 1.0))[0]
    assert abs(result.centres_px[1][0] - initial_x) == pytest.approx(MAX_POSITION_CHANGE_M * PPM, abs=1e-6)
    assert 1 in result.bound_hits.position


# ---------------------------------------------------------------------------
# D7 / D8: end to end
# ---------------------------------------------------------------------------


def test_synthetic_serpentine_a_to_d_recovers_yaws_and_positions() -> None:  # D7
    line_b = _line(4)
    line_c = [SyntheticCamera(c.east_m - 3.3, c.north_m + 27.0, ALTITUDE, -104.3) for c in line_b]
    cameras = dict(enumerate(line_b + line_c))
    pairs = [(k, k + 1) for k in range(3)] + [(k, k + 1) for k in range(4, 7)] + [(k, k + 4) for k in range(4)]
    homographies = {(i, j): plane_homography(cameras[i], cameras[j]) for i, j in pairs}
    latlons = {k: latlon_from_en(c.east_m, c.north_m) for k, c in cameras.items()}

    stage_a = average_rotations(
        [RelativeRotation(i, j, relative_rotation_from_homography(H, IMAGE_SHAPE), 1.0) for (i, j), H in homographies.items()]
    )
    ppm = estimate_pixels_per_meter([PairDisplacement(H, IMAGE_SHAPE, latlons[i], latlons[j]) for (i, j), H in homographies.items()])
    placement = place_by_gps(latlons, ppm)
    shapes = {k: IMAGE_SHAPE for k in cameras}
    aligned = align_to_gps_frame(stage_a, placement, [HeadingEdge(i, j, H, IMAGE_SHAPE, 1.0) for (i, j), H in homographies.items()], ppm, shapes)
    result = refine_poses(aligned, placement, [_edge(cameras, i, j) for i, j in pairs], ppm, shapes)

    origin = cameras[0]
    for k, cam in cameras.items():
        assert np.degrees(_wrap(result.headings_rad[k] - np.radians(cam.yaw_deg))) == pytest.approx(0.0, abs=1e-5)
        assert result.centres_px[k] == pytest.approx(_centre(cam, origin), rel=1e-4, abs=1e-3)


def _run_a_to_d(paths, pair, shape):
    latlons = load_exif_latlons({0: paths[0], 1: paths[1]})
    stage_a = average_rotations([RelativeRotation(0, 1, relative_rotation_from_homography(pair.homography, shape), 1.0)])
    ppm = estimate_pixels_per_meter([PairDisplacement(pair.homography, shape, latlons[0], latlons[1])])
    placement = place_by_gps(latlons, ppm)
    shapes = {0: shape, 1: shape}
    aligned = align_to_gps_frame(stage_a, placement, [HeadingEdge(0, 1, pair.homography, shape, 1.0)], ppm, shapes)
    return refine_poses(aligned, placement, [edge_from_pair_result(pair, shape)], ppm, shapes)


def test_stage_a_to_d_results_are_identical_with_xmp_stripped(tmp_path: Path) -> None:  # D8
    names = ["DJI_20230127131426_0352_W.JPG", "DJI_20230127131429_0353_W.JPG"]
    originals = [FIXTURES / n for n in names]
    stripped = [tmp_path / n for n in names]
    for src, dst in zip(originals, stripped):
        _strip_xmp(src, dst)
    images = [cv2.imread(str(p)) for p in originals]
    pair = match_pair(_SiftRatioMatcher(), images[0], images[1], 0, 1)

    a = _run_a_to_d(originals, pair, images[0].shape)
    b = _run_a_to_d(stripped, pair, images[0].shape)

    assert a.kappa == b.kappa
    assert a.headings_rad == b.headings_rad
    for k in a.poses:
        assert np.array_equal(a.poses[k], b.poses[k])


# ---------------------------------------------------------------------------
# D9: correspondence sampling contract
# ---------------------------------------------------------------------------


def test_farthest_point_sampling_on_a_3x3_grid() -> None:  # D9a
    grid = np.array([(x, y) for y in range(3) for x in range(3)], dtype=float)  # index = 3y + x

    assert list(sample_correspondences(grid, 4)) == [0, 8, 2, 6]


def test_farthest_point_sampling_on_a_line() -> None:  # D9a
    line = np.array([(x, 0.0) for x in range(5)])

    assert list(sample_correspondences(line, 3)) == [0, 4, 2]


def test_fewer_points_than_k_returns_all_in_original_order() -> None:  # D9b
    points = np.array([(5.0, 1.0), (0.0, 0.0), (3.0, 3.0)])

    assert list(sample_correspondences(points, 30)) == [0, 1, 2]


def test_edges_with_fewer_than_two_inliers_are_skipped_and_two_is_enough() -> None:  # D9c
    cameras = dict(enumerate(_line(3)))
    aligned, placement, edges, _, _, _ = _world(cameras, [(0, 1), (1, 2)])
    edges = [
        RefinementEdge(0, 1, edges[0].src_points[:1], edges[0].dst_points[:1], edges[0].homography, IMAGE_SHAPE),
        RefinementEdge(1, 2, edges[1].src_points[:2], edges[1].dst_points[:2], edges[1].homography, IMAGE_SHAPE),
    ]

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    assert result.skipped_edges == {(0, 1): "fewer_than_2_inliers"}


def test_sampling_is_deterministic_and_default_k_is_30() -> None:  # D9d
    k = np.arange(500)
    points = np.stack([(k * 37) % 4000, (k * 91) % 3000], axis=1).astype(float)  # deterministic

    first = sample_correspondences(points, K_POINTS_PER_EDGE)
    second = sample_correspondences(points, K_POINTS_PER_EDGE)

    assert K_POINTS_PER_EDGE == 30
    assert list(first) == list(second)
    assert len(set(first)) == 30


# ---------------------------------------------------------------------------
# D11 / D12 / D13 / D14
# ---------------------------------------------------------------------------


def test_refined_poses_have_unit_scale() -> None:  # D11
    cameras = dict(enumerate(_line(5)))
    pairs = [(k, k + 1) for k in range(4)]
    aligned, placement, edges, _, _, _ = _world(cameras, pairs, edge_bias_deg={p: 0.4 for p in pairs},
                                                 gps_offsets_m={1: (0.3, -0.2)})

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    for pose in result.poses.values():
        assert np.hypot(pose[0, 0], pose[1, 0]) == pytest.approx(1.0, abs=1e-12)
        assert pose[0, 0] == pytest.approx(pose[1, 1], abs=1e-12)
        assert pose[0, 1] == pytest.approx(-pose[1, 0], abs=1e-12)


def test_only_stage_c_posed_nodes_are_refined_and_statuses_pass_through() -> None:  # D12
    cameras = dict(enumerate(_line(4)))
    aligned, placement, edges, _, _, _ = _world(cameras, [(0, 1), (1, 2), (2, 3)])
    del aligned.poses[3], aligned.headings_rad[3]
    aligned.unlocated, aligned.unoriented, aligned.unaligned_components = {3}, {9}, {5}

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    assert set(result.poses) == {0, 1, 2}
    assert result.skipped_edges == {(2, 3): "endpoint_without_stage_c_pose"}
    assert (result.unlocated, result.unoriented, result.unaligned_components) == ({3}, {9}, {5})


@pytest.mark.parametrize("bad", ["nan_points", "mismatched_points", "bad_ppm", "unknown_loss"])
def test_invalid_inputs_raise(bad: str) -> None:  # D13
    cameras = dict(enumerate(_line(2)))
    aligned, placement, edges, _, _, _ = _world(cameras, [(0, 1)])
    edge, ppm = edges[0], PPM
    if bad == "nan_points":
        src = edge.src_points.copy()
        src[0, 0] = np.nan
        edges = [RefinementEdge(0, 1, src, edge.dst_points, edge.homography, IMAGE_SHAPE)]
    elif bad == "mismatched_points":
        edges = [RefinementEdge(0, 1, edge.src_points, edge.dst_points[:-1], edge.homography, IMAGE_SHAPE)]
    elif bad == "bad_ppm":
        ppm = 0.0
    else:
        with pytest.raises(ValueError):
            refine_poses(aligned, placement, edges, ppm, SHAPES, loss="soft_l1")
        return

    with pytest.raises(ValueError):
        refine_poses(aligned, placement, edges, ppm, SHAPES)


def test_k30_matches_using_all_points_within_closed_form_bound() -> None:  # D14
    # With headings fixed, an edge whose dst points carry a rotation bias b about the dst
    # centre estimates m_j - m_i with an error R_j (R_b - I)(mean of its used dst points -
    # c), so K = 30 vs all points can differ by at most 2 sin(b/2) |centroid_30 - centroid_all|
    # per edge. Bound per node: the sum of those over all edges (closed form, conservative).
    cameras = dict(enumerate(_line(6)))
    pairs = [(k, k + 1) for k in range(5)]
    noise = {k: ((0.37, -0.37) if k % 2 == 0 else (-0.37, 0.37)) for k in cameras}
    bias_deg = 0.4
    aligned, placement, edges, _, _, _ = _world(cameras, pairs, edge_bias_deg={p: bias_deg for p in pairs}, gps_offsets_m=noise)

    k30 = refine_poses(aligned, placement, edges, PPM, SHAPES)
    everything = refine_poses(aligned, placement, edges, PPM, SHAPES, k_points=10**9)

    bound = 0.0
    for e in edges:
        chosen = sample_correspondences(e.src_points, K_POINTS_PER_EDGE)
        shift = np.linalg.norm(e.dst_points[chosen].mean(axis=0) - e.dst_points.mean(axis=0))
        bound += 2 * np.sin(np.radians(bias_deg) / 2) * shift
    for k in cameras:
        assert np.linalg.norm(k30.centres_px[k] - everything.centres_px[k]) <= bound + 1e-6
        assert k30.headings_rad[k] == everything.headings_rad[k]


# ---------------------------------------------------------------------------
# IRLS convergence status (same standard as optimize_pose_graph's optimization_status:
# running out of rounds is never reported as converged)
# ---------------------------------------------------------------------------


def _wrong_edge_world():
    cameras = dict(enumerate(_line(8)))
    pairs = [(k, k + 1) for k in range(7)] + [(k, k + 2) for k in range(6)]
    return _world(cameras, pairs, edge_bias_deg={(3, 4): 30.0})


def test_huber_irls_reports_converged_after_several_rounds() -> None:
    import sea_mosaic.refinement as refinement

    assert refinement.IRLS_POSITION_TOL_PX == 1e-9
    assert refinement.IRLS_KAPPA_TOL == 1e-12
    assert refinement.IRLS_MAX_ROUNDS == 100
    aligned, placement, edges, _, _, _ = _wrong_edge_world()

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    assert result.status == "converged"
    assert 2 <= result.irls_rounds < refinement.IRLS_MAX_ROUNDS  # it really iterated


def test_running_out_of_irls_rounds_is_reported_as_not_converged(monkeypatch) -> None:
    import sea_mosaic.refinement as refinement

    monkeypatch.setattr(refinement, "IRLS_MAX_ROUNDS", 2)  # this case needs more than 2
    aligned, placement, edges, _, _, _ = _wrong_edge_world()

    result = refine_poses(aligned, placement, edges, PPM, SHAPES)

    assert result.status == "not_converged"
    assert result.irls_rounds == 2
    assert set(result.poses) == set(aligned.poses)  # results are still returned


def test_linear_loss_converges_in_exactly_one_round() -> None:
    aligned, placement, edges, _, _, _ = _wrong_edge_world()

    result = refine_poses(aligned, placement, edges, PPM, SHAPES, loss="linear")

    assert result.status == "converged"
    assert result.irls_rounds == 1


def test_irls_stopping_rule_requires_all_three_conditions() -> None:
    from sea_mosaic.refinement import IRLS_KAPPA_TOL, IRLS_POSITION_TOL_PX, _irls_converged

    held = {3: 1.0}
    assert _irls_converged(0.0, 0.0, held, dict(held))
    assert _irls_converged(IRLS_POSITION_TOL_PX, IRLS_KAPPA_TOL, held, dict(held))
    assert not _irls_converged(2 * IRLS_POSITION_TOL_PX, 0.0, held, dict(held))
    assert not _irls_converged(0.0, 2 * IRLS_KAPPA_TOL, held, dict(held))
    # Zero steps but the active set changed (a variable was just held or released): not converged.
    assert not _irls_converged(0.0, 0.0, held, {})
    assert not _irls_converged(0.0, 0.0, {}, held)
