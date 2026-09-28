"""Tests for sea_mosaic.global_poses.estimate_global_poses (Stage A->D chained).

estimate_global_poses is pure wiring: it filters edges, estimates pixels_per_meter,
runs Stage B (GPS placement), Stage A (rotation averaging with the chosen heading-anchor
source), Stage C (frame alignment) and Stage D (position refinement), and packs the
result into GlobalTransforms plus diagnostics. The stages themselves are tested in their
own files; here the questions are (1) is the chain wired exactly as validated on real
data (G4 -- spelled out step by step below, compared bit for bit), (2) are the pieces
that only exist at this level right (gimbal sign convention, failure reporting, edge
filtering, output contract).

Expected poses come from the synthetic pinhole camera (tests/synthetic_camera.py) via
_true_pose, derived by hand here -- never from calling a Stage A-D function.
"""

from __future__ import annotations

import numpy as np
import pytest

import sea_mosaic.refinement as refinement
from sea_mosaic.frame_alignment import (
    GPS_HEADING_ANCHOR_WEIGHT,
    HeadingEdge,
    align_to_gps_frame,
    gps_heading_anchors,
)
from sea_mosaic.global_poses import estimate_global_poses
from sea_mosaic.gps_lag import estimate_gps_lag, shift_latlons, travel_directions
from sea_mosaic.gps_placement import PairDisplacement, estimate_pixels_per_meter, place_by_gps
from sea_mosaic.refinement import edge_from_pair_result, refine_poses
from sea_mosaic.rotation_averaging import (
    GIMBAL_YAW_ANCHOR_WEIGHT,
    RelativeRotation,
    average_rotations,
    relative_rotation_from_homography,
)
from sea_mosaic.types import PairResult
from synthetic_camera import (
    FOCAL_PX,
    IMAGE_CENTRE,
    IMAGE_SHAPE,
    SyntheticCamera,
    ground_to_image,
    latlon_from_en,
    plane_homography,
)

ALTITUDE = 100.0
PPM = FOCAL_PX / ALTITUDE  # exact ground scale of an untilted nadir pinhole camera
MAX_PAIR_DISTANCE_M = 40.0

# Tolerances for exact synthetic data. The only inexactness is the lat/lon round trip
# (latlon_from_en uses the fixed ORIGIN_LATLON for cos(lat), geo.projection uses the origin
# node's latitude -- a relative difference ~1e-5 over this ~100 m world, i.e. ~0.03 px).
HEADING_TOL_DEG = 0.002  # 0.002 deg at the image corner (2535 px from centre) is 0.09 px
POINT_TOL_PX = 0.2


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------


def _serpentine_cameras() -> dict[int, SyntheticCamera]:
    """Line B eastbound (6), a two-image U-turn, line C westbound (6), ~27 m apart.

    The U-turn headings (30.8, -29.2 deg, like real 0352/0353) matter: with only the two
    line headings 70.8 and -109.2, a sign-flipped heading anchor set (-70.8, 109.2) differs
    from the truth by one constant (-141.6 deg), which Stage C's per-component offset would
    absorb -- the gimbal sign-convention test could not see the flip. Intermediate headings
    make a flip inconsistent with the edges.
    """
    course_b = np.radians(83.0)
    step = 13.4
    cams = [
        SyntheticCamera(k * step * np.sin(course_b), k * step * np.cos(course_b), ALTITUDE, 70.8)
        for k in range(6)
    ]
    cams += [SyntheticCamera(74.0, 16.0, ALTITUDE, 30.8), SyntheticCamera(72.0, 26.0, ALTITUDE, -29.2)]
    start_c = (cams[5].east_m, cams[5].north_m + 27.0)
    course_c = np.radians(263.0)
    cams += [
        SyntheticCamera(
            start_c[0] + k * step * np.sin(course_c), start_c[1] + k * step * np.cos(course_c), ALTITUDE, -109.2
        )
        for k in range(6)
    ]
    return dict(enumerate(cams))


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


def _pair(cameras, i, j, rng=None, noise_px=0.0, keep_fraction=1.0) -> PairResult:
    src, dst = _correspondences(cameras[i], cameras[j])
    homography = plane_homography(cameras[i], cameras[j])
    if rng is not None:
        keep = rng.random(len(src)) < keep_fraction
        src, dst = src[keep], dst[keep]
        dst = dst + rng.normal(0.0, noise_px, dst.shape)
    return PairResult(i, j, src, dst, np.ones(len(src), dtype=bool), homography)


def _gps_distance_m(a: SyntheticCamera, b: SyntheticCamera) -> float:
    return float(np.hypot(a.east_m - b.east_m, a.north_m - b.north_m))


def _proximity_pairs(cameras) -> list[tuple[int, int]]:
    keys = sorted(cameras)
    return [
        (i, j)
        for n, i in enumerate(keys)
        for j in keys[n + 1 :]
        if _gps_distance_m(cameras[i], cameras[j]) < MAX_PAIR_DISTANCE_M
    ]


def _latlons(cameras, gps_noise_m: float = 0.0, rng=None):
    out = {}
    for k, c in cameras.items():
        de, dn = (rng.normal(0.0, gps_noise_m, 2) if rng is not None else (0.0, 0.0))
        out[k] = latlon_from_en(c.east_m + de, c.north_m + dn)
    return out


def _shapes(cameras):
    return {k: IMAGE_SHAPE for k in cameras}


def _exact_world(cameras=None):
    cameras = cameras if cameras is not None else _serpentine_cameras()
    pairs = [_pair(cameras, i, j) for i, j in _proximity_pairs(cameras)]
    return cameras, pairs, _latlons(cameras), _shapes(cameras)


def _noisy_world(seed: int = 7):
    """Realistic-ish noise so that any change in weights or filtering changes the result:
    varying inlier counts, 0.7 px correspondence noise, homographies re-fit to the noisy
    points, 0.4 m GPS noise."""
    import cv2

    rng = np.random.default_rng(seed)
    cameras = _serpentine_cameras()
    pairs = []
    for i, j in _proximity_pairs(cameras):
        p = _pair(cameras, i, j, rng=rng, noise_px=0.7, keep_fraction=rng.uniform(0.2, 1.0))
        H, _ = cv2.findHomography(p.src_points, p.dst_points, 0)
        pairs.append(PairResult(i, j, p.src_points, p.dst_points, p.inlier_mask, H / H[2, 2]))
    # Edges at and around the filter boundary, so that G4 itself sees the filter: without
    # them every edge has >= ~90 inliers and changing the threshold 4 -> 5 or 4 -> 50, or
    # taking the inlier median over all pairs instead of the usable ones, changes nothing
    # (measured: bit-identical). One valid 4-inlier edge (must be used), one 3-inlier edge
    # and one 10-inlier edge with a non-finite homography (both must be ignored). Both
    # ignored edges sit below the inlier median on purpose: one below and one above would
    # leave the median unchanged, hiding a median-over-all-pairs bug (measured). Node pairs
    # are ~54 m apart, beyond _proximity_pairs, so they are not duplicates.
    for (i, j), n, finite in (((0, 4), 4, True), ((1, 5), 3, True), ((2, 6), 10, False)):
        full = _pair(cameras, i, j)
        keep = rng.choice(len(full.src_points), n, replace=False)
        src, dst = full.src_points[keep], full.dst_points[keep] + rng.normal(0.0, 0.7, (n, 2))
        if not finite:
            H = np.full((3, 3), np.nan)
        elif n >= 4:
            H, _ = cv2.findHomography(src, dst, 0)
            H = H / H[2, 2]
        else:
            H = full.homography
        pairs.append(PairResult(i, j, src, dst, np.ones(n, dtype=bool), H))
    return cameras, pairs, _latlons(cameras, 0.4, rng), _shapes(cameras)


def _true_pose(camera: SyntheticCamera, origin: SyntheticCamera) -> np.ndarray:
    """Image-to-frame pose: pixel -> ground (E, N) -> north-up pixels (x = ppm E, y = -ppm N),
    origin at the origin camera's ground point (hand derivation, independent of Stage A-D)."""
    to_frame = np.array(
        [[PPM, 0.0, -PPM * origin.east_m], [0.0, -PPM, PPM * origin.north_m], [0.0, 0.0, 1.0]]
    )
    pose = to_frame @ np.linalg.inv(ground_to_image(camera))
    return pose / pose[2, 2]


_TEST_POINTS = np.array(
    [IMAGE_CENTRE, [0.0, 0.0], [IMAGE_SHAPE[1], 0.0], [0.0, IMAGE_SHAPE[0]], [IMAGE_SHAPE[1], IMAGE_SHAPE[0]]]
)


def _apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = T @ np.column_stack([points, np.ones(len(points))]).T
    return (p[:2] / p[2]).T


def _heading_deg(T: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def _angle_diff_deg(a: float, b: float) -> float:
    return float((a - b + 180.0) % 360.0 - 180.0)


def _assert_matches_truth(transforms, cameras, origin_index, nodes=None):
    origin = cameras[origin_index]
    for k in nodes if nodes is not None else cameras:
        truth = _true_pose(cameras[k], origin)
        assert abs(_angle_diff_deg(_heading_deg(transforms[k]), _heading_deg(truth))) < HEADING_TOL_DEG, k
        assert np.max(np.abs(_apply(transforms[k], _TEST_POINTS) - _apply(truth, _TEST_POINTS))) < POINT_TOL_PX, k


def _manual_chain(pairs, shapes, latlons, source, gimbal_yaw_deg=None):
    """The validated real-data procedure (CLAUDE.md's Stage A～D 真實資料驗收, accuracy_diag),
    spelled out step by step. estimate_global_poses must reproduce this bit for bit."""
    usable = [p for p in pairs if p.inlier_count >= 4 and np.all(np.isfinite(p.homography))]
    ppm = estimate_pixels_per_meter(
        [
            PairDisplacement(p.homography, shapes[p.src_index], latlons[p.src_index], latlons[p.dst_index])
            for p in usable
            if p.src_index in latlons and p.dst_index in latlons
        ]
    )
    placement = place_by_gps(latlons, ppm, node_indices=shapes.keys())
    median_inliers = float(np.median([p.inlier_count for p in usable]))
    rotation_edges = [
        RelativeRotation(
            p.src_index,
            p.dst_index,
            relative_rotation_from_homography(p.homography, shapes[p.src_index]),
            p.inlier_count / median_inliers,
        )
        for p in usable
    ]
    heading_edges = [HeadingEdge(p.src_index, p.dst_index, p.homography, shapes[p.src_index], 1.0) for p in usable]
    if source == "gps":
        stage_a = average_rotations(
            rotation_edges,
            heading_anchors=gps_heading_anchors(latlons, heading_edges, shapes),
            anchor_weight=GPS_HEADING_ANCHOR_WEIGHT,
        )
    elif source == "gimbal":
        stage_a = average_rotations(
            rotation_edges,
            heading_anchors={k: float(np.radians(v)) for k, v in gimbal_yaw_deg.items()},
            anchor_weight=GIMBAL_YAW_ANCHOR_WEIGHT,
        )
    else:
        stage_a = average_rotations(rotation_edges)
    stage_c = align_to_gps_frame(stage_a, placement, heading_edges, ppm, shapes)
    stage_d = refine_poses(
        stage_c, placement, [edge_from_pair_result(p, shapes[p.src_index]) for p in usable], ppm, shapes
    )
    return ppm, stage_c, stage_d


# ---------------------------------------------------------------------------
# G1-G3: exact synthetic world, each heading-anchor source recovers the truth
# ---------------------------------------------------------------------------


def test_g1_gps_heading_source_recovers_true_poses() -> None:
    cameras, pairs, latlons, shapes = _exact_world()

    result = estimate_global_poses(pairs, shapes, latlons)

    assert result.failure_reason is None
    assert result.heading_anchor_source == "gps"
    assert set(result.global_transforms.transforms) == set(cameras)
    _assert_matches_truth(result.global_transforms.transforms, cameras, origin_index=0)


def test_g2_gimbal_heading_source_uses_compass_yaw_without_sign_flip() -> None:
    """Gimbal readings equal to the cameras' true compass yaws must give the true poses.
    The expected poses come from the pinhole model, not from any conversion function --
    the lesson of the _yaw_target_vector sign bug (CLAUDE.md). The U-turn headings make a
    sign flip visible (see _serpentine_cameras)."""
    cameras, pairs, latlons, shapes = _exact_world()
    gimbal = {k: c.yaw_deg for k, c in cameras.items()}

    result = estimate_global_poses(pairs, shapes, latlons, heading_anchor_source="gimbal", gimbal_yaw_deg=gimbal)

    assert result.failure_reason is None
    assert result.heading_anchor_source == "gimbal"
    _assert_matches_truth(result.global_transforms.transforms, cameras, origin_index=0)


def test_g3_no_heading_anchors_still_recovers_true_poses_and_records_source() -> None:
    cameras, pairs, latlons, shapes = _exact_world()

    result = estimate_global_poses(pairs, shapes, latlons, heading_anchor_source="none")

    assert result.failure_reason is None
    assert result.heading_anchor_source == "none"
    _assert_matches_truth(result.global_transforms.transforms, cameras, origin_index=0)


# ---------------------------------------------------------------------------
# G4: wiring equivalence -- the most important test in this file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["gps", "gimbal", "none"])
def test_g4_wiring_is_bit_identical_to_the_validated_manual_chain(source: str) -> None:
    """On a noisy world (so every weight, filter and parameter matters), the result must be
    bit-identical to _manual_chain: same edge filter, same ppm, same Stage A weights
    (inlier/median) and anchors, Stage C heading-edge weight 1.0, same Stage D edges."""
    cameras, pairs, latlons, shapes = _noisy_world()
    gimbal = {k: c.yaw_deg + 0.3 * ((k % 3) - 1) for k, c in cameras.items()} if source == "gimbal" else None
    ppm, stage_c, stage_d = _manual_chain(pairs, shapes, latlons, source, gimbal)

    result = estimate_global_poses(
        pairs, shapes, latlons, heading_anchor_source=source, gimbal_yaw_deg=gimbal
    )

    assert result.pixels_per_meter == ppm
    assert result.kappa == stage_d.kappa
    assert result.component_offsets_rad == stage_c.component_offsets_rad
    assert result.irls_rounds == stage_d.irls_rounds
    assert result.term_rms == stage_d.term_rms
    assert result.skipped_edges == stage_d.skipped_edges
    assert result.bound_hits == stage_d.bound_hits
    assert result.global_transforms.optimization_status == stage_d.status
    assert set(result.global_transforms.transforms) == set(stage_d.poses)
    for k, pose in stage_d.poses.items():
        assert np.array_equal(result.global_transforms.transforms[k], pose), k


# ---------------------------------------------------------------------------
# G5-G10: failure reporting and edge filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latlons", [None, {}])
def test_g5_no_gps_is_reported_without_raising(latlons) -> None:
    cameras, pairs, _, shapes = _exact_world()

    result = estimate_global_poses(pairs, shapes, latlons)

    assert result.failure_reason == "no_gps"
    assert result.global_transforms.transforms == {}
    assert result.global_transforms.optimization_status == "failed"
    assert np.isnan(result.pixels_per_meter)
    assert set(result.node_failure_reasons) == set(cameras)


def test_g6_all_gps_displacements_below_threshold_reports_ppm_unavailable() -> None:
    cameras = {k: SyntheticCamera(2.0 * k, 0.0, ALTITUDE, 70.8) for k in range(3)}  # 2 m steps
    cameras, pairs, latlons, shapes = _exact_world(cameras)

    result = estimate_global_poses(pairs, shapes, latlons)

    assert result.failure_reason == "pixels_per_meter_unavailable"
    assert result.global_transforms.transforms == {}
    assert result.global_transforms.optimization_status == "failed"
    assert set(result.node_failure_reasons) == set(cameras)


def test_g7_node_without_gps_is_unlocated_and_others_are_unaffected() -> None:
    cameras, pairs, latlons, shapes = _exact_world()
    del latlons[3]

    result = estimate_global_poses(pairs, shapes, latlons)

    assert result.failure_reason is None
    assert result.node_failure_reasons == {3: "unlocated"}
    assert 3 not in result.global_transforms.transforms
    _assert_matches_truth(result.global_transforms.transforms, cameras, 0, nodes=set(cameras) - {3})


@pytest.mark.parametrize("source", ["gps", "gimbal"])
def test_g8_node_with_metadata_but_no_edge_does_not_succeed(source: str) -> None:
    """Metadata alone (GPS, and in gimbal mode also a heading) is not image evidence."""
    cameras, pairs, latlons, shapes = _exact_world()
    lonely = 99
    lonely_camera = SyntheticCamera(30.0, -200.0, ALTITUDE, 70.8)  # far from everything
    shapes[lonely] = IMAGE_SHAPE
    latlons[lonely] = latlon_from_en(lonely_camera.east_m, lonely_camera.north_m)
    gimbal = None
    if source == "gimbal":
        gimbal = {k: c.yaw_deg for k, c in cameras.items()} | {lonely: lonely_camera.yaw_deg}

    result = estimate_global_poses(pairs, shapes, latlons, heading_anchor_source=source, gimbal_yaw_deg=gimbal)

    assert result.node_failure_reasons == {lonely: "no_determined_edge"}
    assert lonely not in result.global_transforms.transforms
    _assert_matches_truth(result.global_transforms.transforms, cameras, 0)


@pytest.mark.parametrize("source", ["gps", "gimbal"])
def test_g8b_metadata_only_node_with_smallest_index_does_not_become_stage_b_origin(source: str) -> None:
    """Stage B's origin defaults to the smallest located index. A metadata-only node
    (GPS, and in gimbal mode a heading, but no edge) with an index below every other node
    must not become that origin: the other nodes' poses must be bit-identical to the run
    without it, and still match the truth with node 0 as origin."""
    cameras, pairs, latlons, shapes = _exact_world()
    gimbal = {k: c.yaw_deg for k, c in cameras.items()} if source == "gimbal" else None
    baseline = estimate_global_poses(pairs, shapes, latlons, heading_anchor_source=source, gimbal_yaw_deg=gimbal)
    lonely = -1  # smaller than every real index
    lonely_camera = SyntheticCamera(30.0, -200.0, ALTITUDE, 70.8)
    shapes = {lonely: IMAGE_SHAPE} | shapes
    latlons = {lonely: latlon_from_en(lonely_camera.east_m, lonely_camera.north_m)} | latlons
    if gimbal is not None:
        gimbal = {lonely: lonely_camera.yaw_deg} | gimbal

    result = estimate_global_poses(pairs, shapes, latlons, heading_anchor_source=source, gimbal_yaw_deg=gimbal)

    assert result.node_failure_reasons == {lonely: "no_determined_edge"}
    assert set(result.global_transforms.transforms) == set(baseline.global_transforms.transforms)
    for k, pose in baseline.global_transforms.transforms.items():
        assert np.array_equal(result.global_transforms.transforms[k], pose), k
    _assert_matches_truth(result.global_transforms.transforms, cameras, origin_index=0)


def test_g9_edges_with_under_four_inliers_or_nonfinite_homography_are_ignored() -> None:
    cameras, pairs, latlons, shapes = _noisy_world()
    baseline = estimate_global_poses(pairs, shapes, latlons)
    rng = np.random.default_rng(3)
    garbage = PairResult(
        0, 13, rng.uniform(0, 3000, (3, 2)), rng.uniform(0, 3000, (3, 2)), np.ones(3, dtype=bool),
        np.array([[0.2, 0.9, 500.0], [-0.8, 0.1, -900.0], [1e-4, 0.0, 1.0]]),
    )
    many = pairs[0]
    nonfinite = PairResult(2, 11, many.src_points, many.dst_points, many.inlier_mask, np.full((3, 3), np.nan))

    result = estimate_global_poses(pairs + [garbage, nonfinite], shapes, latlons)

    assert result.pixels_per_meter == baseline.pixels_per_meter
    assert result.kappa == baseline.kappa
    assert set(result.global_transforms.transforms) == set(baseline.global_transforms.transforms)
    for k, pose in baseline.global_transforms.transforms.items():
        assert np.array_equal(result.global_transforms.transforms[k], pose), k


def test_g10_component_without_usable_alignment_edge_is_unaligned() -> None:
    """A second component far away whose only edge is shorter than 5 m: Stage C has no
    usable edge to align it, so its nodes get no pose; the main component is unaffected."""
    cameras, pairs, latlons, shapes = _exact_world()
    far = {50: SyntheticCamera(400.0, 400.0, ALTITUDE, 10.0), 51: SyntheticCamera(402.0, 400.5, ALTITUDE, 10.0)}
    pairs = pairs + [_pair(far, 50, 51)]
    latlons |= _latlons(far)
    shapes |= _shapes(far)

    result = estimate_global_poses(pairs, shapes, latlons)

    assert result.failure_reason is None
    assert result.node_failure_reasons == {50: "unaligned_component", 51: "unaligned_component"}
    _assert_matches_truth(result.global_transforms.transforms, cameras, 0)


# ---------------------------------------------------------------------------
# G11-G13: argument validation, non-convergence, output contract
# ---------------------------------------------------------------------------


def test_g11_argument_validation() -> None:
    cameras, pairs, latlons, shapes = _exact_world()
    gimbal = {k: c.yaw_deg for k, c in cameras.items()}

    with pytest.raises(ValueError, match="heading_anchor_source"):
        estimate_global_poses(pairs, shapes, latlons, heading_anchor_source="compass")
    with pytest.raises(ValueError, match="gimbal_yaw_deg"):
        estimate_global_poses(pairs, shapes, latlons, heading_anchor_source="gimbal")
    for source in ("gps", "none"):
        with pytest.raises(ValueError, match="gimbal_yaw_deg"):
            estimate_global_poses(pairs, shapes, latlons, heading_anchor_source=source, gimbal_yaw_deg=gimbal)


def test_g12_refinement_not_converged_is_propagated(monkeypatch) -> None:
    cameras, pairs, latlons, shapes = _noisy_world()
    monkeypatch.setattr(refinement, "IRLS_MAX_ROUNDS", 1)  # this world needs more than 1 round

    result = estimate_global_poses(pairs, shapes, latlons)

    assert result.global_transforms.optimization_status == "not_converged"
    assert result.failure_reason == "refinement_not_converged"
    assert result.irls_rounds == 1


def test_g13_output_contract() -> None:
    cameras, pairs, latlons, shapes = _exact_world()

    result = estimate_global_poses(pairs, shapes, latlons)

    gt = result.global_transforms
    assert gt.reference_index is None
    assert np.isnan(gt.residual_error)
    assert gt.optimization_status == "converged"
    assert result.node_failure_reasons == {}
    assert result.pixels_per_meter == pytest.approx(PPM, rel=1e-4)
    assert result.kappa == pytest.approx(1.0, abs=1e-4)
    for T in gt.transforms.values():
        assert np.array_equal(T[2], [0.0, 0.0, 1.0])
        assert T[:2, :2] @ T[:2, :2].T == pytest.approx(np.eye(2), abs=1e-12)  # rotation, scale 1
        assert np.linalg.det(T[:2, :2]) == pytest.approx(1.0, abs=1e-12)  # not a mirror


# ---------------------------------------------------------------------------
# G14-G17: GPS recording-lag correction at the GPS input, shared by every stage
# ---------------------------------------------------------------------------

LAG_TRUE_M = 2.3
INTERVAL_S = 2.5


def _capture_times(cameras):
    """_serpentine_cameras is in flight order."""
    return {k: 1000.0 + INTERVAL_S * k for k in cameras}


def _lagged_latlons(cameras, lag_m):
    """GPS fixes lag_m ahead of the true centre along the true travel direction (central
    difference, one-sided at the ends -- written out here, not via gps_lag)."""
    keys = sorted(cameras)
    out = {}
    for n, k in enumerate(keys):
        a, b = cameras[keys[max(n - 1, 0)]], cameras[keys[min(n + 1, len(keys) - 1)]]
        u = np.array([b.east_m - a.east_m, b.north_m - a.north_m])
        u /= np.linalg.norm(u)
        out[k] = latlon_from_en(cameras[k].east_m + lag_m * u[0], cameras[k].north_m + lag_m * u[1])
    return out


def _centre_errors_px(transforms, cameras, origin_index):
    origin = cameras[origin_index]
    return {
        k: float(np.linalg.norm(_apply(transforms[k], IMAGE_CENTRE[None])[0] - _apply(_true_pose(cameras[k], origin), IMAGE_CENTRE[None])[0]))
        for k in cameras
    }


def test_g14_lag_correction_removes_the_position_bias_in_every_stage() -> None:
    """With the lag applied to GPS, the uncorrected run is off by ~lag * ppm (the bias
    exists); with capture times the estimated lag is removed before Stage A-D and every
    centre lands on the truth. Tolerance: 0.15 m (the lag tolerance of test_gps_lag plus
    the U-turn direction mismatch) in pixels."""
    cameras, pairs, _latlons_exact, shapes = _exact_world()
    lagged = _lagged_latlons(cameras, LAG_TRUE_M)
    times = _capture_times(cameras)

    uncorrected = estimate_global_poses(pairs, shapes, lagged)
    corrected = estimate_global_poses(pairs, shapes, lagged, capture_times_s=times)

    assert corrected.gps_lag.status == "estimated"
    assert corrected.gps_lag.lag_m == pytest.approx(LAG_TRUE_M, abs=0.1)
    # Stage B's origin is the smallest evidenced node; its own GPS also moved, so compare
    # relative placement: truth frame anchored at the origin camera.
    err_u = _centre_errors_px(uncorrected.global_transforms.transforms, cameras, 0)
    err_c = _centre_errors_px(corrected.global_transforms.transforms, cameras, 0)
    assert max(err_u.values()) > 30.0  # ~2 * 2.3 m * 28.7 px/m between opposite lines
    assert max(err_c.values()) < 0.15 * PPM


def test_g15_the_corrected_latlons_feed_every_stage_bit_for_bit() -> None:
    """Wiring: estimate_global_poses == the validated manual chain run on
    shift_latlons(latlons, travel_directions(latlons, times), lag), with lag equal to what
    estimate_gps_lag returns on the usable edges."""
    cameras, pairs, _, shapes = _noisy_world()
    lagged = _lagged_latlons(cameras, LAG_TRUE_M)
    times = _capture_times(cameras)

    result = estimate_global_poses(pairs, shapes, lagged, capture_times_s=times)

    usable = [p for p in pairs if p.inlier_count >= 4 and np.all(np.isfinite(p.homography))]
    lag = estimate_gps_lag(
        [HeadingEdge(p.src_index, p.dst_index, p.homography, shapes[p.src_index], 1.0) for p in usable],
        lagged,
        times,
        shapes,
    )
    assert result.gps_lag == lag
    corrected = shift_latlons(lagged, travel_directions(lagged, times), lag.lag_m)
    ppm, stage_c, stage_d = _manual_chain(pairs, shapes, corrected, "gps")
    assert result.pixels_per_meter == ppm
    assert result.global_transforms.transforms.keys() == stage_d.poses.keys()
    for k, T in stage_d.poses.items():
        assert np.array_equal(result.global_transforms.transforms[k], T), k


def test_g16_without_capture_times_nothing_changes_and_it_is_flagged() -> None:
    cameras, pairs, latlons, shapes = _noisy_world()

    result = estimate_global_poses(pairs, shapes, latlons)

    assert result.gps_lag.status == "no_capture_times"
    assert result.gps_lag.lag_m == 0.0
    assert result.gps_lag.uncorrected_nodes == set(latlons)
    ppm, _stage_c, stage_d = _manual_chain(pairs, shapes, latlons, "gps")
    for k, T in stage_d.poses.items():
        assert np.array_equal(result.global_transforms.transforms[k], T), k


def test_g17_lag_diagnostics_are_present_on_failed_runs_too() -> None:
    cameras, pairs, _, shapes = _exact_world()
    result = estimate_global_poses(pairs, shapes, None, capture_times_s=_capture_times(cameras))
    assert result.failure_reason == "no_gps"
    assert result.gps_lag.status == "not_estimable"
    assert result.gps_lag.reason == "no_gps"
    assert result.gps_lag.lag_m == 0.0
