"""Tests for sea_mosaic.frame_alignment (Stage C: align Stage A's rotations to Stage B's
north-up pixel frame using GPS track bearing, then compose centre-anchored poses).

Per edge (i, j): beta = direction of (p_j - p_i) in the north-up pixel frame, alpha =
direction of travel measured in src image i's own pixel frame (where dst's centre lands),
so beta = phi_i + alpha, and with phi_i = theta_i + delta each edge estimates
delta_ij = beta - alpha - theta_i. delta is the weighted circular mean per Stage A
component. Derived property used as an independent truth: in the north-up pixel frame
(y down), a pose's rotation angle equals the camera's compass yaw.

Per CLAUDE.md's 最小可行版本範圍決定, no GimbalYawDegree or DJI XMP field is used
anywhere; expected values come from hand calculation, closed forms, or the synthetic
pinhole camera model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sea_mosaic.estimate import match_pair
from sea_mosaic.frame_alignment import HeadingEdge, align_to_gps_frame
from sea_mosaic.gps_placement import (
    GpsPlacement,
    PairDisplacement,
    estimate_pixels_per_meter,
    load_exif_latlons,
    place_by_gps,
)
from sea_mosaic.rotation_averaging import (
    RelativeRotation,
    RotationAveragingResult,
    average_rotations,
    relative_rotation_from_homography,
)
from synthetic_camera import (
    FOCAL_PX,
    IMAGE_CENTRE,
    IMAGE_SHAPE,
    SyntheticCamera,
    latlon_from_en,
    plane_homography,
)
from test_gps_placement import FIXTURES, _SiftRatioMatcher, _strip_xmp

PPM = 28.0
SHAPES = {k: IMAGE_SHAPE for k in range(10)}


def _wrap(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2 * np.pi) - np.pi)


def _rot(phi_rad: float) -> np.ndarray:
    return np.array([[np.cos(phi_rad), -np.sin(phi_rad)], [np.sin(phi_rad), np.cos(phi_rad)]])


def _true_pose(phi_deg: float, centre_px: np.ndarray) -> np.ndarray:
    R = _rot(np.radians(phi_deg))
    pose = np.eye(3)
    pose[:2, :2] = R
    pose[:2, 2] = centre_px - R @ IMAGE_CENTRE
    return pose


def _scenario(phis_deg: dict[int, float], centres_m: dict[int, tuple[float, float]], edges, gauge_deg):
    """Ground truth -> (Stage A result with an arbitrary gauge, Stage B placement, edges).

    Homographies come from true centre-anchored poses: H = inv(pose_j) @ pose_i, so dst's
    centre lands at R(-phi_i) (p_j - p_i) + c in src i's pixels (hand derivation)."""
    centres_px = {k: PPM * np.array([e, -n]) for k, (e, n) in centres_m.items()}
    poses = {k: _true_pose(phis_deg[k], centres_px[k]) for k in phis_deg}
    stage_a = RotationAveragingResult(
        angles={k: _wrap(np.radians(phis_deg[k] - gauge_deg)) for k in phis_deg},
        component_of={k: 0 for k in phis_deg},
    )
    placement = GpsPlacement(centres_px=centres_px, unlocated=set(), origin_index=min(centres_px))
    heading_edges = [
        HeadingEdge(i, j, np.linalg.inv(poses[j]) @ poses[i], IMAGE_SHAPE, weight)
        for i, j, weight in edges
    ]
    return stage_a, placement, heading_edges


# ---------------------------------------------------------------------------
# delta estimation
# ---------------------------------------------------------------------------


def test_recovers_gauge_offset_exactly() -> None:
    phis = {0: 70.8, 1: 70.8, 2: 71.5}
    centres = {0: (0.0, 0.0), 1: (12.8, 1.6), 2: (25.6, 3.3)}
    stage_a, placement, edges = _scenario(phis, centres, [(0, 1, 1.0), (1, 2, 1.0)], gauge_deg=-33.0)

    result = align_to_gps_frame(stage_a, placement, edges, PPM, SHAPES)

    assert np.degrees(result.component_offsets_rad[0]) == pytest.approx(-33.0, abs=1e-9)
    for k, phi in phis.items():
        assert np.degrees(_wrap(result.headings_rad[k] - np.radians(phi))) == pytest.approx(0.0, abs=1e-9)


def test_offsets_and_headings_across_plus_minus_180() -> None:
    # Serpentine-like 175 deg flip, gauge chosen so theta and delta both straddle +-180.
    phis = {0: 70.8, 1: 70.8, 2: -104.3, 3: -104.3}
    centres = {0: (0.0, 0.0), 1: (12.8, 1.6), 2: (9.0, 28.0), 3: (-3.8, 26.4)}
    stage_a, placement, edges = _scenario(
        phis, centres, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)], gauge_deg=179.0
    )

    result = align_to_gps_frame(stage_a, placement, edges, PPM, SHAPES)

    assert np.degrees(_wrap(result.component_offsets_rad[0] - np.radians(179.0))) == pytest.approx(
        0.0, abs=1e-9
    )
    for k, phi in phis.items():
        assert np.degrees(_wrap(result.headings_rad[k] - np.radians(phi))) == pytest.approx(0.0, abs=1e-9)


def test_offset_is_weighted_circular_mean_of_per_edge_estimates() -> None:
    # Stage A says theta = 0 everywhere, but the two edges' homographies were generated with
    # src node 0 at +4 deg and -10 deg respectively, so (hand derivation: alpha =
    # beta - phi_src, delta_ij = beta - alpha - theta_0 = phi_src) edge 0->1 implies
    # delta = +4 deg and edge 0->2 implies delta = -10 deg.
    # Closed form: arg(3 e^{i 4deg} + 1 e^{-i 10deg}).
    centres_px = {0: np.zeros(2), 1: PPM * np.array([13.0, 0.0]), 2: PPM * np.array([0.0, -13.0])}
    placement = GpsPlacement(centres_px=centres_px, unlocated=set(), origin_index=0)
    edges = [
        HeadingEdge(0, 1, np.linalg.inv(_true_pose(0.0, centres_px[1])) @ _true_pose(4.0, centres_px[0]), IMAGE_SHAPE, 3.0),
        HeadingEdge(0, 2, np.linalg.inv(_true_pose(0.0, centres_px[2])) @ _true_pose(-10.0, centres_px[0]), IMAGE_SHAPE, 1.0),
    ]
    stage_a = RotationAveragingResult(angles={0: 0.0, 1: 0.0, 2: 0.0}, component_of={0: 0, 1: 0, 2: 0})

    result = align_to_gps_frame(stage_a, placement, edges, PPM, SHAPES)

    expected = np.angle(3.0 * np.exp(1j * np.radians(4.0)) + 1.0 * np.exp(1j * np.radians(-10.0)))
    assert result.component_offsets_rad[0] == pytest.approx(expected, abs=1e-9)


def test_edges_shorter_than_threshold_are_ignored() -> None:
    phis = {0: 70.8, 1: 70.8, 2: 70.8}
    centres = {0: (0.0, 0.0), 1: (12.8, 1.6), 2: (13.5, 2.5)}  # 1->2 is ~1.1 m
    stage_a, placement, edges = _scenario(phis, centres, [(0, 1, 1.0)], gauge_deg=20.0)
    # A short edge whose measured direction is badly wrong (as GPS noise makes it at ~1 m).
    wrong = _true_pose(70.8 + 60.0, placement.centres_px[1])
    edges.append(HeadingEdge(1, 2, np.linalg.inv(_true_pose(70.8, placement.centres_px[2])) @ wrong, IMAGE_SHAPE, 100.0))

    result = align_to_gps_frame(stage_a, placement, edges, PPM, SHAPES)

    assert np.degrees(result.component_offsets_rad[0]) == pytest.approx(20.0, abs=1e-9)


def test_each_component_is_aligned_separately_and_unusable_ones_are_reported() -> None:
    phis = {0: 70.8, 1: 70.8, 5: -104.3, 6: -104.3, 8: 10.0, 9: 10.0}
    centres = {0: (0.0, 0.0), 1: (12.8, 1.6), 5: (0.0, 27.0), 6: (-12.8, 25.4), 8: (50.0, 0.0), 9: (50.5, 0.5)}
    stage_a, placement, edges = _scenario(
        phis, centres, [(0, 1, 1.0), (5, 6, 1.0), (8, 9, 1.0)], gauge_deg=0.0
    )
    stage_a = RotationAveragingResult(
        angles={0: np.radians(70.8 - 10.0), 1: np.radians(70.8 - 10.0),
                5: np.radians(-104.3 + 50.0), 6: np.radians(-104.3 + 50.0),
                8: 0.0, 9: 0.0},
        component_of={0: 0, 1: 0, 5: 1, 6: 1, 8: 2, 9: 2},
    )

    result = align_to_gps_frame(stage_a, placement, edges, PPM, SHAPES)

    assert np.degrees(result.component_offsets_rad[0]) == pytest.approx(10.0, abs=1e-9)
    assert np.degrees(result.component_offsets_rad[1]) == pytest.approx(-50.0, abs=1e-9)
    assert result.unaligned_components == {2}  # its only edge is ~0.7 m
    assert 2 not in result.component_offsets_rad
    assert not {8, 9} & set(result.poses)


def test_node_statuses_are_explicit_and_excluded_from_poses() -> None:
    phis = {0: 70.8, 1: 70.8, 2: 70.8}
    centres = {0: (0.0, 0.0), 1: (12.8, 1.6), 2: (25.6, 3.2)}
    stage_a, placement, edges = _scenario(phis, centres, [(0, 1, 1.0), (1, 2, 1.0)], gauge_deg=0.0)
    # Node 2: Stage A angle but no GPS. Node 7: GPS but no Stage A angle.
    placement = GpsPlacement(
        centres_px={0: placement.centres_px[0], 1: placement.centres_px[1], 7: PPM * np.array([0.0, -40.0])},
        unlocated={2},
        origin_index=0,
    )

    result = align_to_gps_frame(stage_a, placement, edges, PPM, SHAPES)

    assert result.unlocated == {2}
    assert result.unoriented == {7}
    assert set(result.poses) == {0, 1}


# ---------------------------------------------------------------------------
# pose assembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi_deg", [0.0, 70.8, -104.3, 180.0])
def test_poses_are_centre_anchored_with_the_aligned_heading(phi_deg: float) -> None:
    phis = {0: phi_deg, 1: phi_deg}
    east, north = 13.0 * np.sin(np.radians(phi_deg)), 13.0 * np.cos(np.radians(phi_deg))
    stage_a, placement, edges = _scenario(phis, {0: (0.0, 0.0), 1: (east, north)}, [(0, 1, 1.0)], gauge_deg=5.0)

    result = align_to_gps_frame(stage_a, placement, edges, PPM, SHAPES)

    for k in (0, 1):
        pose = result.poses[k]
        assert pose @ np.append(IMAGE_CENTRE, 1.0) == pytest.approx(np.append(placement.centres_px[k], 1.0), abs=1e-6)
        assert np.degrees(_wrap(np.arctan2(pose[1, 0], pose[0, 0]) - np.radians(phi_deg))) == pytest.approx(0.0, abs=1e-9)
        assert np.hypot(pose[0, 0], pose[1, 0]) == pytest.approx(1.0, abs=1e-12)
    if phi_deg == 180.0:
        # The old "anchor on the top-left corner" bug would give t = p; centre anchoring
        # with R = -I gives t = p + c.
        assert result.poses[0][:2, 2] == pytest.approx(placement.centres_px[0] + IMAGE_CENTRE, abs=1e-6)


def test_north_facing_image_keeps_its_top_pointing_north() -> None:
    # North-up regression: phi = 0 must map the image's "up" (0, -1) to north, (0, -1).
    stage_a, placement, edges = _scenario({0: 0.0, 1: 0.0}, {0: (0.0, 0.0), 1: (0.0, 13.0)}, [(0, 1, 1.0)], gauge_deg=-71.0)

    result = align_to_gps_frame(stage_a, placement, edges, PPM, SHAPES)

    up_in_mosaic = result.poses[0][:2, :2] @ np.array([0.0, -1.0])
    assert up_in_mosaic == pytest.approx([0.0, -1.0], abs=1e-9)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["nan_homography", "zero_weight", "bad_ppm", "missing_shape"])
def test_invalid_inputs_raise(bad: str) -> None:
    stage_a, placement, edges = _scenario({0: 0.0, 1: 0.0}, {0: (0.0, 0.0), 1: (0.0, 13.0)}, [(0, 1, 1.0)], gauge_deg=0.0)
    ppm, shapes = PPM, dict(SHAPES)
    if bad == "nan_homography":
        H = edges[0].homography.copy()
        H[0, 0] = np.nan
        edges = [HeadingEdge(0, 1, H, IMAGE_SHAPE, 1.0)]
    elif bad == "zero_weight":
        edges = [HeadingEdge(0, 1, edges[0].homography, IMAGE_SHAPE, 0.0)]
    elif bad == "bad_ppm":
        ppm = -1.0
    else:
        del shapes[1]

    with pytest.raises(ValueError):
        align_to_gps_frame(stage_a, placement, edges, ppm, shapes)


# ---------------------------------------------------------------------------
# Stage A + B + C on synthetic cameras (independent truth: camera positions and yaws)
# ---------------------------------------------------------------------------


def test_synthetic_serpentine_end_to_end_recovers_camera_yaws_and_positions() -> None:
    altitude = 100.0
    line_b = [SyntheticCamera(13.4 * k * np.sin(np.radians(83.0)), 13.4 * k * np.cos(np.radians(83.0)), altitude, 70.8) for k in range(4)]
    line_c = [SyntheticCamera(cam.east_m - 3.3, cam.north_m + 27.0, altitude, -104.3) for cam in line_b]
    cameras = dict(enumerate(line_b + line_c))
    pairs = [(k, k + 1) for k in range(3)] + [(k, k + 1) for k in range(4, 7)] + [(k, k + 4) for k in range(4)]

    homographies = {(i, j): plane_homography(cameras[i], cameras[j]) for i, j in pairs}
    latlons = {k: latlon_from_en(c.east_m, c.north_m) for k, c in cameras.items()}

    stage_a = average_rotations(
        [RelativeRotation(i, j, relative_rotation_from_homography(H, IMAGE_SHAPE), 1.0) for (i, j), H in homographies.items()]
    )
    ppm = estimate_pixels_per_meter(
        [PairDisplacement(H, IMAGE_SHAPE, latlons[i], latlons[j]) for (i, j), H in homographies.items()]
    )
    placement = place_by_gps(latlons, ppm)
    result = align_to_gps_frame(
        stage_a, placement, [HeadingEdge(i, j, H, IMAGE_SHAPE, 1.0) for (i, j), H in homographies.items()],
        ppm, {k: IMAGE_SHAPE for k in cameras},
    )

    assert ppm == pytest.approx(FOCAL_PX / altitude, rel=1e-4)
    origin = cameras[0]
    for k, cam in cameras.items():
        assert np.degrees(_wrap(result.headings_rad[k] - np.radians(cam.yaw_deg))) == pytest.approx(0.0, abs=1e-6)
        true_centre = (FOCAL_PX / altitude) * np.array([cam.east_m - origin.east_m, -(cam.north_m - origin.north_m)])
        mapped_centre = (result.poses[k] @ np.append(IMAGE_CENTRE, 1.0))[:2]
        assert mapped_centre == pytest.approx(true_centre, rel=1e-4, abs=1e-3)


# ---------------------------------------------------------------------------
# Real images: A + B + C results identical with DJI XMP stripped (equality, not accuracy)
# ---------------------------------------------------------------------------


def _run_a_b_c(paths: list[Path], homography: np.ndarray, shape) -> tuple:
    latlons = load_exif_latlons({0: paths[0], 1: paths[1]})
    stage_a = average_rotations([RelativeRotation(0, 1, relative_rotation_from_homography(homography, shape), 1.0)])
    ppm = estimate_pixels_per_meter([PairDisplacement(homography, shape, latlons[0], latlons[1])])
    placement = place_by_gps(latlons, ppm)
    return align_to_gps_frame(stage_a, placement, [HeadingEdge(0, 1, homography, shape, 1.0)], ppm, {0: shape, 1: shape})


def test_stage_a_b_c_results_are_identical_with_xmp_stripped(tmp_path: Path) -> None:
    names = ["DJI_20230127131426_0352_W.JPG", "DJI_20230127131429_0353_W.JPG"]
    originals = [FIXTURES / n for n in names]
    stripped = [tmp_path / n for n in names]
    for src, dst in zip(originals, stripped):
        _strip_xmp(src, dst)
    images = [cv2.imread(str(p)) for p in originals]
    pair = match_pair(_SiftRatioMatcher(), images[0], images[1], 0, 1)

    a = _run_a_b_c(originals, pair.homography, images[0].shape)
    b = _run_a_b_c(stripped, pair.homography, images[0].shape)

    assert a.headings_rad == b.headings_rad
    assert set(a.poses) == set(b.poses) == {0, 1}
    for k in a.poses:
        assert np.array_equal(a.poses[k], b.poses[k])


# ---------------------------------------------------------------------------
# gps_heading_anchors: GPS-derived (beta - alpha) heading anchors for Stage A
# ---------------------------------------------------------------------------
# Per CLAUDE.md's Stage D 改為只精修位置: for the path without GimbalYawDegree, each
# edge's h = beta - alpha (GPS displacement direction minus the travel direction measured
# in the image) is an absolute heading observation of a node; each edge gives one per
# direction. Per node they are combined with weight 1/sigma^2, sigma^2 = sigma_alpha(alpha)^2
# + (0.174 m / d)^2, and Huber reweighting (c = 1.345). No metadata beyond EXIF lat/lon.

import sea_mosaic.frame_alignment as fa  # noqa: E402  (attributes looked up per test)


def _anchor_world(cameras: dict[int, SyntheticCamera], pairs, latlon_offsets_m=None):
    latlons = {}
    for k, cam in cameras.items():
        de, dn = (latlon_offsets_m or {}).get(k, (0.0, 0.0))
        latlons[k] = latlon_from_en(cam.east_m + de, cam.north_m + dn)
    edges = [HeadingEdge(i, j, plane_homography(cameras[i], cameras[j]), IMAGE_SHAPE, 1.0) for i, j in pairs]
    return latlons, edges, {k: IMAGE_SHAPE for k in cameras}


def test_gps_heading_anchor_weight_is_the_neutral_default() -> None:
    assert fa.GPS_HEADING_ANCHOR_WEIGHT == 1.0  # CLAUDE.md: neutral default, not chosen via gimbal


def test_gps_heading_sigma_values() -> None:
    cross_track_m = np.radians(1.4826 * 0.5) * 13.5  # 0.174 m, from the 13.5 m along-track sigma
    along = np.radians(1.4826 * 0.5)
    across = np.radians(1.4826 * np.sqrt(0.5**2 + 57.0))
    for alpha_deg in (90.0, -90.0):
        for d in (6.0, 13.5, 40.0):
            assert fa.gps_heading_sigma_rad(np.radians(alpha_deg), d) == pytest.approx(np.hypot(along, cross_track_m / d), rel=1e-12)
    assert fa.gps_heading_sigma_rad(0.0, 13.5) == pytest.approx(np.hypot(across, cross_track_m / 13.5), rel=1e-12)
    assert fa.gps_heading_sigma_rad(np.radians(90.0), 6.0) > fa.gps_heading_sigma_rad(np.radians(90.0), 40.0)


def test_gps_heading_anchors_equal_camera_yaws_on_exact_data() -> None:
    # Serpentine: two lines 175 deg apart plus cross-line edges; both directions of every
    # edge contribute. Truth: in the north-up pixel frame a node's angle is its compass yaw.
    line_b = [SyntheticCamera(13.4 * k * np.sin(np.radians(83.0)), 13.4 * k * np.cos(np.radians(83.0)), 100.0, 70.8) for k in range(4)]
    line_c = [SyntheticCamera(c.east_m - 3.3, c.north_m + 27.0, 100.0, -104.3) for c in line_b]
    cameras = dict(enumerate(line_b + line_c))
    pairs = [(k, k + 1) for k in range(3)] + [(k, k + 1) for k in range(4, 7)] + [(k, k + 4) for k in range(4)]
    latlons, edges, shapes = _anchor_world(cameras, pairs)

    anchors = fa.gps_heading_anchors(latlons, edges, shapes)

    assert set(anchors) == set(cameras)
    for k, cam in cameras.items():
        assert np.degrees(_wrap(anchors[k] - np.radians(cam.yaw_deg))) == pytest.approx(0.0, abs=1e-3)


def test_robust_combination_resists_a_short_edge_with_gps_error() -> None:
    # The 0351 pathology: node 0 has three correct along-track edges (13.4 / 26.8 / 40.2 m)
    # and one 6 m edge, exactly along the image's vertical axis, whose neighbour's GPS is
    # off by 1.6 m across track (a ~15 deg direction error). Hand estimate: plain weighted
    # mean pulled ~1.0 deg off, Huber-reweighted ~0.2 deg. Capability claim: robust error
    # <= 0.35 x non-robust error; the non-robust value equals the closed-form weighted
    # circular mean of the observations (weights from the separately tested sigma).
    yaw = 83.0  # course == yaw, so along-track edges sit at alpha = -90 deg exactly
    along = lambda d: SyntheticCamera(d * np.sin(np.radians(yaw)), d * np.cos(np.radians(yaw)), 100.0, yaw)
    cameras = {0: along(0.0), 1: along(13.4), 2: along(26.8), 3: along(40.2), 4: along(-6.0)}
    across = (1.6 * np.cos(np.radians(yaw)), -1.6 * np.sin(np.radians(yaw)))
    latlons, edges, shapes = _anchor_world(cameras, [(0, 1), (0, 2), (0, 3), (0, 4)], latlon_offsets_m={4: across})

    robust = fa.gps_heading_anchors(latlons, edges, shapes)[0]
    plain = fa.gps_heading_anchors(latlons, edges, shapes, robust=False)[0]
    truth = np.radians(yaw)

    robust_err, plain_err = abs(_wrap(robust - truth)), abs(_wrap(plain - truth))
    assert np.degrees(plain_err) > 0.5  # the case actually bites
    assert robust_err <= 0.35 * plain_err


def test_edges_shorter_than_threshold_are_ignored_and_unobserved_nodes_absent() -> None:
    cameras = {0: SyntheticCamera(0.0, 0.0, 100.0, 70.8), 1: SyntheticCamera(12.8, 1.6, 100.0, 70.8),
               2: SyntheticCamera(14.9, 2.0, 100.0, 70.8)}  # 1 -> 2 is ~2.1 m
    latlons, edges, shapes = _anchor_world(cameras, [(0, 1), (1, 2)])

    anchors = fa.gps_heading_anchors(latlons, edges, shapes)

    assert set(anchors) == {0, 1}


def test_edges_touching_a_node_without_latlon_are_skipped() -> None:
    cameras = {0: SyntheticCamera(0.0, 0.0, 100.0, 70.8), 1: SyntheticCamera(12.8, 1.6, 100.0, 70.8),
               2: SyntheticCamera(25.6, 3.2, 100.0, 70.8)}
    latlons, edges, shapes = _anchor_world(cameras, [(0, 1), (1, 2)])
    del latlons[2]

    anchors = fa.gps_heading_anchors(latlons, edges, shapes)

    assert set(anchors) == {0, 1}


def test_gps_anchors_suppress_stage_a_drift_end_to_end() -> None:
    # Stage A edges carry +0.4 deg each (loop-consistent drift, 11 * 0.4 = 4.4 deg without
    # anchors); GPS-derived anchors from exact synthetic cameras remove most of it.
    cameras = {k: SyntheticCamera(13.4 * k * np.sin(np.radians(83.0)), 13.4 * k * np.cos(np.radians(83.0)), 100.0, 70.8) for k in range(12)}
    pairs = [(k, k + 1) for k in range(11)]
    latlons, edges, shapes = _anchor_world(cameras, pairs)
    stage_a_edges = [RelativeRotation(i, j, float(np.radians(0.4)), 1.0) for i, j in pairs]  # true relative yaw is 0

    anchored = average_rotations(
        stage_a_edges, heading_anchors=fa.gps_heading_anchors(latlons, edges, shapes), anchor_weight=fa.GPS_HEADING_ANCHOR_WEIGHT
    )

    worst = max(abs(np.degrees(_wrap(anchored.angles[k] - np.radians(70.8)))) for k in cameras)
    assert worst < 1.0


@pytest.mark.parametrize("bad", ["nan_homography", "zero_weight"])
def test_gps_heading_anchors_invalid_inputs_raise(bad: str) -> None:
    cameras = {0: SyntheticCamera(0.0, 0.0, 100.0, 70.8), 1: SyntheticCamera(12.8, 1.6, 100.0, 70.8)}
    latlons, edges, shapes = _anchor_world(cameras, [(0, 1)])
    if bad == "nan_homography":
        H = edges[0].homography.copy()
        H[0, 0] = np.nan
        edges = [HeadingEdge(0, 1, H, IMAGE_SHAPE, 1.0)]
    else:
        edges = [HeadingEdge(0, 1, edges[0].homography, IMAGE_SHAPE, 0.0)]

    with pytest.raises(ValueError):
        fa.gps_heading_anchors(latlons, edges, shapes)
