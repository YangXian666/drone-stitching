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
