"""Tests for sea_mosaic.gps_placement (Stage B: direct GPS placement).

Stage B places each image's CENTRE in a north-up pixel frame (x = ppm * East,
y = -ppm * North; image y points down), with no rotation and no dependence on Stage A,
and estimates pixels_per_meter from the data itself. Per CLAUDE.md's
最小可行版本範圍決定, it uses only EXIF lat/lon and the images -- never GimbalYawDegree or
any DJI XMP field -- and every expected value below comes from hand calculation, an
independent formula (haversine), or the synthetic pinhole camera model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sea_mosaic.estimate import match_pair
from sea_mosaic.gps_placement import (
    MIN_GPS_DISPLACEMENT_M,
    GpsPlacement,
    PairDisplacement,
    estimate_pixels_per_meter,
    load_exif_latlons,
    place_by_gps,
)
from sea_mosaic.io_utils import _read_xmp_drone_dji_fields
from sea_mosaic.matcher import MatchResult
from synthetic_camera import (
    FOCAL_PX,
    IMAGE_CENTRE,
    IMAGE_SHAPE,
    ORIGIN_LATLON,
    SyntheticCamera,
    latlon_from_en,
    plane_homography,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dji_smoke"
PPM = 28.0


# ---------------------------------------------------------------------------
# place_by_gps
# ---------------------------------------------------------------------------


def test_north_displacement_maps_to_negative_y() -> None:
    # Locks out the mirror bug: North must never map to +y (image y points down).
    placement = place_by_gps({0: latlon_from_en(0.0, 0.0), 1: latlon_from_en(0.0, 10.0)}, PPM)

    assert placement.centres_px[1] == pytest.approx([0.0, -10.0 * PPM], abs=1e-6)


def test_east_displacement_maps_to_positive_x() -> None:
    placement = place_by_gps({0: latlon_from_en(0.0, 0.0), 1: latlon_from_en(10.0, 0.0)}, PPM)

    assert placement.centres_px[1] == pytest.approx([10.0 * PPM, 0.0], abs=1e-6)


def test_origin_defaults_to_smallest_located_index_and_can_be_overridden() -> None:
    latlon = {7: latlon_from_en(3.0, 4.0), 2: latlon_from_en(0.0, 0.0), 5: latlon_from_en(-6.0, 1.0)}

    default = place_by_gps(latlon, PPM)
    explicit = place_by_gps(latlon, PPM, origin_index=7)

    assert default.origin_index == 2
    assert default.centres_px[2] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert default.centres_px[7] == pytest.approx([3.0 * PPM, -4.0 * PPM], abs=1e-6)
    assert explicit.origin_index == 7
    assert explicit.centres_px[7] == pytest.approx([0.0, 0.0], abs=1e-9)
    # rel, not abs: with node 7 as origin the projection uses cos(lat of node 7) while
    # latlon_from_en inverts around ORIGIN_LATLON -- an equirectangular-approximation
    # difference of ~5e-7 relative here (far below test_local_projection_matches_haversine's
    # 0.1% claim), not a placement error.
    assert explicit.centres_px[2] == pytest.approx([-3.0 * PPM, 4.0 * PPM], rel=1e-6)


def test_positions_scale_linearly_with_pixels_per_meter() -> None:
    latlon = {0: latlon_from_en(0.0, 0.0), 1: latlon_from_en(12.0, -5.0)}

    a = place_by_gps(latlon, 10.0).centres_px[1]
    b = place_by_gps(latlon, 25.0).centres_px[1]

    assert b == pytest.approx(a * 2.5, rel=1e-12)


@pytest.mark.parametrize("missing_value", [None, (float("nan"), -5.1), (36.4, float("nan"))])
def test_node_without_usable_gps_is_unlocated_never_guessed(missing_value) -> None:
    latlon = {0: latlon_from_en(0.0, 0.0), 1: latlon_from_en(10.0, 0.0)}
    if missing_value is not None:
        latlon[2] = missing_value

    placement = place_by_gps(latlon, PPM, node_indices={0, 1, 2})

    assert placement.unlocated == {2}
    assert 2 not in placement.centres_px


def test_gps_only_node_without_edges_is_still_placed() -> None:
    # Stage B never looks at edges: a node known only through GPS is placed like any other.
    placement = place_by_gps(
        {0: latlon_from_en(0.0, 0.0), 9: latlon_from_en(0.0, -20.0)}, PPM, node_indices={0}
    )

    assert placement.centres_px[9] == pytest.approx([0.0, 20.0 * PPM], abs=1e-6)
    assert placement.unlocated == set()


@pytest.mark.parametrize("ppm", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_pixels_per_meter_raises(ppm: float) -> None:
    with pytest.raises(ValueError):
        place_by_gps({0: latlon_from_en(0.0, 0.0)}, ppm)


def test_origin_without_gps_raises() -> None:
    with pytest.raises(ValueError):
        place_by_gps({0: latlon_from_en(0.0, 0.0)}, PPM, node_indices={0, 3}, origin_index=3)


@pytest.mark.parametrize("bad_latlon", [(91.0, 0.0), (-90.5, 0.0), (0.0, 180.5), (0.0, -181.0)])
def test_out_of_range_latlon_raises(bad_latlon) -> None:
    with pytest.raises(ValueError):
        place_by_gps({0: latlon_from_en(0.0, 0.0), 1: bad_latlon}, PPM)


def test_empty_input_returns_empty_placement() -> None:
    placement = place_by_gps({}, PPM)

    assert placement == GpsPlacement(centres_px={}, unlocated=set(), origin_index=None)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (*a, *b))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371000.0 * np.arcsin(np.sqrt(h)))


@pytest.mark.parametrize("east_m, north_m", [(300.0, 0.0), (0.0, 300.0), (212.0, -212.0)])
def test_local_projection_matches_haversine_within_a_tenth_of_a_percent(
    east_m: float, north_m: float
) -> None:
    far = (ORIGIN_LATLON[0] + north_m / 111_000.0, ORIGIN_LATLON[1] + east_m / 89_000.0)
    placement = place_by_gps({0: ORIGIN_LATLON, 1: far}, PPM)

    projected_m = np.linalg.norm(placement.centres_px[1]) / PPM
    assert projected_m == pytest.approx(_haversine_m(ORIGIN_LATLON, far), rel=1e-3)


# ---------------------------------------------------------------------------
# load_exif_latlons: XMP-sourced lat/lon counts as no GPS
# ---------------------------------------------------------------------------


def test_load_exif_latlons_excludes_xmp_sourced_and_missing_gps() -> None:
    latlons = load_exif_latlons(
        {
            0: FIXTURES / "DJI_20230127131426_0352_W.JPG",
            1: FIXTURES / "exif_stripped_xmp_only.jpg",
            2: FIXTURES / "no_gps.jpg",
        }
    )

    assert set(latlons) == {0}
    assert latlons[0] == pytest.approx((36.4277901, -5.1262460), abs=1e-6)


# ---------------------------------------------------------------------------
# estimate_pixels_per_meter
# ---------------------------------------------------------------------------


def _sim2(theta_deg: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def _pair(pixel_shift: float, gps_east_m: float, theta_deg: float = 0.0) -> PairDisplacement:
    """A similarity homography whose dst centre sits pixel_shift px from the src centre
    (in src coordinates), paired with a GPS displacement of gps_east_m meters."""
    R = _sim2(theta_deg, 0.0, 0.0)[:2, :2]
    offset = R @ np.array([pixel_shift, 0.0])  # arbitrary direction; only length matters
    t = IMAGE_CENTRE - R @ IMAGE_CENTRE + offset
    pose_dst = _sim2(theta_deg, *t)  # pose_src = identity
    return PairDisplacement(
        homography=np.linalg.inv(pose_dst),
        src_image_shape=IMAGE_SHAPE,
        src_latlon=latlon_from_en(0.0, 0.0),
        dst_latlon=latlon_from_en(gps_east_m, 0.0),
    )


def test_exact_similarity_pairs_recover_pixels_per_meter() -> None:
    pairs = [_pair(28.0 * d, d, theta) for d, theta in [(13.5, 0.0), (27.0, 175.0), (9.0, -40.0)]]

    assert estimate_pixels_per_meter(pairs) == pytest.approx(28.0, rel=1e-4)


def test_median_tolerates_a_minority_of_outlier_pairs() -> None:
    good = [_pair(28.0 * d, d) for d in (10.0, 13.5, 27.0)]
    outliers = [_pair(3 * 28.0 * d, d) for d in (12.0, 30.0)]

    assert estimate_pixels_per_meter(good + outliers) == pytest.approx(28.0, rel=1e-4)


def test_pairs_shorter_than_threshold_are_excluded() -> None:
    assert MIN_GPS_DISPLACEMENT_M == 5.0  # CLAUDE.md: model-derived, re-validate in October
    pairs = [_pair(28.0 * 13.5, 13.5), _pair(59.0 * 1.64, 1.64), _pair(59.0 * 1.64, 1.64)]

    assert estimate_pixels_per_meter(pairs) == pytest.approx(28.0, rel=1e-4)


def test_no_pair_above_threshold_raises() -> None:
    with pytest.raises(ValueError):
        estimate_pixels_per_meter([_pair(59.0 * 1.64, 1.64)])


@pytest.mark.parametrize("theta_deg", [0.0, 37.0, 175.0, -120.0])
def test_estimate_is_rotation_invariant(theta_deg: float) -> None:
    assert estimate_pixels_per_meter([_pair(28.0 * 13.5, 13.5, theta_deg)]) == pytest.approx(
        28.0, rel=1e-4
    )


def test_synthetic_nadir_cameras_give_focal_over_altitude() -> None:
    # Independent truth: for untilted nadir pinhole cameras, ground scale = f / altitude.
    altitude = 100.0
    cameras = [
        SyntheticCamera(0.0, 0.0, altitude, 70.8),
        SyntheticCamera(12.8, 1.6, altitude, 70.8),
        SyntheticCamera(4.0, 27.0, altitude, -104.3),  # cross-line, 175 deg flip
    ]
    pairs = [
        PairDisplacement(
            homography=plane_homography(cameras[a], cameras[b]),
            src_image_shape=IMAGE_SHAPE,
            src_latlon=latlon_from_en(cameras[a].east_m, cameras[a].north_m),
            dst_latlon=latlon_from_en(cameras[b].east_m, cameras[b].north_m),
        )
        for a, b in [(0, 1), (0, 2), (1, 2)]
    ]

    assert estimate_pixels_per_meter(pairs) == pytest.approx(FOCAL_PX / altitude, rel=1e-4)


def test_estimate_is_in_src_pixel_units_when_altitudes_differ() -> None:
    # Contract: the pixel displacement is measured in SRC pixel coordinates, so for nadir
    # cameras at different altitudes the estimate is f / src_altitude, not f / dst_altitude.
    # (For equal-scale pairs, |inv(H) c - c| and |H c - c| coincide, so only a scale
    # difference distinguishes the two.)
    src = SyntheticCamera(0.0, 0.0, 100.0, 70.8)
    dst = SyntheticCamera(12.8, 1.6, 110.0, 70.8)
    pair = PairDisplacement(
        homography=plane_homography(src, dst),
        src_image_shape=IMAGE_SHAPE,
        src_latlon=latlon_from_en(src.east_m, src.north_m),
        dst_latlon=latlon_from_en(dst.east_m, dst.north_m),
    )

    assert estimate_pixels_per_meter([pair]) == pytest.approx(FOCAL_PX / 100.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Real images: no dependence on DJI XMP (equality, not accuracy)
# ---------------------------------------------------------------------------


def _strip_xmp(src: Path, dst: Path) -> None:
    """Copy a JPEG, dropping any APP1 segment carrying an XMP packet (EXIF APP1 kept)."""
    data = src.read_bytes()
    out, i = bytearray(data[:2]), 2  # SOI
    while i < len(data):
        marker = data[i : i + 2]
        if marker == b"\xff\xda":  # start of scan: rest is entropy-coded data
            out += data[i:]
            break
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        segment = data[i : i + 2 + length]
        if not (marker == b"\xff\xe1" and segment[4:].startswith(b"http://ns.adobe.com/xap/1.0/")):
            out += segment
        i += 2 + length
    dst.write_bytes(bytes(out))


class _SiftRatioMatcher:
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


def _run_stage_b(path_a: Path, path_b: Path, pair_homography: np.ndarray, image_shape) -> tuple:
    latlons = load_exif_latlons({0: path_a, 1: path_b})
    ppm = estimate_pixels_per_meter(
        [PairDisplacement(pair_homography, image_shape, latlons[0], latlons[1])]
    )
    return latlons, ppm, place_by_gps(latlons, ppm, node_indices={0, 1})


def test_stage_b_results_are_identical_with_xmp_stripped(tmp_path: Path) -> None:
    names = ["DJI_20230127131426_0352_W.JPG", "DJI_20230127131429_0353_W.JPG"]
    originals = [FIXTURES / n for n in names]
    stripped = [tmp_path / n for n in names]
    for src, dst in zip(originals, stripped):
        _strip_xmp(src, dst)
        assert _read_xmp_drone_dji_fields(src) != {}  # the original really has DJI XMP
        assert _read_xmp_drone_dji_fields(dst) == {}  # ... and the copy really has none

    images = [cv2.imread(str(p)) for p in originals]
    for original, copy in zip(images, (cv2.imread(str(p)) for p in stripped)):
        assert np.array_equal(original, copy)  # pixels untouched, so one match suffices
    pair = match_pair(_SiftRatioMatcher(), images[0], images[1], 0, 1)

    latlons_a, ppm_a, placement_a = _run_stage_b(*originals, pair.homography, images[0].shape)
    latlons_b, ppm_b, placement_b = _run_stage_b(*stripped, pair.homography, images[0].shape)

    assert latlons_a == latlons_b
    assert ppm_a == ppm_b
    assert placement_a.unlocated == placement_b.unlocated == set()
    for index in placement_a.centres_px:
        assert np.array_equal(placement_a.centres_px[index], placement_b.centres_px[index])

