"""Stage B of the staged pose-graph architecture: direct GPS placement.

Places each image's CENTRE in a north-up pixel frame -- x = ppm * East, y = -ppm * North
(image y points down, so North maps to -y; mapping it to +y was the mirror bug recorded
in CLAUDE.md) -- directly from GPS, not by iterative solving. Carries no rotation and
does not depend on Stage A; aligning Stage A's rotations to this frame and composing
centre-anchored poses is Stage C's job.

Per CLAUDE.md's 最小可行版本範圍決定, Stage B uses only EXIF lat/lon and the images:
lat/lon that could only be read from DJI XMP counts as no GPS, and pixels_per_meter is
estimated from the data instead of from any altitude field.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sea_mosaic.geo.projection import geodetic_to_local_xy
from sea_mosaic.io_utils import load_latlon_with_source

# Pairs whose GPS displacement is shorter than this are too noisy to use. A model-derived
# interpolation, not a value the data pinned down: data/'s 184 GPS-proximity pairs have
# no edges between 1.64 m and 8.37 m. From the 12-15 m bin, the typical GPS displacement
# error is ~0.35 m, so heading noise ~0.35/d rad (predicted 1.5 deg at 13.5 m vs 1.75 deg
# measured); 5 m keeps that under ~4 deg and the ppm ratio error under ~7%. Re-validate on
# the October dataset (see CLAUDE.md's 最小 GPS 位移閾值與 Stage C 漂移發現).
MIN_GPS_DISPLACEMENT_M = 5.0


@dataclass
class GpsPlacement:
    """Image-centre positions in the north-up pixel frame, origin_index at (0, 0).

    unlocated lists nodes the caller declared (node_indices) that have no usable GPS;
    they are deliberately absent from centres_px rather than given a guessed position.
    """

    centres_px: dict[int, np.ndarray]
    unlocated: set[int]
    origin_index: int | None


@dataclass(frozen=True)
class PairDisplacement:
    """One image pair's pixel and GPS evidence for estimating pixels_per_meter."""

    homography: np.ndarray  # src pixels -> dst pixels
    src_image_shape: tuple[int, ...]  # numpy shape (rows, cols[, channels])
    src_latlon: tuple[float, float]
    dst_latlon: tuple[float, float]


def load_exif_latlons(paths: dict[int, Path]) -> dict[int, tuple[float, float]]:
    """EXIF-sourced (lat, lon) per image index. Images whose coordinates exist only in DJI
    XMP, or that have no GPS at all, are left out -- place_by_gps then reports them as
    unlocated when they are listed in node_indices."""
    latlons = {}
    for index, path in paths.items():
        result = load_latlon_with_source(path)
        if result is not None and result[2] == "exif":
            latlons[index] = (result[0], result[1])
    return latlons


def _usable(latlon: tuple[float, float] | None) -> bool:
    return latlon is not None and bool(np.all(np.isfinite(latlon)))


def place_by_gps(
    latlon: dict[int, tuple[float, float] | None],
    pixels_per_meter: float,
    node_indices: Iterable[int] = (),
    origin_index: int | None = None,
) -> GpsPlacement:
    """Place every node with usable GPS; list node_indices entries without it as unlocated.

    The nodes considered are node_indices plus every key of latlon, so a node known only
    through GPS (no edges) is still placed. A None or non-finite lat/lon counts as no GPS.
    origin_index defaults to the smallest located index. Uses geo.projection's
    equirectangular local-plane approximation (valid for areas up to a few km).

    Raises ValueError for a non-positive or non-finite pixels_per_meter, an out-of-range
    lat/lon, or an origin_index without usable GPS.
    """
    if not (np.isfinite(pixels_per_meter) and pixels_per_meter > 0):
        raise ValueError("pixels_per_meter must be finite and > 0")

    located = {index: value for index, value in latlon.items() if _usable(value)}
    for index, (lat, lon) in located.items():
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"lat/lon out of range for node {index}: {(lat, lon)}")
    unlocated = (set(node_indices) | set(latlon)) - set(located)

    if not located:
        if origin_index is not None:
            raise ValueError(f"origin_index {origin_index} has no usable GPS")
        return GpsPlacement(centres_px={}, unlocated=unlocated, origin_index=None)
    if origin_index is None:
        origin_index = min(located)
    elif origin_index not in located:
        raise ValueError(f"origin_index {origin_index} has no usable GPS")

    origin_lat, origin_lon = located[origin_index]
    centres_px = {}
    for index, (lat, lon) in located.items():
        east_m, north_m = geodetic_to_local_xy(lat, lon, origin_lat, origin_lon)
        centres_px[index] = pixels_per_meter * np.array([east_m, -north_m])

    return GpsPlacement(centres_px=centres_px, unlocated=unlocated, origin_index=origin_index)


def estimate_pixels_per_meter(
    pairs: list[PairDisplacement], min_gps_distance_m: float = MIN_GPS_DISPLACEMENT_M
) -> float:
    """Median over pairs of |pixel displacement| / |GPS displacement|.

    The pixel displacement is where the dst image centre lands in src pixel coordinates
    (inv(H) applied to the src centre, an exact point mapping -- no linearization), minus
    the src centre. Only lengths are compared, so the estimate needs no rotation, altitude
    or field of view. Pairs with GPS displacement below min_gps_distance_m are skipped.

    Raises ValueError when no pair is long enough to use.
    """
    ratios = []
    for pair in pairs:
        east_m, north_m = geodetic_to_local_xy(*pair.dst_latlon, *pair.src_latlon)
        gps_distance_m = float(np.hypot(east_m, north_m))
        if gps_distance_m < min_gps_distance_m:
            continue
        rows, cols = pair.src_image_shape[0], pair.src_image_shape[1]
        centre = np.array([cols / 2.0, rows / 2.0, 1.0])
        mapped = np.linalg.inv(pair.homography) @ centre
        pixel_distance = float(np.linalg.norm(mapped[:2] / mapped[2] - centre[:2]))
        ratios.append(pixel_distance / gps_distance_m)

    if not ratios:
        raise ValueError(f"no pair has a GPS displacement of at least {min_gps_distance_m} m")
    return float(np.median(ratios))
