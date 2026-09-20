"""Local planar projection of GPS coordinates for use as soft pose-graph anchors.

Not a geodetic-accuracy coordinate system: per CLAUDE.md's 已知的暫緩事項, this project
defers full direct-georeferencing calibration, so this module only needs to turn
lat/lon into a locally-consistent metric XY frame good enough to anchor pose-graph
optimization — not to preserve absolute scale or match external geographic datasets.
"""

from __future__ import annotations

import numpy as np

R_EARTH_M = 6371000.0  # spherical mean Earth radius (sphere model, not WGS84 ellipsoid)


def geodetic_to_local_xy(
    lat_deg: float,
    lon_deg: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> np.ndarray:
    """Equirectangular (local tangent-plane) approximation of (lat_deg, lon_deg)
    relative to (origin_lat_deg, origin_lon_deg), returned as [east_m, north_m].
    Valid for small areas (flight lines up to a few km); not a substitute for a real
    geodetic projection (e.g. UTM)."""
    east_m = np.radians(lon_deg - origin_lon_deg) * R_EARTH_M * np.cos(np.radians(origin_lat_deg))
    north_m = np.radians(lat_deg - origin_lat_deg) * R_EARTH_M
    return np.array([east_m, north_m], dtype=np.float64)


def project_gps_positions(
    gps_positions: dict[int, np.ndarray],
    origin_index: int | None = None,
) -> dict[int, np.ndarray]:
    """Project each image's [lat, lon, ...] geodetic reading (as returned by
    io_utils.load_gps_position) onto a shared local XY plane in meters, anchored at
    gps_positions[origin_index].

    origin_index defaults to the smallest key present in gps_positions when not given
    — a simple, dependency-free default for standalone/test use. Pipeline code that
    wants the mosaic's pose-graph reference image as origin should pass
    origin_index=config.reference_index explicitly; this module intentionally has no
    knowledge of PipelineConfig.
    """
    if origin_index is None:
        origin_index = min(gps_positions.keys())

    origin_lat_deg = float(gps_positions[origin_index][0])
    origin_lon_deg = float(gps_positions[origin_index][1])

    return {
        image_index: geodetic_to_local_xy(
            float(reading[0]), float(reading[1]), origin_lat_deg, origin_lon_deg
        )
        for image_index, reading in gps_positions.items()
    }


def estimate_pixels_per_meter(
    altitude_m: float,
    dfov_deg: float,
    width_px: int,
    height_px: int,
) -> float:
    """Rough pinhole-camera estimate of pixels-per-meter ground sampling density.

    This is a crude approximation for compose_global_transforms's GPS anchors (which
    need *some* pixel<->meter conversion to be dimensionally comparable to feature-match
    residuals, even without absolute-scale precision) — it is NOT a substitute for
    geo/camera.py / geo/direct.py's full camera model and precise georeferencing. Sources
    of approximation: (1) an ideal pinhole camera with no real intrinsics/calibration,
    (2) whatever altitude reading the caller passes in as ground truth (see CLAUDE.md's
    documented AbsoluteAltitude vs RelativeAltitude ambiguity — this function takes no
    position on which is correct), (3) no lens distortion correction.

    Method: the diagonal field of view (dfov_deg) and altitude give the ground diagonal
    coverage via a pinhole approximation (2 * altitude_m * tan(dfov_deg/2)); dividing the
    image's diagonal pixel count by that ground diagonal gives pixels_per_meter directly
    (splitting the diagonal into separate ground_width_m/ground_height_m via the pixel
    aspect ratio, as one might do to sanity-check per-axis coverage, cancels out
    algebraically and isn't needed for this ratio).
    """
    diagonal_px = np.hypot(width_px, height_px)
    ground_diagonal_m = 2 * altitude_m * np.tan(np.radians(dfov_deg) / 2)
    return float(diagonal_px / ground_diagonal_m)
