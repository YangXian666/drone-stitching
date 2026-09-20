"""Direct georeferencing via ray-plane reprojection: pixel -> world coordinates.

Skeleton only: not yet connected to the estimate/compose/warp/blend pipeline, and does
not produce a GeoTIFF (see the "direct georeferencing" item in CLAUDE.md's 目前狀態 checklist).
"""

from __future__ import annotations

import numpy as np

from sea_mosaic.geo.camera import CameraIntrinsics, CameraPose


def pixel_to_world_ray(
    pixel: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ray_origin, ray_direction) in world coordinates for a pixel, for ray-plane
    intersection."""
    ...


def intersect_ray_with_plane(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """Intersect a ray with a plane (e.g. the sea surface at z=0); return the world-space
    intersection point, or None if the ray is parallel to the plane."""
    ...


def direct_georeference(
    pixels: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    ground_altitude: float = 0.0,
) -> np.ndarray:
    """Project pixel coordinates to world/geographic coordinates via ray-plane
    reprojection against a flat sea-surface plane.

    Points that cannot be resolved (e.g. ray parallel to the plane) are filled with
    np.nan, never 0.
    """
    ...
