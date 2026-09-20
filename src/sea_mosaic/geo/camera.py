"""Camera intrinsics, distortion, and ground-footprint geometry for direct georeferencing.

Skeleton only: not yet connected to the estimate/compose/warp/blend pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    focal_length_px: tuple[float, float]  # (fx, fy)
    principal_point_px: tuple[float, float]  # (cx, cy)
    image_size: tuple[int, int]  # (width, height)


@dataclass
class DistortionCoefficients:
    """Lens distortion coefficients."""

    radial: tuple[float, float, float]  # k1, k2, k3
    tangential: tuple[float, float]  # p1, p2


@dataclass
class CameraPose:
    """Camera pose in world coordinates, derived from GPS + altitude + gimbal/attitude."""

    position_world: np.ndarray  # shape (3,), world/ENU position
    rotation_world: np.ndarray  # shape (3, 3), rotation matrix


def undistort_points(
    points_px: np.ndarray,
    intrinsics: CameraIntrinsics,
    distortion: DistortionCoefficients,
) -> np.ndarray:
    """Undistort pixel coordinates using the camera intrinsics and distortion coefficients."""
    ...


def compute_ground_footprint(
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    ground_altitude: float = 0.0,
) -> np.ndarray:
    """Compute the ground footprint polygon (image corners projected onto a flat ground
    plane at ground_altitude), returned as world-coordinate points."""
    ...
