"""Synthetic pinhole-camera model shared by Stage B/C tests.

Downward-looking cameras over a flat ground plane z=0 give exact plane-induced
homographies with independently known truth: positions, compass yaws and pixels per
meter (f / altitude for an untilted nadir camera). Nothing here reads real metadata --
per CLAUDE.md's 最小可行版本範圍決定, formal Stage A-D tests take expected values only
from hand calculation, closed forms, or independently constructed synthetic data.

World frame is (East, North, Up) in meters. A camera with yaw psi (compass bearing,
clockwise from north) has its image top pointing along psi.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SHAPE = (3040, 4056)  # (rows, cols), the DJI H20T wide camera
FOCAL_PX = float(np.hypot(IMAGE_SHAPE[1], IMAGE_SHAPE[0]) / 2 / np.tan(np.radians(82.9 / 2)))
IMAGE_CENTRE = np.array([IMAGE_SHAPE[1] / 2.0, IMAGE_SHAPE[0] / 2.0])

_R_EARTH_M = 6371000.0
ORIGIN_LATLON = (36.4277901, -5.1262460)


@dataclass(frozen=True)
class SyntheticCamera:
    east_m: float
    north_m: float
    altitude_m: float
    yaw_deg: float
    tilt_deg: float = 0.0
    tilt_azimuth_deg: float = 0.0


def _world_to_camera(camera: SyntheticCamera) -> np.ndarray:
    p = np.radians(camera.yaw_deg)
    nadir = np.array(
        [[np.cos(p), -np.sin(p), 0.0], [-np.sin(p), -np.cos(p), 0.0], [0.0, 0.0, -1.0]]
    )
    tilt_vec = np.radians(camera.tilt_deg) * np.array(
        [np.cos(np.radians(camera.tilt_azimuth_deg)), np.sin(np.radians(camera.tilt_azimuth_deg)), 0.0]
    )
    tilt, _ = cv2.Rodrigues(tilt_vec)
    return tilt @ nadir


def ground_to_image(camera: SyntheticCamera) -> np.ndarray:
    """3x3 map from ground-plane (east_m, north_m, 1) to this camera's pixels."""
    K = np.array([[FOCAL_PX, 0.0, IMAGE_CENTRE[0]], [0.0, FOCAL_PX, IMAGE_CENTRE[1]], [0.0, 0.0, 1.0]])
    R = _world_to_camera(camera)
    centre = np.array([camera.east_m, camera.north_m, camera.altitude_m])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ centre])


def plane_homography(src: SyntheticCamera, dst: SyntheticCamera) -> np.ndarray:
    """Exact homography mapping src pixels to dst pixels, normalized so H[2,2] = 1."""
    H = ground_to_image(dst) @ np.linalg.inv(ground_to_image(src))
    return H / H[2, 2]


def latlon_from_en(east_m: float, north_m: float) -> tuple[float, float]:
    """Inverse equirectangular approximation around ORIGIN_LATLON (hand-derived inverse of
    the forward formula, not a call into sea_mosaic)."""
    lat0, lon0 = ORIGIN_LATLON
    lat = lat0 + np.degrees(north_m / _R_EARTH_M)
    lon = lon0 + np.degrees(east_m / (_R_EARTH_M * np.cos(np.radians(lat0))))
    return float(lat), float(lon)
