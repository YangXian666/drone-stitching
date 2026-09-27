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


def ground_to_image(
    camera: SyntheticCamera, *, focal_px: float = FOCAL_PX, image_shape: tuple[int, int] = IMAGE_SHAPE
) -> np.ndarray:
    """3x3 map from ground-plane (east_m, north_m, 1) to this camera's pixels. The defaults
    are the full-size DJI camera; the pipeline scene passes the small camera."""
    cx, cy = image_shape[1] / 2.0, image_shape[0] / 2.0
    K = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]])
    R = _world_to_camera(camera)
    centre = np.array([camera.east_m, camera.north_m, camera.altitude_m])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ centre])


def plane_homography(
    src: SyntheticCamera,
    dst: SyntheticCamera,
    *,
    focal_px: float = FOCAL_PX,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
) -> np.ndarray:
    """Exact homography mapping src pixels to dst pixels, normalized so H[2,2] = 1."""
    camera = dict(focal_px=focal_px, image_shape=image_shape)
    H = ground_to_image(dst, **camera) @ np.linalg.inv(ground_to_image(src, **camera))
    return H / H[2, 2]


def latlon_from_en(east_m: float, north_m: float) -> tuple[float, float]:
    """Inverse equirectangular approximation around ORIGIN_LATLON (hand-derived inverse of
    the forward formula, not a call into sea_mosaic)."""
    lat0, lon0 = ORIGIN_LATLON
    lat = lat0 + np.degrees(north_m / _R_EARTH_M)
    lon = lon0 + np.degrees(east_m / (_R_EARTH_M * np.cos(np.radians(lat0))))
    return float(lat), float(lon)


# ---------------------------------------------------------------------------
# Pipeline-level scene: small images, a matcher, lat/lon -- all mutually consistent
# ---------------------------------------------------------------------------

# Small camera with the same diagonal field of view as the DJI camera: identical ground
# footprint (so the same pairs overlap), pixel arrays small enough for fast run_pipeline
# tests (canvas of a few hundred pixels).
PIPELINE_IMAGE_SHAPE = (60, 80)  # (rows, cols)
PIPELINE_FOCAL_PX = float(
    np.hypot(PIPELINE_IMAGE_SHAPE[1], PIPELINE_IMAGE_SHAPE[0]) / 2 / np.tan(np.radians(82.9 / 2))
)


@dataclass
class PipelineScene:
    cameras: dict[int, SyntheticCamera]
    images: dict[int, np.ndarray]  # (rows, cols, 3) uint8, rendered from ground_texture
    latlons: dict[int, tuple[float, float]]  # what load_exif_latlons would return
    gimbal_yaw_deg: dict[int, float]  # what load_gimbal_yaw would return (true compass yaw)
    pairs: list[tuple[int, int]]  # every (i, j), i < j, with GPS distance < max_pair_distance_m
    correspondences: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]  # exact (src, dst) px


def ground_texture(east_m: np.ndarray, north_m: np.ndarray) -> np.ndarray:
    """Smooth colour pattern on the ground plane, (..., 3) float in [0, 255]. Wavelengths of
    40-70 m (22-40 px at the pipeline camera's ~0.57 px/m) keep bilinear resampling
    accurate, so warped images agree with each other to within ~1 grey level."""
    phases = (0.0, 1.3, 2.6)
    channels = [
        128.0
        + 55.0 * np.sin(2 * np.pi * east_m / 47.0 + p) * np.cos(2 * np.pi * north_m / 61.0)
        + 40.0 * np.sin(2 * np.pi * (east_m + 0.7 * north_m) / 71.0 + 2 * p)
        for p in phases
    ]
    return np.clip(np.stack(channels, axis=-1), 0.0, 255.0)


def _render(camera: SyntheticCamera) -> np.ndarray:
    rows, cols = PIPELINE_IMAGE_SHAPE
    x, y = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    ground = np.linalg.inv(ground_to_image(camera, focal_px=PIPELINE_FOCAL_PX, image_shape=PIPELINE_IMAGE_SHAPE)) @ np.stack(
        [x.ravel(), y.ravel(), np.ones(x.size)]
    )
    east, north = ground[0] / ground[2], ground[1] / ground[2]
    return ground_texture(east, north).reshape(rows, cols, 3).round().astype(np.uint8)


def _scene_correspondences(src: SyntheticCamera, dst: SyntheticCamera, spacing_m: float):
    """Ground-grid points projected exactly into both small cameras, kept where both see them."""
    rows, cols = PIPELINE_IMAGE_SHAPE
    lo_e, hi_e = min(src.east_m, dst.east_m) - 80, max(src.east_m, dst.east_m) + 80
    lo_n, hi_n = min(src.north_m, dst.north_m) - 80, max(src.north_m, dst.north_m) + 80
    east, north = np.meshgrid(np.arange(lo_e, hi_e, spacing_m), np.arange(lo_n, hi_n, spacing_m))
    ground = np.stack([east.ravel(), north.ravel(), np.ones(east.size)])

    def project(camera):
        p = ground_to_image(camera, focal_px=PIPELINE_FOCAL_PX, image_shape=PIPELINE_IMAGE_SHAPE) @ ground
        return (p[:2] / p[2]).T

    a, b = project(src), project(dst)
    inside = np.all((a >= 0) & (a < [cols, rows]) & (b >= 0) & (b < [cols, rows]), axis=1)
    return a[inside], b[inside]


def pipeline_scene(
    cameras: dict[int, SyntheticCamera], *, max_pair_distance_m: float = 40.0, grid_spacing_m: float = 5.0
) -> PipelineScene:
    keys = sorted(cameras)
    pairs = [
        (i, j)
        for n, i in enumerate(keys)
        for j in keys[n + 1 :]
        if np.hypot(cameras[i].east_m - cameras[j].east_m, cameras[i].north_m - cameras[j].north_m)
        < max_pair_distance_m
    ]
    return PipelineScene(
        cameras=dict(cameras),
        images={k: _render(c) for k, c in cameras.items()},
        latlons={k: latlon_from_en(c.east_m, c.north_m) for k, c in cameras.items()},
        gimbal_yaw_deg={k: c.yaw_deg for k, c in cameras.items()},
        pairs=pairs,
        correspondences={(i, j): _scene_correspondences(cameras[i], cameras[j], grid_spacing_m) for i, j in pairs},
    )


class SceneMatcher:
    """Matcher test double for a PipelineScene: identifies the two images by object identity
    (run_pipeline passes the caller's arrays through unchanged) and returns the scene's exact
    correspondences. A pair in fail_pairs gets 2 correspondences, below the homography minimum
    of 4, so cv2.findHomography raises inside match_pair -- the real mechanism behind
    run_pipeline's per-pair error isolation. An unknown image or pair raises KeyError."""

    name = "synthetic-scene-matcher"

    def __init__(self, scene: PipelineScene, fail_pairs: set[tuple[int, int]] = frozenset()) -> None:
        self._scene = scene
        self._fail_pairs = set(fail_pairs)
        self._index_of = {id(image): k for k, image in scene.images.items()}

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> "MatchResult":
        from sea_mosaic.matcher import MatchResult

        pair = (self._index_of[id(image_a)], self._index_of[id(image_b)])
        src, dst = self._scene.correspondences[pair]
        if pair in self._fail_pairs:
            src, dst = src[:2], dst[:2]
        return MatchResult(src_points=src.copy(), dst_points=dst.copy(), scores=None)
