"""Tests for the pipeline-level synthetic scene in tests/synthetic_camera.py.

run_pipeline tests need what the Stage A-D tests never did: real (small) image arrays
whose content agrees with the geometry, a Matcher that returns correspondences consistent
with the plane-induced homographies, and EXIF-style lat/lon consistent with both. These
tests check that the scene generator itself keeps those three consistent, so a failing
pipeline test can't be blamed on inconsistent test data.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sea_mosaic.estimate import match_pair
from sea_mosaic.geo.projection import geodetic_to_local_xy
from synthetic_camera import (
    FOCAL_PX,
    IMAGE_SHAPE,
    PIPELINE_FOCAL_PX,
    PIPELINE_IMAGE_SHAPE,
    SceneMatcher,
    SyntheticCamera,
    ground_to_image,
    pipeline_scene,
    plane_homography,
)

ALTITUDE = 100.0


def _line_cameras(n: int = 4, step_m: float = 13.4) -> dict[int, SyntheticCamera]:
    course = np.radians(83.0)
    return {k: SyntheticCamera(k * step_m * np.sin(course), k * step_m * np.cos(course), ALTITUDE, 70.8) for k in range(n)}


def test_pipeline_camera_keeps_the_dji_field_of_view_at_small_size() -> None:
    """Same diagonal field of view (82.9 deg) as the full-size camera, so the ground
    footprint -- and therefore which pairs overlap -- is unchanged; only pixels shrink."""
    full_diag = np.hypot(IMAGE_SHAPE[1], IMAGE_SHAPE[0])
    small_diag = np.hypot(PIPELINE_IMAGE_SHAPE[1], PIPELINE_IMAGE_SHAPE[0])
    assert PIPELINE_FOCAL_PX / small_diag == pytest.approx(FOCAL_PX / full_diag, rel=1e-12)


def test_default_camera_functions_are_unchanged() -> None:
    """The new focal_px/image_shape keywords default to the full-size camera, so every
    existing Stage A-D test keeps its exact geometry."""
    a, b = _line_cameras(2).values()
    assert np.array_equal(ground_to_image(a), ground_to_image(a, focal_px=FOCAL_PX, image_shape=IMAGE_SHAPE))
    assert np.array_equal(plane_homography(a, b), plane_homography(a, b, focal_px=FOCAL_PX, image_shape=IMAGE_SHAPE))


def test_scene_pairs_are_exactly_the_pairs_within_the_distance_limit() -> None:
    cameras = _line_cameras(5)  # 13.4 m steps: (i, i+1), (i, i+2) are < 40 m, (i, i+3) = 40.2 m is not
    scene = pipeline_scene(cameras, max_pair_distance_m=40.0)
    assert scene.pairs == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]


def test_scene_correspondences_satisfy_the_plane_homography_exactly() -> None:
    cameras = _line_cameras(3)
    scene = pipeline_scene(cameras)
    for (i, j), (src, dst) in scene.correspondences.items():
        H = plane_homography(cameras[i], cameras[j], focal_px=PIPELINE_FOCAL_PX, image_shape=PIPELINE_IMAGE_SHAPE)
        mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        assert len(src) >= 30, (i, j)
        assert np.max(np.abs(mapped - dst)) < 1e-9, (i, j)
        rows, cols = PIPELINE_IMAGE_SHAPE
        assert np.all((src >= 0) & (src < [cols, rows])) and np.all((dst >= 0) & (dst < [cols, rows]))


def test_scene_image_content_agrees_with_the_geometry() -> None:
    """Warping image i into image j's pixels with the true homography must reproduce
    image j where they overlap (smooth ground texture, bilinear sampling)."""
    cameras = _line_cameras(2)
    scene = pipeline_scene(cameras)
    H = plane_homography(cameras[0], cameras[1], focal_px=PIPELINE_FOCAL_PX, image_shape=PIPELINE_IMAGE_SHAPE)
    rows, cols = PIPELINE_IMAGE_SHAPE
    warped = cv2.warpPerspective(scene.images[0], H, (cols, rows)).astype(float)
    coverage = cv2.warpPerspective(np.full((rows, cols), 255, np.uint8), H, (cols, rows))
    inner = cv2.erode(coverage, np.ones((5, 5), np.uint8)) > 0
    assert inner.sum() > 0.5 * rows * cols
    assert np.mean(np.abs(warped[inner] - scene.images[1][inner].astype(float))) < 2.0
    assert scene.images[0].shape == (rows, cols, 3) and scene.images[0].dtype == np.uint8
    assert np.std(scene.images[0]) > 20  # real texture, not a flat image


def test_scene_latlons_encode_the_camera_positions() -> None:
    cameras = _line_cameras(3)
    scene = pipeline_scene(cameras)
    for k, c in cameras.items():
        east, north = geodetic_to_local_xy(*scene.latlons[k], *scene.latlons[0])
        assert east == pytest.approx(c.east_m - cameras[0].east_m, abs=1e-3)
        assert north == pytest.approx(c.north_m - cameras[0].north_m, abs=1e-3)
    assert scene.gimbal_yaw_deg == {k: c.yaw_deg for k, c in cameras.items()}


def test_scene_matcher_identifies_images_by_identity_and_feeds_match_pair() -> None:
    cameras = _line_cameras(3)
    scene = pipeline_scene(cameras)
    matcher = SceneMatcher(scene)

    result = match_pair(matcher, scene.images[0], scene.images[1], 0, 1)

    src, _ = scene.correspondences[(0, 1)]
    assert result.inlier_count == len(src)  # exact correspondences: RANSAC keeps all
    H = plane_homography(cameras[0], cameras[1], focal_px=PIPELINE_FOCAL_PX, image_shape=PIPELINE_IMAGE_SHAPE)
    assert np.max(np.abs(result.homography / result.homography[2, 2] - H)) < 1e-6
    with pytest.raises(KeyError):
        matcher.match(scene.images[0].copy(), scene.images[1])  # a copy is not a scene image


def test_scene_matcher_failing_pairs_make_match_pair_raise() -> None:
    """A pair listed in fail_pairs gets only 2 correspondences, below the homography
    minimum of 4: cv2.findHomography raises inside match_pair, the real mechanism behind
    run_pipeline's per-pair error isolation."""
    scene = pipeline_scene(_line_cameras(3))
    matcher = SceneMatcher(scene, fail_pairs={(0, 1)})

    with pytest.raises(cv2.error):
        match_pair(matcher, scene.images[0], scene.images[1], 0, 1)
    assert match_pair(matcher, scene.images[1], scene.images[2], 1, 2).inlier_count >= 30


def test_scene_matcher_rejects_a_pair_without_overlap_data() -> None:
    scene = pipeline_scene(_line_cameras(5))
    matcher = SceneMatcher(scene)
    with pytest.raises(KeyError):
        matcher.match(scene.images[0], scene.images[4])  # 53.6 m apart: not a scene pair
