"""Unit tests for sea_mosaic.warp.compute_canvas_size / warp_images.

Deliberately separates two independent correctness claims (same principle as
match_pair's fake-Matcher tests, which verify RANSAC logic without depending on real
SIFT accuracy): whether warp.py correctly applies a GIVEN 3x3 transform, versus whether
compose_global_transforms's rotation output is trustworthy on real data (it currently
is not -- see CLAUDE.md's 已知的暫緩事項). All transforms here are hand-constructed,
not derived from real compose_global_transforms output.

All test data is hand-built (no np.random), per this repo's existing test convention.
image_shapes / canvas_size are (height, width) throughout, matching
sea_mosaic.types.WarpedImages/WarpedMasks's documented convention. A point's own pixel
coordinates are (x, y) with x=column in [0, width], y=row in [0, height].
"""

from __future__ import annotations

import numpy as np
import pytest

from sea_mosaic.types import GlobalTransforms
from sea_mosaic.warp import compute_canvas_size, warp_images


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _rotation_90() -> np.ndarray:
    """Exactly 90deg, no translation: (x, y) -> (-y, x). Deliberately grid-aligned (not
    an arbitrary angle) so warped pixel positions land exactly on integer coordinates,
    with no interpolation blur to account for in exact-value assertions."""
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _global_transforms(transforms: dict[int, np.ndarray], reference_index: int = 0) -> GlobalTransforms:
    return GlobalTransforms(
        transforms=transforms,
        reference_index=reference_index,
        optimization_status="converged",
        residual_error=0.0,
    )


# --- compute_canvas_size ----------------------------------------------------------------


def test_compute_canvas_size_single_image_identity():
    image_shapes = {0: (100, 200)}  # (height, width)
    transforms = _global_transforms({0: np.eye(3)})

    canvas_size = compute_canvas_size(image_shapes, transforms)

    assert canvas_size == (100, 200)


def test_compute_canvas_size_two_images_partial_overlap_translation():
    image_shapes = {0: (100, 150), 1: (80, 120)}  # (height, width)
    transforms = _global_transforms({0: np.eye(3), 1: _translation(tx=100, ty=50)})

    canvas_size = compute_canvas_size(image_shapes, transforms)

    # image0 corners: (0,0)-(150,100); image1 corners (shifted): (100,50)-(220,130)
    # bounding box: x in [0,220], y in [0,130]
    assert canvas_size == (130, 220)


def test_compute_canvas_size_asymmetric_images_and_offsets_take_extremes_from_different_images():
    """Three differently-sized images at asymmetric (including negative) translations,
    chosen so each of the four bounding-box extremes (min_x, max_x, min_y, max_y) comes
    from a DIFFERENT image -- guards against an implementation that only considers one
    image's extent or misses negative-coordinate corners."""
    image_shapes = {0: (50, 60), 1: (40, 90), 2: (70, 30)}
    transforms = _global_transforms(
        {
            0: _translation(tx=0, ty=0),  # corners: x[0,60], y[0,50]
            1: _translation(tx=-30, ty=20),  # corners: x[-30,60], y[20,60]
            2: _translation(tx=40, ty=-15),  # corners: x[40,70], y[-15,55]
        }
    )

    canvas_size = compute_canvas_size(image_shapes, transforms)

    # min_x=-30 (image1), max_x=70 (image2), min_y=-15 (image2), max_y=60 (image1)
    assert canvas_size == (75, 100)


def test_compute_canvas_size_negative_offset_does_not_change_size():
    """A lone image placed entirely in negative mosaic coordinates must still produce a
    canvas exactly its own size (only the offset used by warp_images changes, not the
    size), and must not error or clamp negative coordinates to zero."""
    image_shapes = {0: (20, 30)}
    transforms = _global_transforms({0: _translation(tx=-10, ty=-5)})

    canvas_size = compute_canvas_size(image_shapes, transforms)

    assert canvas_size == (20, 30)


def test_compute_canvas_size_rotation_swaps_width_and_height():
    """A wide, short image rotated 90deg must produce a tall, narrow canvas -- confirms
    corners are actually transformed, not just shifted by width/height."""
    image_shapes = {0: (40, 100)}  # height=40, width=100
    transforms = _global_transforms({0: _rotation_90()})

    canvas_size = compute_canvas_size(image_shapes, transforms)

    # corners (0,0),(100,0),(100,40),(0,40) -> (0,0),(0,100),(-40,100),(-40,0)
    # bounding box: x in [-40,0], y in [0,100] -> height=100, width=40
    assert canvas_size == (100, 40)


# --- warp_images --------------------------------------------------------------------


def _distinctive_image(height: int, width: int) -> np.ndarray:
    """A small image with a unique value per pixel (not a solid color), so an exact
    equality check also catches an accidental transpose/flip bug that a solid-color
    image would hide."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = [y * 10 % 256, x * 10 % 256, 128]
    return image


def test_warp_images_single_image_identity_matches_original_exactly():
    image = _distinctive_image(10, 15)
    transforms = _global_transforms({0: np.eye(3)})

    warped, masks = warp_images({0: image}, transforms)

    assert warped.canvas_size == (10, 15)
    assert masks.canvas_size == (10, 15)
    assert np.array_equal(warped.images[0], image)
    assert np.all(masks.masks[0] == 255)


def test_warp_images_single_image_translation_matches_original_content():
    """A lone image under a nonzero, non-trivial translation must still warp to content
    identical to the original -- compute_canvas_size's origin offset exactly cancels the
    translation for a single image (there is nothing else to offset against), so this is
    a sanity check that warp_images actually applies that offset rather than ignoring
    translation for a single-image graph."""
    image = _distinctive_image(8, 12)
    transforms = _global_transforms({0: _translation(tx=7, ty=3)})

    warped, masks = warp_images({0: image}, transforms)

    assert warped.canvas_size == (8, 12)
    assert np.array_equal(warped.images[0], image)
    assert np.all(masks.masks[0] == 255)


_TWO_IMAGE_COLOR_A = np.array([10, 20, 30], dtype=np.uint8)
_TWO_IMAGE_COLOR_B = np.array([200, 150, 100], dtype=np.uint8)


def _two_image_translation_scenario() -> tuple[dict[int, np.ndarray], GlobalTransforms]:
    """Two solid-colored 6x8 images at asymmetric translations (including a nonzero
    global bounding-box offset, so warp_images must actually apply that origin shift,
    not just the raw per-edge translation).

    image0 corners: (2,1)-(10,7); image1 corners: (12,5)-(20,11)
    bounds: x[2,20], y[1,11] -> canvas (10, 18); origin offset = (-2,-1)
    image0 placed at canvas rows[0:6], cols[0:8]
    image1 placed at canvas rows[4:10], cols[10:18]
    """
    image_a = np.tile(_TWO_IMAGE_COLOR_A, (6, 8, 1))  # height=6, width=8
    image_b = np.tile(_TWO_IMAGE_COLOR_B, (6, 8, 1))
    transforms = _global_transforms(
        {0: _translation(tx=2, ty=1), 1: _translation(tx=12, ty=5)}
    )
    return {0: image_a, 1: image_b}, transforms


def test_warp_images_two_images_corners_land_at_hand_computed_coordinates():
    images, transforms = _two_image_translation_scenario()

    warped, _masks = warp_images(images, transforms)

    assert warped.canvas_size == (10, 18)
    assert np.array_equal(warped.images[0][0:6, 0:8], images[0])
    assert np.array_equal(warped.images[1][4:10, 10:18], images[1])


def test_warp_images_two_images_mask_and_zero_padding_correctness():
    """Outside each image's own warped footprint, both the image array (zero/black)
    and the mask (0/invalid) must correctly reflect 'no data here' -- tested with
    translations that leave clear empty regions on both sides."""
    images, transforms = _two_image_translation_scenario()

    warped, masks = warp_images(images, transforms)

    image0_full = warped.images[0]
    assert np.array_equal(image0_full[6:10, 0:18], np.zeros((4, 18, 3), dtype=np.uint8))
    assert np.array_equal(image0_full[0:10, 8:18], np.zeros((10, 10, 3), dtype=np.uint8))
    image1_full = warped.images[1]
    assert np.array_equal(image1_full[0:4, 0:18], np.zeros((4, 18, 3), dtype=np.uint8))
    assert np.array_equal(image1_full[0:10, 0:10], np.zeros((10, 10, 3), dtype=np.uint8))

    assert np.all(masks.masks[0][0:6, 0:8] == 255)
    assert np.all(masks.masks[0][6:10, :] == 0)
    assert np.all(masks.masks[0][:, 8:18] == 0)

    assert np.all(masks.masks[1][4:10, 10:18] == 255)
    assert np.all(masks.masks[1][0:4, :] == 0)
    assert np.all(masks.masks[1][:, 0:10] == 0)


def test_warp_images_canvas_size_matches_compute_canvas_size():
    images, transforms = _two_image_translation_scenario()
    image_shapes = {index: image.shape[:2] for index, image in images.items()}

    expected_canvas_size = compute_canvas_size(image_shapes, transforms)
    warped, masks = warp_images(images, transforms)

    assert warped.canvas_size == expected_canvas_size
    assert masks.canvas_size == expected_canvas_size


def test_warp_images_all_entries_share_the_same_canvas_shape():
    images, transforms = _two_image_translation_scenario()

    warped, masks = warp_images(images, transforms)

    for index in images:
        assert warped.images[index].shape[:2] == warped.canvas_size
        assert masks.masks[index].shape[:2] == masks.canvas_size


def test_warp_images_rotation_places_marker_pixel_at_hand_computed_coordinate():
    """A single marker pixel in an otherwise-black image, warped through an exact
    90deg rotation (grid-aligned, no interpolation blur -- see _rotation_90's
    docstring), must land exactly at the hand-computed rotated+offset coordinate."""
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    marker_color = np.array([255, 0, 0], dtype=np.uint8)
    image[5, 15] = marker_color  # row=5 (y=5), col=15 (x=15)

    transforms = _global_transforms({0: _rotation_90()})
    # marker (x=15,y=5) -> (-y,x) = (-5,15)
    # image corners (0,0),(20,0),(20,20),(0,20) -> (0,0),(0,20),(-20,20),(-20,0)
    # bounds: x[-20,0], y[0,20] -> canvas (20,20); origin offset = (20,0)
    # marker mosaic coord (-5,15) + offset(20,0) = (15,15) -> canvas (row=15,col=15)

    warped, masks = warp_images({0: image}, transforms)

    assert warped.canvas_size == (20, 20)
    assert np.array_equal(warped.images[0][15, 15], marker_color)
    assert masks.masks[0][15, 15] == 255
    # canvas (10,10) maps back to source (x=10,y=10), an interior background pixel
    # (only (15,5) is non-zero in the source image) -- unambiguous, not a boundary corner
    assert np.array_equal(warped.images[0][10, 10], np.zeros(3, dtype=np.uint8))
