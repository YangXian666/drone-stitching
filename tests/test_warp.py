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

import gc
import weakref
from unittest import mock

import cv2
import numpy as np
import pytest

from sea_mosaic.types import GlobalTransforms
from sea_mosaic.warp import compute_canvas_size, warp_images, warp_images_streaming


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


# --- warp_images_streaming: bit-exact equivalence to warp_images (gating tests) ---------
#
# warp_images_streaming is a pure memory-management change over warp_images (per-image
# generator instead of an eagerly materialized dict), not a logic change -- it calls the
# identical cv2.warpPerspective per image with no arithmetic reordering, so unlike
# blend_images_streaming's weighted-average math (see test_blend.py), there is no
# floating-point-reassociation caveat here: equivalence must be true bit-exact
# (np.array_equal), with no tolerance.
#
# API convention (per design discussion): the caller computes canvas_size once via the
# existing compute_canvas_size and passes it to warp_images_streaming, rather than
# warp_images_streaming recomputing it internally -- keeps compute_canvas_size's single
# responsibility, no new class/attribute-carrying-generator needed.


def test_warp_images_streaming_matches_warp_images_bitexact_single_image_identity():
    image = _distinctive_image(10, 15)
    transforms = _global_transforms({0: np.eye(3)})
    canvas_size = compute_canvas_size({0: image.shape[:2]}, transforms)

    warped, masks = warp_images({0: image}, transforms)
    streamed = list(warp_images_streaming({0: image}, transforms, canvas_size))

    assert len(streamed) == 1
    index, warped_image, warped_mask = streamed[0]
    assert index == 0
    assert np.array_equal(warped_image, warped.images[0])
    assert np.array_equal(warped_mask, masks.masks[0])


def test_warp_images_streaming_matches_warp_images_bitexact_two_images_translation():
    images, transforms = _two_image_translation_scenario()
    image_shapes = {index: image.shape[:2] for index, image in images.items()}
    canvas_size = compute_canvas_size(image_shapes, transforms)

    warped, masks = warp_images(images, transforms)
    streamed = {
        index: (warped_image, warped_mask)
        for index, warped_image, warped_mask in warp_images_streaming(images, transforms, canvas_size)
    }

    assert set(streamed) == set(images)
    for index in images:
        streamed_image, streamed_mask = streamed[index]
        assert np.array_equal(streamed_image, warped.images[index])
        assert np.array_equal(streamed_mask, masks.masks[index])


def test_warp_images_streaming_matches_warp_images_bitexact_rotation():
    """Rotation exercises actual corner/pixel remapping (not just a translated offset),
    matching why test_warp_images_rotation_places_marker_pixel_at_hand_computed_coordinate
    exists for warp_images itself."""
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5, 15] = [255, 0, 0]
    transforms = _global_transforms({0: _rotation_90()})
    canvas_size = compute_canvas_size({0: image.shape[:2]}, transforms)

    warped, masks = warp_images({0: image}, transforms)
    streamed = list(warp_images_streaming({0: image}, transforms, canvas_size))

    assert len(streamed) == 1
    index, warped_image, warped_mask = streamed[0]
    assert index == 0
    assert np.array_equal(warped_image, warped.images[0])
    assert np.array_equal(warped_mask, masks.masks[0])


def test_warp_images_streaming_canvas_size_matches_warp_images():
    images, transforms = _two_image_translation_scenario()
    image_shapes = {index: image.shape[:2] for index, image in images.items()}
    canvas_size = compute_canvas_size(image_shapes, transforms)

    warped, _masks = warp_images(images, transforms)
    for _index, warped_image, warped_mask in warp_images_streaming(images, transforms, canvas_size):
        assert warped_image.shape[:2] == warped.canvas_size
        assert warped_mask.shape[:2] == warped.canvas_size


# --- warp_images_streaming: actually lazy, not just correct -----------------------------
#
# The equivalence tests above only prove "same math." They cannot catch the failure mode
# this redesign exists to prevent: an implementation that is technically a generator (or
# even yields the right values in the right order) but secretly does all N images' work
# up front, or secretly accumulates already-yielded arrays in some internal cache --
# either of which would silently defeat the whole point (O(canvas_size) memory,
# independent of N) while still passing every equivalence test above.


def _five_image_scenario() -> tuple[dict[int, np.ndarray], GlobalTransforms, tuple[int, int]]:
    images = {i: _distinctive_image(6, 6) for i in range(5)}
    transforms = _global_transforms({i: _translation(tx=i * 3, ty=0) for i in range(5)})
    canvas_size = compute_canvas_size({i: img.shape[:2] for i, img in images.items()}, transforms)
    return images, transforms, canvas_size


def test_warp_images_streaming_only_warps_images_actually_consumed_so_far():
    """Patches the actual expensive canvas-sized operation (cv2.warpPerspective, called
    twice per image: once for the image, once for its mask) rather than inspecting
    internal image-dict access patterns, so this test's validity does not depend on
    whether the implementation iterates via .items(), [key] lookups, or anything else --
    it directly measures the claim: has the expensive per-image work happened only for
    images consumed so far, not all N up front.

    This also catches the specific 'list built eagerly then yielded from' degenerate
    pattern: building `results = [...]` before any `yield` would make the FIRST next()
    call trigger all N images' worth of work at once (call_count jumping straight to
    2*N), not the 2-per-next() progression asserted here."""
    images, transforms, canvas_size = _five_image_scenario()

    call_count = 0
    real_warp_perspective = cv2.warpPerspective

    def _counting_warp_perspective(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_warp_perspective(*args, **kwargs)

    with mock.patch("sea_mosaic.warp.cv2.warpPerspective", side_effect=_counting_warp_perspective):
        stream = warp_images_streaming(images, transforms, canvas_size)
        assert call_count == 0  # constructing the generator must not warp anything yet

        next(stream)
        assert call_count == 2  # exactly one image's (image, mask) pair, not all 5

        next(stream)
        assert call_count == 4

        next(stream)
        assert call_count == 6


def test_warp_images_streaming_does_not_retain_already_yielded_arrays():
    """Guards against a subtler degenerate pattern than the call-counting test above: an
    implementation that IS correctly lazy per next() call (would pass that test) but still
    accidentally accumulates a growing internal history/cache of already-yielded arrays
    (e.g. a debugging leftover) -- which would silently defeat the O(canvas_size) memory
    goal even though external call timing looks correctly lazy.

    Verified via weakref (confirmed to work on plain np.ndarray -- see this session's
    diagnostic): once this test's own (only external) reference to an already-yielded
    array is dropped AND the generator has moved past that item (so its own loop-local
    variable, which legitimately still holds the just-yielded value while paused at a
    yield, has been reassigned to the next item), nothing else in the process should be
    keeping the old array alive."""
    images, transforms, canvas_size = _five_image_scenario()
    stream = warp_images_streaming(images, transforms, canvas_size)

    _index0, warped_image0, warped_mask0 = next(stream)
    weak_image0 = weakref.ref(warped_image0)
    weak_mask0 = weakref.ref(warped_mask0)

    del warped_image0, warped_mask0
    next(stream)  # advance past item 0 -- any well-behaved loop reassigns its locals here
    gc.collect()

    assert weak_image0() is None
    assert weak_mask0() is None
