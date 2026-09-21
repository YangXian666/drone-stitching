"""Unit tests for sea_mosaic.blend.blend_images.

Blending strategy under test: feathering via per-image Euclidean distance-transform
weights (cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)), normalized
per-pixel across all images covering that pixel so they sum to 1. The returned
per-image "seam mask" (second element of blend_images's return tuple) is this
normalized weight -- a continuous float32 array in [0, 1], not a binary partition.
Outside any image's footprint, that image's weight is 0; where no image covers a
pixel at all, every image's weight is 0 and the mosaic pixel is black/0.

All test data is hand-built (no np.random), following this repo's existing convention.
WarpedImages/WarpedMasks entries are constructed directly (not via warp_images, which
is separately tested in test_warp.py) -- each is already canvas-sized with zero
padding outside its own footprint, matching warp_images's actual output shape.

Every canvas here has real black margin on all four sides beyond any image's
footprint, so each image's own distance-transform field is governed purely by its own
mask boundary, not by touching the array's edge (which would otherwise contaminate the
hand-derived numbers below with edge-of-array corner effects).
"""

from __future__ import annotations

import numpy as np
import pytest

from sea_mosaic.blend import blend_images
from sea_mosaic.types import WarpedImages, WarpedMasks


def _distinctive_image(height: int, width: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = [y * 7 % 256, x * 11 % 256, 90]
    return image


def _canvas_image(canvas_size: tuple[int, int], row0: int, col0: int, content: np.ndarray) -> np.ndarray:
    """Paste `content` (height,width,3) onto an otherwise-black canvas-sized array at
    (row0, col0) -- mirrors warp_images's convention of each entry being already
    canvas-sized with zero padding outside its own footprint."""
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    h, w = content.shape[:2]
    canvas[row0 : row0 + h, col0 : col0 + w] = content
    return canvas


def _canvas_mask(canvas_size: tuple[int, int], row0: int, col0: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros(canvas_size, dtype=np.uint8)
    mask[row0 : row0 + height, col0 : col0 + width] = 255
    return mask


# --- 1: single image, no overlap -------------------------------------------------------


def test_blend_images_single_image_matches_original_content():
    canvas_size = (20, 30)
    content = _distinctive_image(12, 20)
    image = _canvas_image(canvas_size, row0=3, col0=5, content=content)
    mask = _canvas_mask(canvas_size, row0=3, col0=5, height=12, width=20)

    warped_images = WarpedImages(images={0: image}, canvas_size=canvas_size)
    warped_masks = WarpedMasks(masks={0: mask}, canvas_size=canvas_size)

    mosaic, seam_masks = blend_images(warped_images, warped_masks)

    assert np.array_equal(mosaic, image)


def test_blend_images_single_image_seam_mask_is_one_inside_footprint_zero_outside():
    canvas_size = (20, 30)
    mask = _canvas_mask(canvas_size, row0=3, col0=5, height=12, width=20)
    image = _canvas_image(canvas_size, row0=3, col0=5, content=_distinctive_image(12, 20))

    warped_images = WarpedImages(images={0: image}, canvas_size=canvas_size)
    warped_masks = WarpedMasks(masks={0: mask}, canvas_size=canvas_size)

    _mosaic, seam_masks = blend_images(warped_images, warped_masks)

    assert np.all(seam_masks[0][3:15, 5:25] == pytest.approx(1.0))
    assert np.all(seam_masks[0][0:3, :] == pytest.approx(0.0))
    assert np.all(seam_masks[0][:, 0:5] == pytest.approx(0.0))


# --- 2: two images, completely non-overlapping ------------------------------------------


def _two_disjoint_images_scenario() -> tuple[WarpedImages, WarpedMasks, dict[str, int]]:
    """Two images placed with a real gap between them (columns 12-17 belong to
    neither), so there is a genuine uncovered band, not just adjacency."""
    canvas_size = (20, 40)
    color_a = np.array([50, 90, 130], dtype=np.uint8)
    color_b = np.array([210, 160, 40], dtype=np.uint8)
    image_a = _canvas_image(canvas_size, row0=2, col0=0, content=np.tile(color_a, (10, 12, 1)))
    image_b = _canvas_image(canvas_size, row0=2, col0=18, content=np.tile(color_b, (10, 12, 1)))
    mask_a = _canvas_mask(canvas_size, row0=2, col0=0, height=10, width=12)
    mask_b = _canvas_mask(canvas_size, row0=2, col0=18, height=10, width=12)

    warped_images = WarpedImages(images={0: image_a, 1: image_b}, canvas_size=canvas_size)
    warped_masks = WarpedMasks(masks={0: mask_a, 1: mask_b}, canvas_size=canvas_size)
    return warped_images, warped_masks, {"canvas_size": canvas_size}


def test_blend_images_two_disjoint_images_each_region_matches_its_own_source():
    warped_images, warped_masks, _ = _two_disjoint_images_scenario()

    mosaic, _seam_masks = blend_images(warped_images, warped_masks)

    assert np.array_equal(mosaic[2:12, 0:12], warped_images.images[0][2:12, 0:12])
    assert np.array_equal(mosaic[2:12, 18:30], warped_images.images[1][2:12, 18:30])


def test_blend_images_two_disjoint_images_gap_and_margins_stay_black():
    warped_images, warped_masks, _ = _two_disjoint_images_scenario()

    mosaic, seam_masks = blend_images(warped_images, warped_masks)

    gap = mosaic[2:12, 12:18]
    assert np.array_equal(gap, np.zeros_like(gap))
    assert np.all(seam_masks[0][2:12, 12:18] == pytest.approx(0.0))
    assert np.all(seam_masks[1][2:12, 12:18] == pytest.approx(0.0))
    assert np.all(mosaic[0:2, :] == 0)  # top margin, uncovered by either image


# --- 3: same-size, mirror-symmetric overlap -> exact 50/50 at the center --------------


def test_blend_images_mirror_symmetric_overlap_is_exact_half_and_half_at_center():
    """A and B are the SAME size, placed mirror-symmetrically about the overlap band's
    center column (odd overlap width, so a single center pixel exists). Reflecting the
    whole configuration about that column swaps A and B, which forces their distance-
    transform values to be equal there regardless of the exact distance metric -- a
    pure symmetry argument, not dependent on distanceTransform's internal formula."""
    canvas_size = (25, 45)
    height, width = 15, 20
    overlap_width = 5  # odd -> single center column
    color_a = np.array([30, 60, 90], dtype=np.uint8)
    color_b = np.array([180, 120, 60], dtype=np.uint8)

    row0 = 5
    col_a0 = 5
    col_b0 = col_a0 + width - overlap_width  # = 20
    image_a = _canvas_image(canvas_size, row0, col_a0, np.tile(color_a, (height, width, 1)))
    image_b = _canvas_image(canvas_size, row0, col_b0, np.tile(color_b, (height, width, 1)))
    mask_a = _canvas_mask(canvas_size, row0, col_a0, height, width)
    mask_b = _canvas_mask(canvas_size, row0, col_b0, height, width)

    warped_images = WarpedImages(images={0: image_a, 1: image_b}, canvas_size=canvas_size)
    warped_masks = WarpedMasks(masks={0: mask_a, 1: mask_b}, canvas_size=canvas_size)

    mosaic, seam_masks = blend_images(warped_images, warped_masks)

    center_row = row0 + height // 2  # row=12, well clear of top/bottom edges
    center_col = col_b0 + overlap_width // 2  # = 22, exact center of overlap columns [20,25)

    assert seam_masks[0][center_row, center_col] == pytest.approx(0.5, abs=1e-6)
    assert seam_masks[1][center_row, center_col] == pytest.approx(0.5, abs=1e-6)
    expected_pixel = ((color_a.astype(np.float64) + color_b.astype(np.float64)) / 2).round()
    assert np.array_equal(mosaic[center_row, center_col].astype(np.float64), expected_pixel)


# --- 4-7: asymmetric-size overlap (shared geometry) -------------------------------------


def _asymmetric_overlap_scenario() -> tuple[WarpedImages, WarpedMasks]:
    """A (height=200, width=50) and B (height=200, width=35), same rows, overlapping in
    columns [40, 60). Real margins on all sides (canvas 250x85) so neither image's mask
    touches the array boundary. Row 125 is used for all scanline checks below: local
    row 100 within both images, distance-to-own-edge = min(101, 100) = 100, far larger
    than any horizontal distance examined (<=25), so the nearest-zero-pixel for every
    column checked is purely horizontal -- no corner effects.
    """
    canvas_size = (250, 85)
    color_a = np.array([40, 80, 120], dtype=np.uint8)
    color_b = np.array([200, 140, 60], dtype=np.uint8)
    image_a = _canvas_image(canvas_size, row0=25, col0=10, content=np.tile(color_a, (200, 50, 1)))
    image_b = _canvas_image(canvas_size, row0=25, col0=40, content=np.tile(color_b, (200, 35, 1)))
    mask_a = _canvas_mask(canvas_size, row0=25, col0=10, height=200, width=50)
    mask_b = _canvas_mask(canvas_size, row0=25, col0=40, height=200, width=35)

    warped_images = WarpedImages(images={0: image_a, 1: image_b}, canvas_size=canvas_size)
    warped_masks = WarpedMasks(masks={0: mask_a, 1: mask_b}, canvas_size=canvas_size)
    return warped_images, warped_masks


_SCANLINE_ROW = 125


def test_blend_images_asymmetric_widths_directional_weight_favors_nearer_image():
    """Near A's side of the overlap (col=40, just entered), A should still dominate;
    near A's far edge (col=59, about to leave A's footprint), B should dominate --
    the asymmetry comes from B being narrower (steeper distance-transform gradient),
    not from any bug: this is a directional/qualitative check, not an exact value."""
    warped_images, warped_masks = _asymmetric_overlap_scenario()

    _mosaic, seam_masks = blend_images(warped_images, warped_masks)

    assert seam_masks[0][_SCANLINE_ROW, 40] > seam_masks[1][_SCANLINE_ROW, 40]
    assert seam_masks[1][_SCANLINE_ROW, 59] > seam_masks[0][_SCANLINE_ROW, 59]


def test_blend_images_seam_mask_weights_sum_to_one_wherever_any_image_covers_a_pixel():
    warped_images, warped_masks = _asymmetric_overlap_scenario()

    _mosaic, seam_masks = blend_images(warped_images, warped_masks)
    row = _SCANLINE_ROW

    # pure background: col=5 (before A's col0=10)
    assert seam_masks[0][row, 5] == pytest.approx(0.0)
    assert seam_masks[1][row, 5] == pytest.approx(0.0)
    # only A: col=20 (before overlap starts at col=40)
    assert seam_masks[0][row, 20] == pytest.approx(1.0)
    assert seam_masks[1][row, 20] == pytest.approx(0.0)
    # overlap: col=45
    assert seam_masks[0][row, 45] + seam_masks[1][row, 45] == pytest.approx(1.0, abs=1e-6)
    # only B: col=68 (after overlap ends at col=60, before B's own edge at col=75)
    assert seam_masks[0][row, 68] == pytest.approx(0.0)
    assert seam_masks[1][row, 68] == pytest.approx(1.0)
    # pure background again: col=80 (after B's col0+width=75)
    assert seam_masks[0][row, 80] == pytest.approx(0.0)
    assert seam_masks[1][row, 80] == pytest.approx(0.0)


def test_blend_images_alpha_transition_step_bounded_by_derived_threshold():
    """Derived (not guessed) bound: 1/17 (~0.058824), the larger of the two overlap-
    boundary transitions for this specific geometry (entering at col 39->40: step
    1/21~=0.047619; leaving A's footprint at col 59->60: step 1/17~=0.058824, larger
    because B is narrower than A and so has a steeper distance-transform gradient at
    its own edge). This range (cols 39-60) covers only the A<->B seam transitions, not
    the outer mosaic-to-background edges (e.g. col 74->75, where weight legitimately
    jumps 1.0 -- that is not a seam feathering is meant to smooth)."""
    warped_images, warped_masks = _asymmetric_overlap_scenario()

    _mosaic, seam_masks = blend_images(warped_images, warped_masks)

    row = _SCANLINE_ROW
    alpha_a = seam_masks[0][row, 39:61].astype(np.float64)
    steps = np.abs(np.diff(alpha_a))

    derived_bound = 1.0 / 17.0
    assert np.max(steps) <= derived_bound + 1e-6


def test_blend_images_alpha_monotonically_decreases_moving_away_from_source_image():
    warped_images, warped_masks = _asymmetric_overlap_scenario()

    _mosaic, seam_masks = blend_images(warped_images, warped_masks)

    row = _SCANLINE_ROW
    alpha_a_in_overlap = seam_masks[0][row, 40:60].astype(np.float64)
    assert np.all(np.diff(alpha_a_in_overlap) <= 1e-6)
