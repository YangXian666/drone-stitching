"""Stage 4 (blend): composite warped images into the final mosaic."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from sea_mosaic.types import WarpedImages, WarpedMasks


def _distance_weight(mask: np.ndarray) -> np.ndarray:
    """Distance-transform feathering weight for one warped mask (float32, 0 outside it).

    Runs with OpenCV's IPP acceleration disabled, restoring the caller's IPP setting
    afterwards (it is process-global state). OpenCV 5.0.0's IPP distanceTransform(DIST_L2,
    DIST_MASK_PRECISE) path was non-deterministic -- identical inputs gave different
    weights from run to run, flipping mosaic pixels that sit on a .5 rounding boundary --
    and not exact (up to 1.9e-6 off the exact Euclidean distance). With IPP off, 40
    pipeline runs in 5 fresh processes gave one mosaic, identical to the one from
    scipy's exact EDT, at about the same speed (0.40 s vs 0.37 s on a 116 Mpx canvas).
    Limiting OpenCV to one thread did NOT help once OpenCV had already run
    multi-threaded in the process (see CLAUDE.md for the full investigation).

    cv2.distanceTransform has no zero pixel to measure to when a mask has no black border
    at all (its nonzero region exactly fills the array) -- it then returns an
    overflow-style sentinel (~1.8e19), not a real distance. Clip to the mask's own
    diagonal: any genuine distance-to-boundary value is bounded by roughly half the
    shorter side, so the full diagonal is a safe, always-larger cap that never affects a
    real (bordered) mask's values, only this sentinel.
    """
    previous_use_ipp = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(False)
    try:
        raw_weight = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    finally:
        cv2.ipp.setUseIPP(previous_use_ipp)
    return np.minimum(raw_weight, float(np.hypot(*mask.shape)))


def blend_images(
    warped_images: WarpedImages,
    warped_masks: WarpedMasks,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Blend warped images into a single mosaic via distance-transform feathering.

    For each image, cv2.distanceTransform(mask, DIST_L2, DIST_MASK_PRECISE) gives its
    raw weight at every pixel (0 outside its own footprint, growing with distance from
    its own mask boundary inside it). Per pixel, weights are normalized across every
    image covering that pixel so they sum to 1; the mosaic pixel is the resulting
    weighted average. Where no image covers a pixel at all, every weight is 0 and the
    mosaic pixel is black.

    Returns the stitched image and, per image index, its normalized weight map (the
    "seam mask") -- a continuous float32 array in [0, 1], not a binary partition. This
    keeps the door open for a future seam-based (graph-cut/multiband) replacement
    without changing the return type.
    """
    canvas_size = warped_images.canvas_size
    indices = sorted(warped_images.images)

    weights = {}
    for index in indices:
        weights[index] = _distance_weight(warped_masks.masks[index])

    total_weight = np.zeros(canvas_size, dtype=np.float64)
    for index in indices:
        total_weight += weights[index]
    covered = total_weight > 0
    safe_total = np.where(covered, total_weight, 1.0)  # never divides by zero below

    seam_masks: dict[int, np.ndarray] = {}
    mosaic = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float64)
    for index in indices:
        alpha = np.where(covered, weights[index] / safe_total, 0.0).astype(np.float32)
        seam_masks[index] = alpha
        mosaic += alpha[:, :, np.newaxis] * warped_images.images[index].astype(np.float64)

    return mosaic.round().astype(np.uint8), seam_masks


def blend_images_streaming(
    warped_stream: Iterable[tuple[int, np.ndarray, np.ndarray]],
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """Streaming form of blend_images: consumes a (index, warped_image, warped_mask)
    stream (as produced by warp.warp_images_streaming) one item at a time and folds each
    image's weighted contribution directly into two fixed-size running accumulators,
    instead of blend_images's normalize-alpha-per-image-first (which requires every
    image's distance-weight to still be alive simultaneously when the second pass runs).
    Peak memory is O(canvas_size), not O(N * canvas_size) -- see CLAUDE.md's
    streaming-accumulator backlog item.

    Mathematically equivalent to blend_images's per-pixel weighted average
    (sum(w_i/W * x_i) == sum(w_i * x_i)/W, where W = sum(w_i)), but NOT guaranteed
    bit-identical at the float64 level: blend_images divides once per image before
    summing, this divides once at the very end after summing -- float division does not
    distribute over addition bit-for-bit. The contract that actually matters, and the one
    tests/test_blend.py's equivalence tests assert with np.array_equal (not a tolerance),
    is the final uint8 mosaic: .round().astype(uint8) absorbs ULP-level float noise except
    exactly at a rounding-boundary pixel (see
    test_blend_images_streaming_matches_blend_images_at_adversarial_rounding_boundary).

    Returns the mosaic only -- no per-image seam_masks dict. seam_masks's only real
    consumer is metrics.compute_seam_error, which docs/task2.md documents as optional
    ("如果 pipeline 本身有 seam finder，請額外保留"); materializing a full seam_masks[index]
    for all N images here would defeat the O(canvas_size) point of streaming in the first
    place. compute_seam_error's own bounding-box-prefiltered redesign (separate backlog
    item) sources per-image data on demand instead.
    """
    weighted_sum = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float64)
    weight_sum = np.zeros(canvas_size, dtype=np.float64)

    for _index, warped_image, warped_mask in warped_stream:
        weight = _distance_weight(warped_mask)
        weight_sum += weight
        weighted_sum += weight[:, :, np.newaxis] * warped_image.astype(np.float64)

    covered = weight_sum > 0
    safe_total = np.where(covered, weight_sum, 1.0)
    mosaic = np.where(
        covered[:, :, np.newaxis], weighted_sum / safe_total[:, :, np.newaxis], 0.0
    )
    return mosaic.round().astype(np.uint8)
