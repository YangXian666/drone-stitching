"""Stage 4 (blend): composite warped images into the final mosaic."""

from __future__ import annotations

import cv2
import numpy as np

from sea_mosaic.types import WarpedImages, WarpedMasks


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

    weights = {
        index: cv2.distanceTransform(warped_masks.masks[index], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for index in indices
    }

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
