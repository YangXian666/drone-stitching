"""Stage 4 (blend): composite warped images into the final mosaic."""

from __future__ import annotations

import numpy as np

from sea_mosaic.types import WarpedImages, WarpedMasks


def blend_images(
    warped_images: WarpedImages,
    warped_masks: WarpedMasks,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Blend warped images into a single mosaic.

    Returns the stitched image and, per image index, the seam mask giving that image's
    final pixel contribution after seam selection.
    """
    ...
