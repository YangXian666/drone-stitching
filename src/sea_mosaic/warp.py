"""Stage 3 (warp): warp each image into the shared mosaic canvas coordinate system."""

from __future__ import annotations

import numpy as np

from sea_mosaic.geo.camera import CameraIntrinsics, CameraPose
from sea_mosaic.types import GlobalTransforms, WarpedImages, WarpedMasks


def compute_canvas_size(
    image_shapes: dict[int, tuple[int, int]],
    global_transforms: GlobalTransforms,
) -> tuple[int, int]:
    """Compute the shared mosaic canvas size (height, width) needed to contain all
    warped images."""
    ...


def warp_images(
    images: dict[int, np.ndarray],
    global_transforms: GlobalTransforms,
    camera_intrinsics: dict[int, CameraIntrinsics] | None = None,
    camera_poses: dict[int, CameraPose] | None = None,
) -> tuple[WarpedImages, WarpedMasks]:
    """Warp each image into the shared mosaic canvas, producing aligned images and
    valid-pixel masks.

    camera_intrinsics / camera_poses are reserved parameters for a future direct
    georeferencing integration (see sea_mosaic.geo.direct), mapping mosaic-canvas
    coordinates to world coordinates. This skeleton does not use them.
    """
    ...
