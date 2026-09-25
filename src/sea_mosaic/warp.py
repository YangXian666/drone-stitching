"""Stage 3 (warp): warp each image into the shared mosaic canvas coordinate system."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from sea_mosaic.geo.camera import CameraIntrinsics, CameraPose
from sea_mosaic.types import GlobalTransforms, WarpedImages, WarpedMasks


def _mosaic_bounds(
    image_shapes: dict[int, tuple[int, int]],
    global_transforms: GlobalTransforms,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_xy, max_xy): the bounding box, in mosaic coordinates, of every
    image's four corners after applying its own global transform.

    Shared by compute_canvas_size (which only needs the resulting size) and
    warp_images (which also needs min_xy as the canvas origin offset), so the corner
    projection logic isn't duplicated between them.
    """
    all_corners = []
    for index, (height, width) in image_shapes.items():
        transform = global_transforms.transforms[index]
        corners = np.array(
            [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
            dtype=np.float64,
        ).T  # shape (3, 4)
        projected = transform @ corners
        all_corners.append(projected[:2, :] / projected[2, :])

    stacked = np.concatenate(all_corners, axis=1)
    return stacked.min(axis=1), stacked.max(axis=1)


def compute_canvas_size(
    image_shapes: dict[int, tuple[int, int]],
    global_transforms: GlobalTransforms,
) -> tuple[int, int]:
    """Compute the shared mosaic canvas size (height, width) needed to contain all
    warped images."""
    min_xy, max_xy = _mosaic_bounds(image_shapes, global_transforms)
    width = int(np.ceil(max_xy[0] - min_xy[0]))
    height = int(np.ceil(max_xy[1] - min_xy[1]))
    return (height, width)


def warp_images(
    images: dict[int, np.ndarray],
    global_transforms: GlobalTransforms,
    camera_intrinsics: dict[int, CameraIntrinsics] | None = None,
    camera_poses: dict[int, CameraPose] | None = None,
) -> tuple[WarpedImages, WarpedMasks]:
    """Warp each image into the shared mosaic canvas, producing aligned images and
    valid-pixel masks.

    Applies each image's full global_transforms.transforms[index] (never just its
    translation component) via cv2.warpPerspective -- see CLAUDE.md's 已知的暫緩事項
    for why the rotation component of that transform is not currently trustworthy on
    data/smoke/, which is a statement about compose_global_transforms's real output on
    this dataset, not a reason for warp_images itself to special-case away rotation.

    camera_intrinsics / camera_poses are reserved parameters for a future direct
    georeferencing integration (see sea_mosaic.geo.direct), mapping mosaic-canvas
    coordinates to world coordinates. This skeleton does not use them.
    """
    image_shapes = {index: image.shape[:2] for index, image in images.items()}
    min_xy, max_xy = _mosaic_bounds(image_shapes, global_transforms)
    canvas_size = (
        int(np.ceil(max_xy[1] - min_xy[1])),
        int(np.ceil(max_xy[0] - min_xy[0])),
    )
    dsize = (canvas_size[1], canvas_size[0])  # cv2 wants (width, height)
    origin_offset = np.array(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]]
    )

    warped_images: dict[int, np.ndarray] = {}
    warped_masks: dict[int, np.ndarray] = {}
    for index, image in images.items():
        transform = origin_offset @ global_transforms.transforms[index]
        warped_images[index] = cv2.warpPerspective(image, transform, dsize)
        full_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_masks[index] = cv2.warpPerspective(full_mask, transform, dsize)

    return (
        WarpedImages(images=warped_images, canvas_size=canvas_size),
        WarpedMasks(masks=warped_masks, canvas_size=canvas_size),
    )


def warp_images_streaming(
    images: dict[int, np.ndarray],
    global_transforms: GlobalTransforms,
    canvas_size: tuple[int, int],
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Per-image generator form of warp_images: yields (index, warped_image, warped_mask)
    one at a time instead of eagerly materializing all N canvas-sized arrays into a dict
    before returning. Exists so a caller (blend_images_streaming, driven by pipeline.py)
    can fold each image's contribution into a running accumulator and let it be garbage
    collected before the next image is warped, keeping peak memory at O(canvas_size)
    instead of warp_images's O(N * canvas_size) -- see CLAUDE.md's streaming-accumulator
    backlog item for the full memory diagnosis this exists to address.

    canvas_size is the caller's responsibility (computed once via compute_canvas_size),
    not recomputed here, matching warp_images's own canvas_size -- keeping
    compute_canvas_size's single responsibility rather than adding a canvas-size-carrying
    iterator class. min_xy (needed for the per-image origin offset, not derivable from
    canvas_size's height/width alone) is still recomputed here via _mosaic_bounds -- a
    cheap O(N) corner-projection, not a canvas-sized allocation, so this redundancy with
    the caller's own compute_canvas_size call does not reintroduce the O(N * canvas_size)
    cost this function exists to avoid.

    Nothing in this function's body runs until the first item is requested (ordinary
    Python generator-function semantics: calling this only constructs a generator object).
    Each loop iteration reassigns warped_image/warped_mask to fresh arrays and yields
    immediately -- no accumulation of previously-yielded arrays anywhere in this function,
    so once a caller drops its own reference to a yielded pair and the generator has moved
    on to the next iteration, nothing here keeps it alive.
    """
    image_shapes = {index: image.shape[:2] for index, image in images.items()}
    min_xy, _max_xy = _mosaic_bounds(image_shapes, global_transforms)
    dsize = (canvas_size[1], canvas_size[0])  # cv2 wants (width, height)
    origin_offset = np.array(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]]
    )

    for index, image in images.items():
        transform = origin_offset @ global_transforms.transforms[index]
        warped_image = cv2.warpPerspective(image, transform, dsize)
        full_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(full_mask, transform, dsize)
        yield index, warped_image, warped_mask
