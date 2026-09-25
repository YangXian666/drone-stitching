"""Pipeline entrypoint: estimate -> compose -> warp -> blend -> evaluate metrics."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sea_mosaic.blend import blend_images_streaming
from sea_mosaic.compose import compose_global_transforms
from sea_mosaic.config import PipelineConfig
from sea_mosaic.estimate import default_inlier_count_reference, match_pair, sequential_pairs
from sea_mosaic.geo.camera import CameraIntrinsics, CameraPose
from sea_mosaic.geo.projection import estimate_pixels_per_meter
from sea_mosaic.matcher import Matcher
from sea_mosaic.metrics import compute_seam_error_streaming, evaluate_stitching_metrics
from sea_mosaic.types import GlobalTransforms, PairResult, ProcessStats
from sea_mosaic.warp import compute_canvas_size, warp_images_streaming

# A homography has 8 degrees of freedom; 4 point correspondences (8 equations) is the
# mathematical minimum needed to determine one uniquely. Below this threshold, the
# system is underdetermined -- there's no meaningful sense in which a result could even
# be evaluated for reliability, it's arbitrary by construction.
#
# This threshold does NOT claim that >=4 inliers means the homography IS reliable.
# CLAUDE.md's Check A diagnostic already proved the opposite: even a well-determined,
# high-inlier edge (inlier_count=717, far above this floor) can still get amplified into
# a collapsed result once folded into the joint pose-graph optimization. "Solvable" and
# "trustworthy" are different claims; this constant only rules out the specific failure
# mode of "there wasn't even enough data to ask the question at all."
_MIN_INLIERS_FOR_DETERMINED_HOMOGRAPHY = 4


def _empty_process_stats(input_image_count: int, elapsed: float) -> ProcessStats:
    return ProcessStats(
        pipeline_status="failed",
        input_image_count=input_image_count,
        successful_image_count=0,
        failed_image_count=input_image_count,
        failed_image_indices=list(range(input_image_count)),
        total_processing_time_sec=elapsed,
        avg_processing_time_per_image_sec=(
            elapsed / input_image_count if input_image_count > 0 else np.nan
        ),
    )


def run_pipeline(
    images: dict[int, np.ndarray],
    matcher: Matcher,
    config: PipelineConfig,
    camera_intrinsics: dict[int, CameraIntrinsics] | None = None,
    camera_poses: dict[int, CameraPose] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run the full estimate -> compose -> warp -> blend pipeline and evaluate metrics.

    images are already-loaded arrays (image_index -> ndarray) -- run_pipeline's
    responsibility boundary is explicitly "receives already-loaded image arrays", not
    file IO. Loading from disk (io_utils.load_image/load_images, still stubs) is a
    separate, independently-scoped concern for the caller, not something run_pipeline
    does on the caller's behalf (see CLAUDE.md).

    Returns (stitched_image, metrics_df), where metrics_df is the one-row DataFrame from
    metrics.evaluate_stitching_metrics.

    camera_intrinsics / camera_poses are reserved parameters for a future direct
    georeferencing integration (see sea_mosaic.geo.camera / sea_mosaic.geo.direct); this
    skeleton does not use them. metrics_df's 'method' column is populated from
    matcher.name.
    """
    start_time = time.perf_counter()
    input_image_count = len(images)

    if input_image_count == 0:
        process_stats = _empty_process_stats(0, time.perf_counter() - start_time)
        metrics_df = evaluate_stitching_metrics(
            pair_results=[],
            global_transforms=GlobalTransforms(
                transforms={}, reference_index=config.reference_index,
                optimization_status="failed", residual_error=np.nan,
            ),
            image_shapes={},
            process_stats=process_stats,
            method=matcher.name,
        )
        return np.zeros((0, 0, 3), dtype=np.uint8), metrics_df

    all_indices = sorted(images)
    pairs = config.pairs if config.pairs is not None else sequential_pairs(images)

    # --- estimate: run_pipeline is the system boundary, so a single bad pair (e.g.
    # too few raw matches -> cv2.error from cv2.findHomography) is caught and skipped
    # here, not allowed to crash the whole run. match_pair/estimate_all_pairs
    # themselves stay untouched, simple, exception-free contracts.
    pair_results: list[PairResult] = []
    for src_index, dst_index in pairs:
        try:
            pair_result = match_pair(
                matcher, images[src_index], images[dst_index], src_index, dst_index,
                ransac_threshold=config.ransac_threshold,
            )
        except Exception:
            continue
        pair_results.append(pair_result)

    edges_touching: dict[int, list[PairResult]] = {}
    for pair_result in pair_results:
        edges_touching.setdefault(pair_result.src_index, []).append(pair_result)
        edges_touching.setdefault(pair_result.dst_index, []).append(pair_result)

    def has_determined_edge(image_index: int) -> bool:
        return any(
            pr.inlier_count >= _MIN_INLIERS_FOR_DETERMINED_HOMOGRAPHY
            for pr in edges_touching.get(image_index, [])
        )

    # reference_index must always be discoverable by build_pose_graph's node_indices
    # union, even with zero edges and no caller-supplied GPS anchor (a single-image
    # "mosaic", or every edge above failing) -- this default is exactly what
    # build_pose_graph already falls back to internally for an anchor-less node
    # (identity at the origin), so it changes no computed value, it only makes the
    # reference node discoverable instead of silently absent from the graph.
    gps_positions = {config.reference_index: np.zeros(2), **(config.gps_positions or {})}

    ref_height, ref_width = images[config.reference_index].shape[:2]
    pixels_per_meter = estimate_pixels_per_meter(
        altitude_m=config.altitude_m, dfov_deg=config.dfov_deg,
        width_px=ref_width, height_px=ref_height,
    )
    inlier_count_reference = default_inlier_count_reference(pair_results)

    global_transforms = compose_global_transforms(
        pair_results,
        gps_positions=gps_positions,
        reference_index=config.reference_index,
        pixels_per_meter=pixels_per_meter,
        inlier_count_reference=inlier_count_reference,
        gimbal_yaw=config.gimbal_yaw,
        yaw_anchor_weight=config.yaw_anchor_weight,
    )

    if global_transforms.optimization_status != "converged":
        # Do not trust warp/blend to a GlobalTransforms the optimizer itself doesn't
        # believe in (see CLAUDE.md) -- the whole run is failed, downstream stages
        # are never invoked.
        elapsed = time.perf_counter() - start_time
        process_stats = _empty_process_stats(input_image_count, elapsed)
        metrics_df = evaluate_stitching_metrics(
            pair_results=pair_results,
            global_transforms=global_transforms,
            image_shapes={i: images[i].shape[:2] for i in all_indices},
            process_stats=process_stats,
            method=matcher.name,
        )
        placeholder = np.zeros_like(images[config.reference_index])
        return placeholder, metrics_df

    # --- warp + classify + blend (streaming): filter non-finite transforms BEFORE
    # canvas sizing, so one bad transform can't poison compute_canvas_size's shared
    # bounding-box computation for every other image. A finite-but-degenerate transform
    # (e.g. scale=0) never raises in cv2.warpPerspective -- it silently produces an
    # all-black mask, already correctly excluded below by the "must have >=1 nonzero
    # pixel" rule; no separate isolation mechanism is needed for that case.
    #
    # This used to be three separate passes over all N images (warp everything, THEN
    # classify each by inspecting the fully materialized warped_masks dict, THEN blend
    # only the survivors) -- each pass needing every image's canvas-sized warped data
    # alive at once (O(N * canvas_size)). It is now one fused streaming pass:
    # warp_images_streaming yields one image at a time, _successful_only classifies it
    # immediately (discarding a failed image's warped arrays right there -- they are
    # never yielded onward, never touch blend_images_streaming's accumulator, and are
    # not stored anywhere else in this function either) and forwards only survivors into
    # blend_images_streaming, which folds each one into its running accumulator and lets
    # it be freed before the next arrives. Peak memory is O(canvas_size), not
    # O(N * canvas_size) -- see CLAUDE.md's streaming-accumulator backlog item.
    usable_transforms = {
        index: transform
        for index, transform in global_transforms.transforms.items()
        if np.all(np.isfinite(transform))
    }

    canvas_size = (0, 0)
    successful_indices: set[int] = set()
    mosaic = np.zeros((0, 0, 3), dtype=np.uint8)
    seam_error_value = float(np.nan)

    if usable_transforms:
        usable_images = {index: images[index] for index in usable_transforms}
        usable_global_transforms = GlobalTransforms(
            transforms=usable_transforms,
            reference_index=global_transforms.reference_index,
            optimization_status=global_transforms.optimization_status,
            residual_error=global_transforms.residual_error,
        )
        usable_shapes = {index: image.shape[:2] for index, image in usable_images.items()}
        canvas_size = compute_canvas_size(usable_shapes, usable_global_transforms)

        # is-reference / has-determined-edge / has-anchor are all computable from
        # pair_results/config alone, independent of any warped pixel data -- precomputed
        # once here so the streaming loop below only has to check the ONE thing that
        # genuinely requires the warped mask: whether it ended up non-empty.
        structurally_qualifies = {
            index: (
                index == config.reference_index
                or has_determined_edge(index)
                or (config.gps_positions is not None and index in config.gps_positions)
            )
            for index in usable_transforms
        }

        def _successful_only(stream):
            for index, warped_image, warped_mask in stream:
                if structurally_qualifies[index] and np.any(warped_mask > 0):
                    successful_indices.add(index)
                    yield index, warped_image, warped_mask
                # else: warped_image/warped_mask fall out of scope right here -- never
                # yielded downstream, never entered into blend_images_streaming's
                # accumulator, and this generator holds no history of past iterations.

        warped_stream = warp_images_streaming(usable_images, usable_global_transforms, canvas_size)
        mosaic = blend_images_streaming(_successful_only(warped_stream), canvas_size)

        if successful_indices:
            successful_images = {index: usable_images[index] for index in successful_indices}
            seam_error_value = compute_seam_error_streaming(
                successful_images, global_transforms, canvas_size
            )
        else:
            # blend_images_streaming ran over an empty (fully-filtered-out) stream above,
            # producing a canvas-sized all-zero mosaic -- not this function's documented
            # "no successful images" contract, which is an empty (0, 0, 3) array.
            mosaic = np.zeros((0, 0, 3), dtype=np.uint8)

    failed_indices = sorted(set(all_indices) - successful_indices)
    successful_image_count = len(successful_indices)

    if successful_image_count == len(all_indices):
        pipeline_status = "success"
    elif len(all_indices) > 1 and successful_image_count >= 2:
        pipeline_status = "partial_success"
    else:
        pipeline_status = "failed"

    elapsed = time.perf_counter() - start_time
    process_stats = ProcessStats(
        pipeline_status=pipeline_status,
        input_image_count=input_image_count,
        successful_image_count=successful_image_count,
        failed_image_count=len(failed_indices),
        failed_image_indices=failed_indices,
        total_processing_time_sec=elapsed,
        avg_processing_time_per_image_sec=elapsed / input_image_count,
    )

    # warped_images/warped_masks/seam_masks are intentionally omitted here (defaulting to
    # None inside evaluate_stitching_metrics, which returns seam_error=np.nan for that
    # case) -- the streaming path never materializes those eager, all-N-canvas-sized
    # structures at all. seam_error_value was already computed above via
    # compute_seam_error_streaming, so it is patched into the one column that would
    # otherwise be affected, immediately afterward.
    metrics_df = evaluate_stitching_metrics(
        pair_results=pair_results,
        global_transforms=global_transforms,
        image_shapes={i: images[i].shape[:2] for i in all_indices},
        process_stats=process_stats,
        method=matcher.name,
        loops=config.loops,
    )
    metrics_df["seam_error"] = seam_error_value

    return mosaic, metrics_df
