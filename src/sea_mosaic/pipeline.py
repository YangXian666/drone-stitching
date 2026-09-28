"""Pipeline entrypoint: estimate -> global poses (Stage A->D) -> warp -> blend -> evaluate metrics."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from sea_mosaic.blend import blend_images_streaming
from sea_mosaic.config import PipelineConfig
from sea_mosaic.estimate import gps_proximity_pairs, match_pair, sequential_pairs
from sea_mosaic.geo.camera import CameraIntrinsics, CameraPose
from sea_mosaic.global_poses import GlobalPoseEstimate, estimate_global_poses
from sea_mosaic.matcher import Matcher
from sea_mosaic.metrics import compute_seam_error_streaming, evaluate_stitching_metrics
from sea_mosaic.types import GlobalTransforms, PairResult, ProcessStats
from sea_mosaic.warp import compute_canvas_size, warp_images_streaming

logger = logging.getLogger(__name__)


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


def _single_image_transforms(index: int) -> GlobalTransforms:
    """docs/task2.md §3.1: a single input image that can be output as-is is a success.
    Stage A-D needs at least one pair (pixels_per_meter is estimated from pairs), so the
    single-image case bypasses it: identity transform, the image itself is the frame."""
    return GlobalTransforms(
        transforms={index: np.eye(3)},
        reference_index=index,
        optimization_status="converged",
        residual_error=float(np.nan),
    )


def _log_pose_estimate(estimate: GlobalPoseEstimate) -> None:
    """metrics_df's columns are fixed by docs/task2.md, so why images got no pose is
    reported here (see CLAUDE.md: run_pipeline's return signature is unchanged for now)."""
    if estimate.failure_reason is not None:
        logger.warning("global pose estimation: %s", estimate.failure_reason)
    if estimate.node_failure_reasons:
        logger.warning(
            "images without a global pose: %s", dict(sorted(estimate.node_failure_reasons.items()))
        )
    lag = estimate.gps_lag
    if lag.status != "estimated":
        logger.warning(
            "GPS lag not corrected: %s", lag.status + (f" ({lag.reason})" if lag.reason else "")
        )
    elif lag.uncorrected_nodes:
        logger.warning(
            "GPS lag not applied to images without a travel direction: %s", sorted(lag.uncorrected_nodes)
        )
    else:
        logger.info("GPS lag corrected: %.2f m after %d rounds", lag.lag_m, lag.rounds)
    rejected = estimate.edge_check.rejected
    if rejected:
        counts: dict[str, int] = {}
        for reason in rejected.values():
            counts[reason] = counts.get(reason, 0) + 1
        logger.warning("edge consistency check rejected %d edges: %s", len(rejected), dict(sorted(counts.items())))
    hits = estimate.bound_hits
    if hits is not None and (hits.position or hits.kappa):
        # Guard rails only catch gross failure; every hit is a diagnostic signal.
        logger.warning(
            "Stage D guard rail hit: positions %s, kappa %s", sorted(hits.position), hits.kappa
        )


def run_pipeline(
    images: dict[int, np.ndarray],
    matcher: Matcher,
    config: PipelineConfig,
    camera_intrinsics: dict[int, CameraIntrinsics] | None = None,
    camera_poses: dict[int, CameraPose] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run the full estimate -> global poses -> warp -> blend pipeline and evaluate metrics.

    images are already-loaded arrays (image_index -> ndarray) -- run_pipeline's
    responsibility boundary is explicitly "receives already-loaded image arrays", not
    file IO. Loading from disk (io_utils.load_image/load_images, still stubs) is a
    separate, independently-scoped concern for the caller, not something run_pipeline
    does on the caller's behalf (see CLAUDE.md). The same holds for metadata: pass EXIF
    lat/lon (gps_placement.load_exif_latlons) and, for heading_anchor_source="gimbal",
    GimbalYawDegree (io_utils.load_gimbal_yaw) through config.

    Global poses come from global_poses.estimate_global_poses (Stage A->D) in the north-up
    pixel frame. An image counts as successful when it got a finite global pose AND its
    warped mask contributes at least one pixel. estimate_global_poses already withholds a
    pose from every image without a usable edge (metadata alone is not image evidence), so
    no separate edge or anchor rule is needed here. Without GPS no image can be placed and
    the run fails (the reason is logged). A single input image bypasses Stage A-D and is
    output as-is (docs/task2.md §3.1).

    Returns (stitched_image, metrics_df), where metrics_df is the one-row DataFrame from
    metrics.evaluate_stitching_metrics. stitched_image is an empty (0, 0, 3) array when no
    image succeeded.

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
                transforms={}, reference_index=None,
                optimization_status="failed", residual_error=np.nan,
            ),
            image_shapes={},
            process_stats=process_stats,
            method=matcher.name,
        )
        return np.zeros((0, 0, 3), dtype=np.uint8), metrics_df

    all_indices = sorted(images)
    pair_results: list[PairResult] = []

    if input_image_count == 1:
        global_transforms = _single_image_transforms(all_indices[0])
    else:
        # Default pairing: with GPS, every pair closer than 40 m -- sequential pairs give each
        # node <= 2 edges, which the edge-consistency check can never verify (it needs
        # CONSENSUS_MIN_AGREE observations per node). Without GPS the run fails anyway
        # (no_gps); matching stays sequential so the matching metrics still mean something.
        if config.pairs is not None:
            pairs = config.pairs
        elif config.latlons:
            pairs = gps_proximity_pairs({k: v for k, v in config.latlons.items() if k in images})
        else:
            pairs = sequential_pairs(images)

        # --- estimate: run_pipeline is the system boundary, so a single bad pair (e.g.
        # too few raw matches -> cv2.error from cv2.findHomography) is caught and skipped
        # here, not allowed to crash the whole run. match_pair/estimate_all_pairs
        # themselves stay untouched, simple, exception-free contracts.
        for src_index, dst_index in pairs:
            try:
                pair_result = match_pair(
                    matcher, images[src_index], images[dst_index], src_index, dst_index,
                    ransac_threshold=config.ransac_threshold,
                )
            except Exception:
                continue
            pair_results.append(pair_result)

        # --- global poses: Stage A->D. Edge filtering (>= 4 inliers, finite homography),
        # pixels_per_meter and the Stage A edge weights are all decided inside
        # estimate_global_poses; argument errors (heading_anchor_source vs gimbal_yaw_deg)
        # raise there, data-dependent failures are reported.
        estimate = estimate_global_poses(
            pair_results,
            {index: images[index].shape for index in all_indices},
            config.latlons,
            heading_anchor_source=config.heading_anchor_source,
            gimbal_yaw_deg=config.gimbal_yaw_deg,
            capture_times_s=config.capture_times_s,
        )
        _log_pose_estimate(estimate)
        global_transforms = estimate.global_transforms

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
        return np.zeros((0, 0, 3), dtype=np.uint8), metrics_df

    # --- warp + classify + blend (streaming): filter non-finite transforms BEFORE
    # canvas sizing, so one bad transform can't poison compute_canvas_size's shared
    # bounding-box computation for every other image. A finite-but-degenerate transform
    # (e.g. scale=0) never raises in cv2.warpPerspective -- it silently produces an
    # all-black mask, already correctly excluded below by the "must have >=1 nonzero
    # pixel" rule; no separate isolation mechanism is needed for that case.
    #
    # One fused streaming pass: warp_images_streaming yields one image at a time,
    # _successful_only classifies it immediately (discarding a failed image's warped
    # arrays right there -- they are never yielded onward, never touch
    # blend_images_streaming's accumulator, and are not stored anywhere else in this
    # function either) and forwards only survivors into blend_images_streaming, which
    # folds each one into its running accumulator and lets it be freed before the next
    # arrives. Peak memory is O(canvas_size), not O(N * canvas_size) -- see CLAUDE.md's
    # streaming-accumulator backlog item.
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

        def _successful_only(stream):
            for index, warped_image, warped_mask in stream:
                if np.any(warped_mask > 0):
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
