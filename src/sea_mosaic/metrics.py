"""Stitching quality metrics + pipeline process statistics, merged into one pandas
DataFrame, per docs/task2.md.

Any metric that cannot be computed must be filled with np.nan, never 0 (CLAUDE.md
architectural constraint 5 / task2.md section 10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from sea_mosaic.types import GlobalTransforms, PairResult, ProcessStats, WarpedImages, WarpedMasks
from sea_mosaic.warp import _mosaic_bounds

_METRICS_SCHEMA_COLUMNS = [
    # Pipeline process statistics
    "pipeline_status",
    "input_image_count",
    "successful_image_count",
    "failed_image_count",
    "stitch_success_rate",
    "failed_image_indices",
    "total_processing_time_sec",
    "avg_processing_time_per_image_sec",
    # Stitching quality metrics
    "reprojection_error_px",
    "inlier_ratio",
    "inlier_count",
    "cycle_loop_error_px",
    "seam_error",
    "distortion",
]

# Homography denominator below this magnitude is treated as an invalid projection
# (task2.md section 11: numerical stability).
_HOMOGRAPHY_DEGENERACY_EPS = 1e-12

_CYCLE_REFERENCE_POINTS = np.array(
    [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [50.0, 50.0]],
    dtype=np.float64,
)

_DISTORTION_GRID_ROWS = 10
_DISTORTION_GRID_COLS = 10
_DISTORTION_SIGMA_EPS = 1e-9


def _normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > _HOMOGRAPHY_DEGENERACY_EPS:
        H = H / H[2, 2]
    return H


def _project_points(H: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx2 points through homography H. Returns (projected, valid_mask); rows
    with a degenerate denominator or non-finite result are marked invalid."""
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points, ones])
    proj = (H @ points_h.T).T
    w = proj[:, 2]
    valid = np.abs(w) >= _HOMOGRAPHY_DEGENERACY_EPS
    out = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    out[valid] = proj[valid, :2] / w[valid, None]
    finite = np.all(np.isfinite(out), axis=1)
    valid &= finite
    return out, valid


def compute_reprojection_error(pair_results: list[PairResult]) -> float:
    """Global RMSE reprojection error in pixels, using only RANSAC inliers across all
    pairs. np.nan if there are no valid inliers."""
    all_errors_sq: list[np.ndarray] = []

    for pair in pair_results:
        mask = np.asarray(pair.inlier_mask, dtype=bool)
        if not np.any(mask):
            continue

        src = np.asarray(pair.src_points, dtype=np.float64)[mask]
        dst = np.asarray(pair.dst_points, dtype=np.float64)[mask]
        if src.shape[0] == 0:
            continue

        H = _normalize_homography(pair.homography)
        proj, valid = _project_points(H, src)
        if not np.any(valid):
            continue

        diff = proj[valid] - dst[valid]
        all_errors_sq.append(np.sum(diff**2, axis=1))

    if not all_errors_sq:
        return np.nan

    combined = np.concatenate(all_errors_sq)
    if combined.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(combined)))


def compute_inlier_statistics(pair_results: list[PairResult]) -> dict[str, float]:
    """Return {'inlier_ratio': total_inlier_count / total_match_count, 'inlier_count':
    total_inlier_count}, aggregated globally (not averaged per pair). inlier_ratio is
    np.nan if total_match_count == 0."""
    total_match_count = 0
    total_inlier_count = 0

    for pair in pair_results:
        mask = np.asarray(pair.inlier_mask, dtype=bool)
        total_match_count += mask.shape[0]
        total_inlier_count += int(np.sum(mask))

    inlier_ratio = (
        total_inlier_count / total_match_count if total_match_count > 0 else np.nan
    )

    return {"inlier_ratio": inlier_ratio, "inlier_count": total_inlier_count}


def compute_cycle_loop_error(
    pair_results: list[PairResult],
    loops: list[list[int]] | None,
) -> float:
    """Mean cycle-closure RMSE in pixels over the given image-index loops. np.nan if
    loops is None or empty (no loop closure available)."""
    if not loops:
        return np.nan

    edge_homographies: dict[tuple[int, int], np.ndarray] = {}
    for pair in pair_results:
        edge_homographies[(pair.src_index, pair.dst_index)] = _normalize_homography(
            pair.homography
        )

    def get_edge_homography(src_index: int, dst_index: int) -> np.ndarray | None:
        if (src_index, dst_index) in edge_homographies:
            return edge_homographies[(src_index, dst_index)]
        if (dst_index, src_index) in edge_homographies:
            try:
                H_inv = np.linalg.inv(edge_homographies[(dst_index, src_index)])
            except np.linalg.LinAlgError:
                return None
            return _normalize_homography(H_inv)
        return None

    loop_rmses: list[float] = []

    for loop in loops:
        if len(loop) < 2:
            continue

        H_cycle = np.eye(3, dtype=np.float64)
        loop_valid = True
        for src_index, dst_index in zip(loop[:-1], loop[1:]):
            H_edge = get_edge_homography(src_index, dst_index)
            if H_edge is None:
                loop_valid = False
                break
            H_cycle = H_edge @ H_cycle
        if not loop_valid:
            continue

        proj, valid = _project_points(H_cycle, _CYCLE_REFERENCE_POINTS)
        if not np.any(valid):
            continue

        diff = proj[valid] - _CYCLE_REFERENCE_POINTS[valid]
        errors_sq = np.sum(diff**2, axis=1)
        loop_rmses.append(float(np.sqrt(np.mean(errors_sq))))

    if not loop_rmses:
        return np.nan

    return float(np.mean(loop_rmses))


def _accumulate_pair_seam_diff(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    band: np.ndarray | None = None,
) -> tuple[float, int]:
    """Mean-absolute-RGB-difference contribution from one image pair's overlap region
    (optionally further restricted to a seam band), shared verbatim by compute_seam_error
    (band from seam_masks when given) and compute_seam_error_streaming (band always None,
    since the streaming blend path never produces a seam_masks dict -- see
    compute_seam_error_streaming's docstring). Extracted so both implementations run the
    literal same pixel-comparison code, not two independently-written copies that could
    silently diverge -- what a caller compares (whole overlap vs. a specific pair) differs,
    this arithmetic does not.

    Returns (0.0, 0) when there is no actual pixel overlap -- deliberately a real
    contribution of zero, not a sentinel -- so a caller can sum this across many pairs
    (including ones the caller only suspects might overlap, e.g. a bounding-box-prefiltered
    candidate that turns out not to) without special-casing "did this pair contribute."
    """
    overlap = (np.asarray(mask_a) > 0) & (np.asarray(mask_b) > 0)
    if band is not None:
        overlap = overlap & band

    if not np.any(overlap):
        return 0.0, 0

    a_f = img_a.astype(np.float32) / 255.0
    b_f = img_b.astype(np.float32) / 255.0
    diff = np.abs(a_f - b_f)
    pixel_diff = diff[overlap]

    return float(np.sum(pixel_diff)), int(pixel_diff.size)


def compute_seam_error(
    warped_images: WarpedImages | None,
    warped_masks: WarpedMasks | None,
    seam_masks: dict[int, np.ndarray] | None = None,
) -> float:
    """Mean absolute RGB difference in [0, 1] over overlapping seam/overlap pixels
    between warped images. np.nan if there is no valid overlap."""
    if warped_images is None or warped_masks is None:
        return np.nan

    indices = sorted(warped_images.images.keys())
    total_abs_diff = 0.0
    total_pixel_count = 0

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_a, idx_b = indices[i], indices[j]
            img_a = warped_images.images.get(idx_a)
            img_b = warped_images.images.get(idx_b)
            mask_a = warped_masks.masks.get(idx_a)
            mask_b = warped_masks.masks.get(idx_b)
            if img_a is None or img_b is None or mask_a is None or mask_b is None:
                continue

            band = None
            if seam_masks is not None:
                band_a = seam_masks.get(idx_a)
                band_b = seam_masks.get(idx_b)
                band_parts = [np.asarray(b) > 0 for b in (band_a, band_b) if b is not None]
                if band_parts:
                    band = band_parts[0]
                    for extra in band_parts[1:]:
                        band = band | extra

            abs_diff, pixel_count = _accumulate_pair_seam_diff(img_a, img_b, mask_a, mask_b, band)
            total_abs_diff += abs_diff
            total_pixel_count += pixel_count

    if total_pixel_count == 0:
        return np.nan

    return total_abs_diff / total_pixel_count


def _image_canvas_bbox(
    image_shape: tuple[int, int], transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box (min_xy, max_xy) of one image's four corners after applying its own
    transform, in mosaic/canvas coordinates. Per-image analog of warp._mosaic_bounds
    (which combines all images into one shared bounding box) -- deliberately a small,
    self-contained duplicate of that corner-projection math rather than reusing
    _mosaic_bounds via a single-entry dict, since that would mean constructing a throwaway
    GlobalTransforms just to project one image's corners."""
    height, width = image_shape
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
        dtype=np.float64,
    ).T
    projected = transform @ corners
    projected_xy = projected[:2, :] / projected[2, :]
    return projected_xy.min(axis=1), projected_xy.max(axis=1)


def _bboxes_overlap(
    bbox_a: tuple[np.ndarray, np.ndarray], bbox_b: tuple[np.ndarray, np.ndarray]
) -> bool:
    """Whether two axis-aligned bounding boxes (min_xy, max_xy) overlap, INCLUSIVE of a
    shared boundary (a touching edge or corner counts as overlapping). This is
    deliberately the safe direction for a prefilter whose only correctness requirement is
    "never produce a false negative" (see _candidate_overlapping_pairs): an over-included
    pair costs a little wasted work, checked away precisely by the existing pixel-level
    overlap test in _accumulate_pair_seam_diff; an excluded (missed) pair would silently
    drop a real seam-error contribution, which is not recoverable downstream."""
    min_a, max_a = bbox_a
    min_b, max_b = bbox_b
    return bool(
        min_a[0] <= max_b[0]
        and min_b[0] <= max_a[0]
        and min_a[1] <= max_b[1]
        and min_b[1] <= max_a[1]
    )


def _candidate_overlapping_pairs(
    image_shapes: dict[int, tuple[int, int]], global_transforms: GlobalTransforms
) -> list[tuple[int, int]]:
    """Bounding-box-prefiltered candidate pairs for compute_seam_error_streaming: a safe
    superset (never misses a real overlap, see _bboxes_overlap) of pairs whose images
    might genuinely overlap in canvas space, computed entirely from cheap per-image corner
    projections -- no canvas-sized arrays, no assumption that only index-adjacent images
    can overlap (a loop-closure revisit, already anticipated by PipelineConfig.loops, can
    make far-apart indices spatially coincide). Iterates sorted indices in the same
    ascending (i, j) order compute_seam_error's own all-pairs loop uses, so a caller that
    processes only the contributing subset of these pairs accumulates in the identical
    order the eager implementation does."""
    indices = sorted(image_shapes)
    bboxes = {
        index: _image_canvas_bbox(image_shapes[index], global_transforms.transforms[index])
        for index in indices
    }
    pairs = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_a, idx_b = indices[i], indices[j]
            if _bboxes_overlap(bboxes[idx_a], bboxes[idx_b]):
                pairs.append((idx_a, idx_b))
    return pairs


def compute_seam_error_streaming(
    images: dict[int, np.ndarray],
    global_transforms: GlobalTransforms,
    canvas_size: tuple[int, int],
) -> float:
    """Bounding-box-prefiltered, on-demand-rewarp form of compute_seam_error: instead of
    requiring a fully materialized WarpedImages/WarpedMasks (all N canvas-sized images
    alive simultaneously -- the O(N * canvas_size) cost warp_images_streaming/
    blend_images_streaming exist to avoid), this sources each candidate pair's pixel data
    by re-warping only the 2 source images involved, on demand, discarding them before
    moving to the next pair. Trades some redundant CPU (an image touching K candidate
    pairs gets re-warped K times) for bounded memory: at most 2 images' worth of
    canvas-sized data alive at once, independent of N.

    Candidate pairs come from _candidate_overlapping_pairs. The existing pixel-level
    overlap guard (shared verbatim with compute_seam_error via _accumulate_pair_seam_diff)
    still precisely excludes any bbox-only candidate that turns out to have no real pixel
    overlap, so this produces the exact same total (same contributing pairs, same order,
    same arithmetic) as compute_seam_error given the equivalent eagerly-warped input.

    `images` may legitimately be a SUBSET of a larger warped set (e.g. only the images
    that passed run_pipeline's success classification, matching compute_seam_error's own
    historical behavior of only considering images that made it into the blend) without
    needing the caller to also supply the full set's canvas origin: origin_offset here is
    always a pure translation applied uniformly to every image being compared, so it
    cannot change their relative alignment to each other, and a subset's own bounding
    span is always contained within whatever larger span canvas_size was sized from --
    so re-anchoring to the subset's own origin can never clip anything a full-set origin
    wouldn't have. Verified empirically (not just derived): computing this with a
    subset's own origin vs. an explicitly-supplied full-set origin gave the bit-identical
    seam_error value.

    No seam_masks parameter: blend_images_streaming deliberately does not produce one (see
    its docstring), and docs/task2.md documents seam_masks as optional for the seam metric
    ("如果 pipeline 本身有 seam finder，請額外保留") -- this always falls back to the same
    overlap-region behavior as compute_seam_error(..., seam_masks=None).
    """
    image_shapes = {index: image.shape[:2] for index, image in images.items()}
    candidate_pairs = _candidate_overlapping_pairs(image_shapes, global_transforms)
    if not candidate_pairs:
        return float(np.nan)

    min_xy, _max_xy = _mosaic_bounds(image_shapes, global_transforms)
    dsize = (canvas_size[1], canvas_size[0])
    origin_offset = np.array(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]]
    )

    def _warp_one(index: int) -> tuple[np.ndarray, np.ndarray]:
        image = images[index]
        transform = origin_offset @ global_transforms.transforms[index]
        warped_image = cv2.warpPerspective(image, transform, dsize)
        full_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(full_mask, transform, dsize)
        return warped_image, warped_mask

    total_abs_diff = 0.0
    total_pixel_count = 0
    for idx_a, idx_b in candidate_pairs:
        img_a, mask_a = _warp_one(idx_a)
        img_b, mask_b = _warp_one(idx_b)
        abs_diff, pixel_count = _accumulate_pair_seam_diff(img_a, img_b, mask_a, mask_b)
        total_abs_diff += abs_diff
        total_pixel_count += pixel_count

    if total_pixel_count == 0:
        return float(np.nan)

    return total_abs_diff / total_pixel_count


def _warp_point(H: np.ndarray, x: float, y: float) -> np.ndarray | None:
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    w = p[2]
    if abs(w) < _HOMOGRAPHY_DEGENERACY_EPS:
        return None
    q = p[:2] / w
    if not np.all(np.isfinite(q)):
        return None
    return q


def _local_jacobian(H: np.ndarray, x: float, y: float, eps: float) -> np.ndarray | None:
    p_xp = _warp_point(H, x + eps, y)
    p_xm = _warp_point(H, x - eps, y)
    p_yp = _warp_point(H, x, y + eps)
    p_ym = _warp_point(H, x, y - eps)
    if p_xp is None or p_xm is None or p_yp is None or p_ym is None:
        return None

    d_dx = (p_xp - p_xm) / (2.0 * eps)
    d_dy = (p_yp - p_ym) / (2.0 * eps)
    J = np.column_stack([d_dx, d_dy])
    if not np.all(np.isfinite(J)):
        return None
    return J


def compute_distortion(
    global_transforms: GlobalTransforms,
    image_shapes: dict[int, tuple[int, int]],
) -> float:
    """Mean local Jacobian anisotropic distortion (abs(log(sigma_1 / sigma_2))) over a
    sampling grid, averaged across all successfully placed images. np.nan if no valid
    transform is available."""
    distortions: list[float] = []

    for image_index, H in global_transforms.transforms.items():
        shape = image_shapes.get(image_index)
        if shape is None:
            continue

        height, width = shape
        if height <= 1 or width <= 1:
            continue

        H_norm = _normalize_homography(H)
        eps = max(1.0, min(width, height) * 0.01)
        xs = np.linspace(0.0, float(width - 1), _DISTORTION_GRID_COLS)
        ys = np.linspace(0.0, float(height - 1), _DISTORTION_GRID_ROWS)

        for y in ys:
            for x in xs:
                J = _local_jacobian(H_norm, float(x), float(y), eps)
                if J is None:
                    continue
                try:
                    singular_values = np.linalg.svd(J, compute_uv=False)
                except np.linalg.LinAlgError:
                    continue

                sigma1, sigma2 = float(singular_values[0]), float(singular_values[-1])
                if not np.isfinite(sigma1) or not np.isfinite(sigma2):
                    continue
                if sigma2 <= _DISTORTION_SIGMA_EPS:
                    continue

                distortions.append(abs(np.log(sigma1 / sigma2)))

    if not distortions:
        return np.nan

    return float(np.mean(distortions))


def build_metrics_dataframe(fields: dict[str, Any]) -> pd.DataFrame:
    """Assemble a one-row DataFrame in the fixed schema order from docs/task2.md.

    In addition to the task2.md-required fields, `fields` should include a 'method'
    entry (matcher name) so that runs across different matchers/pipelines can be
    compared later; this is an extension column beyond the task2.md schema.
    """
    ordered: dict[str, Any] = {
        column: fields[column] for column in _METRICS_SCHEMA_COLUMNS if column in fields
    }
    extra = {key: value for key, value in fields.items() if key not in _METRICS_SCHEMA_COLUMNS}
    ordered.update(extra)

    return pd.DataFrame([ordered])


def evaluate_stitching_metrics(
    pair_results: list[PairResult],
    global_transforms: GlobalTransforms,
    image_shapes: dict[int, tuple[int, int]],
    process_stats: ProcessStats,
    method: str,
    warped_images: WarpedImages | None = None,
    warped_masks: WarpedMasks | None = None,
    seam_masks: dict[int, np.ndarray] | None = None,
    loops: list[list[int]] | None = None,
) -> pd.DataFrame:
    """Compute all stitching quality metrics, merge them with process_stats, and return
    a one-row pandas DataFrame per docs/task2.md section 2.

    method is the matcher name used for this run (e.g. matcher.name); it is added as a
    'method' column so results can be compared across matchers/runs later.
    """
    inlier_stats = compute_inlier_statistics(pair_results)

    fields = {
        "pipeline_status": process_stats.pipeline_status,
        "input_image_count": process_stats.input_image_count,
        "successful_image_count": process_stats.successful_image_count,
        "failed_image_count": process_stats.failed_image_count,
        "stitch_success_rate": process_stats.stitch_success_rate,
        "failed_image_indices": process_stats.failed_image_indices,
        "total_processing_time_sec": process_stats.total_processing_time_sec,
        "avg_processing_time_per_image_sec": process_stats.avg_processing_time_per_image_sec,
        "reprojection_error_px": compute_reprojection_error(pair_results),
        "inlier_ratio": inlier_stats["inlier_ratio"],
        "inlier_count": int(inlier_stats["inlier_count"]),
        "cycle_loop_error_px": compute_cycle_loop_error(pair_results, loops),
        "seam_error": compute_seam_error(warped_images, warped_masks, seam_masks),
        "distortion": compute_distortion(global_transforms, image_shapes),
        "method": method,
    }

    return build_metrics_dataframe(fields)


def save_metrics_txt(metrics_df: pd.DataFrame, result_image_path: str | Path) -> Path:
    """Write metrics_df.to_string(index=False) to metrics.txt next to result_image_path,
    and return the written path. Must not silently swallow I/O errors."""
    result_image_path = Path(result_image_path)
    metrics_path = result_image_path.parent / "metrics.txt"
    metrics_path.write_text(metrics_df.to_string(index=False), encoding="utf-8")
    return metrics_path
