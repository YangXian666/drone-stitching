"""Stitching quality metrics + pipeline process statistics, merged into one pandas
DataFrame, per docs/task2.md.

Any metric that cannot be computed must be filled with np.nan, never 0 (CLAUDE.md
architectural constraint 5 / task2.md section 10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sea_mosaic.types import GlobalTransforms, PairResult, ProcessStats, WarpedImages, WarpedMasks

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

            overlap = (np.asarray(mask_a) > 0) & (np.asarray(mask_b) > 0)

            if seam_masks is not None:
                band_a = seam_masks.get(idx_a)
                band_b = seam_masks.get(idx_b)
                band_parts = [np.asarray(b) > 0 for b in (band_a, band_b) if b is not None]
                if band_parts:
                    band = band_parts[0]
                    for extra in band_parts[1:]:
                        band = band | extra
                    overlap = overlap & band

            if not np.any(overlap):
                continue

            a_f = img_a.astype(np.float32) / 255.0
            b_f = img_b.astype(np.float32) / 255.0
            diff = np.abs(a_f - b_f)
            pixel_diff = diff[overlap]

            total_abs_diff += float(np.sum(pixel_diff))
            total_pixel_count += pixel_diff.size

    if total_pixel_count == 0:
        return np.nan

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
