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


def compute_reprojection_error(pair_results: list[PairResult]) -> float:
    """Global RMSE reprojection error in pixels, using only RANSAC inliers across all
    pairs. np.nan if there are no valid inliers."""
    ...


def compute_inlier_statistics(pair_results: list[PairResult]) -> dict[str, float]:
    """Return {'inlier_ratio': total_inlier_count / total_match_count, 'inlier_count':
    total_inlier_count}, aggregated globally (not averaged per pair). inlier_ratio is
    np.nan if total_match_count == 0."""
    ...


def compute_cycle_loop_error(
    pair_results: list[PairResult],
    loops: list[list[int]] | None,
) -> float:
    """Mean cycle-closure RMSE in pixels over the given image-index loops. np.nan if
    loops is None or empty (no loop closure available)."""
    ...


def compute_seam_error(
    warped_images: WarpedImages | None,
    warped_masks: WarpedMasks | None,
    seam_masks: dict[int, np.ndarray] | None = None,
) -> float:
    """Mean absolute RGB difference in [0, 1] over overlapping seam/overlap pixels
    between warped images. np.nan if there is no valid overlap."""
    ...


def compute_distortion(
    global_transforms: GlobalTransforms,
    image_shapes: dict[int, tuple[int, int]],
) -> float:
    """Mean local Jacobian anisotropic distortion (abs(log(sigma_1 / sigma_2))) over a
    sampling grid, averaged across all successfully placed images. np.nan if no valid
    transform is available."""
    ...


def build_metrics_dataframe(fields: dict[str, Any]) -> pd.DataFrame:
    """Assemble a one-row DataFrame in the fixed schema order from docs/task2.md.

    In addition to the task2.md-required fields, `fields` should include a 'method'
    entry (matcher name) so that runs across different matchers/pipelines can be
    compared later; this is an extension column beyond the task2.md schema.
    """
    ...


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
    ...


def save_metrics_txt(metrics_df: pd.DataFrame, result_image_path: str | Path) -> Path:
    """Write metrics_df.to_string(index=False) to metrics.txt next to result_image_path,
    and return the written path. Must not silently swallow I/O errors."""
    ...
