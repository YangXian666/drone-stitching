"""Core intermediate data structures shared across the stitching pipeline stages.

These dataclasses are the four structures CLAUDE.md requires the pipeline to preserve:
pair_results, global_transforms, warped_images, warped_masks — plus ProcessStats, which
backs the pipeline execution statistics consumed by metrics.evaluate_stitching_metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairResult:
    """Result of matching and geometrically verifying one image pair (src -> dst)."""

    src_index: int
    dst_index: int
    src_points: np.ndarray  # shape (N, 2), float64
    dst_points: np.ndarray  # shape (N, 2), float64
    inlier_mask: np.ndarray  # shape (N,), bool, aligned to src_points/dst_points
    homography: np.ndarray  # shape (3, 3), float64, src -> dst

    @property
    def match_count(self) -> int:
        """Total number of correspondences found for this pair, before RANSAC filtering."""
        ...

    @property
    def inlier_count(self) -> int:
        """Number of correspondences that passed RANSAC geometric verification."""
        ...


@dataclass
class GlobalTransforms:
    """Image-to-mosaic transforms produced by pose-graph optimization (never pairwise
    homography chain multiplication — see CLAUDE.md architectural constraint 4)."""

    transforms: dict[int, np.ndarray]  # image_index -> 3x3 H_image_to_mosaic
    reference_index: int
    optimization_status: str
    residual_error: float  # np.nan if the optimizer residual is unavailable


@dataclass
class WarpedImages:
    """Per-image pixel data warped into the shared mosaic canvas coordinate system."""

    images: dict[int, np.ndarray]  # image_index -> warped image
    canvas_size: tuple[int, int]  # (height, width)


@dataclass
class WarpedMasks:
    """Per-image valid-pixel masks aligned to the same canvas as WarpedImages."""

    masks: dict[int, np.ndarray]  # image_index -> boolean/uint8 mask
    canvas_size: tuple[int, int]  # (height, width)


@dataclass
class ProcessStats:
    """Pipeline execution statistics, per docs/task2.md section 3."""

    pipeline_status: str
    input_image_count: int
    successful_image_count: int
    failed_image_count: int
    failed_image_indices: list[int]
    total_processing_time_sec: float
    avg_processing_time_per_image_sec: float

    @property
    def stitch_success_rate(self) -> float:
        """successful_image_count / input_image_count; np.nan if input_image_count == 0."""
        if self.input_image_count == 0:
            return np.nan
        return self.successful_image_count / self.input_image_count
