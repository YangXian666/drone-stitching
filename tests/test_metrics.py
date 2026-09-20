"""Unit tests for sea_mosaic.metrics, per docs/task2.md section 14.

All data here is synthetic and hand-built (no real images, no
tests/fixtures/synthetic.py helpers). metrics.py's functions are currently stubs, so
these tests are expected to fail until they are implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sea_mosaic.metrics import (
    build_metrics_dataframe,
    compute_cycle_loop_error,
    compute_distortion,
    compute_inlier_statistics,
    compute_reprojection_error,
    compute_seam_error,
    save_metrics_txt,
)
from sea_mosaic.types import GlobalTransforms, PairResult, ProcessStats, WarpedImages, WarpedMasks


def _pair_result(
    src_index: int,
    dst_index: int,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    inlier_mask: np.ndarray,
    homography: np.ndarray,
) -> PairResult:
    return PairResult(
        src_index=src_index,
        dst_index=dst_index,
        src_points=np.asarray(src_points, dtype=np.float64),
        dst_points=np.asarray(dst_points, dtype=np.float64),
        inlier_mask=np.asarray(inlier_mask, dtype=bool),
        homography=np.asarray(homography, dtype=np.float64),
    )


# --- Test 1: Identity Homography -------------------------------------------------


def test_reprojection_error_identity_homography():
    H = np.eye(3, dtype=np.float64)
    points = np.array(
        [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0], [30.0, 30.0]],
        dtype=np.float64,
    )
    pair = _pair_result(0, 1, points, points, np.ones(5, dtype=bool), H)

    error = compute_reprojection_error([pair])

    assert error == pytest.approx(0.0, abs=1e-9)


# --- Test 2: Known Translation ----------------------------------------------------


def test_reprojection_error_known_translation():
    H = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    src = np.array(
        [[0.0, 0.0], [20.0, 5.0], [40.0, 10.0], [5.0, 40.0], [15.0, 15.0]],
        dtype=np.float64,
    )
    dst = src + np.array([10.0, 20.0])

    # Outlier: wrong destination, must be excluded from the RMSE.
    src_outlier = np.array([100.0, 100.0])
    dst_outlier = src_outlier + np.array([10.0, 20.0]) + np.array([100.0, 100.0])

    all_src = np.vstack([src, src_outlier])
    all_dst = np.vstack([dst, dst_outlier])
    inlier_mask = np.array([True, True, True, True, True, False])

    pair = _pair_result(0, 1, all_src, all_dst, inlier_mask, H)

    error = compute_reprojection_error([pair])

    assert error == pytest.approx(0.0, abs=1e-9)


# --- Test 3: Inlier Ratio ----------------------------------------------------------


def test_inlier_ratio_and_count():
    H = np.eye(3, dtype=np.float64)

    # Single pair: 100 matches, 80 inliers.
    mask_a = np.zeros(100, dtype=bool)
    mask_a[:80] = True
    pair_a = _pair_result(0, 1, np.zeros((100, 2)), np.zeros((100, 2)), mask_a, H)

    stats_single = compute_inlier_statistics([pair_a])
    assert stats_single["inlier_ratio"] == pytest.approx(0.8)
    assert stats_single["inlier_count"] == 80

    # Two pairs: global aggregation, not naive per-pair average.
    mask_b = np.zeros(10, dtype=bool)
    mask_b[:2] = True
    pair_b = _pair_result(1, 2, np.zeros((10, 2)), np.zeros((10, 2)), mask_b, H)

    stats_multi = compute_inlier_statistics([pair_a, pair_b])
    assert stats_multi["inlier_ratio"] == pytest.approx(82 / 110)
    assert stats_multi["inlier_ratio"] != pytest.approx((0.8 + 0.2) / 2)
    assert stats_multi["inlier_count"] == 82


# --- Test 4: Identity Cycle ---------------------------------------------------------


def test_cycle_loop_error_identity_cycle():
    rng_h01 = np.array(
        [[1.0, 0.0, 15.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rng_h12 = np.array(
        [[0.0, -1.0, 20.0], [1.0, 0.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    h20 = np.linalg.inv(rng_h12 @ rng_h01)

    dummy_points = np.zeros((4, 2), dtype=np.float64)
    dummy_mask = np.ones(4, dtype=bool)

    pair_01 = _pair_result(0, 1, dummy_points, dummy_points, dummy_mask, rng_h01)
    pair_12 = _pair_result(1, 2, dummy_points, dummy_points, dummy_mask, rng_h12)
    pair_20 = _pair_result(2, 0, dummy_points, dummy_points, dummy_mask, h20)

    error = compute_cycle_loop_error([pair_01, pair_12, pair_20], loops=[[0, 1, 2, 0]])

    assert error == pytest.approx(0.0, abs=1e-6)


# --- Test 5: No Loop -----------------------------------------------------------------


def test_cycle_loop_error_no_loop():
    H = np.eye(3, dtype=np.float64)
    dummy_points = np.zeros((4, 2), dtype=np.float64)
    dummy_mask = np.ones(4, dtype=bool)
    pair = _pair_result(0, 1, dummy_points, dummy_points, dummy_mask, H)

    assert np.isnan(compute_cycle_loop_error([pair], loops=[]))
    assert np.isnan(compute_cycle_loop_error([pair], loops=None))


# --- Test 6: Identical Overlap --------------------------------------------------------


def test_seam_error_identical_overlap():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[10:40, 10:40] = 200
    image[20:30, 5:15] = 100

    mask = np.zeros((50, 50), dtype=bool)
    mask[5:45, 5:45] = True

    warped_images = WarpedImages(images={0: image, 1: image.copy()}, canvas_size=(50, 50))
    warped_masks = WarpedMasks(masks={0: mask, 1: mask.copy()}, canvas_size=(50, 50))

    error = compute_seam_error(warped_images, warped_masks)

    assert error == pytest.approx(0.0, abs=1e-9)


# --- Test 7: Identity Warp Distortion --------------------------------------------------


def test_distortion_identity_warp():
    global_transforms = GlobalTransforms(
        transforms={0: np.eye(3, dtype=np.float64)},
        reference_index=0,
        optimization_status="converged",
        residual_error=0.0,
    )
    image_shapes = {0: (100, 100)}

    distortion = compute_distortion(global_transforms, image_shapes)

    assert distortion == pytest.approx(0.0, abs=1e-6)


# --- Test 8: metrics.txt Output ----------------------------------------------------------


def test_save_metrics_txt_roundtrip(tmp_path):
    metrics_df = pd.DataFrame(
        [
            {
                "pipeline_status": "partial_success",
                "input_image_count": 10,
                "successful_image_count": 8,
                "failed_image_count": 2,
                "stitch_success_rate": 0.8,
                "failed_image_indices": [3, 7],
                "total_processing_time_sec": 12.34,
                "avg_processing_time_per_image_sec": 1.234,
                "reprojection_error_px": 1.5,
                "inlier_ratio": 0.75,
                "inlier_count": 300,
                "cycle_loop_error_px": np.nan,
                "seam_error": 0.05,
                "distortion": 0.02,
            }
        ]
    )
    result_image_path = tmp_path / "stitched.png"

    metrics_path = save_metrics_txt(metrics_df, result_image_path)

    assert metrics_path == tmp_path / "metrics.txt"
    assert metrics_path.exists()
    assert metrics_path.read_text(encoding="utf-8") == metrics_df.to_string(index=False)


# --- Test 9: Process Statistics -----------------------------------------------------------


def test_process_statistics_normal():
    stats = ProcessStats(
        pipeline_status="partial_success",
        input_image_count=10,
        successful_image_count=8,
        failed_image_count=2,
        failed_image_indices=[7, 9],
        total_processing_time_sec=5.0,
        avg_processing_time_per_image_sec=0.5,
    )

    assert stats.stitch_success_rate == pytest.approx(0.8)


def test_process_statistics_empty_input():
    stats = ProcessStats(
        pipeline_status="failed",
        input_image_count=0,
        successful_image_count=0,
        failed_image_count=0,
        failed_image_indices=[],
        total_processing_time_sec=0.0,
        avg_processing_time_per_image_sec=np.nan,
    )

    assert np.isnan(stats.stitch_success_rate)
    assert np.isnan(stats.avg_processing_time_per_image_sec)


def test_process_statistics_anchor_only():
    stats = ProcessStats(
        pipeline_status="failed",
        input_image_count=10,
        successful_image_count=1,
        failed_image_count=9,
        failed_image_indices=list(range(1, 10)),
        total_processing_time_sec=3.0,
        avg_processing_time_per_image_sec=0.3,
    )

    assert stats.stitch_success_rate == pytest.approx(0.1)


def test_process_statistics_dataframe_passthrough():
    stats = ProcessStats(
        pipeline_status="partial_success",
        input_image_count=10,
        successful_image_count=8,
        failed_image_count=2,
        failed_image_indices=[3, 7],
        total_processing_time_sec=12.34,
        avg_processing_time_per_image_sec=1.234,
    )

    fields = {
        "pipeline_status": stats.pipeline_status,
        "input_image_count": stats.input_image_count,
        "successful_image_count": stats.successful_image_count,
        "failed_image_count": stats.failed_image_count,
        "stitch_success_rate": stats.stitch_success_rate,
        "failed_image_indices": stats.failed_image_indices,
        "total_processing_time_sec": stats.total_processing_time_sec,
        "avg_processing_time_per_image_sec": stats.avg_processing_time_per_image_sec,
        "reprojection_error_px": np.nan,
        "inlier_ratio": np.nan,
        "inlier_count": 0,
        "cycle_loop_error_px": np.nan,
        "seam_error": np.nan,
        "distortion": np.nan,
        "method": "dummy_matcher",
    }

    metrics_df = build_metrics_dataframe(fields)

    row = metrics_df.iloc[0]
    assert row["pipeline_status"] == "partial_success"
    assert row["input_image_count"] == 10
    assert row["successful_image_count"] == 8
    assert row["failed_image_count"] == 2
    assert row["stitch_success_rate"] == pytest.approx(0.8)
