"""Unit tests for sea_mosaic.metrics, per docs/task2.md section 14.

All data here is synthetic and hand-built (no real images, no
tests/fixtures/synthetic.py helpers). metrics.py's functions are currently stubs, so
these tests are expected to fail until they are implemented.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd
import pytest

from sea_mosaic.metrics import (
    _bboxes_overlap,
    _candidate_overlapping_pairs,
    _image_canvas_bbox,
    build_metrics_dataframe,
    compute_cycle_loop_error,
    compute_distortion,
    compute_inlier_statistics,
    compute_reprojection_error,
    compute_seam_error,
    compute_seam_error_streaming,
    save_metrics_txt,
)
from sea_mosaic.types import GlobalTransforms, PairResult, ProcessStats, WarpedImages, WarpedMasks
from sea_mosaic.warp import compute_canvas_size, warp_images


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


# --- compute_seam_error bounding-box prefilter helpers (streaming redesign, gating) -----
#
# _image_canvas_bbox / _bboxes_overlap are the low-level building blocks for the
# bounding-box-prefiltered candidate-pair selection that will replace compute_seam_error's
# current all-pairs O(N^2) loop over fully materialized warped_images/warped_masks (see
# CLAUDE.md's streaming-accumulator backlog item). Tested here in isolation, matching this
# repo's existing convention of unit-testing small private geometry helpers directly (e.g.
# posegraph.py's _yaw_target_vector) rather than only through the public function that
# calls them.
#
# _bboxes_overlap's boundary convention is deliberately INCLUSIVE (a shared edge/corner
# counts as overlapping): this filter's only correctness requirement is "never produce a
# false negative" (see the design's "safe superset, not exact" principle) -- an
# over-included pair costs a little wasted work, checked away precisely by the existing
# pixel-level `(mask_a>0)&(mask_b>0)` overlap test inside compute_seam_error itself, while
# an under-included (missed) pair would silently drop a real seam-error contribution.


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _rotation_90() -> np.ndarray:
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_image_canvas_bbox_identity_transform():
    min_xy, max_xy = _image_canvas_bbox((50, 80), np.eye(3))  # (height, width)

    assert np.array_equal(min_xy, [0.0, 0.0])
    assert np.array_equal(max_xy, [80.0, 50.0])


def test_image_canvas_bbox_translation_shifts_both_corners():
    min_xy, max_xy = _image_canvas_bbox((50, 80), _translation(tx=10, ty=5))

    assert np.array_equal(min_xy, [10.0, 5.0])
    assert np.array_equal(max_xy, [90.0, 55.0])


def test_image_canvas_bbox_rotation_swaps_width_and_height_extent():
    """Same hand-computed geometry as
    test_warp.test_compute_canvas_size_rotation_swaps_width_and_height, since both
    functions project the same four corners through the same transform -- reusing an
    already independently-verified computation, not a fresh guess."""
    min_xy, max_xy = _image_canvas_bbox((40, 100), _rotation_90())  # height=40, width=100

    # corners (0,0),(100,0),(100,40),(0,40) -> (0,0),(0,100),(-40,100),(-40,0)
    assert np.array_equal(min_xy, [-40.0, 0.0])
    assert np.array_equal(max_xy, [0.0, 100.0])


def test_bboxes_overlap_fully_overlapping():
    bbox_a = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    bbox_b = (np.array([5.0, 5.0]), np.array([15.0, 15.0]))

    assert _bboxes_overlap(bbox_a, bbox_b) is True


def test_bboxes_overlap_fully_disjoint():
    bbox_a = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    bbox_b = (np.array([20.0, 20.0]), np.array([30.0, 30.0]))

    assert _bboxes_overlap(bbox_a, bbox_b) is False


def test_bboxes_overlap_one_contained_in_the_other():
    bbox_a = (np.array([0.0, 0.0]), np.array([100.0, 100.0]))
    bbox_b = (np.array([10.0, 10.0]), np.array([20.0, 20.0]))

    assert _bboxes_overlap(bbox_a, bbox_b) is True


def test_bboxes_overlap_edge_touching_counts_as_overlapping():
    """Locks the inclusive boundary convention explicitly (see module docstring above):
    bbox_a's right edge (x=10) exactly meets bbox_b's left edge (x=10) -- zero-area shared
    boundary, deliberately treated as an overlap (True), not a disjoint case (False)."""
    bbox_a = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    bbox_b = (np.array([10.0, 0.0]), np.array([20.0, 10.0]))

    assert _bboxes_overlap(bbox_a, bbox_b) is True


def test_bboxes_overlap_degenerate_zero_area_bbox():
    point_bbox = (np.array([5.0, 5.0]), np.array([5.0, 5.0]))
    containing_bbox = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    disjoint_bbox = (np.array([6.0, 6.0]), np.array([10.0, 10.0]))

    assert _bboxes_overlap(point_bbox, containing_bbox) is True
    assert _bboxes_overlap(point_bbox, disjoint_bbox) is False


# --- _candidate_overlapping_pairs: filter correctness (gating) --------------------------


def _global_transforms(transforms: dict[int, np.ndarray], reference_index: int = 0) -> GlobalTransforms:
    return GlobalTransforms(
        transforms=transforms, reference_index=reference_index,
        optimization_status="converged", residual_error=0.0,
    )


def _rotation(angle_deg: float) -> np.ndarray:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_candidate_overlapping_pairs_includes_non_adjacent_loop_closure():
    """The property that actually matters: PipelineConfig.loops already anticipates
    non-sequential revisits, so a filter that only checked index-adjacent pairs would
    silently miss a real overlap. Image 4 is placed at EXACTLY image 0's position --
    genuinely overlapping despite being 4 apart by index -- while image 0 vs image 3
    (clearly far apart in both index AND space: bbox x[0,50] vs x[120,170], no overlap)
    proves this isn't a no-op that just returns every pair."""
    image_shapes = {i: (50, 50) for i in range(5)}
    transforms = _global_transforms({
        0: _translation(0, 0),
        1: _translation(40, 0),
        2: _translation(80, 0),
        3: _translation(120, 0),
        4: _translation(0, 0),  # loop closure: revisits image 0's exact position
    })

    pairs = _candidate_overlapping_pairs(image_shapes, transforms)

    assert (0, 4) in pairs  # the loop-closure pair -- the whole point of this test
    assert (0, 3) not in pairs  # clearly disjoint -- proves genuine filtering, not "all pairs"
    assert (0, 1) in pairs  # sanity: the ordinary adjacent-overlap case still works
    assert (1, 2) in pairs


# --- compute_seam_error_streaming: bit-exact equivalence to compute_seam_error (gating) -


def _rotated_bbox_overlap_pixel_disjoint_scenario() -> tuple[dict[int, np.ndarray], GlobalTransforms]:
    """Deliberately adversarial (see this session's design discussion on "safe superset,
    not exact"): image 1 is a 40x40 square rotated 45deg about its own (0,0) corner, which
    turns its bounding box into a diamond's containing rectangle -- x in [-28.28,28.28],
    y in [0,56.57] (hand-derived: corner (40,40) -> (0,56.57) is the y-max vertex, corner
    (0,40) -> (-28.28,28.28) is the x-min vertex, etc). A 45deg-rotated square's actual
    footprint is a diamond inscribed in that rectangle -- touching only the midpoints of
    each rectangle edge, leaving all four rectangle CORNERS empty by construction (a
    standard geometric fact about a square rotated 45deg, not specific to this codebase).
    Image 0 (a plain, unrotated 5x5 square at translation(20,0), occupying x[20,25],
    y[0,5]) sits entirely inside the empty top-right corner wedge of that bounding
    rectangle: verified via the diamond's L1-ball inequality |x|+|y-28.28|<=28.28 -- every
    corner of image 0's 5x5 footprint gives a value >43 (>>28.28), i.e. outside the
    diamond. So bbox_0 and bbox_1 genuinely overlap (x[20,25] subset of [-28.28,28.28],
    y[0,5] subset of [0,56.57]) while their real warped pixel footprints do not -- exactly
    the case _candidate_overlapping_pairs's bbox filter cannot distinguish, and which
    compute_seam_error's/compute_seam_error_streaming's existing pixel-level
    `(mask_a>0)&(mask_b>0)` guard must still correctly zero out."""
    images = {
        0: np.full((5, 5, 3), 100, dtype=np.uint8),
        1: np.full((40, 40, 3), 200, dtype=np.uint8),
    }
    transforms = _global_transforms({0: _translation(20, 0), 1: _rotation(45)})
    return images, transforms


def _rotated_scenario_empirically_has_no_pixel_overlap(
    images: dict[int, np.ndarray], transforms: GlobalTransforms
) -> bool:
    """Verifies the adversarial fixture's geometry empirically against the actual,
    already-trusted warp_images output, rather than trusting the hand derivation alone."""
    image_shapes = {i: img.shape[:2] for i, img in images.items()}
    canvas_size = compute_canvas_size(image_shapes, transforms)
    _warped, masks = warp_images(images, transforms)
    overlap = (masks.masks[0] > 0) & (masks.masks[1] > 0)
    return not np.any(overlap)


_EQUIVALENCE_SCENARIOS = {
    "two_images_full_overlap": lambda: (
        {0: np.full((30, 30, 3), 80, dtype=np.uint8), 1: np.full((30, 30, 3), 180, dtype=np.uint8)},
        _global_transforms({0: _translation(0, 0), 1: _translation(0, 0)}),
    ),
    "two_images_partial_overlap": lambda: (
        {0: np.full((30, 40, 3), 60, dtype=np.uint8), 1: np.full((30, 40, 3), 220, dtype=np.uint8)},
        _global_transforms({0: _translation(0, 0), 1: _translation(20, 0)}),
    ),
    "far_apart_no_overlap": lambda: (
        {0: np.full((20, 20, 3), 50, dtype=np.uint8), 1: np.full((20, 20, 3), 150, dtype=np.uint8)},
        _global_transforms({0: _translation(0, 0), 1: _translation(500, 500)}),
    ),
    "five_image_line_with_loop_closure": lambda: (
        {i: np.full((50, 50, 3), (i + 1) * 30, dtype=np.uint8) for i in range(5)},
        _global_transforms({
            0: _translation(0, 0), 1: _translation(40, 0), 2: _translation(80, 0),
            3: _translation(120, 0), 4: _translation(0, 0),
        }),
    ),
    "bbox_overlap_pixel_disjoint_adversarial": _rotated_bbox_overlap_pixel_disjoint_scenario,
}


@pytest.mark.parametrize(
    "scenario_name", list(_EQUIVALENCE_SCENARIOS), ids=list(_EQUIVALENCE_SCENARIOS)
)
def test_compute_seam_error_streaming_matches_compute_seam_error_bitexact(scenario_name):
    """Bit-exact (not tolerance-based), matching this session's earlier conclusion that
    compute_seam_error's math is a plain running sum over the SAME set of contributing
    pairs in the SAME order for both implementations (no per-pixel division like
    blend_images_streaming has), so unlike that function there is no floating-point-
    reassociation caveat here -- old and new must agree exactly."""
    images, transforms = _EQUIVALENCE_SCENARIOS[scenario_name]()
    image_shapes = {i: img.shape[:2] for i, img in images.items()}
    canvas_size = compute_canvas_size(image_shapes, transforms)
    warped_images, warped_masks = warp_images(images, transforms)

    old_error = compute_seam_error(warped_images, warped_masks)
    new_error = compute_seam_error_streaming(images, transforms, canvas_size)

    if np.isnan(old_error):
        assert np.isnan(new_error)
    else:
        assert new_error == old_error


def test_bbox_overlap_pixel_disjoint_scenario_is_empirically_verified_adversarial():
    """Guards the adversarial fixture itself, not compute_seam_error_streaming: confirms
    the hand-derived '45deg-rotated diamond leaves its bbox corners empty' geometry
    actually holds for this exact fixture (via real warp_images output), so the
    equivalence test case above is known to be exercising the safe-superset property it
    claims to, not silently degenerating into an ordinary overlapping case."""
    images, transforms = _rotated_bbox_overlap_pixel_disjoint_scenario()

    assert _rotated_scenario_empirically_has_no_pixel_overlap(images, transforms)

    image_shapes = {i: img.shape[:2] for i, img in images.items()}
    bbox_0 = _image_canvas_bbox(image_shapes[0], transforms.transforms[0])
    bbox_1 = _image_canvas_bbox(image_shapes[1], transforms.transforms[1])
    assert _bboxes_overlap(bbox_0, bbox_1)  # bbox says "maybe" ...
    # ... but the pixel-level check above says "no" -- this is the case the bbox filter
    # cannot distinguish, and downstream must still handle correctly (see the
    # equivalence test's "bbox_overlap_pixel_disjoint_adversarial" case).


# --- compute_seam_error_streaming: peak memory stays flat as N grows --------------------


def _synthetic_sparse_line_scenario(
    n: int, total_span: float = 200.0, image_height: int = 20
) -> tuple[dict[int, np.ndarray], GlobalTransforms]:
    """n images spread along a FIXED-length line (total_span does not grow with n), each
    only overlapping its immediate neighbor by half its own width -- matching a realistic
    flight line's sparse overlap (each image touches ~2 neighbors, not all n-1 others).

    An earlier version of this fixture placed all n images at the identical position to
    hold canvas_size fixed; that made every pair a candidate (n*(n-1)/2 -- ~4950 for
    n=100), and for a small canvas the resulting Python-level candidate-pair LIST itself
    (not per-image pixel data) dominated the traced memory, an honest but different cost
    from the one this test means to isolate. This version decouples the two properties
    that matter instead of conflating them: image_width shrinks as n grows
    (image_width = 2 * total_span/n) so canvas_size stays close to total_span regardless
    of n (verified: canvas width 220px at n=10 vs 202px at n=100, not the ~10x growth an
    n-proportional line would produce), while overlap stays sparse (each image still only
    overlaps ~2 immediate neighbors, so candidate-pair count grows ~linearly with n --
    verified: 17 pairs at n=10 vs 197 at n=100, not n^2/2). warpPerspective's OUTPUT array
    size is always canvas_size regardless of the SOURCE image's width, so shrinking source
    width does not itself reduce the per-pair memory cost being measured -- it only keeps
    overlap sparse, which is the property needed here."""
    step = total_span / n
    image_width = int(round(step * 2))
    images = {
        i: np.full((image_height, image_width, 3), (i % 200) + 1, dtype=np.uint8) for i in range(n)
    }
    transforms = _global_transforms({i: _translation(i * step, 0) for i in range(n)})
    return images, transforms


def _peak_traced_bytes_for_seam_error(n: int) -> int:
    images, transforms = _synthetic_sparse_line_scenario(n)
    image_shapes = {i: img.shape[:2] for i, img in images.items()}
    canvas_size = compute_canvas_size(image_shapes, transforms)

    tracemalloc.start()
    try:
        tracemalloc.clear_traces()
        compute_seam_error_streaming(images, transforms, canvas_size)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_compute_seam_error_streaming_peak_memory_is_flat_not_linear_in_n():
    """Same acceptance-criterion methodology as blend_images_streaming's flatness test:
    N grows 10x (10 -> 100); peak memory should stay far below a 10x increase, since only
    2 images' canvas-sized pixel data should ever be alive at once (the current candidate
    pair being evaluated), never all N."""
    peak_10 = _peak_traced_bytes_for_seam_error(10)
    peak_100 = _peak_traced_bytes_for_seam_error(100)

    ratio = peak_100 / peak_10
    assert ratio < 2.0, (
        f"peak traced memory scaled {ratio:.2f}x going from N=10 to N=100 "
        f"(peak_10={peak_10} bytes, peak_100={peak_100} bytes) -- expected roughly flat "
        f"(bounded by 2 images' worth of canvas-sized data), not O(N)"
    )


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
