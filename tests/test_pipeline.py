"""Unit tests for sea_mosaic.pipeline.run_pipeline.

run_pipeline's job here is tested at the ORCHESTRATION level: given a scripted Matcher
(no real SIFT) and, where needed, a mocked compose_global_transforms, does run_pipeline
correctly classify images (success/partial_success/failed per docs/task2.md), isolate
per-pair and per-image failures without crashing, and wire pixels_per_meter/
inlier_count_reference/metrics correctly? This mirrors the same layering principle used
throughout this project (fake Matcher for match_pair's RANSAC logic, synthetic data for
warp/blend geometry): each test isolates ONE claim, not "does the whole real pipeline
produce a good-looking mosaic".

All test data is hand-built (no np.random). Images are tiny (3x3 or 5x5) arrays tagged
with their own index value, recognized by the scripted matcher below -- no real image
content or SIFT is involved anywhere in this file.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sea_mosaic.config import PipelineConfig
from sea_mosaic.estimate import default_inlier_count_reference
from sea_mosaic.geo.projection import estimate_pixels_per_meter
from sea_mosaic.matcher import MatchResult
from sea_mosaic.pipeline import run_pipeline
from sea_mosaic.types import GlobalTransforms


def _tagged_image(index: int, size: int = 3) -> np.ndarray:
    """A tiny image array whose every pixel equals its own index -- lets
    _ScriptedMatcher below identify which pair it's being asked to match, without any
    real image content or SIFT involved."""
    return np.full((size, size, 3), index, dtype=np.uint8)


def _good_match_result(tx: float = 3.0, ty: float = 2.0) -> MatchResult:
    """5 well-spread points (4 corners + center, not collinear), all consistent with a
    pure translation -- cv2.findHomography/RANSAC accepts all 5 as inliers (verified
    empirically: real cv2.findHomography never reports fewer inliers than the minimal
    4-point sample it fits, so a small, fully-consistent point set like this reliably
    gives inlier_count == match_count == 5, not something lower)."""
    src = np.array([[10.0, 10.0], [50.0, 10.0], [10.0, 50.0], [50.0, 50.0], [30.0, 30.0]])
    dst = src + np.array([tx, ty])
    return MatchResult(src_points=src, dst_points=dst, scores=None)


def _too_few_points_match_result() -> MatchResult:
    """Only 2 correspondences -- below the mathematical minimum of 4 for a homography.
    match_pair's cv2.findHomography call raises a real cv2.error for this (verified
    empirically), not a contrived/injected exception. This is the SAME real mechanism
    behind both the per-pair error-isolation tests and the "no valid edge" classification
    tests below -- match_count<4 (raises) and inlier_count<4 (reported by a successful
    RANSAC call) turned out, empirically, to be the same practical boundary: a
    successful cv2.findHomography call on real data never reports fewer than 4 inliers."""
    return MatchResult(
        src_points=np.array([[10.0, 10.0], [20.0, 20.0]]),
        dst_points=np.array([[13.0, 12.0], [23.0, 22.0]]),
        scores=None,
    )


class _ScriptedMatcher:
    """Matcher test double: returns a pre-scripted MatchResult per (src_index,
    dst_index) pair, identifying the pair from _tagged_image's own pixel value (not
    from any real feature content)."""

    name = "scripted-test-matcher"

    def __init__(self, results_by_pair: dict[tuple[int, int], MatchResult]) -> None:
        self._results_by_pair = results_by_pair

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        src_index = int(image_a.flat[0])
        dst_index = int(image_b.flat[0])
        return self._results_by_pair[(src_index, dst_index)]


def _base_config(**overrides) -> PipelineConfig:
    defaults = dict(altitude_m=100.0, dfov_deg=82.9)
    defaults.update(overrides)
    return PipelineConfig(**defaults)


# --- A: happy-path classification -------------------------------------------------------


def test_run_pipeline_all_images_well_matched_is_success() -> None:
    images = {i: _tagged_image(i) for i in range(3)}
    matcher = _ScriptedMatcher(
        {(0, 1): _good_match_result(), (1, 2): _good_match_result()}
    )

    _mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["pipeline_status"][0] == "success"
    assert metrics_df["successful_image_count"][0] == 3
    assert metrics_df["failed_image_count"][0] == 0


def test_run_pipeline_single_image_is_success() -> None:
    images = {0: _tagged_image(0)}
    matcher = _ScriptedMatcher({})

    _mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["pipeline_status"][0] == "success"
    assert metrics_df["successful_image_count"][0] == 1


# --- B: the three classification paths (edge / GPS anchor / neither) -------------------


def _isolated_node_scenario() -> tuple[dict[int, np.ndarray], _ScriptedMatcher]:
    """5 images; node 2 is sandwiched between two edges that BOTH fail (too few
    points), isolating it from feature-based positioning entirely, while nodes 1 and 3
    each keep one good edge of their own (to 0 and 4 respectively) and are unaffected."""
    images = {i: _tagged_image(i) for i in range(5)}
    matcher = _ScriptedMatcher(
        {
            (0, 1): _good_match_result(),
            (1, 2): _too_few_points_match_result(),
            (2, 3): _too_few_points_match_result(),
            (3, 4): _good_match_result(),
        }
    )
    return images, matcher


def test_run_pipeline_isolated_node_without_gps_anchor_fails_but_others_succeed() -> None:
    images, matcher = _isolated_node_scenario()

    _mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["pipeline_status"][0] == "partial_success"
    assert metrics_df["successful_image_count"][0] == 4
    assert metrics_df["failed_image_indices"][0] == [2]


def test_run_pipeline_isolated_node_with_gps_anchor_still_succeeds() -> None:
    """The one case where option A (structural-only) and option B (quality-gated)
    disagree: node 2 has no valid feature-based edge, but DOES have a GPS anchor --
    per docs/task2.md 3.2's "reference/anchor image...算成功", it must still count."""
    images, matcher = _isolated_node_scenario()
    gps_positions = {2: np.array([0.0, 0.0])}

    _mosaic, metrics_df = run_pipeline(
        images, matcher, _base_config(gps_positions=gps_positions)
    )

    assert metrics_df["pipeline_status"][0] == "success"
    assert metrics_df["successful_image_count"][0] == 5
    assert metrics_df["failed_image_indices"][0] == []


def test_run_pipeline_mixed_isolation_failed_image_indices_exactly_the_unrescued_one() -> None:
    """Asymmetric graph, not a mirrored/repeated pattern: node 2 is isolated by being
    SANDWICHED between two failed edges (both its neighbors otherwise have their own
    good edges elsewhere); node 5 is isolated by being a DEAD END with only one edge,
    which fails (there is no symmetric edge on its other side at all, unlike node 2).
    Only the sandwiched one (2) gets a GPS anchor; the dead-end one (5) does not."""
    images = {i: _tagged_image(i) for i in range(6)}
    matcher = _ScriptedMatcher(
        {
            (0, 1): _good_match_result(),
            (1, 2): _too_few_points_match_result(),
            (2, 3): _too_few_points_match_result(),
            (3, 4): _good_match_result(),
            (4, 5): _too_few_points_match_result(),
        }
    )
    gps_positions = {2: np.array([0.0, 0.0])}

    _mosaic, metrics_df = run_pipeline(
        images, matcher, _base_config(gps_positions=gps_positions)
    )

    assert metrics_df["pipeline_status"][0] == "partial_success"
    assert metrics_df["successful_image_count"][0] == 5
    assert metrics_df["failed_image_indices"][0] == [5]


# --- C: whole-run failure paths ---------------------------------------------------------


def test_run_pipeline_only_reference_survives_is_failed() -> None:
    images = {i: _tagged_image(i) for i in range(3)}
    matcher = _ScriptedMatcher(
        {
            (0, 1): _too_few_points_match_result(),
            (1, 2): _too_few_points_match_result(),
        }
    )

    _mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["pipeline_status"][0] == "failed"
    assert metrics_df["successful_image_count"][0] == 1
    assert metrics_df["failed_image_indices"][0] == [1, 2]


def test_run_pipeline_not_converged_compose_forces_failed_and_skips_warp_blend() -> None:
    """optimize_pose_graph's own not_converged status must not be trusted downstream --
    run_pipeline treats it as a whole-run failure and must not call warp_images/
    blend_images at all on a GlobalTransforms the optimizer itself doesn't believe in."""
    images = {i: _tagged_image(i) for i in range(2)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result()})

    not_converged = GlobalTransforms(
        transforms={0: np.eye(3), 1: np.eye(3)},
        reference_index=0,
        optimization_status="not_converged",
        residual_error=np.nan,
    )

    with patch("sea_mosaic.pipeline.compose_global_transforms", return_value=not_converged):
        with patch("sea_mosaic.pipeline.warp_images") as mock_warp:
            with patch("sea_mosaic.pipeline.blend_images") as mock_blend:
                _mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["pipeline_status"][0] == "failed"
    assert metrics_df["successful_image_count"][0] == 0
    assert metrics_df["failed_image_indices"][0] == [0, 1]
    assert mock_warp.call_count == 0
    assert mock_blend.call_count == 0


def test_run_pipeline_empty_input_is_failed_without_crashing() -> None:
    _mosaic, metrics_df = run_pipeline({}, _ScriptedMatcher({}), _base_config())

    assert metrics_df["pipeline_status"][0] == "failed"
    assert metrics_df["input_image_count"][0] == 0
    assert np.isnan(metrics_df["stitch_success_rate"][0])
    assert np.isnan(metrics_df["avg_processing_time_per_image_sec"][0])


# --- D: error isolation ------------------------------------------------------------------


def test_run_pipeline_estimate_stage_pair_exception_is_isolated_not_fatal() -> None:
    """(1,2) has too few points and makes match_pair raise a REAL cv2.error (not an
    injected/mocked exception) -- run_pipeline must not crash; that edge is simply
    excluded, node 2 (with no other edge and no GPS anchor) fails, node 0/1 succeed."""
    images = {i: _tagged_image(i) for i in range(3)}
    matcher = _ScriptedMatcher(
        {(0, 1): _good_match_result(), (1, 2): _too_few_points_match_result()}
    )

    _mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["pipeline_status"][0] == "partial_success"
    assert metrics_df["successful_image_count"][0] == 2
    assert metrics_df["failed_image_indices"][0] == [2]


def test_run_pipeline_nonfinite_transform_isolated_without_poisoning_other_images() -> None:
    """Regression guard for the "contamination radius" problem found empirically while
    designing this isolation: compute_canvas_size converts bounds via int(np.ceil(...)),
    which raises ValueError on NaN/inf -- and if canvas_size were computed from ALL
    images' transforms in one call, node 1's NaN transform would poison that single
    shared computation for nodes 0 and 2 too, not just fail node 1 alone.

    The fix is NOT a try/except around a per-image warp call (cv2.warpPerspective
    itself never raises for bad matrices -- see the other test below): it's an
    up-front np.all(np.isfinite(transform)) filter applied BEFORE canvas sizing, so a
    non-finite transform is excluded from that shared computation entirely. This test
    locks in that node 0 and node 2 still get a sane, uncorrupted mosaic despite node
    1's transform being unusable."""
    images = {0: _tagged_image(0, size=5), 1: _tagged_image(1, size=5), 2: _tagged_image(2, size=5)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})

    nan_transform = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    poisoned = GlobalTransforms(
        transforms={
            0: np.eye(3),
            1: nan_transform,
            2: np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        },
        reference_index=0,
        optimization_status="converged",
        residual_error=0.1,
    )

    with patch("sea_mosaic.pipeline.compose_global_transforms", return_value=poisoned):
        mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["failed_image_indices"][0] == [1]
    assert metrics_df["successful_image_count"][0] == 2
    assert np.all(np.isfinite(mosaic.astype(np.float64)))
    # canvas reflects only nodes 0 and 2's (5x5, offset by 10) footprints, not a NaN-
    # corrupted computation -- small and sane, not degenerate
    assert mosaic.shape[0] <= 20 and mosaic.shape[1] <= 20


def test_run_pipeline_finite_degenerate_transform_needs_no_special_isolation_mechanism() -> None:
    """This test's point is NOT to verify some hidden try/except or pre-check catching
    a warp failure -- it's the opposite: it locks in the empirical finding that
    cv2.warpPerspective never raises for a finite-but-singular (scale=0) transform, it
    silently produces an all-black warped image/mask. No new isolation mechanism is
    needed for this case at all: it is already correctly handled by the ordinary
    "warped_masks.masks[i] must have >=1 nonzero pixel to count as successful" rule
    from part B's tests above. If this test ever starts failing because
    cv2.warpPerspective's behavior changes to raise instead, that would be the actual
    interesting finding -- not a bug in run_pipeline's isolation logic."""
    images = {0: _tagged_image(0, size=5), 1: _tagged_image(1, size=5), 2: _tagged_image(2, size=5)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})

    singular_transform = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 1.0]])
    degenerate = GlobalTransforms(
        transforms={
            0: np.eye(3),
            1: singular_transform,
            2: np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        },
        reference_index=0,
        optimization_status="converged",
        residual_error=0.1,
    )

    with patch("sea_mosaic.pipeline.compose_global_transforms", return_value=degenerate):
        _mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["failed_image_indices"][0] == [1]
    assert metrics_df["successful_image_count"][0] == 2


# --- E: wiring correctness ---------------------------------------------------------------


def test_run_pipeline_metrics_df_matches_computed_classification() -> None:
    images, matcher = _isolated_node_scenario()

    _mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert metrics_df["input_image_count"][0] == 5
    assert (
        metrics_df["stitch_success_rate"][0]
        == pytest.approx(metrics_df["successful_image_count"][0] / 5)
    )
    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 1


def test_run_pipeline_computes_pixels_per_meter_and_inlier_count_reference_itself() -> None:
    """Neither value is a caller-supplied parameter (see CLAUDE.md/PipelineConfig) --
    both must be computed internally from real inputs (altitude_m/dfov_deg and this
    run's own pair_results), not passed through as arbitrary constants."""
    images = {i: _tagged_image(i) for i in range(3)}
    matcher = _ScriptedMatcher(
        {(0, 1): _good_match_result(), (1, 2): _good_match_result()}
    )
    config = _base_config(altitude_m=120.0, dfov_deg=82.9)

    from sea_mosaic.compose import compose_global_transforms as real_compose

    with patch("sea_mosaic.pipeline.compose_global_transforms", wraps=real_compose) as spy:
        run_pipeline(images, matcher, config)

    expected_ppm = estimate_pixels_per_meter(
        altitude_m=120.0, dfov_deg=82.9, width_px=3, height_px=3
    )
    called_kwargs = spy.call_args.kwargs
    assert called_kwargs["pixels_per_meter"] == pytest.approx(expected_ppm)
    # both edges give inlier_count=5 (see _good_match_result's docstring) -> median=5
    assert called_kwargs["inlier_count_reference"] == pytest.approx(5.0)


def test_run_pipeline_without_gimbal_yaw_degrades_gracefully() -> None:
    images = {i: _tagged_image(i) for i in range(3)}
    matcher = _ScriptedMatcher(
        {(0, 1): _good_match_result(), (1, 2): _good_match_result()}
    )
    config = _base_config(gimbal_yaw=None, yaw_anchor_weight=None)

    _mosaic, metrics_df = run_pipeline(images, matcher, config)

    assert metrics_df["pipeline_status"][0] == "success"
