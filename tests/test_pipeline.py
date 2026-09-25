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

import pickle
import tracemalloc
from pathlib import Path
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
    run_pipeline treats it as a whole-run failure and must not call warp_images_streaming/
    blend_images_streaming at all on a GlobalTransforms the optimizer itself doesn't
    believe in.

    Patch targets updated from warp_images/blend_images to their streaming replacements
    when run_pipeline was wired to call the streaming accumulator path (see CLAUDE.md's
    streaming-accumulator backlog item) -- same test name, same assertions, same intent;
    only the two patch-target strings changed, tracking what run_pipeline actually calls
    now. Deliberately not left pointing at the old (no longer called) names: doing so
    would have made mock_warp.call_count == 0 trivially true regardless of whether
    run_pipeline's not-converged guard worked at all, since neither old name is on the
    execution path anymore -- a silently gutted, always-green assertion, not a real one."""
    images = {i: _tagged_image(i) for i in range(2)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result()})

    not_converged = GlobalTransforms(
        transforms={0: np.eye(3), 1: np.eye(3)},
        reference_index=0,
        optimization_status="not_converged",
        residual_error=np.nan,
    )

    with patch("sea_mosaic.pipeline.compose_global_transforms", return_value=not_converged):
        with patch("sea_mosaic.pipeline.warp_images_streaming") as mock_warp:
            with patch("sea_mosaic.pipeline.blend_images_streaming") as mock_blend:
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


# --- F: streaming-accumulator integration regression (golden fixtures) ------------------
#
# These compare run_pipeline's output against fixtures captured from run_pipeline BEFORE
# the streaming refactor (tests/fixtures/golden_pipeline/capture_golden.py) -- they only
# ever READ those committed golden files, never regenerate them (an auto-regenerating
# "golden" would silently launder a real regression into a new baseline instead of
# catching it). Unlike the equivalence tests in stages 1-3 of this same redesign, these
# are expected to PASS right now, before any pipeline.py change: they're regression
# scaffolding, not a red-first check for a not-yet-written function -- their job starts
# once pipeline.py's internals actually change.

_GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden_pipeline"
_TIMING_COLUMNS = {"total_processing_time_sec", "avg_processing_time_per_image_sec"}


def _load_golden(name: str) -> tuple[np.ndarray, pd.DataFrame]:
    mosaic = np.load(_GOLDEN_DIR / f"{name}.npy")
    with open(_GOLDEN_DIR / f"{name}_metrics.pkl", "rb") as f:
        metrics_df = pickle.load(f)
    return mosaic, metrics_df


def _assert_metrics_df_matches_golden(new_df: pd.DataFrame, golden_df: pd.DataFrame) -> None:
    """Excludes wall-clock timing columns (legitimately different every run, not a
    regression signal). Everything else is asserted with a tight rtol rather than bare
    == -- not because any column is expected to need it (none of reprojection_error_px/
    inlier_ratio/inlier_count/cycle_loop_error_px/distortion derive from blended pixel
    values at all, and seam_error was proven bit-exact, not tolerance-based, in the
    compute_seam_error_streaming equivalence tests) but as a documented safety margin; a
    genuine mismatch here needs investigating as a real difference, not silently
    loosened further."""
    assert list(new_df.columns) == list(golden_df.columns)
    for col in new_df.columns:
        if col in _TIMING_COLUMNS:
            continue
        new_val, golden_val = new_df[col][0], golden_df[col][0]
        if isinstance(golden_val, float) and np.isnan(golden_val):
            assert isinstance(new_val, float) and np.isnan(new_val), f"{col}: expected NaN, got {new_val!r}"
        elif isinstance(golden_val, list):
            assert new_val == golden_val, f"{col}: {new_val!r} != {golden_val!r}"
        elif isinstance(golden_val, float):
            assert new_val == pytest.approx(golden_val, rel=1e-9, abs=1e-12), (
                f"{col}: {new_val!r} != {golden_val!r}"
            )
        else:
            assert new_val == golden_val, f"{col}: {new_val!r} != {golden_val!r}"


def test_run_pipeline_matches_golden_all_images_well_matched() -> None:
    images = {i: _tagged_image(i) for i in range(3)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
    golden_mosaic, golden_metrics = _load_golden("all_images_well_matched")

    mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert np.array_equal(mosaic, golden_mosaic)
    _assert_metrics_df_matches_golden(metrics_df, golden_metrics)


def test_run_pipeline_matches_golden_isolated_node_without_gps_anchor() -> None:
    """Exercises the classification path (edge/anchor-based) that has to move from a
    post-hoc filter over a fully materialized warped_masks dict into an inline filter
    over the warp stream -- the part of this integration that's more than a rename."""
    images, matcher = _isolated_node_scenario()
    golden_mosaic, golden_metrics = _load_golden("isolated_node_without_gps_anchor")

    mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert np.array_equal(mosaic, golden_mosaic)
    _assert_metrics_df_matches_golden(metrics_df, golden_metrics)


def test_run_pipeline_matches_golden_nonfinite_transform_isolated() -> None:
    """Exercises the pre-warp finite-transform filter combined with mask-emptiness
    classification -- distinct from the edge/anchor-based classification above."""
    images = {0: _tagged_image(0, size=5), 1: _tagged_image(1, size=5), 2: _tagged_image(2, size=5)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
    nan_transform = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    poisoned = GlobalTransforms(
        transforms={
            0: np.eye(3),
            1: nan_transform,
            2: np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        },
        reference_index=0, optimization_status="converged", residual_error=0.1,
    )
    golden_mosaic, golden_metrics = _load_golden("nonfinite_transform_isolated")

    with patch("sea_mosaic.pipeline.compose_global_transforms", return_value=poisoned):
        mosaic, metrics_df = run_pipeline(images, matcher, _base_config())

    assert np.array_equal(mosaic, golden_mosaic)
    _assert_metrics_df_matches_golden(metrics_df, golden_metrics)


# --- G: warp/blend/seam_error memory scaling (NOT all of run_pipeline -- see the test's
# own docstring for why compose_global_transforms is deliberately mocked out) -----------


def _synthetic_pipeline_scenario(
    n: int, total_span: float = 200.0, image_height: int = 10
) -> tuple[dict[int, np.ndarray], _ScriptedMatcher]:
    """n images along a FIXED-length span (image width shrinks as n grows, same lesson
    as the compute_seam_error_streaming fixture in stage 3) so canvas_size stays roughly
    constant as n grows -- empirically verified against the REAL compose_global_transforms
    (not assumed, since run_pipeline runs real pose-graph optimization, not a hand-fed
    GlobalTransforms): mosaic.shape was IDENTICAL, (11, 201, 3), at both n=10 and n=100.
    Sequential MatchResults (_good_match_result(tx=step, ty=0)) give consistent pairwise
    translations regardless of the tiny image dimensions -- findHomography works on the
    raw point correspondences, not on whether they fall within the image's own bounds."""
    step = total_span / n
    image_width = max(2, int(round(step * 2)))
    images = {
        i: np.full((image_height, image_width, 3), (i % 200) + 1, dtype=np.uint8) for i in range(n)
    }
    matcher = _ScriptedMatcher(
        {(i, i + 1): _good_match_result(tx=step, ty=0.0) for i in range(n - 1)}
    )
    return images, matcher


def _hand_fed_global_transforms(n: int, total_span: float = 200.0) -> GlobalTransforms:
    """A cheap, non-optimized GlobalTransforms matching _synthetic_pipeline_scenario's own
    geometry exactly (translation by i*step), for mocking sea_mosaic.pipeline.
    compose_global_transforms in the memory-scaling test below. See that test's docstring
    for why: compose_global_transforms's real pose-graph solver has its own, separate,
    O(N)-ish-or-worse memory cost (see CLAUDE.md's dedicated backlog item) that has
    nothing to do with warp_images_streaming/blend_images_streaming/
    compute_seam_error_streaming -- this mock isolates exactly the three stages this
    integration is actually responsible for."""
    step = total_span / n
    return GlobalTransforms(
        transforms={i: np.array([[1.0, 0.0, i * step], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) for i in range(n)},
        reference_index=0, optimization_status="converged", residual_error=0.0,
    )


def _peak_traced_bytes_for_pipeline(n: int) -> int:
    images, matcher = _synthetic_pipeline_scenario(n)
    gt = _hand_fed_global_transforms(n)
    tracemalloc.start()
    try:
        tracemalloc.clear_traces()
        with patch("sea_mosaic.pipeline.compose_global_transforms", return_value=gt):
            run_pipeline(images, matcher, _base_config())
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_run_pipeline_warp_blend_seam_error_peak_memory_is_flat_not_linear_in_n() -> None:
    """Scope note (renamed from an earlier "end_to_end" name specifically to prevent this
    misreading): this test verifies ONLY warp_images_streaming/blend_images_streaming/
    compute_seam_error_streaming's memory behavior through run_pipeline's real
    orchestration -- it does NOT cover run_pipeline's memory behavior as a whole.
    compose_global_transforms is mocked out (see _hand_fed_global_transforms) because its
    real pose-graph solver has a separate, much larger, and entirely independent memory
    cost that was never in scope for this streaming-accumulator redesign: measured
    directly, compose_global_transforms alone accounted for 15,948KB of a 16,388KB total
    peak at n=100 (97%) when left real -- see CLAUDE.md's dedicated backlog item for that
    finding, which is NOT considered fixed or covered by this test passing.

    With compose_global_transforms isolated out, this is the actual acceptance criterion
    for the three streaming stages this redesign covers, measured through the real public
    entry point, not just the lower-level functions in isolation. Deliberately verified
    (via `git stash` of pipeline.py, not guessed) to fail against the PRE-refactor
    run_pipeline with this exact same compose mock already in place: peak_10=546548
    bytes, peak_100=2831789 bytes (ratio ~5.18x, fails the <2.0 bound below) using
    warp_images/blend_images/eager compute_seam_error's all-pairs O(N^2) call --
    confirming the compose mock alone does not trivially pass this test, only the actual
    streaming wiring does."""
    peak_10 = _peak_traced_bytes_for_pipeline(10)
    peak_100 = _peak_traced_bytes_for_pipeline(100)

    ratio = peak_100 / peak_10
    assert ratio < 2.0, (
        f"peak traced memory scaled {ratio:.2f}x going from N=10 to N=100 "
        f"(peak_10={peak_10} bytes, peak_100={peak_100} bytes) -- expected roughly flat, "
        f"not O(N)"
    )
