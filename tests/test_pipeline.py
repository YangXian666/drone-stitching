"""Unit tests for sea_mosaic.pipeline.run_pipeline (Stage A->D global poses).

run_pipeline is tested at the ORCHESTRATION level: does it classify images correctly
(success/partial_success/failed per docs/task2.md), isolate per-pair and per-image
failures without crashing, handle the single-image and no-GPS cases, forward the
heading-anchor configuration, and log why images got no pose? Stage A-D itself is tested
in tests/test_global_poses.py and the per-stage test files.

Most tests use tests/synthetic_camera.py's PipelineScene: small (60x80) images rendered
from a ground texture through a pinhole camera, exact correspondences served by
SceneMatcher, and EXIF-style lat/lon -- all mutually consistent (checked in
tests/test_synthetic_scene.py). Tests that mock estimate_global_poses out (they are about
what run_pipeline does with a given GlobalTransforms) keep the older, simpler tagged-image
helpers: there the image content never reaches pose estimation.
"""

from __future__ import annotations

import logging
import pickle
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sea_mosaic.pipeline as pipeline_module
from sea_mosaic.config import PipelineConfig
from sea_mosaic.global_poses import GlobalPoseEstimate
from sea_mosaic.gps_lag import GpsLagEstimate
from sea_mosaic.matcher import MatchResult
from sea_mosaic.pipeline import run_pipeline
from sea_mosaic.refinement import BoundHits
from sea_mosaic.types import GlobalTransforms
from synthetic_camera import (
    PIPELINE_FOCAL_PX,
    PIPELINE_IMAGE_SHAPE,
    SceneMatcher,
    SyntheticCamera,
    ground_texture,
    pipeline_scene,
)

ALTITUDE = 100.0


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _line_cameras(n: int, step_m: float = 13.4) -> dict[int, SyntheticCamera]:
    """One flight line like real line B: course 83 deg, camera yaw 70.8 deg."""
    course = np.radians(83.0)
    return {
        k: SyntheticCamera(k * step_m * np.sin(course), k * step_m * np.cos(course), ALTITUDE, 70.8) for k in range(n)
    }


def _scene(n: int):
    return pipeline_scene(_line_cameras(n))


def _config(scene, **overrides) -> PipelineConfig:
    fields = dict(latlons=dict(scene.latlons), pairs=list(scene.pairs))
    fields.update(overrides)
    return PipelineConfig(**fields)


def _isolated_node_scene():
    """5 images on a line; node 2 is sandwiched between two edges that both fail (2
    correspondences -> cv2.findHomography raises inside match_pair), so it has no usable
    edge. A (1, 3) bypass keeps {3, 4} connected to {0, 1}, so node 2 is the only image
    without image evidence (without the bypass {3, 4} would be a separate component --
    fine for the new architecture, but it would test something else)."""
    scene = _scene(5)
    matcher = SceneMatcher(scene, fail_pairs={(1, 2), (2, 3)})
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]
    return scene, matcher, pairs


def _estimate_returning(global_transforms: GlobalTransforms, **overrides) -> GlobalPoseEstimate:
    """A GlobalPoseEstimate wrapping a hand-built GlobalTransforms, for tests that mock
    estimate_global_poses out."""
    fields = dict(
        global_transforms=global_transforms,
        failure_reason=None,
        node_failure_reasons={},
        heading_anchor_source="gps",
        pixels_per_meter=1.0,
        kappa=1.0,
        component_offsets_rad={},
        bound_hits=BoundHits(),
        skipped_edges={},
        term_rms={},
        irls_rounds=1,
        gps_lag=GpsLagEstimate(
            lag_m=2.3, status="estimated", reason=None, rounds=2, history_m=[0.0, 2.3, 2.3],
            trusted_nodes=3, uncorrected_nodes=set(),
        ),
    )
    fields.update(overrides)
    return GlobalPoseEstimate(**fields)


# --- tagged-image helpers, only for tests that mock estimate_global_poses --------------


def _tagged_image(index: int, size: int = 3) -> np.ndarray:
    """A tiny image array whose every pixel equals its own index -- lets
    _ScriptedMatcher identify the pair without any real image content."""
    return np.full((size, size, 3), index, dtype=np.uint8)


def _good_match_result(tx: float = 3.0, ty: float = 2.0) -> MatchResult:
    """5 well-spread points consistent with a pure translation (RANSAC keeps all 5)."""
    src = np.array([[10.0, 10.0], [50.0, 10.0], [10.0, 50.0], [50.0, 50.0], [30.0, 30.0]])
    return MatchResult(src_points=src, dst_points=src + np.array([tx, ty]), scores=None)


class _ScriptedMatcher:
    name = "scripted-test-matcher"

    def __init__(self, results_by_pair: dict[tuple[int, int], MatchResult]) -> None:
        self._results_by_pair = results_by_pair

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        return self._results_by_pair[(int(image_a.flat[0]), int(image_b.flat[0]))]


# ---------------------------------------------------------------------------
# A: happy path and the single-image special case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["gps", "gimbal", "none"])
def test_run_pipeline_all_images_well_matched_is_success(source: str) -> None:
    scene = _scene(6)
    gimbal = scene.gimbal_yaw_deg if source == "gimbal" else None

    mosaic, metrics_df = run_pipeline(
        scene.images, SceneMatcher(scene), _config(scene, heading_anchor_source=source, gimbal_yaw_deg=gimbal)
    )

    assert metrics_df["pipeline_status"][0] == "success"
    assert metrics_df["successful_image_count"][0] == 6
    assert metrics_df["failed_image_count"][0] == 0
    assert mosaic.ndim == 3 and mosaic.shape[0] > PIPELINE_IMAGE_SHAPE[0]


def test_run_pipeline_mosaic_is_the_north_up_ground_texture() -> None:
    """End to end against independent truth: each mosaic pixel shows the ground texture
    at the point the north-up frame puts there (x = ppm*E, y = -ppm*N, origin at node 0,
    shifted by the canvas origin computed from the TRUE image footprints). Checks the
    frame convention and the whole chain, not only the classification."""
    scene = _scene(6)
    mosaic, metrics_df = run_pipeline(scene.images, SceneMatcher(scene), _config(scene))
    assert metrics_df["pipeline_status"][0] == "success"

    ppm = PIPELINE_FOCAL_PX / ALTITUDE
    origin = scene.cameras[0]
    rows, cols = PIPELINE_IMAGE_SHAPE
    corners = []
    for camera in scene.cameras.values():
        # true pose: rotation by the compass yaw about the image centre, centre at the camera
        yaw = np.radians(camera.yaw_deg)
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        centre = ppm * np.array([camera.east_m - origin.east_m, -(camera.north_m - origin.north_m)])
        local = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]], float) - [cols / 2, rows / 2]
        corners.append(local @ R.T + centre)
    min_xy = np.min(np.vstack(corners), axis=0)

    v, u = np.nonzero(mosaic.sum(axis=2) > 0)
    x, y = u + min_xy[0], v + min_xy[1]
    expected = ground_texture(x / ppm + origin.east_m, -y / ppm + origin.north_m)
    interior = (u > 2) & (v > 2) & (u < mosaic.shape[1] - 3) & (v < mosaic.shape[0] - 3)
    error = np.abs(mosaic[v, u].astype(float) - expected)[interior]
    # Median, not mean. Measured when written: median 0.354, mean 3.51. The mean is
    # dominated by the 1-2 px band along every image edge (mean error 8.07 there):
    # bilinear warping blends the out-of-image black into edge pixels, which keep a small
    # blend weight (a warp/blend property, see CLAUDE.md). 2+ px from any edge the error
    # is 0.31-0.44, the single-image resampling floor (0.34). Against the same mosaic,
    # wrong frame conventions give medians of 29.5 (y = +ppm*N), 40.7 (x = -ppm*E), 14.2
    # (rotated 5 deg), 5.9 (1 px shift), 2.3 (0.5 px shift), 1.8 (scale off by 1%).
    assert np.median(error) < 1.0


def test_run_pipeline_single_image_is_output_as_is_with_nan_residual() -> None:
    """docs/task2.md §3.1: one input image that can be output is a success. Stage A-D
    (which needs pairs) is bypassed: no matching, no estimate_global_poses call; the
    GlobalTransforms is identity, converged, residual_error NaN, reference_index = the
    image's own index."""
    scene = _scene(1)
    image = scene.images[0]
    matcher = MagicMock()
    matcher.name = "never-called"

    with patch.object(pipeline_module, "estimate_global_poses") as mock_estimate, patch.object(
        pipeline_module, "evaluate_stitching_metrics", wraps=pipeline_module.evaluate_stitching_metrics
    ) as spy_metrics:
        mosaic, metrics_df = run_pipeline({7: image}, matcher, PipelineConfig())

    assert metrics_df["pipeline_status"][0] == "success"
    assert metrics_df["successful_image_count"][0] == 1
    assert np.array_equal(mosaic, image)
    assert matcher.match.call_count == 0
    assert mock_estimate.call_count == 0
    gt = spy_metrics.call_args.kwargs["global_transforms"]
    assert np.array_equal(gt.transforms[7], np.eye(3))
    assert gt.reference_index == 7
    assert gt.optimization_status == "converged"
    assert np.isnan(gt.residual_error)


# ---------------------------------------------------------------------------
# B: per-image classification (image evidence is required; metadata is not enough)
# ---------------------------------------------------------------------------


def test_run_pipeline_image_without_edge_or_gps_fails_but_others_succeed(caplog) -> None:
    scene, matcher, pairs = _isolated_node_scene()
    latlons = {k: v for k, v in scene.latlons.items() if k != 2}

    with caplog.at_level(logging.WARNING, logger="sea_mosaic.pipeline"):
        _mosaic, metrics_df = run_pipeline(scene.images, matcher, _config(scene, latlons=latlons, pairs=pairs))

    assert metrics_df["pipeline_status"][0] == "partial_success"
    assert metrics_df["successful_image_count"][0] == 4
    assert metrics_df["failed_image_indices"][0] == [2]
    assert "{2: 'no_determined_edge'}" in caplog.text


@pytest.mark.parametrize("source", ["gps", "gimbal"])
def test_run_pipeline_image_with_gps_but_no_edge_still_fails(source: str) -> None:
    """Expectation REVERSED from the old architecture, where a GPS anchor alone made an
    image count as successful. Now metadata alone -- GPS, and in gimbal mode also a
    heading -- is not image evidence, so node 2 fails either way."""
    scene, matcher, pairs = _isolated_node_scene()
    gimbal = scene.gimbal_yaw_deg if source == "gimbal" else None

    _mosaic, metrics_df = run_pipeline(
        scene.images, matcher, _config(scene, pairs=pairs, heading_anchor_source=source, gimbal_yaw_deg=gimbal)
    )

    assert metrics_df["pipeline_status"][0] == "partial_success"
    assert metrics_df["successful_image_count"][0] == 4
    assert metrics_df["failed_image_indices"][0] == [2]


def test_run_pipeline_mixed_failures_report_exactly_the_failed_images_and_reasons(caplog) -> None:
    """Three different ways to fail in one run: node 2 sandwiched between two failed edges,
    node 5 a dead end whose only edge fails (both "no_determined_edge"), node 4 with a good
    edge but no GPS ("unlocated")."""
    scene = pipeline_scene(_line_cameras(6))
    matcher = SceneMatcher(scene, fail_pairs={(1, 2), (2, 3), (4, 5)})
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 3)]
    latlons = {k: v for k, v in scene.latlons.items() if k != 4}

    with caplog.at_level(logging.WARNING, logger="sea_mosaic.pipeline"):
        _mosaic, metrics_df = run_pipeline(scene.images, matcher, _config(scene, latlons=latlons, pairs=pairs))

    assert metrics_df["pipeline_status"][0] == "partial_success"
    assert metrics_df["successful_image_count"][0] == 3
    assert metrics_df["failed_image_indices"][0] == [2, 4, 5]
    assert "{2: 'no_determined_edge', 4: 'unlocated', 5: 'no_determined_edge'}" in caplog.text


# ---------------------------------------------------------------------------
# C: whole-run failure paths
# ---------------------------------------------------------------------------


def test_run_pipeline_no_usable_edge_is_failed(caplog) -> None:
    """Replaces the old "only the reference image survives" test: that state no longer
    exists. There is no reference image, and a pose needs a Stage C alignment edge whose
    two endpoints both get poses, so a multi-image run poses either 0 or >= 2 images."""
    scene = _scene(3)
    matcher = SceneMatcher(scene, fail_pairs=set(scene.pairs))

    with caplog.at_level(logging.WARNING, logger="sea_mosaic.pipeline"):
        mosaic, metrics_df = run_pipeline(scene.images, matcher, _config(scene))

    assert metrics_df["pipeline_status"][0] == "failed"
    assert metrics_df["successful_image_count"][0] == 0
    assert metrics_df["failed_image_indices"][0] == [0, 1, 2]
    assert mosaic.shape == (0, 0, 3)
    assert "pixels_per_meter_unavailable" in caplog.text


def test_run_pipeline_without_gps_is_failed_and_logs_why(caplog) -> None:
    """Replaces the old "computes pixels_per_meter from altitude itself" test: there is no
    altitude any more. Without GPS no image can be placed (agreed design), and the reason
    is logged because metrics_df's columns are fixed."""
    scene = _scene(4)

    with caplog.at_level(logging.WARNING, logger="sea_mosaic.pipeline"):
        mosaic, metrics_df = run_pipeline(scene.images, SceneMatcher(scene), PipelineConfig(pairs=scene.pairs))

    assert metrics_df["pipeline_status"][0] == "failed"
    assert metrics_df["successful_image_count"][0] == 0
    assert mosaic.shape == (0, 0, 3)
    assert "global pose estimation: no_gps" in caplog.text
    # matching still ran: its metrics stay meaningful without GPS
    assert metrics_df["inlier_count"][0] > 0


def test_run_pipeline_not_converged_forces_failed_and_skips_warp_blend() -> None:
    """A Stage D not_converged status must not be trusted downstream: whole-run failure,
    warp/blend never called, empty (0, 0, 3) output (changed from the old
    zeros_like(reference image): there is no reference image any more)."""
    images = {i: _tagged_image(i) for i in range(2)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result()})
    not_converged = GlobalTransforms(
        transforms={0: np.eye(3), 1: np.eye(3)},
        reference_index=None,
        optimization_status="not_converged",
        residual_error=np.nan,
    )
    estimate = _estimate_returning(not_converged, failure_reason="refinement_not_converged")

    with patch.object(pipeline_module, "estimate_global_poses", return_value=estimate):
        with patch.object(pipeline_module, "warp_images_streaming") as mock_warp:
            with patch.object(pipeline_module, "blend_images_streaming") as mock_blend:
                mosaic, metrics_df = run_pipeline(images, matcher, PipelineConfig())

    assert metrics_df["pipeline_status"][0] == "failed"
    assert metrics_df["successful_image_count"][0] == 0
    assert metrics_df["failed_image_indices"][0] == [0, 1]
    assert mosaic.shape == (0, 0, 3)
    assert mock_warp.call_count == 0
    assert mock_blend.call_count == 0


def test_run_pipeline_empty_input_is_failed_without_crashing() -> None:
    _mosaic, metrics_df = run_pipeline({}, _ScriptedMatcher({}), PipelineConfig())

    assert metrics_df["pipeline_status"][0] == "failed"
    assert metrics_df["input_image_count"][0] == 0
    assert np.isnan(metrics_df["stitch_success_rate"][0])
    assert np.isnan(metrics_df["avg_processing_time_per_image_sec"][0])


# ---------------------------------------------------------------------------
# D: error isolation
# ---------------------------------------------------------------------------


def test_run_pipeline_estimate_stage_pair_exception_is_isolated_not_fatal() -> None:
    """(1, 2) gets 2 correspondences and makes match_pair raise a REAL cv2.error --
    run_pipeline must not crash; that edge is excluded, node 2 (no other edge) fails,
    nodes 0 and 1 succeed."""
    scene = _scene(3)
    matcher = SceneMatcher(scene, fail_pairs={(1, 2)})

    _mosaic, metrics_df = run_pipeline(scene.images, matcher, _config(scene, pairs=[(0, 1), (1, 2)]))

    assert metrics_df["pipeline_status"][0] == "partial_success"
    assert metrics_df["successful_image_count"][0] == 2
    assert metrics_df["failed_image_indices"][0] == [2]


def _poisoned_transforms(bad: np.ndarray) -> GlobalTransforms:
    return GlobalTransforms(
        transforms={0: np.eye(3), 1: bad, 2: np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
        reference_index=None,
        optimization_status="converged",
        residual_error=np.nan,
    )


def test_run_pipeline_nonfinite_transform_isolated_without_poisoning_other_images() -> None:
    """compute_canvas_size raises on NaN/inf; if the canvas were sized from ALL transforms
    at once, node 1's NaN would poison it for nodes 0 and 2 too. The up-front
    np.all(np.isfinite(transform)) filter excludes node 1 before canvas sizing."""
    images = {k: _tagged_image(k, size=5) for k in range(3)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
    nan_transform = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    with patch.object(
        pipeline_module, "estimate_global_poses", return_value=_estimate_returning(_poisoned_transforms(nan_transform))
    ):
        mosaic, metrics_df = run_pipeline(images, matcher, PipelineConfig())

    assert metrics_df["failed_image_indices"][0] == [1]
    assert metrics_df["successful_image_count"][0] == 2
    assert np.all(np.isfinite(mosaic.astype(np.float64)))
    assert mosaic.shape[0] <= 20 and mosaic.shape[1] <= 20


def test_run_pipeline_finite_degenerate_transform_needs_no_special_isolation_mechanism() -> None:
    """cv2.warpPerspective never raises for a finite-but-singular (scale=0) transform; it
    produces an all-black mask, which the ordinary "mask must have >= 1 nonzero pixel"
    rule already classifies as failed. If this starts failing because cv2 begins to
    raise instead, that is the interesting finding, not a run_pipeline bug."""
    images = {k: _tagged_image(k, size=5) for k in range(3)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
    singular = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 1.0]])

    with patch.object(
        pipeline_module, "estimate_global_poses", return_value=_estimate_returning(_poisoned_transforms(singular))
    ):
        _mosaic, metrics_df = run_pipeline(images, matcher, PipelineConfig())

    assert metrics_df["failed_image_indices"][0] == [1]
    assert metrics_df["successful_image_count"][0] == 2


# ---------------------------------------------------------------------------
# E: wiring correctness
# ---------------------------------------------------------------------------


def test_run_pipeline_metrics_df_matches_computed_classification() -> None:
    scene, matcher, pairs = _isolated_node_scene()

    _mosaic, metrics_df = run_pipeline(scene.images, matcher, _config(scene, pairs=pairs))

    assert metrics_df["input_image_count"][0] == 5
    assert metrics_df["stitch_success_rate"][0] == pytest.approx(metrics_df["successful_image_count"][0] / 5)
    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 1


def test_run_pipeline_forwards_heading_configuration_to_estimate_global_poses() -> None:
    """Replaces the old "degrades gracefully without gimbal_yaw" test. The default source
    is "gps"; heading_anchor_source and gimbal_yaw_deg reach estimate_global_poses as
    given, together with EXIF lat/lon and every image's shape."""
    scene = _scene(4)
    real = pipeline_module.estimate_global_poses

    with patch.object(pipeline_module, "estimate_global_poses", wraps=real) as spy:
        run_pipeline(scene.images, SceneMatcher(scene), _config(scene))
        run_pipeline(
            scene.images,
            SceneMatcher(scene),
            _config(scene, heading_anchor_source="gimbal", gimbal_yaw_deg=scene.gimbal_yaw_deg),
        )

    default_call, gimbal_call = spy.call_args_list
    assert default_call.kwargs == {"heading_anchor_source": "gps", "gimbal_yaw_deg": None, "capture_times_s": None}
    assert gimbal_call.kwargs == {
        "heading_anchor_source": "gimbal",
        "gimbal_yaw_deg": scene.gimbal_yaw_deg,
        "capture_times_s": None,
    }
    _pairs, shapes, latlons = default_call.args
    assert shapes == {k: image.shape for k, image in scene.images.items()}
    assert latlons == scene.latlons


def test_run_pipeline_forwards_capture_times_to_estimate_global_poses() -> None:
    """capture_times_s (gps_lag.load_exif_capture_times) reaches estimate_global_poses,
    which corrects the GPS recording lag at the GPS input for every stage."""
    scene = _scene(4)
    times = {k: 1000.0 + 2.5 * k for k in scene.images}
    real = pipeline_module.estimate_global_poses

    with patch.object(pipeline_module, "estimate_global_poses", wraps=real) as spy:
        run_pipeline(scene.images, SceneMatcher(scene), _config(scene, capture_times_s=times))

    assert spy.call_args.kwargs["capture_times_s"] == times


@pytest.mark.parametrize(
    "lag, expected",
    [
        (
            GpsLagEstimate(0.0, "not_estimable", "no_trusted_edges", 0, [0.0], 0, {0, 1, 2}),
            "GPS lag not corrected: not_estimable (no_trusted_edges)",
        ),
        (
            GpsLagEstimate(0.0, "no_capture_times", None, 0, [0.0], 0, {0, 1, 2}),
            "GPS lag not corrected: no_capture_times",
        ),
    ],
)
def test_run_pipeline_logs_when_gps_lag_is_not_corrected(caplog, lag, expected) -> None:
    """The lag diagnostics must reach the run's output even though metrics_df has no
    column for them (its columns are fixed by docs/task2.md)."""
    images = {k: _tagged_image(k, size=5) for k in range(3)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
    shift = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    gt = GlobalTransforms({0: np.eye(3), 1: shift, 2: shift @ shift}, None, "converged", np.nan)

    with caplog.at_level(logging.WARNING, logger="sea_mosaic.pipeline"):
        with patch.object(pipeline_module, "estimate_global_poses", return_value=_estimate_returning(gt, gps_lag=lag)):
            run_pipeline(images, matcher, PipelineConfig())

    assert expected in caplog.text


def test_run_pipeline_logs_nodes_left_uncorrected(caplog) -> None:
    images = {k: _tagged_image(k, size=5) for k in range(3)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
    shift = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    gt = GlobalTransforms({0: np.eye(3), 1: shift, 2: shift @ shift}, None, "converged", np.nan)
    lag = GpsLagEstimate(2.3, "estimated", None, 2, [0.0, 2.3, 2.3], 3, {2})

    with caplog.at_level(logging.WARNING, logger="sea_mosaic.pipeline"):
        with patch.object(pipeline_module, "estimate_global_poses", return_value=_estimate_returning(gt, gps_lag=lag)):
            run_pipeline(images, matcher, PipelineConfig())

    assert "GPS lag not applied to images without a travel direction: [2]" in caplog.text
    assert "GPS lag not corrected" not in caplog.text


def test_run_pipeline_gimbal_source_without_gimbal_data_raises() -> None:
    """A configuration error, not a data problem: raised (by estimate_global_poses), not
    reported as a failed run."""
    scene = _scene(3)
    with pytest.raises(ValueError, match="gimbal_yaw_deg"):
        run_pipeline(scene.images, SceneMatcher(scene), _config(scene, heading_anchor_source="gimbal"))


def test_run_pipeline_logs_stage_d_guard_rail_hits(caplog) -> None:
    """Guard rails only catch gross failure; every hit is a diagnostic signal and must be
    visible even though metrics_df has no column for it."""
    images = {k: _tagged_image(k, size=5) for k in range(3)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
    shift = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    gt = GlobalTransforms({0: np.eye(3), 1: shift, 2: shift @ shift}, None, "converged", np.nan)
    estimate = _estimate_returning(gt, bound_hits=BoundHits(position={1}, kappa=True))

    with caplog.at_level(logging.WARNING, logger="sea_mosaic.pipeline"):
        with patch.object(pipeline_module, "estimate_global_poses", return_value=estimate):
            run_pipeline(images, matcher, PipelineConfig())

    assert "Stage D guard rail hit: positions [1], kappa True" in caplog.text


# ---------------------------------------------------------------------------
# F: golden-fixture regression
# ---------------------------------------------------------------------------
#
# These compare run_pipeline's output against committed fixtures
# (tests/fixtures/golden_pipeline/, written by capture_golden.py) and only ever READ them
# (an auto-regenerating "golden" would launder a regression into a new baseline). The
# baseline was deliberately reset when run_pipeline switched to Stage A->D: the old
# scenarios had no GPS and would all be "failed" now (see CLAUDE.md).

_GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden_pipeline"
_TIMING_COLUMNS = {"total_processing_time_sec", "avg_processing_time_per_image_sec"}


def _load_golden(name: str) -> tuple[np.ndarray, pd.DataFrame]:
    mosaic = np.load(_GOLDEN_DIR / f"{name}.npy")
    with open(_GOLDEN_DIR / f"{name}_metrics.pkl", "rb") as f:
        metrics_df = pickle.load(f)
    return mosaic, metrics_df


def _assert_metrics_df_matches_golden(new_df: pd.DataFrame, golden_df: pd.DataFrame) -> None:
    """Excludes wall-clock timing columns; everything else with a tight rtol as a
    documented safety margin -- a mismatch is a real difference to investigate."""
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
            assert new_val == pytest.approx(golden_val, rel=1e-9, abs=1e-12), f"{col}: {new_val!r} != {golden_val!r}"
        else:
            assert new_val == golden_val, f"{col}: {new_val!r} != {golden_val!r}"


def golden_all_images_well_matched():
    scene = _scene(6)
    return scene.images, SceneMatcher(scene), _config(scene)


def golden_isolated_node_without_edge():
    scene, matcher, pairs = _isolated_node_scene()
    return scene.images, matcher, _config(scene, pairs=pairs)


def golden_nonfinite_transform_isolated():
    images = {k: _tagged_image(k, size=5) for k in range(3)}
    matcher = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
    nan_transform = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return images, matcher, PipelineConfig(), _estimate_returning(_poisoned_transforms(nan_transform))


def test_run_pipeline_matches_golden_all_images_well_matched() -> None:
    images, matcher, config = golden_all_images_well_matched()
    golden_mosaic, golden_metrics = _load_golden("all_images_well_matched")

    mosaic, metrics_df = run_pipeline(images, matcher, config)

    assert np.array_equal(mosaic, golden_mosaic)
    _assert_metrics_df_matches_golden(metrics_df, golden_metrics)


def test_run_pipeline_matches_golden_isolated_node_without_edge() -> None:
    images, matcher, config = golden_isolated_node_without_edge()
    golden_mosaic, golden_metrics = _load_golden("isolated_node_without_edge")

    mosaic, metrics_df = run_pipeline(images, matcher, config)

    assert np.array_equal(mosaic, golden_mosaic)
    _assert_metrics_df_matches_golden(metrics_df, golden_metrics)


def test_run_pipeline_matches_golden_nonfinite_transform_isolated() -> None:
    images, matcher, config, estimate = golden_nonfinite_transform_isolated()
    golden_mosaic, golden_metrics = _load_golden("nonfinite_transform_isolated")

    with patch.object(pipeline_module, "estimate_global_poses", return_value=estimate):
        mosaic, metrics_df = run_pipeline(images, matcher, config)

    assert np.array_equal(mosaic, golden_mosaic)
    _assert_metrics_df_matches_golden(metrics_df, golden_metrics)


# ---------------------------------------------------------------------------
# G: warp/blend/seam_error memory scaling (pose estimation mocked out)
# ---------------------------------------------------------------------------


def _synthetic_pipeline_scenario(
    n: int, total_span: float = 200.0, image_height: int = 10
) -> tuple[dict[int, np.ndarray], _ScriptedMatcher]:
    """n images along a FIXED-length span (image width shrinks as n grows) so canvas_size
    stays roughly constant as n grows."""
    step = total_span / n
    image_width = max(2, int(round(step * 2)))
    images = {i: np.full((image_height, image_width, 3), (i % 200) + 1, dtype=np.uint8) for i in range(n)}
    matcher = _ScriptedMatcher({(i, i + 1): _good_match_result(tx=step, ty=0.0) for i in range(n - 1)})
    return images, matcher


def _hand_fed_estimate(n: int, total_span: float = 200.0) -> GlobalPoseEstimate:
    """A cheap GlobalTransforms matching _synthetic_pipeline_scenario's geometry exactly
    (translation by i*step), so the test measures only warp/blend/seam_error."""
    step = total_span / n
    return _estimate_returning(
        GlobalTransforms(
            transforms={i: np.array([[1.0, 0.0, i * step], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) for i in range(n)},
            reference_index=None,
            optimization_status="converged",
            residual_error=np.nan,
        )
    )


def _peak_traced_bytes_for_pipeline(n: int) -> int:
    images, matcher = _synthetic_pipeline_scenario(n)
    estimate = _hand_fed_estimate(n)
    tracemalloc.start()
    try:
        tracemalloc.clear_traces()
        with patch.object(pipeline_module, "estimate_global_poses", return_value=estimate):
            run_pipeline(images, matcher, PipelineConfig())
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_run_pipeline_warp_blend_seam_error_peak_memory_is_flat_not_linear_in_n() -> None:
    """Scope note: verifies ONLY warp_images_streaming/blend_images_streaming/
    compute_seam_error_streaming's memory behaviour through run_pipeline's real
    orchestration. Pose estimation is mocked out (formerly compose_global_transforms,
    now estimate_global_poses); its own memory cost is a separate question (see
    CLAUDE.md). When this test was introduced it was verified (via `git stash` of
    pipeline.py) to fail against the pre-streaming run_pipeline with the same mock in
    place: peak_10=546548 bytes, peak_100=2831789 bytes (~5.18x)."""
    peak_10 = _peak_traced_bytes_for_pipeline(10)
    peak_100 = _peak_traced_bytes_for_pipeline(100)

    ratio = peak_100 / peak_10
    assert ratio < 2.0, (
        f"peak traced memory scaled {ratio:.2f}x going from N=10 to N=100 "
        f"(peak_10={peak_10} bytes, peak_100={peak_100} bytes) -- expected roughly flat, not O(N)"
    )
