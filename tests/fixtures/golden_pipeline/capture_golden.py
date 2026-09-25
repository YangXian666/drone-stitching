"""Golden-fixture capture for run_pipeline's streaming-accumulator regression test.

Run ONCE, by hand, against pipeline.py BEFORE the streaming refactor:

    poetry run python3 tests/fixtures/golden_pipeline/capture_golden.py

RECAPTURE HISTORY -- isolated_node_without_gps_anchor was recaptured once, AFTER
run_pipeline was wired to the streaming path, deliberately deviating from "goldens are
only ever captured against the pre-refactor code": its original (eager-blend) golden
differed from the streaming implementation's real output at exactly 1 of 19,530 values
(pixel (row=50, col=27), all 3 channels: eager gave [2,2,2], streaming gives [3,3,3]).
Investigated directly, not assumed: the true weighted average at that pixel is
~2.50000015 in BOTH formulations (they agree to 7 significant figures) -- it sits almost
exactly on the float rounding boundary, and blend_images (eager) explicitly downcasts
its normalized alpha to float32 before multiplying, while blend_images_streaming never
computes a normalized ratio at all (it divides once, in float64, at the very end) -- two
equally legitimate floating-point paths that happen to land on opposite sides of .5 for
this one pixel. This is exactly the one-ULP rounding-boundary case documented (and
deliberately tested for, adversarially) when blend_images_streaming was designed --
this is its first naturally-occurring (not deliberately constructed) instance. The
streaming path is what run_pipeline actually calls from now on, so its arithmetic is the
correct baseline going forward, not the eager path being retired -- recapturing this one
golden reflects that, it is not masking a real regression. The other two golden files
(all_images_well_matched, nonfinite_transform_isolated) matched the streaming
implementation bit-exact on first try and were left untouched.

Writes <scenario>.npy (mosaic, via np.save -- lossless, exact roundtrip; a PNG/JPEG
would be wrong here, it's lossy) and <scenario>_metrics.pkl (metrics_df, via pickle --
preserves the DataFrame exactly, including the list-valued failed_image_indices column
and NaN entries; this is an internal test fixture read back by the same test suite, not
a user-facing artifact, so pickle's usual portability/security caveats don't apply).

These files are committed to the repo and read-only from every test's perspective --
tests/test_pipeline.py's golden-comparison test only ever loads them, never regenerates
them (an auto-regenerating "golden" file would silently launder a real regression into a
new baseline instead of catching it).

Scenarios chosen deliberately, not just the happy path -- see CLAUDE.md/this session's
design discussion: streaming requires restructuring run_pipeline's per-image
success/failure classification (currently a post-hoc filter over a fully materialized
warped_masks dict) into an inline filter over the warp stream, so the golden set has to
actually exercise that classification logic, not just confirm a trivial all-succeed case
still produces the same pixels.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))  # for `tests` import
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tests.test_pipeline import (  # noqa: E402
    _base_config,
    _good_match_result,
    _isolated_node_scenario,
    _ScriptedMatcher,
    _tagged_image,
)
from sea_mosaic.pipeline import run_pipeline  # noqa: E402
from sea_mosaic.types import GlobalTransforms  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent


def _capture(name: str, images, matcher, config) -> None:
    mosaic, metrics_df = run_pipeline(images, matcher, config)
    np.save(OUTPUT_DIR / f"{name}.npy", mosaic)
    with open(OUTPUT_DIR / f"{name}_metrics.pkl", "wb") as f:
        pickle.dump(metrics_df, f)
    print(f"captured {name}: mosaic.shape={mosaic.shape}, "
          f"pipeline_status={metrics_df['pipeline_status'][0]}, "
          f"successful_image_count={metrics_df['successful_image_count'][0]}")


def main() -> None:
    # Optional argv filter, e.g. `capture_golden.py isolated_node_without_gps_anchor`,
    # for a targeted recapture of one scenario without touching the others -- used for
    # exactly that when isolated_node_without_gps_anchor was recaptured (see this file's
    # module docstring, RECAPTURE HISTORY).
    only = set(sys.argv[1:]) or None

    # 1. all_images_well_matched -- happy path baseline
    if only is None or "all_images_well_matched" in only:
        _capture(
            "all_images_well_matched",
            {i: _tagged_image(i) for i in range(3)},
            _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()}),
            _base_config(),
        )

    # 2. isolated_node_without_gps_anchor -- exercises per-image success/failure
    # classification (the part of run_pipeline that has to move from a post-hoc filter
    # over a fully materialized dict to an inline filter over the warp stream)
    if only is None or "isolated_node_without_gps_anchor" in only:
        images, matcher = _isolated_node_scenario()
        _capture("isolated_node_without_gps_anchor", images, matcher, _base_config())

    # 3. nonfinite_transform_isolated -- pre-warp finite-transform filter
    # (compute_canvas_size raises on NaN if not excluded first) combined with
    # mask-emptiness classification, distinct from scenario 2's edge/anchor-based
    # classification. Reuses test_pipeline.py's exact mocked-poisoned-GlobalTransforms
    # scenario (compose_global_transforms mocked identically here and in the comparison
    # test, so both capture and comparison exercise the SAME non-reproducible-by-real-
    # optimization NaN transform).
    if only is None or "nonfinite_transform_isolated" in only:
        images3 = {0: _tagged_image(0, size=5), 1: _tagged_image(1, size=5), 2: _tagged_image(2, size=5)}
        matcher3 = _ScriptedMatcher({(0, 1): _good_match_result(), (1, 2): _good_match_result()})
        nan_transform = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        poisoned = GlobalTransforms(
            transforms={
                0: np.eye(3),
                1: nan_transform,
                2: np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            },
            reference_index=0, optimization_status="converged", residual_error=0.1,
        )
        with patch("sea_mosaic.pipeline.compose_global_transforms", return_value=poisoned):
            _capture("nonfinite_transform_isolated", images3, matcher3, _base_config())

    print("done")


if __name__ == "__main__":
    main()
