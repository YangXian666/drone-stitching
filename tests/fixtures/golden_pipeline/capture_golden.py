"""Golden-fixture capture for tests/test_pipeline.py's run_pipeline regression tests.

Usage (scenario names are required -- there is no "capture everything" default, so a
fixture is only ever rewritten on purpose):

    poetry run python3 tests/fixtures/golden_pipeline/capture_golden.py SCENARIO [SCENARIO ...]

Scenarios are defined once, in tests/test_pipeline.py (golden_<name> functions), and used
both here and by the comparison tests, so capture and comparison cannot drift apart.

REQUIRED AFTER EVERY RECAPTURE -- determinism check. A golden is only meaningful if the
pipeline reproduces it bit for bit. Recapture in at least 5 separate, fresh processes and
compare sha256 of the .npy files; every run must give the same hash:

    for i in 1 2 3 4 5; do
        poetry run python3 tests/fixtures/golden_pipeline/capture_golden.py SCENARIO
        sha256sum tests/fixtures/golden_pipeline/SCENARIO.npy
    done

(The _metrics.pkl files legitimately differ: they contain wall-clock timing columns.) Use
fresh processes, not repeated calls in one process, and sha256, not Python's hash()
(randomized per process). This check is what exposed the non-deterministic OpenCV IPP
distanceTransform path on 2026-09-27 (see CLAUDE.md): a single capture looked fine.

Writes <scenario>.npy (mosaic, np.save -- lossless exact roundtrip; PNG/JPEG would be
lossy) and <scenario>_metrics.pkl (metrics_df, pickle -- keeps the list-valued
failed_image_indices column and NaN entries exactly; an internal fixture read back by the
same test suite, not a user-facing artifact).

These files are committed and read-only from every test's perspective: the tests only
load them, never regenerate them.

HISTORY
- Streaming refactor (earlier): goldens first captured against the pre-streaming
  run_pipeline. isolated_node_without_gps_anchor was recaptured once afterwards because
  of a one-ULP rounding-boundary difference between eager and streaming blending (one
  pixel, true value ~2.50000015; the streaming path is the correct baseline).
- Stage A->D switch (2026-09-27): baseline deliberately reset. run_pipeline's behaviour
  changed on purpose (global poses from estimate_global_poses in a north-up frame; image
  evidence required for success), and the old scenarios had no GPS, so they would all be
  "failed" now. Recaptured with new, GPS-consistent synthetic scenes:
  all_images_well_matched (replaced) and isolated_node_without_edge (new name; the old
  isolated_node_without_gps_anchor files are obsolete). nonfinite_transform_isolated was
  NOT recaptured: it mocks pose estimation out and its old golden still matches bit for
  bit, which shows that everything after pose estimation (warp, blend, classification,
  metrics) is unchanged -- not that the old and new pose estimation agree.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "tests"))  # test_pipeline, synthetic_camera
sys.path.insert(0, str(_REPO_ROOT / "src"))

import test_pipeline as tp  # noqa: E402
import sea_mosaic.pipeline as pipeline_module  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent


def _write(name: str, mosaic: np.ndarray, metrics_df) -> None:
    np.save(OUTPUT_DIR / f"{name}.npy", mosaic)
    with open(OUTPUT_DIR / f"{name}_metrics.pkl", "wb") as f:
        pickle.dump(metrics_df, f)
    print(
        f"captured {name}: mosaic.shape={mosaic.shape}, "
        f"pipeline_status={metrics_df['pipeline_status'][0]}, "
        f"successful_image_count={metrics_df['successful_image_count'][0]}, "
        f"failed_image_indices={metrics_df['failed_image_indices'][0]}"
    )


def capture_all_images_well_matched() -> None:
    images, matcher, config = tp.golden_all_images_well_matched()
    _write("all_images_well_matched", *pipeline_module.run_pipeline(images, matcher, config))


def capture_isolated_node_without_edge() -> None:
    images, matcher, config = tp.golden_isolated_node_without_edge()
    _write("isolated_node_without_edge", *pipeline_module.run_pipeline(images, matcher, config))


def capture_nonfinite_transform_isolated() -> None:
    images, matcher, config, estimate = tp.golden_nonfinite_transform_isolated()
    with patch.object(pipeline_module, "estimate_global_poses", return_value=estimate):
        _write("nonfinite_transform_isolated", *pipeline_module.run_pipeline(images, matcher, config))


SCENARIOS = {
    "all_images_well_matched": capture_all_images_well_matched,
    "isolated_node_without_edge": capture_isolated_node_without_edge,
    "nonfinite_transform_isolated": capture_nonfinite_transform_isolated,
}


def main() -> None:
    names = sys.argv[1:]
    unknown = [n for n in names if n not in SCENARIOS]
    if not names or unknown:
        sys.exit(f"usage: capture_golden.py SCENARIO [SCENARIO ...]; scenarios: {', '.join(SCENARIOS)}"
                 + (f"; unknown: {unknown}" if unknown else ""))
    for name in names:
        SCENARIOS[name]()


if __name__ == "__main__":
    main()
