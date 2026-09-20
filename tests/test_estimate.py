"""Unit tests for sea_mosaic.estimate.

match_pair/estimate_all_pairs are still stubs (SIFT/RANSAC implementation is separate
follow-up work) — the tests below are written against those stub signatures and are
expected to fail until they are implemented, per this repo's test-first convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from sea_mosaic.estimate import estimate_all_pairs, match_pair, sequential_pairs
from sea_mosaic.matcher import MatchResult


def test_sequential_pairs_contiguous_zero_based_keys() -> None:
    images = {i: np.zeros((2, 2)) for i in range(5)}

    pairs = sequential_pairs(images)

    assert pairs == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_sequential_pairs_non_contiguous_keys_sorted_by_value() -> None:
    images = {352: np.zeros((2, 2)), 355: np.zeros((2, 2)), 353: np.zeros((2, 2))}

    pairs = sequential_pairs(images)

    assert pairs == [(352, 353), (353, 355)]


def test_sequential_pairs_length_is_n_minus_one() -> None:
    images = {i: np.zeros((2, 2)) for i in range(10)}

    pairs = sequential_pairs(images)

    assert len(pairs) == len(images) - 1


def test_sequential_pairs_single_image_returns_empty() -> None:
    images = {0: np.zeros((2, 2))}

    assert sequential_pairs(images) == []


def test_sequential_pairs_empty_images_returns_empty() -> None:
    assert sequential_pairs({}) == []


class _FixedMatcher:
    """Matcher test double that ignores image content and returns a pre-built MatchResult."""

    name = "fixed-test-matcher"

    def __init__(self, match_result: MatchResult) -> None:
        self._match_result = match_result

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        return self._match_result


def _apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points via homogeneous coordinates (no cv2 needed)."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed = homogeneous @ homography.T
    return transformed[:, :2] / transformed[:, 2:3]


_TRUE_HOMOGRAPHY = np.array(
    [
        [1.05, 0.03, 12.0],
        [-0.02, 0.98, -6.0],
        [0.0002, -0.0001, 1.0],
    ],
    dtype=np.float64,
)

_DUMMY_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)


def _grid_points(n_per_axis: int = 6) -> np.ndarray:
    xs = np.linspace(0, 500, n_per_axis)
    ys = np.linspace(0, 400, n_per_axis)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def test_match_pair_recovers_known_homography_without_outliers() -> None:
    src_points = _grid_points()
    dst_points = _apply_homography(src_points, _TRUE_HOMOGRAPHY)
    matcher = _FixedMatcher(MatchResult(src_points=src_points, dst_points=dst_points))

    result = match_pair(matcher, _DUMMY_IMAGE, _DUMMY_IMAGE, src_index=0, dst_index=1)

    normalized_estimated = result.homography / result.homography[2, 2]
    normalized_true = _TRUE_HOMOGRAPHY / _TRUE_HOMOGRAPHY[2, 2]
    assert normalized_estimated == pytest.approx(normalized_true, abs=1e-3)
    assert result.inlier_mask.all()
    assert result.match_count == len(src_points)
    assert result.inlier_count == len(src_points)


def _mixed_inlier_outlier_match_result() -> tuple[MatchResult, np.ndarray]:
    """30 correspondences that satisfy _TRUE_HOMOGRAPHY, plus 6 hand-placed outliers
    whose dst_points are shifted far enough to violate it. Returns the MatchResult and
    the ground-truth boolean mask (True = should be marked inlier)."""
    src_points = _grid_points()
    dst_points = _apply_homography(src_points, _TRUE_HOMOGRAPHY).copy()

    outlier_indices = [0, 5, 10, 15, 20, 25]
    dst_points[outlier_indices] += 80.0  # far outside the default ransac_threshold=3.0

    expected_inlier_mask = np.ones(len(src_points), dtype=bool)
    expected_inlier_mask[outlier_indices] = False

    return MatchResult(src_points=src_points, dst_points=dst_points), expected_inlier_mask


def test_match_pair_inlier_mask_flags_correct_correspondences() -> None:
    match_result, expected_inlier_mask = _mixed_inlier_outlier_match_result()
    matcher = _FixedMatcher(match_result)

    result = match_pair(matcher, _DUMMY_IMAGE, _DUMMY_IMAGE, src_index=0, dst_index=1)

    assert np.array_equal(result.inlier_mask, expected_inlier_mask)
    assert result.match_count == len(match_result.src_points)
    assert result.inlier_count == int(expected_inlier_mask.sum())


def test_match_pair_ransac_excludes_outliers_from_homography_estimate() -> None:
    match_result, _ = _mixed_inlier_outlier_match_result()
    matcher = _FixedMatcher(match_result)

    result = match_pair(matcher, _DUMMY_IMAGE, _DUMMY_IMAGE, src_index=0, dst_index=1)

    normalized_estimated = result.homography / result.homography[2, 2]
    normalized_true = _TRUE_HOMOGRAPHY / _TRUE_HOMOGRAPHY[2, 2]
    assert normalized_estimated == pytest.approx(normalized_true, abs=1e-2)


def test_estimate_all_pairs_respects_explicit_pairs_and_labels_indices() -> None:
    images = {0: _DUMMY_IMAGE, 1: _DUMMY_IMAGE, 2: _DUMMY_IMAGE}
    src_points = _grid_points()
    dst_points = _apply_homography(src_points, _TRUE_HOMOGRAPHY)
    matcher = _FixedMatcher(MatchResult(src_points=src_points, dst_points=dst_points))

    results = estimate_all_pairs(images, matcher, pairs=[(0, 1), (1, 2)])

    assert [(r.src_index, r.dst_index) for r in results] == [(0, 1), (1, 2)]


def test_estimate_all_pairs_defaults_to_exhaustive_all_pairs() -> None:
    images = {0: _DUMMY_IMAGE, 1: _DUMMY_IMAGE, 2: _DUMMY_IMAGE, 3: _DUMMY_IMAGE}
    src_points = _grid_points()
    dst_points = _apply_homography(src_points, _TRUE_HOMOGRAPHY)
    matcher = _FixedMatcher(MatchResult(src_points=src_points, dst_points=dst_points))

    results = estimate_all_pairs(images, matcher, pairs=None)

    assert sorted((r.src_index, r.dst_index) for r in results) == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    ]
