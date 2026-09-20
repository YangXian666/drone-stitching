"""Unit tests for sea_mosaic.estimate.sequential_pairs.

match_pair/estimate_all_pairs are still stubs (SIFT/RANSAC implementation is separate
follow-up work); only the pair-selection helper is covered here.
"""

from __future__ import annotations

import numpy as np

from sea_mosaic.estimate import sequential_pairs


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
