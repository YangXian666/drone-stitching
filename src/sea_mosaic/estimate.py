"""Stage 1 (estimate): feature matching + RANSAC geometric verification per image pair.

Uses an injected Matcher (see matcher.py) — never a hardcoded matcher implementation.
"""

from __future__ import annotations

import numpy as np

from sea_mosaic.matcher import Matcher
from sea_mosaic.types import PairResult


def sequential_pairs(images: dict[int, np.ndarray]) -> list[tuple[int, int]]:
    """Return consecutive-neighbor pairs (i, i+1) in ascending key order.

    Pairing is based on sorted dict-key order, not on key values being
    contiguous — image indices need not be 0..N-1 or gap-free. Intended as an
    explicit, opt-in strategy (e.g. for a linear flight-sequence smoke test),
    not the default behind estimate_all_pairs's pairs=None (which stays
    exhaustive all-pairs).
    """
    sorted_keys = sorted(images)
    return list(zip(sorted_keys, sorted_keys[1:]))


def match_pair(
    matcher: Matcher,
    image_a: np.ndarray,
    image_b: np.ndarray,
    src_index: int,
    dst_index: int,
    ransac_threshold: float = 3.0,
) -> PairResult:
    """Match one image pair with the given matcher and estimate a RANSAC homography,
    returning a PairResult."""
    ...


def estimate_all_pairs(
    images: dict[int, np.ndarray],
    matcher: Matcher,
    pairs: list[tuple[int, int]] | None = None,
    ransac_threshold: float = 3.0,
) -> list[PairResult]:
    """Estimate PairResults for all (or the given) image pairs using the injected matcher."""
    ...
