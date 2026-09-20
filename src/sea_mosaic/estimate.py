"""Stage 1 (estimate): feature matching + RANSAC geometric verification per image pair.

Uses an injected Matcher (see matcher.py) — never a hardcoded matcher implementation.
"""

from __future__ import annotations

import numpy as np

from sea_mosaic.matcher import Matcher
from sea_mosaic.types import PairResult


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
