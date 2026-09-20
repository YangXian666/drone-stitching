"""Matcher protocol: pipeline code depends on this Protocol only, never on a concrete
matcher implementation (CLAUDE.md architectural constraint 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class MatchResult:
    """Raw keypoint correspondences between two images, before geometric verification."""

    src_points: np.ndarray  # shape (N, 2)
    dst_points: np.ndarray  # shape (N, 2)
    scores: np.ndarray | None = None  # shape (N,), per-match confidence, optional


class Matcher(Protocol):
    """Structural interface any feature matcher (SIFT+BF, LoFTR, SuperGlue, ...) must satisfy."""

    name: str  # matcher name, used as the "method" column in metrics output

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Return raw keypoint correspondences between two images."""
        ...
