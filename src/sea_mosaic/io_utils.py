"""Image loading and GPS metadata extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load a single image from disk as an array."""
    ...


def load_images(paths: list[Path]) -> dict[int, np.ndarray]:
    """Load a sequence of images, keyed by their index in the input list."""
    ...


def load_gps_position(path: Path) -> np.ndarray | None:
    """Extract a GPS anchor position from image EXIF/XMP metadata; return None if
    unavailable. Full EXIF/XMP parsing is a separate roadmap item (see CLAUDE.md)."""
    ...
