"""Pose-graph construction and GPS-anchored optimization for global image-to-mosaic transforms.

Per CLAUDE.md architectural constraint 4: global transforms must come from optimizing a
pose graph with GPS positions as anchors, never from chaining pairwise homographies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sea_mosaic.types import GlobalTransforms, PairResult


@dataclass
class GPSAnchor:
    """A GPS-derived position prior for one image, used as an anchor in pose-graph optimization."""

    image_index: int
    position_xy: np.ndarray  # shape (2,) or (3,), local ENU/projected GPS position
    weight: float  # anchor confidence used in the optimization objective


@dataclass
class PoseGraphNode:
    """One image's pose node in the pose graph."""

    image_index: int
    initial_pose: np.ndarray  # 3x3 initial pose guess in the mosaic frame
    gps_anchor: GPSAnchor | None


@dataclass
class PoseGraphEdge:
    """A relative-pose constraint between two images, derived from a PairResult."""

    src_index: int
    dst_index: int
    relative_pose: np.ndarray  # 3x3 relative transform
    information: np.ndarray  # information/weight matrix for this edge


@dataclass
class PoseGraph:
    """The full pose graph to be optimized: nodes, pairwise edges, and GPS anchors."""

    nodes: list[PoseGraphNode]
    edges: list[PoseGraphEdge]
    anchors: list[GPSAnchor]


def build_pose_graph(
    pair_results: list[PairResult],
    gps_positions: dict[int, np.ndarray] | None = None,
) -> PoseGraph:
    """Build a pose graph from pairwise estimates and optional GPS anchors.

    Edges come from pair_results' homographies (not chained); anchors come from
    gps_positions when provided.
    """
    ...


def optimize_pose_graph(graph: PoseGraph, reference_index: int = 0) -> GlobalTransforms:
    """Optimize the GPS-anchored pose graph to produce global image-to-mosaic transforms.

    residual_error is np.nan if the optimizer does not converge or cannot be run.
    """
    ...
