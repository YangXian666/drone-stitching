"""Pose-graph construction and GPS-anchored optimization for global image-to-mosaic transforms.

Per CLAUDE.md architectural constraint 4: global transforms must come from optimizing a
pose graph with GPS positions as anchors, never from chaining pairwise homographies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from sea_mosaic.types import GlobalTransforms, PairResult

# Number of free Sim(2) parameters per node: (a, b, tx, ty), where
# T = [[a, -b, tx], [b, a, ty], [0, 0, 1]] (a = s*cos(theta), b = s*sin(theta)).
_PARAMS_PER_NODE = 4


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
    *,
    pixels_per_meter: float,
    inlier_count_reference: float,
) -> PoseGraph:
    """Build a pose graph from pairwise estimates and optional GPS anchors.

    Edges come from pair_results' homographies (not chained; relative_pose is each
    PairResult's homography as-is, since both already follow the src -> dst
    convention). Each edge's information is
    (pair_result.inlier_count / inlier_count_reference) * eye(6) — normalized so a
    "typical" edge for this batch of pair_results carries a weight of about 1.0,
    comparable to each GPS anchor's fixed weight of 1.0.

    Anchors come from gps_positions when provided: gps_positions is expected in local
    planar meters (geo.projection.project_gps_positions's output), and is converted to
    pixel-equivalent units via pixels_per_meter before becoming each node's GPSAnchor
    and initial_pose translation (a node with no GPS entry falls back to identity at the
    origin as its initial pose guess).

    pixels_per_meter and inlier_count_reference are both required, with no default:
    both are numbers specific to the input dataset (camera/altitude geometry, and this
    batch's inlier_count distribution respectively) — silently defaulting either one
    would risk reproducing the exact unit/weight imbalance already measured and
    documented in CLAUDE.md's 已知的限制 for data/smoke/.
    """
    edges = [
        PoseGraphEdge(
            src_index=pair_result.src_index,
            dst_index=pair_result.dst_index,
            relative_pose=pair_result.homography,
            information=(pair_result.inlier_count / inlier_count_reference) * np.eye(6),
        )
        for pair_result in pair_results
    ]

    node_indices = {pair_result.src_index for pair_result in pair_results}
    node_indices |= {pair_result.dst_index for pair_result in pair_results}
    if gps_positions is not None:
        node_indices |= set(gps_positions)

    anchors_by_index: dict[int, GPSAnchor] = {}
    if gps_positions is not None:
        for image_index, position_m in gps_positions.items():
            anchors_by_index[image_index] = GPSAnchor(
                image_index=image_index,
                position_xy=position_m * pixels_per_meter,
                weight=1.0,
            )

    nodes = []
    for image_index in sorted(node_indices):
        gps_anchor = anchors_by_index.get(image_index)
        if gps_anchor is not None:
            initial_pose = _params_to_pose(
                np.array([1.0, 0.0, gps_anchor.position_xy[0], gps_anchor.position_xy[1]])
            )
        else:
            initial_pose = _params_to_pose(np.array([1.0, 0.0, 0.0, 0.0]))
        nodes.append(
            PoseGraphNode(image_index=image_index, initial_pose=initial_pose, gps_anchor=gps_anchor)
        )

    return PoseGraph(nodes=nodes, edges=edges, anchors=list(anchors_by_index.values()))


def _pose_to_params(pose: np.ndarray) -> np.ndarray:
    """Extract (a, b, tx, ty) from a Sim(2) pose matrix."""
    return np.array([pose[0, 0], pose[1, 0], pose[0, 2], pose[1, 2]])


def _params_to_pose(params: np.ndarray) -> np.ndarray:
    """Build a Sim(2) pose matrix from (a, b, tx, ty)."""
    a, b, tx, ty = params
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


def optimize_pose_graph(graph: PoseGraph, reference_index: int = 0) -> GlobalTransforms:
    """Optimize the GPS-anchored pose graph to produce global image-to-mosaic transforms.

    residual_error is np.nan if the optimizer does not converge or cannot be run.
    """
    node_by_index = {node.image_index: node for node in graph.nodes}
    if reference_index not in node_by_index:
        return GlobalTransforms(
            transforms={},
            reference_index=reference_index,
            optimization_status="failed",
            residual_error=np.nan,
        )

    reference_pose = node_by_index[reference_index].initial_pose
    # Fixed, stable ordering for every node except the reference: this order is the only
    # thing that maps a slice of the flat parameter vector back to an image index, so it
    # must be identical every time it's derived from node_by_index.
    optimizable_indices = sorted(index for index in node_by_index if index != reference_index)

    initial_params = np.concatenate(
        [_pose_to_params(node_by_index[index].initial_pose) for index in optimizable_indices]
    )

    def poses_from_flat(flat_params: np.ndarray) -> dict[int, np.ndarray]:
        poses = {reference_index: reference_pose}
        for position, index in enumerate(optimizable_indices):
            start = position * _PARAMS_PER_NODE
            poses[index] = _params_to_pose(flat_params[start : start + _PARAMS_PER_NODE])
        return poses

    def residuals(flat_params: np.ndarray) -> np.ndarray:
        poses = poses_from_flat(flat_params)
        residual_terms = []
        for edge in graph.edges:
            predicted = np.linalg.inv(poses[edge.dst_index]) @ poses[edge.src_index]
            diff = (predicted - edge.relative_pose)[:2, :].flatten()
            residual_terms.append(edge.information @ diff)
        for anchor in graph.anchors:
            position_error = poses[anchor.image_index][:2, 2] - anchor.position_xy[:2]
            residual_terms.append(anchor.weight * position_error)
        if not residual_terms:
            return np.zeros(0)
        return np.concatenate(residual_terms)

    if not optimizable_indices:
        # Every node is pinned to the reference; there's nothing to optimize.
        residual_vector = residuals(initial_params)
        residual_error = (
            float(np.sqrt(np.mean(residual_vector**2))) if residual_vector.size else np.nan
        )
        return GlobalTransforms(
            transforms={reference_index: reference_pose},
            reference_index=reference_index,
            optimization_status="converged",
            residual_error=residual_error,
        )

    result = least_squares(residuals, initial_params)
    transforms = poses_from_flat(result.x)

    if not result.success:
        return GlobalTransforms(
            transforms=transforms,
            reference_index=reference_index,
            optimization_status="not_converged",
            residual_error=np.nan,
        )

    n_residuals = result.fun.size
    residual_error = float(np.sqrt(2 * result.cost / n_residuals)) if n_residuals else np.nan

    return GlobalTransforms(
        transforms=transforms,
        reference_index=reference_index,
        optimization_status="converged",
        residual_error=residual_error,
    )
