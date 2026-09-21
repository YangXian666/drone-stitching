"""Pose-graph construction and GPS-anchored optimization for global image-to-mosaic transforms.

Per CLAUDE.md architectural constraint 4: global transforms must come from optimizing a
pose graph with GPS positions as anchors, never from chaining pairwise homographies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
class YawAnchor:
    """A GimbalYawDegree-derived rotation prior for one image, used as an anchor in
    pose-graph optimization -- independent of GPSAnchor, which only constrains
    translation (pose[:2,2]), never rotation/scale (pose[:2,0:2]).

    target_vector is [cos(theta_target), sin(theta_target)] -- already including the
    empirically-validated sign flip from relative GimbalYawDegree to this pose graph's
    Sim(2) rotation convention (see _yaw_target_vector). Being a unit vector, comparing
    a node's (a,b) against it in the optimization objective implicitly constrains
    scale~=1 too, with no separate scale field needed."""

    image_index: int
    target_vector: np.ndarray  # shape (2,), unit vector [cos(theta_target), sin(theta_target)]
    weight: float  # anchor confidence used in the optimization objective


@dataclass
class PoseGraphNode:
    """One image's pose node in the pose graph."""

    image_index: int
    initial_pose: np.ndarray  # 3x3 initial pose guess in the mosaic frame
    gps_anchor: GPSAnchor | None
    yaw_anchor: YawAnchor | None = None


@dataclass
class PoseGraphEdge:
    """A relative-pose constraint between two images, derived from a PairResult."""

    src_index: int
    dst_index: int
    relative_pose: np.ndarray  # 3x3 relative transform
    information: np.ndarray  # information/weight matrix for this edge


@dataclass
class PoseGraph:
    """The full pose graph to be optimized: nodes, pairwise edges, GPS anchors, and yaw
    anchors."""

    nodes: list[PoseGraphNode]
    edges: list[PoseGraphEdge]
    anchors: list[GPSAnchor]
    yaw_anchors: list[YawAnchor] = field(default_factory=list)


def build_pose_graph(
    pair_results: list[PairResult],
    gps_positions: dict[int, np.ndarray] | None = None,
    *,
    pixels_per_meter: float,
    inlier_count_reference: float,
    gimbal_yaw: dict[int, float] | None = None,
    yaw_anchor_weight: float | None = None,
) -> PoseGraph:
    """Build a pose graph from pairwise estimates and optional GPS/yaw anchors.

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

    Yaw anchors come from gimbal_yaw when provided: gimbal_yaw is expected in degrees,
    relative to some origin (geo.projection.project_gimbal_yaw_degrees's output),
    mirroring how gps_positions is expected already-projected. gimbal_yaw is optional
    (None means no yaw anchors at all — a legitimate, fully-supported pose graph, same
    as gps_positions=None; see CLAUDE.md's YawAnchor 依賴風險), but yaw_anchor_weight is
    required whenever gimbal_yaw is given: its magnitude is dataset-specific (see
    CLAUDE.md's weight validation) and must never silently default, same principle as
    pixels_per_meter/inlier_count_reference. A node without its own gimbal_yaw entry
    simply gets no YawAnchor (yaw_anchor=None), even when other nodes do — spotty
    metadata coverage must not raise.
    """
    if gimbal_yaw is not None and yaw_anchor_weight is None:
        raise ValueError("yaw_anchor_weight is required when gimbal_yaw is given")

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
    if gimbal_yaw is not None:
        node_indices |= set(gimbal_yaw)

    anchors_by_index: dict[int, GPSAnchor] = {}
    if gps_positions is not None:
        for image_index, position_m in gps_positions.items():
            anchors_by_index[image_index] = GPSAnchor(
                image_index=image_index,
                position_xy=position_m * pixels_per_meter,
                weight=1.0,
            )

    yaw_anchors_by_index: dict[int, YawAnchor] = {}
    if gimbal_yaw is not None:
        for image_index, relative_yaw_deg in gimbal_yaw.items():
            yaw_anchors_by_index[image_index] = YawAnchor(
                image_index=image_index,
                target_vector=_yaw_target_vector(relative_yaw_deg),
                weight=yaw_anchor_weight,
            )

    nodes = []
    for image_index in sorted(node_indices):
        gps_anchor = anchors_by_index.get(image_index)
        yaw_anchor = yaw_anchors_by_index.get(image_index)
        if gps_anchor is not None:
            initial_pose = _params_to_pose(
                np.array([1.0, 0.0, gps_anchor.position_xy[0], gps_anchor.position_xy[1]])
            )
        else:
            initial_pose = _params_to_pose(np.array([1.0, 0.0, 0.0, 0.0]))
        nodes.append(
            PoseGraphNode(
                image_index=image_index,
                initial_pose=initial_pose,
                gps_anchor=gps_anchor,
                yaw_anchor=yaw_anchor,
            )
        )

    return PoseGraph(
        nodes=nodes,
        edges=edges,
        anchors=list(anchors_by_index.values()),
        yaw_anchors=list(yaw_anchors_by_index.values()),
    )


def _pose_to_params(pose: np.ndarray) -> np.ndarray:
    """Extract (a, b, tx, ty) from a Sim(2) pose matrix."""
    return np.array([pose[0, 0], pose[1, 0], pose[0, 2], pose[1, 2]])


def _params_to_pose(params: np.ndarray) -> np.ndarray:
    """Build a Sim(2) pose matrix from (a, b, tx, ty)."""
    a, b, tx, ty = params
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


def _yaw_target_vector(relative_yaw_deg: float) -> np.ndarray:
    """Convert a relative GimbalYawDegree reading (geo.projection.project_gimbal_yaw_
    degrees's output: a compass-bearing delta, degrees, relative to some origin image)
    into this pose graph's Sim(2) rotation target, [cos(theta_target), sin(theta_target)].

    theta_target = relative_yaw_deg directly, with NO extra sign flip -- see CLAUDE.md's
    已知的限制 for the full derivation and the story of an earlier, buggy version of this
    function that used -relative_yaw_deg instead. In short: CLAUDE.md's validated
    `H_angle ~= -relative_yaw` is about a single EDGE's own relative_pose (inv(pose_dst)
    @ pose_src) decomposition, one matrix inversion away from a NODE's absolute rotation
    angle (pose_dst = pose_src @ inv(relative_pose), and inverting a pure rotation
    negates its angle again) -- the two sign flips cancel, leaving
    node_angle = relative_yaw directly. Confirmed against data/smoke/'s real edge-only
    (no anchors at all) optimize_pose_graph run: nodes 1/2/3 converged to
    -61.584/-92.584/-140.876deg, matching their real relative_yaw of
    -62.200/-93.600/-143.300deg directly, not negated.

    This is a validation covering a single straight-line flight at one altitude and
    should be re-checked if a future dataset's flight pattern is more complex (e.g.
    turns, climbs).

    The result is always a unit vector, which is why YawAnchor needs no separate scale
    field: comparing a node's (a,b) against this target implicitly constrains scale~=1.
    """
    theta_target_rad = np.radians(relative_yaw_deg)
    return np.array([np.cos(theta_target_rad), np.sin(theta_target_rad)])


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
            # Derived from relative_pose ~= inv(pose_dst) @ pose_src by left-multiplying
            # both sides by pose_dst: pose_src ~= pose_dst @ relative_pose. Never inverts
            # a decision variable (R_dst) -- only relative_pose's own fixed R_rel/t_rel
            # are read from data. This avoids the classic Sim(2) pose-graph coupling trap
            # where inverting pose_dst introduces a 1/scale^2 term that lets a GPS
            # anchor's translation pull leak into the rotation/scale subspace (see
            # CLAUDE.md's 已知的限制 for the full derivation and diagnosis).
            R_dst = poses[edge.dst_index][:2, :2]
            t_dst = poses[edge.dst_index][:2, 2]
            R_rel = edge.relative_pose[:2, :2]
            t_rel = edge.relative_pose[:2, 2]
            predicted_src = np.hstack([R_dst @ R_rel, (R_dst @ t_rel + t_dst).reshape(2, 1)])
            diff = (predicted_src - poses[edge.src_index][:2, :]).flatten()
            residual_terms.append(edge.information @ diff)
        for anchor in graph.anchors:
            position_error = poses[anchor.image_index][:2, 2] - anchor.position_xy[:2]
            residual_terms.append(anchor.weight * position_error)
        for yaw_anchor in graph.yaw_anchors:
            rotation_error = poses[yaw_anchor.image_index][:2, 0] - yaw_anchor.target_vector
            residual_terms.append(yaw_anchor.weight * rotation_error)
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
