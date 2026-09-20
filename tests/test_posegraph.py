"""Unit tests for sea_mosaic.posegraph.optimize_pose_graph's core least-squares math.

build_pose_graph (constructing a PoseGraph from real PairResults + GPS positions) is out
of scope here — these tests construct PoseGraph objects directly so the optimizer's math
can be verified in full isolation with hand-built, deterministic synthetic data (no
np.random, per this repo's existing test convention).
"""

from __future__ import annotations

import numpy as np

from sea_mosaic.posegraph import (
    GPSAnchor,
    PoseGraph,
    PoseGraphEdge,
    PoseGraphNode,
    optimize_pose_graph,
)

_TRUE_POSITIONS = {
    0: (0.0, 0.0),
    1: (100.0, 0.0),
    2: (200.0, 0.0),
    3: (300.0, 0.0),
    4: (400.0, 0.0),
}

_ANCHOR_OFFSETS = {
    0: (0.3, -0.2),
    1: (-0.4, 0.5),
    2: (0.2, 0.4),
    3: (-0.5, -0.3),
    4: (0.4, 0.3),
}

_EDGE_TRANSLATION_NOISE = {
    (0, 1): (3.0, 1.0),
    (1, 2): (3.0, -1.0),
    (2, 3): (3.0, 1.0),
    (3, 4): (3.0, -1.0),
}


def _similarity_matrix(a: float, b: float, tx: float, ty: float) -> np.ndarray:
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _true_pose(index: int) -> np.ndarray:
    x, y = _TRUE_POSITIONS[index]
    return _similarity_matrix(1.0, 0.0, x, y)


def _observed_relative_pose(
    src_index: int, dst_index: int, translation_offset: tuple[float, float]
) -> np.ndarray:
    """relative_pose satisfies p_dst ~= relative_pose @ p_src, i.e. inv(T_dst) @ T_src in
    the noise-free case. For pure-translation ground truth this is
    (x_src - x_dst, y_src - y_dst), plus the injected noise offset."""
    x_src, y_src = _TRUE_POSITIONS[src_index]
    x_dst, y_dst = _TRUE_POSITIONS[dst_index]
    true_tx, true_ty = x_src - x_dst, y_src - y_dst
    offset_x, offset_y = translation_offset
    return _similarity_matrix(1.0, 0.0, true_tx + offset_x, true_ty + offset_y)


def _build_chain_graph(
    edge_translation_noise: dict[tuple[int, int], tuple[float, float]],
    include_anchors: bool,
) -> PoseGraph:
    anchors: list[GPSAnchor] = []
    nodes: list[PoseGraphNode] = []
    for index in _TRUE_POSITIONS:
        gps_anchor = None
        if include_anchors:
            offset_x, offset_y = _ANCHOR_OFFSETS[index]
            x, y = _TRUE_POSITIONS[index]
            gps_anchor = GPSAnchor(
                image_index=index,
                position_xy=np.array([x + offset_x, y + offset_y]),
                weight=1.0,
            )
            anchors.append(gps_anchor)
        nodes.append(
            PoseGraphNode(image_index=index, initial_pose=_true_pose(index), gps_anchor=gps_anchor)
        )

    edges = [
        PoseGraphEdge(
            src_index=src,
            dst_index=dst,
            relative_pose=_observed_relative_pose(src, dst, edge_translation_noise[(src, dst)]),
            information=np.eye(6),
        )
        for src, dst in sorted(edge_translation_noise)
    ]

    return PoseGraph(nodes=nodes, edges=edges, anchors=anchors)


def _position_errors(transforms: dict[int, np.ndarray]) -> dict[int, float]:
    return {
        index: float(np.linalg.norm(transforms[index][:2, 2] - np.array(_TRUE_POSITIONS[index])))
        for index in _TRUE_POSITIONS
    }


def test_optimize_pose_graph_converges_near_truth_with_gps_anchors_no_drift() -> None:
    graph = _build_chain_graph(_EDGE_TRANSLATION_NOISE, include_anchors=True)

    result = optimize_pose_graph(graph, reference_index=0)

    assert result.optimization_status == "converged"
    assert np.isfinite(result.residual_error)

    errors = _position_errors(result.transforms)
    assert all(error < 2.0 for error in errors.values())
    # No drift accumulation: error at the far end of the chain (node 4) is not
    # meaningfully larger than near the reference (node 1) — naive chaining would give
    # roughly e1=3, e2=6, e3=9, e4=12 here.
    assert abs(errors[4] - errors[1]) < 1.0


def test_optimize_pose_graph_gps_anchor_rescues_badly_degraded_edge() -> None:
    corrupted_noise = dict(_EDGE_TRANSLATION_NOISE)
    corrupted_noise[(1, 2)] = (50.0, -1.0)  # simulates a low-inlier, badly-estimated pair

    graph_with_anchors = _build_chain_graph(corrupted_noise, include_anchors=True)
    graph_without_anchors = _build_chain_graph(corrupted_noise, include_anchors=False)

    result_with_anchors = optimize_pose_graph(graph_with_anchors, reference_index=0)
    result_without_anchors = optimize_pose_graph(graph_without_anchors, reference_index=0)

    max_error_with_anchors = max(_position_errors(result_with_anchors.transforms).values())
    max_error_without_anchors = max(_position_errors(result_without_anchors.transforms).values())

    # Without anchors, a tree graph has zero redundancy — the bad edge's error propagates
    # to every downstream node essentially unfiltered (hand-solvable: node 2 inherits
    # ~53px, node 4 ~59px, from the exact zero-residual chain solution).
    assert max_error_without_anchors > 30.0
    # With anchors, the same corrupted edge should be pulled back into a reasonable range,
    # and the gap between the two runs should be decisive, not incidental.
    assert max_error_with_anchors < 15.0
    assert max_error_with_anchors < 0.3 * max_error_without_anchors
