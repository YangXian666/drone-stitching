"""Unit tests for sea_mosaic.posegraph.optimize_pose_graph's core least-squares math.

build_pose_graph (constructing a PoseGraph from real PairResults + GPS positions) is out
of scope here — these tests construct PoseGraph objects directly so the optimizer's math
can be verified in full isolation with hand-built, deterministic synthetic data (no
np.random, per this repo's existing test convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from sea_mosaic.posegraph import (
    GPSAnchor,
    PoseGraph,
    PoseGraphEdge,
    PoseGraphNode,
    build_pose_graph,
    optimize_pose_graph,
)
from sea_mosaic.types import PairResult

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


# --- build_pose_graph -----------------------------------------------------------------


def _pair_result(src_index: int, dst_index: int, homography: np.ndarray, match_count: int, inlier_count: int) -> PairResult:
    inlier_mask = np.zeros(match_count, dtype=bool)
    inlier_mask[:inlier_count] = True
    return PairResult(
        src_index=src_index,
        dst_index=dst_index,
        src_points=np.zeros((match_count, 2)),
        dst_points=np.zeros((match_count, 2)),
        inlier_mask=inlier_mask,
        homography=homography,
    )


_HOMOGRAPHY_01 = _similarity_matrix(1.0, 0.0, 100.0, 0.0)
_HOMOGRAPHY_12 = _similarity_matrix(1.0, 0.0, 50.0, 0.0)
_PIXELS_PER_METER = 10.0
_INLIER_COUNT_REFERENCE = 4.0

_PAIR_RESULT_01 = _pair_result(0, 1, _HOMOGRAPHY_01, match_count=10, inlier_count=8)
_PAIR_RESULT_12 = _pair_result(1, 2, _HOMOGRAPHY_12, match_count=20, inlier_count=4)

_GPS_POSITIONS_M = {0: np.array([0.0, 0.0]), 1: np.array([5.0, 0.0]), 2: np.array([9.0, 0.0])}


def test_build_pose_graph_maps_pair_results_to_edges_with_normalized_information() -> None:
    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        _GPS_POSITIONS_M,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
    )

    edges_by_pair = {(edge.src_index, edge.dst_index): edge for edge in graph.edges}
    assert set(edges_by_pair) == {(0, 1), (1, 2)}

    edge_01 = edges_by_pair[(0, 1)]
    assert np.array_equal(edge_01.relative_pose, _HOMOGRAPHY_01)
    assert edge_01.information == pytest.approx(2.0 * np.eye(6))  # 8 / 4

    edge_12 = edges_by_pair[(1, 2)]
    assert np.array_equal(edge_12.relative_pose, _HOMOGRAPHY_12)
    assert edge_12.information == pytest.approx(1.0 * np.eye(6))  # 4 / 4


def test_build_pose_graph_information_scales_proportionally_for_real_measured_disparity() -> None:
    """Locks in the actual finding from CLAUDE.md's 已知的限制 diagnosis: real
    data/smoke/ edges span inlier_count 40 to 3061 against a median reference of 1861
    (~46x spread) — this must produce a proportionally low/high information, not just
    "the function runs" with a mild, non-representative 2x spread like the other test
    above uses for its edge-mapping check."""
    low_inlier_edge = _pair_result(0, 1, _HOMOGRAPHY_01, match_count=40, inlier_count=40)
    high_inlier_edge = _pair_result(1, 2, _HOMOGRAPHY_12, match_count=3061, inlier_count=3061)
    inlier_count_reference = 1861.0

    graph = build_pose_graph(
        [low_inlier_edge, high_inlier_edge],
        gps_positions=None,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=inlier_count_reference,
    )

    edges_by_pair = {(edge.src_index, edge.dst_index): edge for edge in graph.edges}
    assert edges_by_pair[(0, 1)].information == pytest.approx((40 / 1861) * np.eye(6))
    assert edges_by_pair[(1, 2)].information == pytest.approx((3061 / 1861) * np.eye(6))

    low_scale = edges_by_pair[(0, 1)].information[0, 0]
    high_scale = edges_by_pair[(1, 2)].information[0, 0]
    # The real measured disparity: the low-inlier edge should be weighted roughly 1/76th
    # of the high-inlier edge (40/1861 vs 3061/1861), not some incidental small gap.
    assert high_scale / low_scale == pytest.approx(3061 / 40, rel=1e-9)
    assert low_scale < 0.05  # well below 1.0 — this edge should be sharply down-weighted
    assert high_scale > 1.5  # well above 1.0 — this edge should carry noticeably more say


def test_build_pose_graph_creates_gps_anchors_in_pixel_units() -> None:
    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        _GPS_POSITIONS_M,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
    )

    anchors_by_index = {anchor.image_index: anchor for anchor in graph.anchors}
    assert set(anchors_by_index) == {0, 1, 2}
    for index, meters in _GPS_POSITIONS_M.items():
        anchor = anchors_by_index[index]
        assert anchor.position_xy[:2] == pytest.approx(meters * _PIXELS_PER_METER)
        assert anchor.weight == 1.0


def test_build_pose_graph_sets_initial_pose_translation_from_gps_anchor() -> None:
    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        _GPS_POSITIONS_M,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
    )

    nodes_by_index = {node.image_index: node for node in graph.nodes}
    for index, meters in _GPS_POSITIONS_M.items():
        pose = nodes_by_index[index].initial_pose
        assert pose[:2, 2] == pytest.approx(meters * _PIXELS_PER_METER)
        assert pose[0, 0] == pytest.approx(1.0)
        assert pose[1, 0] == pytest.approx(0.0)


def test_build_pose_graph_without_gps_positions_has_no_anchors_and_origin_initial_pose() -> None:
    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        gps_positions=None,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
    )

    assert graph.anchors == []
    for node in graph.nodes:
        assert node.gps_anchor is None
        assert node.initial_pose == pytest.approx(_similarity_matrix(1.0, 0.0, 0.0, 0.0))


def test_build_pose_graph_includes_gps_only_node_with_no_edges() -> None:
    gps_positions = dict(_GPS_POSITIONS_M)
    gps_positions[3] = np.array([20.0, 0.0])  # image 3 has GPS but no successful match_pair

    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        gps_positions,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
    )

    node_indices = {node.image_index for node in graph.nodes}
    assert node_indices == {0, 1, 2, 3}
    assert not any(edge.src_index == 3 or edge.dst_index == 3 for edge in graph.edges)
    anchors_by_index = {anchor.image_index: anchor for anchor in graph.anchors}
    assert anchors_by_index[3].position_xy[:2] == pytest.approx(np.array([200.0, 0.0]))
