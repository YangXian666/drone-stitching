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
    YawAnchor,
    _yaw_target_vector,
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


# --- _yaw_target_vector (private sign-flip + unit-vector helper) ----------------------


def test_yaw_target_vector_zero_relative_yaw_is_identity_rotation() -> None:
    target = _yaw_target_vector(0.0)

    assert target == pytest.approx([1.0, 0.0], abs=1e-12)


def test_yaw_target_vector_applies_validated_sign_flip() -> None:
    """theta_target = relative_yaw_deg directly, with NO extra sign flip.

    This is deliberately NOT theta_target = -relative_yaw_deg, even though CLAUDE.md's
    已知的限制 validated `H_angle ~= -relative_yaw` for a single edge's own homography
    decomposition. That relationship is about the EDGE's relative_pose (inv(pose_dst) @
    pose_src), one matrix inversion away from the NODE's own absolute rotation angle
    that YawAnchor actually targets: pose_dst = pose_src @ inv(relative_pose), and
    inverting a pure rotation negates its angle again, so
    node_angle = -H_angle = -(-relative_yaw) = +relative_yaw -- the two sign flips
    cancel out. (An earlier version of this function skipped that second inversion step
    and used theta_target = -relative_yaw_deg instead -- see CLAUDE.md's 已知的限制
    for the full story of that bug and why this test's old expected value didn't catch
    it: it was derived with the same missing-inversion mistake, so it matched the buggy
    implementation instead of exposing it.)

    Confirmed directly against real, already-measured data: the very first edge-only
    (no GPS/yaw anchor at all) optimize_pose_graph run on data/smoke/'s real 9-edge
    chain converged node 1/2/3 to -61.584/-92.584/-140.876deg, matching their real
    relative_yaw of -62.200/-93.600/-143.300deg directly -- not negated."""
    target = _yaw_target_vector(-62.2)

    theta_target_rad = np.radians(-62.2)
    expected = np.array([np.cos(theta_target_rad), np.sin(theta_target_rad)])
    assert target == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("relative_yaw_deg", [0.0, -62.2, -93.6, -143.3, 90.0, -180.0, 179.9])
def test_yaw_target_vector_is_unit_length(relative_yaw_deg: float) -> None:
    """Locks in the "implicitly constrains scale~=1" design rationale from CLAUDE.md:
    the target is always a unit vector, regardless of angle."""
    target = _yaw_target_vector(relative_yaw_deg)

    assert np.linalg.norm(target) == pytest.approx(1.0, rel=1e-9)


# --- build_pose_graph: gimbal_yaw / yaw_anchor_weight ----------------------------------


def test_build_pose_graph_without_gimbal_yaw_has_empty_yaw_anchors() -> None:
    """Graceful-degradation guarantee (CLAUDE.md's YawAnchor 依賴風險 todo): no
    gimbal_yaw at all must not error and must produce an empty yaw_anchors list, not a
    partially-built or exceptional state."""
    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        _GPS_POSITIONS_M,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
    )

    assert graph.yaw_anchors == []
    for node in graph.nodes:
        assert node.yaw_anchor is None


def test_build_pose_graph_requires_yaw_anchor_weight_when_gimbal_yaw_given() -> None:
    gimbal_yaw = {0: 0.0, 1: -62.2, 2: -93.6}

    with pytest.raises(ValueError):
        build_pose_graph(
            [_PAIR_RESULT_01, _PAIR_RESULT_12],
            _GPS_POSITIONS_M,
            pixels_per_meter=_PIXELS_PER_METER,
            inlier_count_reference=_INLIER_COUNT_REFERENCE,
            gimbal_yaw=gimbal_yaw,
        )


def test_build_pose_graph_creates_yaw_anchors_from_gimbal_yaw() -> None:
    gimbal_yaw = {0: 0.0, 1: -62.2, 2: -93.6}
    yaw_anchor_weight = 0.02

    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        _GPS_POSITIONS_M,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
        gimbal_yaw=gimbal_yaw,
        yaw_anchor_weight=yaw_anchor_weight,
    )

    yaw_anchors_by_index = {anchor.image_index: anchor for anchor in graph.yaw_anchors}
    assert set(yaw_anchors_by_index) == {0, 1, 2}
    for index, relative_yaw_deg in gimbal_yaw.items():
        anchor = yaw_anchors_by_index[index]
        assert anchor.weight == pytest.approx(yaw_anchor_weight)
        assert anchor.target_vector == pytest.approx(_yaw_target_vector(relative_yaw_deg))

    nodes_by_index = {node.image_index: node for node in graph.nodes}
    for index in gimbal_yaw:
        assert nodes_by_index[index].yaw_anchor is yaw_anchors_by_index[index]


def test_build_pose_graph_yaw_only_node_with_no_edges() -> None:
    """image 3 has GimbalYawDegree but no GPS position and no successful match_pair —
    mirrors test_build_pose_graph_includes_gps_only_node_with_no_edges but for the yaw
    axis, confirming node discovery unions gimbal_yaw's keys independently of
    gps_positions's."""
    gimbal_yaw = {0: 0.0, 1: -62.2, 3: -143.3}
    yaw_anchor_weight = 0.02

    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        _GPS_POSITIONS_M,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
        gimbal_yaw=gimbal_yaw,
        yaw_anchor_weight=yaw_anchor_weight,
    )

    node_indices = {node.image_index for node in graph.nodes}
    assert node_indices == {0, 1, 2, 3}
    assert not any(edge.src_index == 3 or edge.dst_index == 3 for edge in graph.edges)
    nodes_by_index = {node.image_index: node for node in graph.nodes}
    assert nodes_by_index[3].gps_anchor is None
    assert nodes_by_index[3].yaw_anchor is not None
    assert nodes_by_index[3].yaw_anchor.target_vector == pytest.approx(_yaw_target_vector(-143.3))


def test_build_pose_graph_partial_gimbal_yaw_coverage() -> None:
    """Only some nodes have a GimbalYawDegree reading (spotty metadata is the realistic
    failure mode flagged in CLAUDE.md's YawAnchor 依賴風險 todo, not just "all or
    nothing") — the nodes without one must get yaw_anchor=None without build_pose_graph
    raising, while the nodes with one still get a correct YawAnchor."""
    gimbal_yaw = {0: 0.0, 2: -93.6}  # node 1 has no gimbal yaw reading
    yaw_anchor_weight = 0.02

    graph = build_pose_graph(
        [_PAIR_RESULT_01, _PAIR_RESULT_12],
        _GPS_POSITIONS_M,
        pixels_per_meter=_PIXELS_PER_METER,
        inlier_count_reference=_INLIER_COUNT_REFERENCE,
        gimbal_yaw=gimbal_yaw,
        yaw_anchor_weight=yaw_anchor_weight,
    )

    nodes_by_index = {node.image_index: node for node in graph.nodes}
    assert nodes_by_index[0].yaw_anchor is not None
    assert nodes_by_index[1].yaw_anchor is None
    assert nodes_by_index[2].yaw_anchor is not None
    assert {anchor.image_index for anchor in graph.yaw_anchors} == {0, 2}


# --- optimize_pose_graph: yaw anchor residuals -----------------------------------------


def test_optimize_pose_graph_empty_yaw_anchors_behaves_exactly_as_before() -> None:
    """Regression guard for the day residuals() starts consulting graph.yaw_anchors:
    an explicit empty list must be a true no-op on the existing GPS-anchor-only
    behavior (same assertions as
    test_optimize_pose_graph_converges_near_truth_with_gps_anchors_no_drift)."""
    base_graph = _build_chain_graph(_EDGE_TRANSLATION_NOISE, include_anchors=True)
    graph = PoseGraph(nodes=base_graph.nodes, edges=base_graph.edges, anchors=base_graph.anchors, yaw_anchors=[])

    result = optimize_pose_graph(graph, reference_index=0)

    assert result.optimization_status == "converged"
    assert np.isfinite(result.residual_error)
    errors = _position_errors(result.transforms)
    assert all(error < 2.0 for error in errors.values())
    assert abs(errors[4] - errors[1]) < 1.0


def test_optimize_pose_graph_reference_only_graph_converges_without_crashing() -> None:
    """Regression guard: a graph containing ONLY the reference node (no other nodes,
    no edges, no anchors at all) must not crash. Found while implementing
    pipeline.py's run_pipeline: `optimizable_indices` is empty in this case, and
    `np.concatenate([])` (called unconditionally, before the "nothing to optimize"
    branch could return early) raised `ValueError: need at least one array to
    concatenate` -- a real, previously-uncaught bug, not a run_pipeline mistake, since
    none of this file's other tests ever built a graph with only one node total."""
    graph = PoseGraph(
        nodes=[PoseGraphNode(image_index=0, initial_pose=_similarity_matrix(1.0, 0.0, 0.0, 0.0), gps_anchor=None)],
        edges=[],
        anchors=[],
    )

    result = optimize_pose_graph(graph, reference_index=0)

    assert result.optimization_status == "converged"
    assert np.array_equal(result.transforms[0], _similarity_matrix(1.0, 0.0, 0.0, 0.0))
    assert np.isnan(result.residual_error)  # no edges, no anchors -> nothing to measure


def test_optimize_pose_graph_reference_only_graph_with_anchor_converges_with_zero_residual() -> None:
    """Same single-node scenario, but the reference itself carries a GPSAnchor whose
    position exactly matches its own initial_pose translation (mirrors run_pipeline's
    synthetic default-anchor-for-the-reference pattern) -- residual_error must be a
    finite ~0, not NaN or a crash."""
    anchor = GPSAnchor(image_index=0, position_xy=np.array([0.0, 0.0]), weight=1.0)
    graph = PoseGraph(
        nodes=[PoseGraphNode(image_index=0, initial_pose=_similarity_matrix(1.0, 0.0, 0.0, 0.0), gps_anchor=anchor)],
        edges=[],
        anchors=[anchor],
    )

    result = optimize_pose_graph(graph, reference_index=0)

    assert result.optimization_status == "converged"
    assert result.residual_error == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("theta_target_deg", [30.0, -47.0])
def test_optimize_pose_graph_yaw_anchor_pulls_isolated_node_rotation_to_target(theta_target_deg: float) -> None:
    """No edges, no GPS anchors at all — the only residual term touching node 1 is its
    YawAnchor, so (a,b) should converge almost exactly to target_vector. Parametrized
    over an asymmetric negative angle (-47deg) in addition to a "nice" positive one
    (30deg), to rule out a sign/quadrant bug that a single convenient angle could hide —
    same rationale as this file's existing asymmetric synthetic data for node-index-
    mapping checks."""
    theta_target_rad = np.radians(theta_target_deg)
    target_vector = np.array([np.cos(theta_target_rad), np.sin(theta_target_rad)])
    yaw_anchor = YawAnchor(image_index=1, target_vector=target_vector, weight=1.0)
    graph = PoseGraph(
        nodes=[
            PoseGraphNode(image_index=0, initial_pose=_similarity_matrix(1.0, 0.0, 0.0, 0.0), gps_anchor=None),
            PoseGraphNode(
                image_index=1,
                initial_pose=_similarity_matrix(1.0, 0.0, 0.0, 0.0),
                gps_anchor=None,
                yaw_anchor=yaw_anchor,
            ),
        ],
        edges=[],
        anchors=[],
        yaw_anchors=[yaw_anchor],
    )

    result = optimize_pose_graph(graph, reference_index=0)

    assert result.optimization_status == "converged"
    optimized_ab = np.array([result.transforms[1][0, 0], result.transforms[1][1, 0]])
    assert optimized_ab == pytest.approx(target_vector, abs=1e-4)


def test_optimize_pose_graph_yaw_anchor_rescues_severely_corrupted_edge_rotation() -> None:
    """Corruption magnitude matches the real severity observed in CLAUDE.md's diagnosis
    (rotation angle discontinuous / sign-flipped by tens of degrees, not small-angle
    noise): node 0 is the reference (rotation=0deg), node 1's true rotation is 40deg, but
    edge (0,1) is corrupted to imply a relative rotation of 90deg instead of the correct
    -40deg -- a full quadrant error.

    Without any yaw anchor this is an exactly-determined tree (1 edge, 1 free node, zero
    redundancy), so it converges to *exactly* satisfy the corrupted edge: node 1 lands at
    -90deg, a 130deg error from truth.

    yaw_weight=5.0 (not the production-tuned 0.02 from CLAUDE.md, which was calibrated
    against real edges' info-weighted *rotation sub-block* residual of order 0.02-0.46 --
    a much weaker opposing pull than this synthetic edge's full eye(6) weight) was found
    by directly solving this exact residual formula (least_squares on the same equations
    optimize_pose_graph implements, not by trial-and-error against the implementation):
    weights below ~1.2 stay in the corrupted edge's basin (this is a genuinely non-convex
    problem -- there are two competing local minima 130deg apart), while weight=5.0 sits
    comfortably in the rescued basin (angle error ~3deg), verified stable across a
    4.0-10.0 sweep."""
    pose0 = _similarity_matrix(1.0, 0.0, 0.0, 0.0)  # true rotation 0deg
    true_rotation_1_deg = 40.0
    corrupted_deg = 90.0  # true relative rotation would be 0 - 40 = -40deg

    def _rotation_matrix(deg: float) -> np.ndarray:
        rad = np.radians(deg)
        return _similarity_matrix(np.cos(rad), np.sin(rad), 0.0, 0.0)

    corrupted_edge = PoseGraphEdge(
        src_index=0, dst_index=1, relative_pose=_rotation_matrix(corrupted_deg), information=np.eye(6)
    )
    target_vector = np.array(
        [np.cos(np.radians(true_rotation_1_deg)), np.sin(np.radians(true_rotation_1_deg))]
    )

    def _graph(with_yaw_anchor: bool) -> PoseGraph:
        yaw_anchor = YawAnchor(image_index=1, target_vector=target_vector, weight=5.0)
        node1 = PoseGraphNode(
            image_index=1,
            initial_pose=_similarity_matrix(1.0, 0.0, 0.0, 0.0),
            gps_anchor=None,
            yaw_anchor=yaw_anchor if with_yaw_anchor else None,
        )
        return PoseGraph(
            nodes=[PoseGraphNode(image_index=0, initial_pose=pose0, gps_anchor=None), node1],
            edges=[corrupted_edge],
            anchors=[],
            yaw_anchors=[yaw_anchor] if with_yaw_anchor else [],
        )

    def _angle_deg(pose: np.ndarray) -> float:
        return float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))

    result_without = optimize_pose_graph(_graph(with_yaw_anchor=False), reference_index=0)
    result_with = optimize_pose_graph(_graph(with_yaw_anchor=True), reference_index=0)

    angle_without = _angle_deg(result_without.transforms[1])
    angle_with = _angle_deg(result_with.transforms[1])
    error_without = abs(angle_without - true_rotation_1_deg)
    error_with = abs(angle_with - true_rotation_1_deg)

    assert angle_without == pytest.approx(-90.0, abs=1e-3)
    assert error_without > 45.0  # matches the real severity: not small-angle noise

    assert error_with < 10.0
    assert error_with < 0.3 * error_without


def test_optimize_pose_graph_production_yaw_weight_cannot_rescue_severe_rotation_error() -> None:
    """Locks in a known capability boundary of the weight~=0.02 design (CLAUDE.md's
    YawAnchor 依賴風險 section): it was calibrated against real edges' info-weighted
    *rotation sub-block* residual (order 0.02-0.46 -- mild disagreement from an edge
    whose rotation is itself basically correct but down-weighted by a low inlier_count),
    not against a severely wrong rotation like the 130deg error used in
    test_optimize_pose_graph_yaw_anchor_rescues_severely_corrupted_edge_rotation.

    At this production weight, the same severe corruption is NOT rescued: the result
    stays in the corrupted edge's basin, essentially indistinguishable from having no
    yaw anchor at all (verified directly via least_squares on this exact residual
    formula before writing this assertion: weight=0.02 gives ~-89.99deg, vs. -90.0deg
    with no anchor at all -- a 0.01deg difference).

    If a future change makes this test start passing (production weight starts
    rescuing severe errors too), that is a signal the weight has drifted away from the
    rot_wtd-median-derived magnitude this test intentionally pins down -- not an
    improvement to celebrate without first re-checking why."""
    pose0 = _similarity_matrix(1.0, 0.0, 0.0, 0.0)
    true_rotation_1_deg = 40.0
    corrupted_deg = 90.0
    production_yaw_weight = 0.02  # CLAUDE.md-recommended value (range 0.01~0.05)

    def _rotation_matrix(deg: float) -> np.ndarray:
        rad = np.radians(deg)
        return _similarity_matrix(np.cos(rad), np.sin(rad), 0.0, 0.0)

    corrupted_edge = PoseGraphEdge(
        src_index=0, dst_index=1, relative_pose=_rotation_matrix(corrupted_deg), information=np.eye(6)
    )
    target_vector = np.array(
        [np.cos(np.radians(true_rotation_1_deg)), np.sin(np.radians(true_rotation_1_deg))]
    )
    yaw_anchor = YawAnchor(image_index=1, target_vector=target_vector, weight=production_yaw_weight)
    node1 = PoseGraphNode(
        image_index=1,
        initial_pose=_similarity_matrix(1.0, 0.0, 0.0, 0.0),
        gps_anchor=None,
        yaw_anchor=yaw_anchor,
    )
    graph = PoseGraph(
        nodes=[PoseGraphNode(image_index=0, initial_pose=pose0, gps_anchor=None), node1],
        edges=[corrupted_edge],
        anchors=[],
        yaw_anchors=[yaw_anchor],
    )

    result = optimize_pose_graph(graph, reference_index=0)

    angle = float(np.degrees(np.arctan2(result.transforms[1][1, 0], result.transforms[1][0, 0])))
    error_from_truth = abs(angle - true_rotation_1_deg)
    error_from_no_anchor_result = abs(angle - (-90.0))

    # Stuck in the corrupted edge's basin: barely nudged from the no-anchor result.
    assert error_from_truth > 100.0
    assert error_from_no_anchor_result < 5.0
