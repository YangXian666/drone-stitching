"""Stage D of the staged pose-graph architecture: bounded refinement of positions only.

Headings are NOT refined: each node's heading phi_i comes from Stage A+C unchanged (see
CLAUDE.md's Stage D 改為只精修位置 -- edge weights only describe random noise, so real
edges claim a relative-rotation precision 30-60x better than their actual ~0.3 deg
systematic bias; letting Stage D move headings pulled a perfect start back to the edges'
drift). The unknowns are each node's image-centre position m_i in Stage B's north-up pixel
frame plus one global kappa multiplying the GPS targets (absorbing pixels_per_meter
estimation error); pose_i(x) = R(phi_i)(x - c_i) + m_i with scale fixed at 1.

Residual terms, each divided by its own sigma:
- edges: K sampled RANSAC inlier correspondences per edge. With R_i, R_j fixed,
  pose_i(x_src) - pose_j(x_dst) = (m_i - m_j) - d_k, d_k = R_j(x_dst - c_j) - R_i(x_src - c_i)
  a known constant per point -- a pure relative-translation constraint. No homography is
  linearized anywhere (the recurring trap behind the mirror bug, the linearization-point
  bias and H[:2,2] ignoring the perspective row; see CLAUDE.md).
- GPS: m_i - kappa * p_i.
The problem is linear in (m, kappa). Bounds relative to Stage C are guard rails only; every
hit is reported. No GimbalYawDegree or DJI XMP field is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr

from sea_mosaic.frame_alignment import AlignedPoses
from sea_mosaic.gps_placement import GpsPlacement
from sea_mosaic.types import PairResult

K_POINTS_PER_EDGE = 30
KAPPA_BOUND = 0.077  # 3 x the largest between-subset deviation of the ppm median (2.56%)
MAX_POSITION_CHANGE_M = 5.0  # per axis, ~13 x GPS_SIGMA_M: only catches gross failure

MAD_TO_SIGMA = 1.4826  # standard deviation per median absolute deviation (normal)
GPS_SIGMA_M = 0.35 * MAD_TO_SIGMA / np.sqrt(2.0)  # 0.35 m is a two-fix displacement MAD
EDGE_SIGMA_FLOOR_PX = 0.1  # order of SIFT keypoint localization accuracy
HUBER_F_SCALE = 1.345  # Huber constant for 95% efficiency under normal noise

# IRLS convergence (stage 2). Converged only when, in one round, every position moves at
# most IRLS_POSITION_TOL_PX, kappa at most IRLS_KAPPA_TOL, AND the set of variables held
# at a guard rail did not change. Reaching IRLS_MAX_ROUNDS otherwise is reported as
# "not_converged" -- never as converged (the same standard as optimize_pose_graph's
# optimization_status).
IRLS_POSITION_TOL_PX = 1e-9
IRLS_KAPPA_TOL = 1e-12
IRLS_MAX_ROUNDS = 100

_PARAMS_PER_NODE = 2  # m_x, m_y


@dataclass(frozen=True)
class RefinementEdge:
    """One image pair's evidence for Stage D: RANSAC inlier correspondences (src pixels,
    dst pixels) and the pair's homography, used only for the edge's own reprojection-error
    sigma."""

    src_index: int
    dst_index: int
    src_points: np.ndarray  # (N, 2)
    dst_points: np.ndarray  # (N, 2)
    homography: np.ndarray  # src pixels -> dst pixels
    src_image_shape: tuple[int, ...]  # numpy shape (rows, cols[, channels])


@dataclass
class BoundHits:
    position: set[int] = field(default_factory=set)
    kappa: bool = False


@dataclass
class RefinedPoses:
    """Refined centre-anchored poses (scale 1) plus diagnostics.

    headings_rad are Stage C's, passed through unchanged. Only nodes that had a Stage C
    pose are refined; Stage C's unlocated / unoriented / unaligned_components are passed
    through. skipped_edges maps (src, dst) to the reason an edge contributed nothing.
    """

    poses: dict[int, np.ndarray]
    headings_rad: dict[int, float]
    centres_px: dict[int, np.ndarray]
    kappa: float
    bound_hits: BoundHits
    skipped_edges: dict[tuple[int, int], str]
    term_rms: dict[str, float]  # RMS of sigma-normalized residuals per term; nan if empty
    status: str  # "converged" or "not_converged" (IRLS stopping rule, see IRLS_* constants)
    irls_rounds: int  # IRLS rounds actually run (0 when there was nothing to solve)
    unlocated: set[int]
    unoriented: set[int]
    unaligned_components: set[int]


def edge_from_pair_result(pair: PairResult, src_image_shape: tuple[int, ...]) -> RefinementEdge:
    """Build a RefinementEdge from a PairResult, keeping only its RANSAC inliers."""
    mask = np.asarray(pair.inlier_mask, dtype=bool)
    return RefinementEdge(
        pair.src_index,
        pair.dst_index,
        np.asarray(pair.src_points, dtype=np.float64)[mask],
        np.asarray(pair.dst_points, dtype=np.float64)[mask],
        np.asarray(pair.homography, dtype=np.float64),
        src_image_shape,
    )


def sample_correspondences(src_points: np.ndarray, k: int) -> np.ndarray:
    """Indices of at most k correspondences, spread out by farthest-point sampling.

    Contract: if n <= k, return all indices in original order. Otherwise the first pick is
    the point farthest from the centroid of all src points; each next pick maximizes the
    minimum Euclidean distance to the points already picked; ties go to the smallest
    index. Indices are returned in pick order. Deterministic -- no randomness.
    """
    points = np.asarray(src_points, dtype=np.float64)
    n = len(points)
    if n <= k:
        return np.arange(n)

    distance_to_centroid = np.linalg.norm(points - points.mean(axis=0), axis=1)
    first = int(np.argmax(distance_to_centroid))  # argmax returns the smallest index on ties
    picked = [first]
    min_distance = np.linalg.norm(points - points[first], axis=1)
    min_distance[first] = -np.inf
    while len(picked) < k:
        nxt = int(np.argmax(min_distance))
        picked.append(nxt)
        min_distance = np.minimum(min_distance, np.linalg.norm(points - points[nxt], axis=1))
        min_distance[picked] = -np.inf
    return np.array(picked)


def _image_centre(image_shape: tuple[int, ...]) -> np.ndarray:
    return np.array([image_shape[1] / 2.0, image_shape[0] / 2.0])


def _rotation(phi: float) -> np.ndarray:
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def _validate(edges: list[RefinementEdge], pixels_per_meter: float) -> None:
    if not (np.isfinite(pixels_per_meter) and pixels_per_meter > 0):
        raise ValueError("pixels_per_meter must be finite and > 0")
    for edge in edges:
        src, dst = np.asarray(edge.src_points), np.asarray(edge.dst_points)
        name = f"edge {edge.src_index}->{edge.dst_index}"
        if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape:
            raise ValueError(f"{name}: src_points and dst_points must both be (N, 2) with equal N")
        if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
            raise ValueError(f"{name}: non-finite correspondence")
        if not np.all(np.isfinite(edge.homography)):
            raise ValueError(f"{name}: non-finite homography")


def _edge_sigma_px(edge: RefinementEdge) -> float:
    """RMS reprojection error of all this edge's inliers under its own homography."""
    src = np.column_stack([edge.src_points, np.ones(len(edge.src_points))])
    projected = (edge.homography @ src.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    rms = float(np.sqrt(np.mean(np.sum((projected - edge.dst_points) ** 2, axis=1))))
    return max(rms, EDGE_SIGMA_FLOOR_PX)


def _irls_converged(
    position_step: float, kappa_step: float, held: dict[int, float], previous_held: dict[int, float]
) -> bool:
    """IRLS stopping rule: every position moved at most IRLS_POSITION_TOL_PX, kappa at most
    IRLS_KAPPA_TOL, and the set of variables held at a guard rail is unchanged."""
    return position_step <= IRLS_POSITION_TOL_PX and kappa_step <= IRLS_KAPPA_TOL and held == previous_held


def _exact_lsqr(A: csr_matrix, b: np.ndarray) -> np.ndarray:
    return lsqr(A, b, atol=1e-15, btol=1e-15, conlim=1e12, iter_lim=100 * max(A.shape[1], 1))[0]


def _solve_with_guard_rails(
    A: csr_matrix, b: np.ndarray, weights: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    held: dict[int, float],
) -> tuple[np.ndarray, dict[int, float]]:
    """Exact weighted linear least squares with box guard rails, by an active set.

    held maps a variable index to the bound it is held at. Solve for the free variables;
    hold any that land outside their box at the violated bound and re-solve; release a held
    variable whose gradient points back inside the box and re-solve; stop when neither
    applies. Returns the solution and the final held set.
    """
    n = A.shape[1]
    sqrt_w = np.sqrt(weights)
    held = dict(held)
    for _ in range(2 * n + 1):
        x = np.zeros(n)
        for index, value in held.items():
            x[index] = value
        free = np.array([k for k in range(n) if k not in held], dtype=int)
        if free.size:
            weighted = csr_matrix(A[:, free].multiply(sqrt_w[:, None]))
            x[free] = _exact_lsqr(weighted, sqrt_w * (b - A @ x))
        outside = [k for k in free if x[k] < lower[k] or x[k] > upper[k]]
        if outside:
            for k in outside:
                held[k] = lower[k] if x[k] < lower[k] else upper[k]
            continue
        gradient = A.T @ (weights * (A @ x - b))
        inward = [k for k, v in held.items() if (v == lower[k] and gradient[k] < 0) or (v == upper[k] and gradient[k] > 0)]
        if inward:
            for k in inward:
                del held[k]
            continue
        return x, held
    return x, held


def refine_poses(
    aligned: AlignedPoses,
    placement: GpsPlacement,
    edges: list[RefinementEdge],
    pixels_per_meter: float,
    image_shapes: dict[int, tuple[int, ...]],
    *,
    k_points: int = K_POINTS_PER_EDGE,
    loss: str = "huber",
) -> RefinedPoses:
    """Refine Stage C's positions (not headings) by bounded least squares; see the module
    docstring.

    loss exists so a capability test can compare against a plain least-squares loss;
    production callers use the default. Raises ValueError for invalid correspondences, a
    non-finite homography or a non-positive pixels_per_meter.
    """
    _validate(edges, pixels_per_meter)
    if loss not in ("huber", "linear"):
        raise ValueError(f"loss must be 'huber' or 'linear', got {loss!r}")

    nodes = sorted(aligned.poses)
    column = {node: position * _PARAMS_PER_NODE for position, node in enumerate(nodes)}
    kappa_column = len(nodes) * _PARAMS_PER_NODE
    centre_of = {node: _image_centre(image_shapes[node]) for node in nodes}
    rotation_of = {node: _rotation(aligned.headings_rad[node]) for node in nodes}
    gps = placement.centres_px

    skipped: dict[tuple[int, int], str] = {}
    point_terms = []  # (i, j, d (K,2), sigma): residual (m_i - m_j - d_k) / sigma
    for edge in edges:
        key = (edge.src_index, edge.dst_index)
        if edge.src_index not in column or edge.dst_index not in column:
            skipped[key] = "endpoint_without_stage_c_pose"
            continue
        if len(edge.src_points) < 2:
            skipped[key] = "fewer_than_2_inliers"
            continue
        chosen = sample_correspondences(edge.src_points, k_points)
        i, j = edge.src_index, edge.dst_index
        d = (edge.dst_points[chosen] - centre_of[j]) @ rotation_of[j].T - (edge.src_points[chosen] - centre_of[i]) @ rotation_of[i].T
        point_terms.append((i, j, d, _edge_sigma_px(edge)))

    gps_nodes = [node for node in nodes if node in gps]
    gps_sigma_px = GPS_SIGMA_M * pixels_per_meter

    x0 = np.zeros(kappa_column + 1)
    for node in nodes:
        x0[column[node] : column[node] + 2] = (aligned.poses[node] @ np.append(centre_of[node], 1.0))[:2]
    x0[kappa_column] = 1.0

    position_bound = MAX_POSITION_CHANGE_M * pixels_per_meter
    lower, upper = x0 - position_bound, x0 + position_bound
    lower[kappa_column], upper[kappa_column] = 1.0 - KAPPA_BOUND, 1.0 + KAPPA_BOUND

    # The problem is linear: residuals = A @ x - b with a constant sparse A. Rows: for each
    # sampled correspondence, (m_i - m_j - d_k) / sigma_edge per axis; for each GPS node,
    # (m - kappa * p) / sigma_gps per axis.
    n_rows = sum(2 * len(d) for _, _, d, _ in point_terms) + 2 * len(gps_nodes)
    A = lil_matrix((max(n_rows, 1), len(x0)))
    b = np.zeros(max(n_rows, 1))
    row = 0
    term_rows = {"edge": [], "gps": []}
    for i, j, d, sigma in point_terms:
        for k in range(len(d)):
            for axis in range(2):
                A[row, column[i] + axis] = 1.0 / sigma
                A[row, column[j] + axis] = -1.0 / sigma
                b[row] = d[k, axis] / sigma
                term_rows["edge"].append(row)
                row += 1
    for node in gps_nodes:
        for axis in range(2):
            A[row, column[node] + axis] = 1.0 / gps_sigma_px
            A[row, kappa_column] = -gps[node][axis] / gps_sigma_px
            term_rows["gps"].append(row)
            row += 1
    A = A.tocsr()

    if n_rows == 0:
        x, held, status, rounds = x0, {}, "converged", 0
    else:
        # Two-stage solve (CLAUDE.md's Stage D solver findings). Stage 1: the exact
        # unbounded linear least-squares solution. Stage 2: IRLS -- each round recomputes
        # Huber weights (w = 1 for |r| <= c, else c/|r|; the same minimizer as a Huber loss
        # with f_scale = c) and solves the weighted linear problem exactly with the guard
        # rails as an active set. An approximate trust-region solver (lsmr) stalled at
        # ~10 px on the one-wrong-edge test; the exact sparse solve reaches 0.0125 px.
        margin = 1e-9 * (upper - lower)
        x = np.clip(_exact_lsqr(A, b), lower + margin, upper - margin)
        held: dict[int, float] = {}
        status, rounds = "not_converged", 0
        for rounds in range(1, IRLS_MAX_ROUNDS + 1):
            r = A @ x - b
            if loss == "linear":
                weights = np.ones_like(r)
            else:
                weights = np.where(np.abs(r) <= HUBER_F_SCALE, 1.0, HUBER_F_SCALE / np.maximum(np.abs(r), 1e-300))
            previous_held = dict(held)
            x_new, held = _solve_with_guard_rails(A, b, weights, lower, upper, held)
            position_step = float(np.max(np.abs(x_new[:kappa_column] - x[:kappa_column]))) if kappa_column else 0.0
            kappa_step = abs(float(x_new[kappa_column] - x[kappa_column]))
            x = x_new
            if _irls_converged(position_step, kappa_step, held, previous_held):
                status = "converged"
                break
    active = np.zeros(len(x0), dtype=int)
    for index in held:
        active[index] = 1

    hits = BoundHits(kappa=bool(active[kappa_column] != 0))
    poses, headings, centres = {}, {}, {}
    for node in nodes:
        c = column[node]
        if np.any(active[c : c + 2] != 0):
            hits.position.add(node)
        R = rotation_of[node]
        pose = np.eye(3)
        pose[:2, :2] = R
        pose[:2, 2] = x[c : c + 2] - R @ centre_of[node]
        poses[node], headings[node], centres[node] = pose, aligned.headings_rad[node], x[c : c + 2].copy()

    final = (A @ x - b) if n_rows else np.zeros(0)
    term_rms = {
        name: (float(np.sqrt(np.mean(final[rows] ** 2))) if rows else float("nan"))
        for name, rows in term_rows.items()
    }
    return RefinedPoses(
        poses=poses,
        headings_rad=headings,
        centres_px=centres,
        kappa=float(x[kappa_column]),
        bound_hits=hits,
        skipped_edges=skipped,
        term_rms=term_rms,
        status=status,
        irls_rounds=rounds,
        unlocated=set(aligned.unlocated),
        unoriented=set(aligned.unoriented),
        unaligned_components=set(aligned.unaligned_components),
    )
