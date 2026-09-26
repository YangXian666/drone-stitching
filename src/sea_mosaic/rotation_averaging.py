"""Stage A of the staged pose-graph architecture: pure rotation averaging.

Consumes edges' relative rotations, never GPS (see CLAUDE.md's 分階段架構決定). Each
node's orientation is a unit complex number, so there is no scale degree of freedom to
collapse. Without anchors the result's gauge (which node sits at angle 0) is arbitrary and
is replaced later by Stage C's alignment to the GPS frame.

Optional heading anchors (e.g. GimbalYawDegree, per CLAUDE.md's 範圍調整 of 2026-09-26)
enter as weak edges from an internal ground node at angle 0, so anchored components come
out in the anchors' absolute frame. The anchors suppress the drift that is self-cancelling
around loops but accumulates along lines. Without anchors the behaviour is exactly as
before; this module never reads metadata itself -- callers pass anchor values in radians.

Angle convention: a node's angle is the rotation of its image-to-mosaic pose, and an
edge's theta_rad is theta_dst - theta_src.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Anchor weight for GimbalYawDegree, relative to edge weights of inlier_count / median.
# Chosen from data/'s real-data sweep (0.01 / 0.1 / 1 / 10): at 1 the line-B drift slope
# fell from -0.496 to -0.033 deg/node, and 10 gave no further improvement. Data-derived,
# like the 5 m threshold: re-validate on the October dataset.
GIMBAL_YAW_ANCHOR_WEIGHT = 1.0

# Internal node id for the anchors' ground node; never returned to callers. Being the
# smallest id, it becomes the gauge (angle 0) of every component it joins.
_GROUND = -(1 << 62)


@dataclass(frozen=True)
class RelativeRotation:
    """One edge's relative-rotation measurement: theta_rad ~= theta_dst - theta_src."""

    src_index: int
    dst_index: int
    theta_rad: float
    weight: float  # must be finite and > 0


@dataclass
class RotationAveragingResult:
    """Per-node angles (radians, wrapped to [-pi, pi)) and connected-component labels.

    Angles are only meaningful relative to other nodes in the same component: each
    component is solved on its own, with its smallest node index as the gauge (angle 0).
    Component ids are 0, 1, ... in order of each component's smallest node index.
    """

    angles: dict[int, float]
    component_of: dict[int, int]


def relative_rotation_from_homography(
    homography: np.ndarray, image_shape: tuple[int, ...]
) -> float:
    """theta_ab (radians) for a pair whose homography maps src pixels to dst pixels.

    The rotation is read from the homography's local Jacobian at the src image centre,
    via polar decomposition -- not from homography[:2,:2], which (whenever the
    perspective row is non-zero) is not the local linearization at any point, and on real
    data was biased by a consistent-sign ~+1 deg per edge (see CLAUDE.md's 第二個座標系
    陷阱). For a 2x2 matrix with positive determinant, the polar-decomposition rotation
    angle is atan2(J10 - J01, J00 + J11). H ~= inv(pose_dst) @ pose_src, so
    theta_ab = theta_dst - theta_src = -angle(J).

    image_shape is the src image's numpy shape, (rows, cols[, channels]).

    Raises ValueError for a non-finite homography, or one whose local Jacobian at the
    centre is orientation-reversing or singular (a reflection is not a rotation, and
    silently returning an angle for it would hide exactly the kind of frame error
    CLAUDE.md records as the mirror bug).
    """
    H = np.asarray(homography, dtype=np.float64)
    if not np.all(np.isfinite(H)):
        raise ValueError("homography must be finite")

    rows, cols = image_shape[0], image_shape[1]
    centre = np.array([cols / 2.0, rows / 2.0, 1.0])
    projected = H @ centre
    w = projected[2]
    if not np.isfinite(w) or abs(w) < 1e-12:
        raise ValueError("homography maps the image centre to infinity")
    jacobian = (H[:2, :2] * w - np.outer(projected[:2], H[2, :2])) / w**2

    if np.linalg.det(jacobian) <= 0:
        raise ValueError("homography is orientation-reversing or singular at the image centre")

    angle = np.arctan2(jacobian[1, 0] - jacobian[0, 1], jacobian[0, 0] + jacobian[1, 1])
    return float(-angle)


def _wrap(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2 * np.pi) - np.pi)


def _validate(edges: list[RelativeRotation]) -> None:
    for edge in edges:
        if edge.src_index == edge.dst_index:
            raise ValueError(f"self-loop edge on node {edge.src_index}")
        if not np.isfinite(edge.theta_rad):
            raise ValueError(f"non-finite theta_rad on edge {edge.src_index}->{edge.dst_index}")
        if not (np.isfinite(edge.weight) and edge.weight > 0):
            raise ValueError(f"weight must be finite and > 0 on edge {edge.src_index}->{edge.dst_index}")


def _connected_components(edges: list[RelativeRotation]) -> list[list[int]]:
    """Connected components as sorted node lists, ordered by their smallest node."""
    neighbours: dict[int, set[int]] = {}
    for edge in edges:
        neighbours.setdefault(edge.src_index, set()).add(edge.dst_index)
        neighbours.setdefault(edge.dst_index, set()).add(edge.src_index)

    seen: set[int] = set()
    components = []
    for start in sorted(neighbours):
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            members.append(node)
            for other in neighbours[node] - seen:
                seen.add(other)
                stack.append(other)
        components.append(sorted(members))
    return components


def average_rotations(
    edges: list[RelativeRotation],
    heading_anchors: dict[int, float] | None = None,
    anchor_weight: float | None = None,
) -> RotationAveragingResult:
    """Spectral rotation averaging, optionally with weak absolute heading anchors.

    heading_anchors maps node -> absolute angle (radians) in the anchors' frame; each
    becomes an edge ground -> node with theta = anchor value and weight anchor_weight, which
    is required whenever anchors are given (no silent default). A node may appear through
    an anchor alone. Every component containing an anchored node is solved in the absolute
    frame; when its anchors disagree (sensor noise), the orientation is the least-squares
    compromise of all of them, not any single anchor's value. Components without anchors
    keep the smallest-index gauge. Component ids count the ground node, so components
    joined only through anchors share one id.
    """
    if heading_anchors is None:
        return _average_rotations(edges)
    if anchor_weight is None or not (np.isfinite(anchor_weight) and anchor_weight > 0):
        raise ValueError("anchor_weight must be given, finite and > 0 when heading_anchors are given")
    for node, value in heading_anchors.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite heading anchor for node {node}")
    anchor_edges = [
        RelativeRotation(_GROUND, node, float(value), float(anchor_weight))
        for node, value in heading_anchors.items()
    ]
    result = _average_rotations(list(edges) + anchor_edges)
    result.angles.pop(_GROUND, None)
    result.component_of.pop(_GROUND, None)
    return result


def _average_rotations(edges: list[RelativeRotation]) -> RotationAveragingResult:
    """Spectral rotation averaging over each connected component separately.

    Minimizes sum_edges w * |z_dst - e^{i theta} z_src|^2 over unit complex z, relaxed to
    sum |z|^2 = n: the relaxed solution is the eigenvector of the connection Laplacian
    L = D - W with the smallest eigenvalue (W[dst,src] = w e^{i theta}, Hermitian; D the
    weighted degrees), then each entry is projected back onto the unit circle. On
    consistent (noise-free) measurements that eigenvector is exactly z, with equal
    magnitude in every entry, for any connected graph and any positive weights.

    Not the principal eigenvector of the plain W: with heterogeneous weights (real
    inlier_count/median ranges ~0.01..11) that eigenvector localizes around the heaviest
    edges and, on a long chain, decays until entries underflow to exactly 0 and lose their
    phase -- found on data/'s 52-image run (see CLAUDE.md) and locked in by
    test_noise_free_long_chain_with_heterogeneous_weights_is_recovered_exactly.

    Nodes are exactly those appearing in edges; a node with no edges is outside Stage A's
    concern. Components are solved independently because their relative phase is
    unconstrained -- solving them jointly would silently produce meaningless relative
    angles between components.
    """
    _validate(edges)

    angles: dict[int, float] = {}
    component_of: dict[int, int] = {}
    for component_id, members in enumerate(_connected_components(edges)):
        position = {node: k for k, node in enumerate(members)}
        laplacian = np.zeros((len(members), len(members)), dtype=np.complex128)
        for edge in edges:
            if edge.src_index not in position:
                continue
            s, d = position[edge.src_index], position[edge.dst_index]
            measurement = edge.weight * np.exp(1j * edge.theta_rad)
            laplacian[d, s] -= measurement
            laplacian[s, d] -= np.conj(measurement)
            laplacian[s, s] += edge.weight
            laplacian[d, d] += edge.weight

        _, eigenvectors = np.linalg.eigh(laplacian)
        z = eigenvectors[:, 0]  # eigh sorts eigenvalues ascending
        gauge = np.angle(z[0])
        for node, k in position.items():
            angles[node] = _wrap(float(np.angle(z[k]) - gauge))
            component_of[node] = component_id

    return RotationAveragingResult(angles=angles, component_of=component_of)
