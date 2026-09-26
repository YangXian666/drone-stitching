"""Stage C of the staged pose-graph architecture: align Stage A's rotations to Stage B's
north-up pixel frame using GPS track bearing, then compose centre-anchored poses.

For an edge (i, j): beta_ij is the direction of Stage B's p_j - p_i in the north-up pixel
frame, and alpha_ij is the direction of travel measured in src image i's own pixel frame
(where dst's centre lands, inv(H) applied to i's centre). A pose maps image vectors into
the frame by R(phi_i), so beta_ij = phi_i + alpha_ij; with phi_i = theta_i + delta (Stage
A's angle plus one offset per component), each edge estimates
delta_ij = beta_ij - alpha_ij - theta_i. delta is their weighted circular mean (complex
sum, no wraparound). In this frame (y down) a pose's rotation angle equals the camera's
compass yaw.

One delta per component removes only Stage A's arbitrary gauge -- not its along-line
drift; that is left to Stage D (see CLAUDE.md's 最小 GPS 位移閾值與 Stage C 漂移發現).
Per CLAUDE.md's 最小可行版本範圍決定, no GimbalYawDegree or DJI XMP field is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sea_mosaic.gps_placement import MIN_GPS_DISPLACEMENT_M, GpsPlacement
from sea_mosaic.rotation_averaging import RotationAveragingResult


@dataclass(frozen=True)
class HeadingEdge:
    """One image pair's evidence for aligning its src node's heading to GPS track bearing."""

    src_index: int
    dst_index: int
    homography: np.ndarray  # src pixels -> dst pixels
    src_image_shape: tuple[int, ...]  # numpy shape (rows, cols[, channels])
    weight: float  # must be finite and > 0; how edges are weighted is the caller's choice


@dataclass
class AlignedPoses:
    """Centre-anchored image-to-frame poses for every node whose position AND heading are
    known, plus explicit lists of every node or component that could not get one.

    poses[k] @ (image centre) == Stage B's centre for k, with rotation headings_rad[k] and
    scale 1 (per-node scale is Stage D's concern).
    """

    poses: dict[int, np.ndarray]
    headings_rad: dict[int, float]
    component_offsets_rad: dict[int, float]  # delta per Stage A component id
    unaligned_components: set[int]  # Stage A components with no usable edge
    unlocated: set[int]  # Stage A angle, but no Stage B position
    unoriented: set[int]  # Stage B position, but no Stage A angle


def _image_centre(image_shape: tuple[int, ...]) -> np.ndarray:
    return np.array([image_shape[1] / 2.0, image_shape[0] / 2.0])


def _wrap(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2 * np.pi) - np.pi)


def _validate(edges: list[HeadingEdge], pixels_per_meter: float) -> None:
    if not (np.isfinite(pixels_per_meter) and pixels_per_meter > 0):
        raise ValueError("pixels_per_meter must be finite and > 0")
    for edge in edges:
        if not np.all(np.isfinite(edge.homography)):
            raise ValueError(f"non-finite homography on edge {edge.src_index}->{edge.dst_index}")
        if not (np.isfinite(edge.weight) and edge.weight > 0):
            raise ValueError(f"weight must be finite and > 0 on edge {edge.src_index}->{edge.dst_index}")


def align_to_gps_frame(
    stage_a: RotationAveragingResult,
    placement: GpsPlacement,
    edges: list[HeadingEdge],
    pixels_per_meter: float,
    image_shapes: dict[int, tuple[int, ...]],
    min_gps_distance_m: float = MIN_GPS_DISPLACEMENT_M,
) -> AlignedPoses:
    """Estimate one heading offset per Stage A component and compose centre-anchored poses.

    An edge is usable when both endpoints have a Stage A angle in the same component and a
    Stage B position, and their GPS displacement is at least min_gps_distance_m (converted
    to pixels with pixels_per_meter). A component with no usable edge is reported in
    unaligned_components and its nodes get no pose.

    Raises ValueError for a non-finite homography, a non-positive weight or
    pixels_per_meter, or a node that would get a pose but has no entry in image_shapes.
    """
    _validate(edges, pixels_per_meter)
    centres = placement.centres_px
    min_distance_px = min_gps_distance_m * pixels_per_meter

    sums: dict[int, complex] = {}
    for edge in edges:
        i, j = edge.src_index, edge.dst_index
        if i not in stage_a.angles or j not in stage_a.angles or i not in centres or j not in centres:
            continue
        component = stage_a.component_of[i]
        if stage_a.component_of[j] != component:
            continue
        gps_vector = centres[j] - centres[i]
        if np.linalg.norm(gps_vector) < min_distance_px:
            continue
        centre = _image_centre(edge.src_image_shape)
        mapped = np.linalg.inv(edge.homography) @ np.append(centre, 1.0)
        image_vector = mapped[:2] / mapped[2] - centre
        beta = np.arctan2(gps_vector[1], gps_vector[0])
        alpha = np.arctan2(image_vector[1], image_vector[0])
        delta_ij = beta - alpha - stage_a.angles[i]
        sums[component] = sums.get(component, 0j) + edge.weight * np.exp(1j * delta_ij)

    offsets = {component: float(np.angle(total)) for component, total in sums.items()}
    components = set(stage_a.component_of.values())

    poses: dict[int, np.ndarray] = {}
    headings: dict[int, float] = {}
    for node, theta in stage_a.angles.items():
        component = stage_a.component_of[node]
        if component not in offsets or node not in centres:
            continue
        if node not in image_shapes:
            raise ValueError(f"no image shape for node {node}")
        phi = _wrap(theta + offsets[component])
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        pose = np.eye(3)
        pose[:2, :2] = R
        pose[:2, 2] = centres[node] - R @ _image_centre(image_shapes[node])
        poses[node] = pose
        headings[node] = phi

    return AlignedPoses(
        poses=poses,
        headings_rad=headings,
        component_offsets_rad=offsets,
        unaligned_components=components - set(offsets),
        unlocated={node for node in stage_a.angles if node not in centres},
        unoriented={node for node in centres if node not in stage_a.angles},
    )
