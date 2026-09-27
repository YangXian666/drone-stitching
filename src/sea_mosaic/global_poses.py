"""Stage A->D chained into one call: pair results + EXIF lat/lon -> GlobalTransforms.

This is pure wiring, reproducing exactly the procedure validated on real data (CLAUDE.md's
Stage A～D 真實資料驗收): the numbers recorded there are this path's quality evidence, so no
weight or parameter may change here without re-validating.

1. Edge filter: an edge is used only with >= MIN_INLIERS_FOR_DETERMINED_HOMOGRAPHY inliers
   and a finite homography. Only nodes with at least one used edge ("evidenced" nodes)
   take part in any stage -- metadata alone (GPS, a gimbal reading) is not image evidence
   and must not give a node a pose, nor move the Stage B origin.
2. pixels_per_meter estimated from the data (gps_placement.estimate_pixels_per_meter).
3. Stage B: place_by_gps, origin = smallest evidenced node with GPS.
4. Stage A: average_rotations, edge weight inlier_count / median(inlier_count of used edges),
   heading anchors from heading_anchor_source ("gps": gps_heading_anchors; "gimbal":
   GimbalYawDegree in radians, no sign flip -- in the north-up pixel frame an image's pose
   angle is its compass bearing; "none": no anchors).
5. Stage C: align_to_gps_frame, every heading edge weight 1.0.
6. Stage D: refine_poses on the used edges' inliers.

The output frame is north-up pixels (x = East, y = -North), so GlobalTransforms has no
reference image: reference_index is None, residual_error is np.nan (Stage D reports two
separate term RMS values, kept in term_rms). Every node without a pose is listed in
node_failure_reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sea_mosaic.frame_alignment import (
    GPS_HEADING_ANCHOR_WEIGHT,
    HeadingEdge,
    align_to_gps_frame,
    gps_heading_anchors,
)
from sea_mosaic.gps_placement import PairDisplacement, estimate_pixels_per_meter, place_by_gps
from sea_mosaic.refinement import BoundHits, edge_from_pair_result, refine_poses
from sea_mosaic.rotation_averaging import (
    GIMBAL_YAW_ANCHOR_WEIGHT,
    RelativeRotation,
    average_rotations,
    relative_rotation_from_homography,
)
from sea_mosaic.types import GlobalTransforms, PairResult

# A homography has 8 degrees of freedom; 4 point correspondences (8 equations) is the
# mathematical minimum needed to determine one uniquely. Below this threshold, the
# system is underdetermined -- there's no meaningful sense in which a result could even
# be evaluated for reliability, it's arbitrary by construction.
#
# This threshold does NOT claim that >=4 inliers means the homography IS reliable.
# CLAUDE.md's Check A diagnostic already proved the opposite: even a well-determined,
# high-inlier edge (inlier_count=717, far above this floor) can still get amplified into
# a collapsed result once folded into a joint optimization. "Solvable" and "trustworthy"
# are different claims; this constant only rules out the specific failure mode of "there
# wasn't even enough data to ask the question at all." (Moved here from pipeline.py,
# which no longer filters edges itself.)
MIN_INLIERS_FOR_DETERMINED_HOMOGRAPHY = 4

HEADING_ANCHOR_SOURCES = ("gps", "gimbal", "none")


@dataclass
class GlobalPoseEstimate:
    """GlobalTransforms for downstream warp/blend/metrics, plus Stage A-D diagnostics.

    failure_reason is None on success, otherwise one of "no_gps",
    "pixels_per_meter_unavailable", "no_posed_nodes", "refinement_not_converged" (in the
    last case the not-converged poses are still returned, as refine_poses does).
    node_failure_reasons maps every node without a pose to "no_determined_edge",
    "unlocated", "unoriented", "unaligned_component", or the run's failure_reason when the
    whole run failed before any node could be posed.
    """

    global_transforms: GlobalTransforms
    failure_reason: str | None
    node_failure_reasons: dict[int, str]
    heading_anchor_source: str
    pixels_per_meter: float  # np.nan when it could not be estimated
    kappa: float  # np.nan when Stage D did not run
    component_offsets_rad: dict[int, float]
    bound_hits: BoundHits | None
    skipped_edges: dict[tuple[int, int], str]
    term_rms: dict[str, float]
    irls_rounds: int


def _validate(heading_anchor_source: str, gimbal_yaw_deg: dict[int, float] | None) -> None:
    if heading_anchor_source not in HEADING_ANCHOR_SOURCES:
        raise ValueError(
            f"heading_anchor_source must be one of {HEADING_ANCHOR_SOURCES}, got {heading_anchor_source!r}"
        )
    if heading_anchor_source == "gimbal" and gimbal_yaw_deg is None:
        raise ValueError('heading_anchor_source="gimbal" requires gimbal_yaw_deg')
    if heading_anchor_source != "gimbal" and gimbal_yaw_deg is not None:
        raise ValueError(
            f'gimbal_yaw_deg was given but heading_anchor_source is {heading_anchor_source!r}; '
            'pass heading_anchor_source="gimbal" to use it (it is never silently ignored)'
        )


def _failed(
    reason: str,
    node_failure_reasons: dict[int, str],
    heading_anchor_source: str,
    pixels_per_meter: float = float(np.nan),
) -> GlobalPoseEstimate:
    return GlobalPoseEstimate(
        global_transforms=GlobalTransforms(
            transforms={}, reference_index=None, optimization_status="failed", residual_error=float(np.nan)
        ),
        failure_reason=reason,
        node_failure_reasons=node_failure_reasons,
        heading_anchor_source=heading_anchor_source,
        pixels_per_meter=pixels_per_meter,
        kappa=float(np.nan),
        component_offsets_rad={},
        bound_hits=None,
        skipped_edges={},
        term_rms={},
        irls_rounds=0,
    )


def estimate_global_poses(
    pair_results: list[PairResult],
    image_shapes: dict[int, tuple[int, ...]],
    latlons: dict[int, tuple[float, float]] | None,
    *,
    heading_anchor_source: Literal["gps", "gimbal", "none"] = "gps",
    gimbal_yaw_deg: dict[int, float] | None = None,
) -> GlobalPoseEstimate:
    """Run Stage A->D (see the module docstring for the exact procedure).

    image_shapes must list every input image, including ones without any edge. latlons
    must be EXIF-sourced (gps_placement.load_exif_latlons); gimbal_yaw_deg (degrees,
    io_utils.load_gimbal_yaw) is used only, and required, with heading_anchor_source
    "gimbal". Raises ValueError for an invalid heading_anchor_source / gimbal_yaw_deg
    combination; every data-dependent failure is reported, not raised.
    """
    _validate(heading_anchor_source, gimbal_yaw_deg)

    usable = [
        p
        for p in pair_results
        if p.inlier_count >= MIN_INLIERS_FOR_DETERMINED_HOMOGRAPHY and np.all(np.isfinite(p.homography))
    ]
    evidenced = {p.src_index for p in usable} | {p.dst_index for p in usable}
    reasons = {k: "no_determined_edge" for k in image_shapes if k not in evidenced}
    located = {k: v for k, v in (latlons or {}).items() if k in evidenced}

    if not latlons:
        return _failed("no_gps", reasons | {k: "unlocated" for k in evidenced}, heading_anchor_source)

    try:
        pixels_per_meter = estimate_pixels_per_meter(
            [
                PairDisplacement(p.homography, image_shapes[p.src_index], located[p.src_index], located[p.dst_index])
                for p in usable
                if p.src_index in located and p.dst_index in located
            ]
        )
    except ValueError:
        return _failed(
            "pixels_per_meter_unavailable",
            reasons | {k: "pixels_per_meter_unavailable" for k in evidenced},
            heading_anchor_source,
        )

    placement = place_by_gps(located, pixels_per_meter, node_indices=evidenced)

    median_inliers = float(np.median([p.inlier_count for p in usable]))
    rotation_edges = [
        RelativeRotation(
            p.src_index,
            p.dst_index,
            relative_rotation_from_homography(p.homography, image_shapes[p.src_index]),
            p.inlier_count / median_inliers,
        )
        for p in usable
    ]
    heading_edges = [
        HeadingEdge(p.src_index, p.dst_index, p.homography, image_shapes[p.src_index], 1.0) for p in usable
    ]
    if heading_anchor_source == "gps":
        stage_a = average_rotations(
            rotation_edges,
            heading_anchors=gps_heading_anchors(located, heading_edges, image_shapes),
            anchor_weight=GPS_HEADING_ANCHOR_WEIGHT,
        )
    elif heading_anchor_source == "gimbal":
        stage_a = average_rotations(
            rotation_edges,
            heading_anchors={k: float(np.radians(v)) for k, v in gimbal_yaw_deg.items() if k in evidenced},
            anchor_weight=GIMBAL_YAW_ANCHOR_WEIGHT,
        )
    else:
        stage_a = average_rotations(rotation_edges)

    stage_c = align_to_gps_frame(stage_a, placement, heading_edges, pixels_per_meter, image_shapes)
    stage_d = refine_poses(
        stage_c,
        placement,
        [edge_from_pair_result(p, image_shapes[p.src_index]) for p in usable],
        pixels_per_meter,
        image_shapes,
    )

    unaligned_nodes = {k for k, c in stage_a.component_of.items() if c in stage_c.unaligned_components}
    for k in sorted(evidenced - set(stage_d.poses)):
        if k in placement.unlocated:
            reasons[k] = "unlocated"
        elif k in unaligned_nodes:
            reasons[k] = "unaligned_component"
        else:
            reasons[k] = "unoriented"

    if not stage_d.poses:
        return _failed("no_posed_nodes", reasons, heading_anchor_source, pixels_per_meter)

    return GlobalPoseEstimate(
        global_transforms=GlobalTransforms(
            transforms=dict(stage_d.poses),
            reference_index=None,
            optimization_status=stage_d.status,
            residual_error=float(np.nan),
        ),
        failure_reason=None if stage_d.status == "converged" else "refinement_not_converged",
        node_failure_reasons=reasons,
        heading_anchor_source=heading_anchor_source,
        pixels_per_meter=pixels_per_meter,
        kappa=stage_d.kappa,
        component_offsets_rad=dict(stage_c.component_offsets_rad),
        bound_hits=stage_d.bound_hits,
        skipped_edges=dict(stage_d.skipped_edges),
        term_rms=dict(stage_d.term_rms),
        irls_rounds=stage_d.irls_rounds,
    )
