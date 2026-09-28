"""Edge-consistency check: the second line of defence for edges, independent of the matcher.

A homography that RANSAC accepted is not yet trusted: on open water SIFT+RANSAC returns
6-11 spurious inliers that pass the ">= 4 inliers" floor (CLAUDE.md's 738 張資料集結構與開闊
海面). Each edge is checked against GPS, in this order, and the FIRST failing check is its
rejection reason (REJECTION_ORDER):

  no_gps                 an end has no lat/lon
  too_short              GPS distance < MIN_GPS_DISPLACEMENT_M (5 m): no heading observation
  untrusted_node         an end's per-node heading consensus is not trusted (gps_lag's
                         heading_consensus on frame_alignment's GPS heading observations;
                         includes nodes with fewer than CONSENSUS_MIN_AGREE observations)
  heading_disagrees      an end's consensus does not agree with this edge's observation
  length                 |log(ppm_ij / ppm_mode)| > LENGTH_TOL_LOG, ppm_mode being the densest
                         +-PPM_MODE_WINDOW_LOG window over the edges that passed the heading
                         checks (so false edges cannot move it)
  orientation_reversing  the homography's Jacobian at the image centre has det <= 0
  rotation               |theta_ij - (phi_dst - phi_src)| > ROTATION_TOL_DEG, theta_ij the
                         centre-Jacobian rotation, phi the two ends' consensus headings

Parameters: CLAUDE.md's 邊一致性檢查：參數定案 (all data-derived except CONSENSUS_MIN_AGREE).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from sea_mosaic.frame_alignment import HeadingEdge, gps_heading_observations
from sea_mosaic.geo.projection import geodetic_to_local_xy
from sea_mosaic.gps_lag import HeadingObservation, NodeConsensus, heading_consensus
from sea_mosaic.gps_placement import MIN_GPS_DISPLACEMENT_M
from sea_mosaic.rotation_averaging import relative_rotation_from_homography
from sea_mosaic.types import PairResult

PPM_MODE_WINDOW_LOG = 0.08
LENGTH_TOL_LOG = 0.15
# 1.5 x the largest rotation residual (8.30 deg) among heading+length survivors whose BOTH
# ends have land >= 5% -- survivors judged true by land, not by any heading or rotation
# quantity (the first derivation, 14.8 deg, used mixed-stratum survivors and was overturned
# by one false edge at 17.3 deg; CLAUDE.md).
ROTATION_TOL_DEG = 12.45

REJECTION_ORDER = (
    "no_gps", "too_short", "untrusted_node", "heading_disagrees", "length", "orientation_reversing", "rotation"
)


@dataclass
class EdgeCheck:
    """accepted: the edges that passed, in input order. rejected: (src, dst) -> first failing
    reason. consensus: per-node heading consensus used by the heading and rotation checks.
    ppm_mode: NaN when no edge passed the heading checks; ppm_support: edges inside its window."""

    accepted: list[PairResult]
    rejected: dict[tuple[int, int], str]
    consensus: dict[int, NodeConsensus]
    ppm_mode: float
    ppm_support: int


def check_edge_consistency(
    pairs: list[PairResult],
    latlons: Mapping[int, tuple[float, float]] | None,
    image_shapes: Mapping[int, tuple[int, ...]],
) -> EdgeCheck:
    """Run the checks described in the module docstring on every pair."""
    latlons = latlons or {}
    located = {k for k, v in latlons.items() if v is not None and bool(np.all(np.isfinite(v)))}
    rejected: dict[tuple[int, int], str] = {}
    distance: dict[tuple[int, int], float] = {}
    for p in pairs:
        key = (p.src_index, p.dst_index)
        if p.src_index not in located or p.dst_index not in located:
            rejected[key] = "no_gps"
            continue
        distance[key] = float(np.hypot(*geodetic_to_local_xy(*latlons[p.dst_index], *latlons[p.src_index])))
        if distance[key] < MIN_GPS_DISPLACEMENT_M:
            rejected[key] = "too_short"

    candidates = [p for p in pairs if (p.src_index, p.dst_index) not in rejected]
    heading_edges = [HeadingEdge(p.src_index, p.dst_index, p.homography, image_shapes[p.src_index], 1.0) for p in candidates]
    observations = gps_heading_observations(
        {k: latlons[k] for k in located}, heading_edges, dict(image_shapes), min_gps_distance_m=MIN_GPS_DISPLACEMENT_M
    )
    consensus = {
        node: heading_consensus([HeadingObservation(edge, h, sigma) for edge, h, sigma, _w in obs])
        for node, obs in observations.items()
    }

    heading_ok = []
    for p in candidates:
        key = (p.src_index, p.dst_index)
        ends = [consensus.get(n) for n in key]
        if any(c is None or not c.trusted for c in ends):
            rejected[key] = "untrusted_node"
        elif not all(c.agrees.get(key, False) for c in ends):
            rejected[key] = "heading_disagrees"
        else:
            heading_ok.append(p)

    ppm = {(p.src_index, p.dst_index): _pixels_per_meter(p, image_shapes, distance) for p in heading_ok}
    ppm_mode, ppm_support = _densest_log_window(list(ppm.values()), PPM_MODE_WINDOW_LOG)
    for p in heading_ok:
        key = (p.src_index, p.dst_index)
        if abs(np.log(ppm[key] / ppm_mode)) > LENGTH_TOL_LOG:
            rejected[key] = "length"
            continue
        try:
            theta = relative_rotation_from_homography(p.homography, image_shapes[p.src_index])
        except ValueError:  # orientation-reversing (or singular) at the image centre
            rejected[key] = "orientation_reversing"
            continue
        expected = consensus[p.dst_index].heading_rad - consensus[p.src_index].heading_rad
        residual = (theta - expected + np.pi) % (2 * np.pi) - np.pi
        if abs(np.degrees(residual)) > ROTATION_TOL_DEG:
            rejected[key] = "rotation"

    return EdgeCheck(
        accepted=[p for p in pairs if (p.src_index, p.dst_index) not in rejected],
        rejected=rejected,
        consensus=consensus,
        ppm_mode=ppm_mode,
        ppm_support=ppm_support,
    )


def _pixels_per_meter(p: PairResult, image_shapes, distance) -> float:
    """|where the dst centre lands in src pixels - the src centre| / GPS distance, the same
    measurement as gps_placement.estimate_pixels_per_meter."""
    rows, cols = image_shapes[p.src_index][0], image_shapes[p.src_index][1]
    centre = np.array([cols / 2.0, rows / 2.0, 1.0])
    mapped = np.linalg.inv(p.homography) @ centre
    return float(np.linalg.norm(mapped[:2] / mapped[2] - centre[:2])) / distance[(p.src_index, p.dst_index)]


def _densest_log_window(values: list[float], half_width: float) -> tuple[float, int]:
    """The value (one of the inputs) whose +-half_width window in log holds the most values;
    ties -> the smallest. (nan, 0) for no input."""
    if not values:
        return float("nan"), 0
    logs = np.sort(np.log(values))
    counts = [int(np.sum(np.abs(logs - v) <= half_width)) for v in logs]
    best = int(np.argmax(counts))
    return float(np.exp(logs[best])), counts[best]
