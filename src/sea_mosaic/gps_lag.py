"""GPS recording lag: estimate it from the data and remove it at the GPS input.

Each GPS fix sits L metres ahead of where the image was exposed, along the direction of
travel (CLAUDE.md's GPS 記錄時間延遲：影響全 pipeline 的根因: ~2.2-2.35 m on data/, found
twice independently). Fixes on the same line shift alike and cancel in any bearing between
them; across lines (opposite directions) and through turns they do not, which bends the GPS
bearing beta used by Stage A's GPS heading anchors and Stage C, and moves Stage B's
placement and Stage D's GPS targets. The correction is applied once, to the lat/lon every
stage receives:

    p' = p - L * u        (u: the node's unit travel direction, East/North)

u comes from the node's time-adjacent GPS fixes within one flight segment (central
difference, one-sided at segment ends). Capture time (EXIF DateTimeOriginal, plus
SubSecTimeOriginal when present) only orders the fixes and splits segments at gaps; it
never enters the geometry.

L is estimated per run, never hard-coded (a different aircraft or speed gives a different
L): L0 = 0 -> per-node heading consensus on h = beta - alpha (frame_alignment's GPS heading
observations; beta from lag-corrected GPS) -> trusted edges -> the grid L minimizing the sum
over the trusted observations of Huber(z), z = wrap(h - node's 1/sigma^2-weighted circular
mean) / sigma -> repeat until L moves by at most one grid step. (A median-of-node-medians
objective was tried first and dropped: it only reacts once opposite-direction edges are the
majority at a node, so no edge-count threshold could transfer between datasets.) Validated
on data/ and the 738-image set: converges within three rounds from L0 = 0, 1 and 4 m to
2.15 m for both the full sample and the 52 coastal images; no trusted node (open water) ->
not estimable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from sea_mosaic.frame_alignment import HeadingEdge, gps_heading_observations
from sea_mosaic.geo.projection import R_EARTH_M, geodetic_to_local_xy
from sea_mosaic.io_utils import load_capture_time_s

# Split flight segments where consecutive capture times differ by more than this many times
# the median interval. On data/ and the 738-image set any factor in (1.67, 7) gives the same
# split (median 3 s, normal intervals <= 5 s, gaps 21, 36, 2616 s); 3 lies inside. Like the
# 5 m threshold: the data does not pin the value down; re-validate on the October dataset.
CAPTURE_GAP_FACTOR = 3.0

# Per-node heading consensus. PROVISIONAL: these are the values the lag estimate was
# validated with; the final values are decided with the edge-consistency check, after
# Stage A-D is re-validated on lag-corrected GPS (CLAUDE.md).
CONSENSUS_Z = 3.0  # an observation agrees if |h - consensus| <= Z * its own sigma
CONSENSUS_MIN_AGREE = 3  # a node needs at least this many agreeing observations ...
CONSENSUS_MIN_SHARE = 0.6  # ... making up at least this share of its observations

# Lag search: grid over [min, max] in steps; stop when L moves by <= one step.
LAG_SEARCH_MIN_M = -10.0
LAG_SEARCH_MAX_M = 10.0
LAG_SEARCH_STEP_M = 0.05
LAG_MAX_ROUNDS = 10
# The lag is observable only through edges whose ends travel in different directions
# (same-direction fixes shift alike): require this many trusted edges between nodes
# travelling in opposite directions (u_i . u_j < 0: cross-line or across a U-turn).
# Criterion fixed in advance: lag error p95 <= the per-fix GPS sigma (0.37 m). Synthetic
# surveys (2 lines x 12 and 3 lines x 10 nodes, 0.37 m GPS noise, 0.7 px points, 100 trials
# per count): 10 is the smallest count with p95 AND p99 <= 0.37 m in both geometries (7 and
# 8 pass p95 but not p99 in the 3-line survey). Real data has 69-116. Re-validate on the
# October dataset (CLAUDE.md).
LAG_MIN_OPPOSITE_EDGES = 10
HUBER_C = 1.345  # same constant as frame_alignment (95% efficiency under normal noise)

EdgeKey = tuple[int, int]


@dataclass(frozen=True)
class HeadingObservation:
    """One edge's estimate h of one end node's heading (radians, north-up pixel frame)."""

    edge: EdgeKey
    h_rad: float
    sigma_rad: float


@dataclass(frozen=True)
class NodeConsensus:
    """heading_rad: 1/sigma^2-weighted circular mean of the largest agreeing subset;
    agrees: per edge, whether its observation agrees with heading_rad; trusted: at least
    CONSENSUS_MIN_AGREE agreeing observations and a share of at least CONSENSUS_MIN_SHARE.
    An untrusted node fails as a whole: none of its edges count as trusted."""

    heading_rad: float
    agrees: dict[EdgeKey, bool]
    n_agree: int
    n_obs: int
    trusted: bool


@dataclass
class GpsLagEstimate:
    """lag_m is the lag actually applied: 0.0 unless status == "estimated".

    status: "estimated" | "not_estimable" | "no_capture_times"
    reason (for "not_estimable"): "no_gps", "no_trusted_edges", "not_identifiable",
        "search_bound" (optimum on the grid edge), "not_converged" (LAG_MAX_ROUNDS used up)
    history_m: L per round, starting with L0 = 0.0; rounds = len(history_m) - 1
    trusted_nodes: trusted nodes in the last consensus round
    uncorrected_nodes: nodes with GPS that got no correction (no capture time, or no
        travel direction) -- all of them when the lag was not applied
    """

    lag_m: float
    status: Literal["estimated", "not_estimable", "no_capture_times"]
    reason: str | None
    rounds: int
    history_m: list[float]
    trusted_nodes: int
    uncorrected_nodes: set[int] = field(default_factory=set)


def load_exif_capture_times(paths: Mapping[int, Path]) -> dict[int, float]:
    """Capture time in seconds per image index (io_utils.load_capture_time_s: EXIF only).
    Images without an EXIF capture time are left out."""
    times = {}
    for index, path in paths.items():
        t = load_capture_time_s(path)
        if t is not None:
            times[index] = t
    return times


def flight_segments(capture_times_s: Mapping[int, float]) -> list[list[int]]:
    """Node indices in capture-time order (ties by index), split where the interval to the
    next node exceeds CAPTURE_GAP_FACTOR * the median interval (equal does not split)."""
    order = sorted(capture_times_s, key=lambda k: (capture_times_s[k], k))
    if len(order) < 2:
        return [order] if order else []
    intervals = np.diff([capture_times_s[k] for k in order])
    threshold = CAPTURE_GAP_FACTOR * float(np.median(intervals))
    segments, current = [], [order[0]]
    for node, interval in zip(order[1:], intervals):
        if interval > threshold:
            segments.append(current)
            current = []
        current.append(node)
    segments.append(current)
    return segments


def _usable(latlon) -> bool:
    return latlon is not None and bool(np.all(np.isfinite(latlon)))


def travel_directions(
    latlons: Mapping[int, tuple[float, float]], capture_times_s: Mapping[int, float]
) -> dict[int, np.ndarray]:
    """Unit (East, North) travel direction per node that has both GPS and a capture time
    and is not alone in its flight segment; segments are built over those nodes only."""
    nodes = {k: capture_times_s[k] for k in capture_times_s if k in latlons and _usable(latlons[k])}
    directions = {}
    for segment in flight_segments(nodes):
        if len(segment) < 2:
            continue
        for n, node in enumerate(segment):
            before, after = segment[max(n - 1, 0)], segment[min(n + 1, len(segment) - 1)]
            v = geodetic_to_local_xy(*latlons[after], *latlons[before])
            norm = float(np.hypot(*v))
            if norm > 0:
                directions[node] = v / norm
    return directions


def shift_latlons(
    latlons: Mapping[int, tuple[float, float]], directions: Mapping[int, np.ndarray], lag_m: float
) -> dict[int, tuple[float, float]]:
    """p' = p - lag_m * u for nodes with a direction; others unchanged. Uses the exact
    inverse of geo.projection's local equirectangular approximation around each point."""
    out = {}
    for node, latlon in latlons.items():
        if node not in directions or lag_m == 0.0:
            out[node] = latlon
            continue
        lat, lon = latlon
        east_m, north_m = -lag_m * np.asarray(directions[node], dtype=np.float64)
        out[node] = (
            lat + float(np.degrees(north_m / R_EARTH_M)),
            lon + float(np.degrees(east_m / (R_EARTH_M * np.cos(np.radians(lat))))),
        )
    return out


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _weighted_circular_mean(h: np.ndarray, sigma: np.ndarray) -> float:
    w = 1.0 / sigma**2
    return float(np.arctan2(np.sum(w * np.sin(h)), np.sum(w * np.cos(h))))


def heading_consensus(observations: list[HeadingObservation]) -> NodeConsensus:
    """Per-node consensus over the node's heading observations (see NodeConsensus).

    The largest subset within CONSENSUS_Z sigma of any one observation (ties: the first)
    seeds the consensus; its weighted circular mean is the heading, and agreement is then
    re-evaluated around that heading."""
    if not observations:
        return NodeConsensus(float("nan"), {}, 0, 0, False)
    h = np.array([o.h_rad for o in observations])
    sigma = np.array([o.sigma_rad for o in observations])
    counts = [int(np.sum(np.abs(_wrap(h - h[k])) / sigma <= CONSENSUS_Z)) for k in range(len(h))]
    seed = np.abs(_wrap(h - h[int(np.argmax(counts))])) / sigma <= CONSENSUS_Z
    heading = _weighted_circular_mean(h[seed], sigma[seed])
    agree = np.abs(_wrap(h - heading)) / sigma <= CONSENSUS_Z
    n_agree, n_obs = int(agree.sum()), len(h)
    return NodeConsensus(
        heading_rad=heading,
        agrees={o.edge: bool(a) for o, a in zip(observations, agree)},
        n_agree=n_agree,
        n_obs=n_obs,
        trusted=n_agree >= CONSENSUS_MIN_AGREE and n_agree / n_obs >= CONSENSUS_MIN_SHARE,
    )


def _observations(heading_edges, latlons, image_shapes) -> dict[int, list[HeadingObservation]]:
    return {
        node: [HeadingObservation(edge, h, sigma) for edge, h, sigma, _w in obs]
        for node, obs in gps_heading_observations(dict(latlons), heading_edges, dict(image_shapes)).items()
    }


def _huber_objective(observations: dict[int, list[HeadingObservation]]) -> float:
    total = 0.0
    for obs in observations.values():
        if len(obs) < CONSENSUS_MIN_AGREE:
            continue
        h = np.array([o.h_rad for o in obs])
        sigma = np.array([o.sigma_rad for o in obs])
        z = np.abs(_wrap(h - _weighted_circular_mean(h, sigma))) / sigma
        total += float(np.sum(np.where(z <= HUBER_C, 0.5 * z**2, HUBER_C * (z - 0.5 * HUBER_C))))
    return total


def _not_estimable(reason, history, trusted_nodes, located) -> GpsLagEstimate:
    return GpsLagEstimate(
        lag_m=0.0, status="not_estimable", reason=reason, rounds=len(history) - 1,
        history_m=list(history), trusted_nodes=trusted_nodes, uncorrected_nodes=set(located),
    )


def estimate_gps_lag(
    heading_edges: list[HeadingEdge],
    latlons: Mapping[int, tuple[float, float]] | None,
    capture_times_s: Mapping[int, float] | None,
    image_shapes: Mapping[int, tuple[int, ...]],
) -> GpsLagEstimate:
    """The iterative estimate described in the module docstring. Never raises for data
    reasons: every failure is a status/reason with lag_m = 0.0."""
    located = {k for k, v in (latlons or {}).items() if _usable(v)}
    if not located:
        return _not_estimable("no_gps", [0.0], 0, located)
    if not capture_times_s:
        return GpsLagEstimate(0.0, "no_capture_times", None, 0, [0.0], 0, set(located))

    directions = travel_directions(latlons, capture_times_s)
    grid = np.round(np.arange(LAG_SEARCH_MIN_M, LAG_SEARCH_MAX_M + LAG_SEARCH_STEP_M / 2, LAG_SEARCH_STEP_M), 10)
    lag, history, trusted_nodes = 0.0, [0.0], 0
    for _ in range(LAG_MAX_ROUNDS):
        consensus = {n: heading_consensus(o) for n, o in _observations(heading_edges, shift_latlons(latlons, directions, lag), image_shapes).items()}
        trusted_nodes = sum(c.trusted for c in consensus.values())
        trusted = [
            e for e in heading_edges
            if all(
                n in consensus and consensus[n].trusted and consensus[n].agrees.get((e.src_index, e.dst_index), False)
                for n in (e.src_index, e.dst_index)
            )
        ]
        if not trusted:
            return _not_estimable("no_trusted_edges", history, trusted_nodes, located)
        opposite = sum(
            1 for e in trusted
            if e.src_index in directions and e.dst_index in directions
            and float(directions[e.src_index] @ directions[e.dst_index]) < 0
        )
        if opposite < LAG_MIN_OPPOSITE_EDGES:
            return _not_estimable("not_identifiable", history, trusted_nodes, located)
        values = [_huber_objective(_observations(trusted, shift_latlons(latlons, directions, float(g)), image_shapes)) for g in grid]
        best = int(np.argmin(values))  # first minimum on ties
        new = float(grid[best])
        history.append(new)
        if best in (0, len(grid) - 1):
            return _not_estimable("search_bound", history, trusted_nodes, located)
        if abs(new - lag) <= LAG_SEARCH_STEP_M + 1e-9:
            return GpsLagEstimate(
                lag_m=new, status="estimated", reason=None, rounds=len(history) - 1, history_m=history,
                trusted_nodes=trusted_nodes, uncorrected_nodes=located - set(directions),
            )
        lag = new
    return _not_estimable("not_converged", history, trusted_nodes, located)
