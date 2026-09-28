"""Tests for sea_mosaic.gps_lag: estimating and removing the GPS recording lag.

Each GPS fix sits L metres ahead of where the image was actually exposed, along the
direction of travel (CLAUDE.md's GPS 記錄時間延遲：影響全 pipeline 的根因). The correction
is p' = p - L * u, with u the node's travel direction from its time-adjacent GPS fixes.
L is estimated per run: L0 = 0 -> per-node heading consensus -> the L that minimizes the
heading spread on the trusted edges -> repeat.

Expected values come from the synthetic pinhole camera (tests/synthetic_camera.py) and
hand derivations, never from calling a gps_lag function to produce its own expectation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sea_mosaic.frame_alignment import HeadingEdge
from sea_mosaic.geo.projection import geodetic_to_local_xy
from sea_mosaic.gps_lag import (
    CAPTURE_GAP_FACTOR,
    CONSENSUS_MIN_AGREE,
    CONSENSUS_MIN_SHARE,
    CONSENSUS_Z,
    LAG_MIN_OPPOSITE_EDGES,
    LAG_SEARCH_STEP_M,
    HeadingObservation,
    estimate_gps_lag,
    flight_segments,
    heading_consensus,
    load_exif_capture_times,
    shift_latlons,
    travel_directions,
)
from synthetic_camera import IMAGE_SHAPE, SyntheticCamera, ground_to_image, latlon_from_en, plane_homography

ALTITUDE = 100.0
STEP_M = 13.4  # along-track spacing, like data/
INTERVAL_S = 2.5
MAX_PAIR_DISTANCE_M = 40.0
# Lag recovery tolerance: two search-grid steps. One covers the grid quantization; the
# other the U-turn, where the estimator's travel directions come from the lagged GPS while
# the synthetic lag was applied along the true directions (they differ only at turns).
LAG_TOL_M = 2 * LAG_SEARCH_STEP_M


# ---------------------------------------------------------------------------
# Synthetic survey
# ---------------------------------------------------------------------------


def _line(start_en, course_deg, n, yaw_deg):
    c = math.radians(course_deg)
    return [
        SyntheticCamera(start_en[0] + k * STEP_M * math.sin(c), start_en[1] + k * STEP_M * math.cos(c), ALTITUDE, yaw_deg)
        for k in range(n)
    ]


def _serpentine(n_per_line: int = 8) -> list[SyntheticCamera]:
    """Eastbound line, two-image U-turn, westbound line 27 m north, in flight order.
    Camera yaw differs from the course by ~12 deg (crab), like data/'s line B."""
    line_b = _line((0.0, 0.0), 83.0, n_per_line, 70.8)
    end = line_b[-1]
    turn = [
        SyntheticCamera(end.east_m + 8.0, end.north_m + 8.0, ALTITUDE, 30.8),
        SyntheticCamera(end.east_m + 6.0, end.north_m + 19.0, ALTITUDE, -29.2),
    ]
    line_c = _line((end.east_m, end.north_m + 27.0), 263.0, n_per_line, -109.2)
    return line_b + turn + line_c


def _true_directions(cams: list[SyntheticCamera]) -> list[np.ndarray]:
    """Unit (E, N) travel direction per camera: central difference, one-sided at the ends
    (hand-written here, independent of gps_lag.travel_directions)."""
    out = []
    for k in range(len(cams)):
        a, b = cams[max(k - 1, 0)], cams[min(k + 1, len(cams) - 1)]
        v = np.array([b.east_m - a.east_m, b.north_m - a.north_m])
        out.append(v / np.linalg.norm(v))
    return out


def _world(lag_m: float, cams: list[SyntheticCamera] | None = None):
    """(heading_edges, latlons with the lag applied, capture_times_s, image_shapes)."""
    cams = cams if cams is not None else _serpentine()
    dirs = _true_directions(cams)
    latlons = {
        k: latlon_from_en(c.east_m + lag_m * dirs[k][0], c.north_m + lag_m * dirs[k][1]) for k, c in enumerate(cams)
    }
    times = {k: 1000.0 + INTERVAL_S * k for k in range(len(cams))}
    edges = [
        HeadingEdge(i, j, plane_homography(cams[i], cams[j]), IMAGE_SHAPE, 1.0)
        for i in range(len(cams))
        for j in range(i + 1, len(cams))
        if math.hypot(cams[i].east_m - cams[j].east_m, cams[i].north_m - cams[j].north_m) < MAX_PAIR_DISTANCE_M
    ]
    return edges, latlons, times, {k: IMAGE_SHAPE for k in range(len(cams))}


def _random_homography(rng) -> np.ndarray:
    """A plausible-looking but wrong image-to-image map: random rotation and shift."""
    t = rng.uniform(-np.pi, np.pi)
    return np.array(
        [[math.cos(t), -math.sin(t), rng.uniform(-3000, 3000)], [math.sin(t), math.cos(t), rng.uniform(-3000, 3000)], [0, 0, 1.0]]
    )


def _en(latlon, ref):
    return np.array(geodetic_to_local_xy(*latlon, *ref))


# ---------------------------------------------------------------------------
# L1: flight segments (capture-time gaps)
# ---------------------------------------------------------------------------


def test_l1_gap_factor_is_the_data_derived_value() -> None:
    """Any factor in (1.67, 7) splits both data/ and the 738-image set identically
    (CLAUDE.md); 3 was chosen inside that window. Changing it needs re-validation."""
    assert CAPTURE_GAP_FACTOR == 3.0


def test_l1_segments_split_where_interval_exceeds_factor_times_median() -> None:
    # intervals 3, 2, 3, 21, 3, 3 -> median 3 -> split above 9 s
    times = {0: 0.0, 1: 3.0, 2: 5.0, 3: 8.0, 4: 29.0, 5: 32.0, 6: 35.0}
    assert flight_segments(times) == [[0, 1, 2, 3], [4, 5, 6]]


def test_l1_interval_equal_to_threshold_does_not_split() -> None:
    times = {0: 0.0, 1: 3.0, 2: 6.0, 3: 15.0, 4: 18.0}  # intervals 3, 3, 9, 3: 9 == 3 * 3
    assert flight_segments(times) == [[0, 1, 2, 3, 4]]


def test_l1_segments_follow_time_order_not_index_order_ties_by_index() -> None:
    assert flight_segments({5: 0.0, 2: 3.0, 9: 6.0}) == [[5, 2, 9]]
    assert flight_segments({3: 10.0, 1: 10.0, 2: 13.0}) == [[1, 3, 2]]


def test_l1_degenerate_inputs() -> None:
    assert flight_segments({}) == []
    assert flight_segments({4: 7.0}) == [[4]]


# ---------------------------------------------------------------------------
# L2: travel directions
# ---------------------------------------------------------------------------


def _latlons_en(points):
    return {k: latlon_from_en(e, n) for k, (e, n) in points.items()}


def test_l2_straight_eastbound_line_points_east() -> None:
    latlons = _latlons_en({k: (10.0 * k, 0.0) for k in range(5)})
    dirs = travel_directions(latlons, {k: 2.5 * k for k in range(5)})
    assert set(dirs) == set(range(5))
    for d in dirs.values():
        assert d == pytest.approx([1.0, 0.0], abs=1e-6)


def test_l2_central_difference_inside_one_sided_at_ends() -> None:
    latlons = _latlons_en({0: (0.0, 0.0), 1: (10.0, 0.0), 2: (10.0, 10.0)})
    dirs = travel_directions(latlons, {0: 0.0, 1: 3.0, 2: 6.0})
    assert dirs[0] == pytest.approx([1.0, 0.0], abs=1e-6)
    assert dirs[1] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)], abs=1e-6)
    assert dirs[2] == pytest.approx([0.0, 1.0], abs=1e-6)


def test_l2_no_differencing_across_a_gap_and_singletons_get_no_direction() -> None:
    # segment A eastbound (0, 1, 2); long gap; segment B northbound (3, 4); gap; lone 5
    latlons = _latlons_en({0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (20, 30), 4: (20, 40), 5: (99, 99)})
    times = {0: 0.0, 1: 3.0, 2: 6.0, 3: 100.0, 4: 103.0, 5: 500.0}
    dirs = travel_directions(latlons, times)
    assert dirs[2] == pytest.approx([1.0, 0.0], abs=1e-6)  # not pulled toward node 3
    assert dirs[3] == pytest.approx([0.0, 1.0], abs=1e-6)  # not pulled toward node 2
    assert 5 not in dirs


def test_l2_nodes_missing_time_or_gps_are_left_out_and_skipped_as_neighbours() -> None:
    latlons = _latlons_en({0: (0, 0), 1: (10, 0), 3: (30, 0), 9: (0, 50)})
    times = {0: 0.0, 1: 3.0, 2: 6.0, 3: 9.0, 4: 12.0}  # 2 and 4 have no GPS; 9 has no time
    dirs = travel_directions(latlons, times)
    assert set(dirs) == {0, 1, 3}
    assert dirs[1] == pytest.approx([1.0, 0.0], abs=1e-6)  # neighbours 0 and 3, not 2


# ---------------------------------------------------------------------------
# L3: applying the correction
# ---------------------------------------------------------------------------


def test_l3_shift_moves_each_fix_back_by_lag_along_its_direction() -> None:
    latlons = _latlons_en({0: (0.0, 0.0), 1: (40.0, 25.0)})
    dirs = {0: np.array([1.0, 0.0]), 1: np.array([0.6, 0.8])}
    shifted = shift_latlons(latlons, dirs, 2.5)
    for k in (0, 1):
        assert _en(shifted[k], latlons[k]) == pytest.approx(-2.5 * dirs[k], abs=1e-6)


def test_l3_nodes_without_direction_and_zero_lag_are_unchanged() -> None:
    latlons = _latlons_en({0: (0.0, 0.0), 1: (40.0, 25.0)})
    dirs = {0: np.array([1.0, 0.0])}
    shifted = shift_latlons(latlons, dirs, 2.5)
    assert shifted[1] == latlons[1]
    assert shift_latlons(latlons, dirs, 0.0) == latlons


# ---------------------------------------------------------------------------
# L4: per-node heading consensus (the building block the lag estimate is built on)
# ---------------------------------------------------------------------------


def _obs(headings_deg, sigma_deg=1.0):
    return [HeadingObservation((0, k + 1), math.radians(h), math.radians(sigma_deg)) for k, h in enumerate(headings_deg)]


def test_s5_consensus_parameters_are_the_finalized_values() -> None:
    """Z = 6.31 (held-out p99 under the constant sigma) and share 50% are final (CLAUDE.md's
    邊一致性檢查：參數定案); MIN_AGREE = 3 is the one unvalidated, PROVISIONAL value."""
    assert (CONSENSUS_Z, CONSENSUS_MIN_AGREE, CONSENSUS_MIN_SHARE) == (6.31, 3, 0.5)


def test_l4_agreeing_observations_are_trusted_with_weighted_circular_mean() -> None:
    obs = [
        HeadingObservation((0, 1), math.radians(10.0), math.radians(1.0)),
        HeadingObservation((0, 2), math.radians(12.0), math.radians(2.0)),
        HeadingObservation((0, 3), math.radians(11.0), math.radians(1.0)),
    ]
    result = heading_consensus(obs)
    w = np.array([1.0, 0.25, 1.0])  # 1 / sigma^2 with sigma in degrees -- same ratios in radians
    h = np.radians([10.0, 12.0, 11.0])
    expected = math.atan2(np.sum(w * np.sin(h)), np.sum(w * np.cos(h)))
    assert result.trusted
    assert (result.n_agree, result.n_obs) == (3, 3)
    assert result.heading_rad == pytest.approx(expected, abs=1e-12)
    assert all(result.agrees.values())


def test_l4_no_consensus_marks_the_whole_node_failed() -> None:
    """The boundary behaviour the user asked to pin down: headings spread around the
    circle, no subset agrees -> the node is untrusted, whatever its largest subset is."""
    result = heading_consensus(_obs([0, 45, 90, 135, 180, 225, 270, 315]))
    assert not result.trusted
    assert result.n_agree < CONSENSUS_MIN_AGREE
    assert result.n_obs == 8


def test_s6_share_and_count_boundaries() -> None:
    """Share >= 50% trusts (exactly half included); fewer than 3 agreeing never trusts."""
    three_of_six = heading_consensus(_obs([0, 0.5, -0.5, 90, -90, 180]))
    three_of_seven = heading_consensus(_obs([0, 0.5, -0.5, 90, -90, 180, 45]))
    two_of_two = heading_consensus(_obs([0, 0.5]))
    assert three_of_six.trusted and (three_of_six.n_agree, three_of_six.n_obs) == (3, 6)
    assert not three_of_seven.trusted and (three_of_seven.n_agree, three_of_seven.n_obs) == (3, 7)
    assert not two_of_two.trusted  # fewer than CONSENSUS_MIN_AGREE observations
    assert [three_of_six.agrees[(0, k)] for k in range(1, 7)] == [True, True, True, False, False, False]


def test_s6_agreement_gate_is_z_times_each_observations_own_sigma() -> None:
    """Z = 6.31: 6 sigma agrees, 7 sigma does not (the old Z = 3 rejected both)."""
    inside = heading_consensus(_obs([0, 0, 0]) + [HeadingObservation((0, 9), math.radians(6.0), math.radians(1.0))])
    outside = heading_consensus(_obs([0, 0, 0]) + [HeadingObservation((0, 9), math.radians(7.0), math.radians(1.0))])
    looser = heading_consensus(_obs([0, 0, 0]) + [HeadingObservation((0, 9), math.radians(7.0), math.radians(2.0))])
    assert inside.agrees[(0, 9)]
    assert not outside.agrees[(0, 9)]
    assert looser.agrees[(0, 9)]  # 3.5 sigma


def test_l4_wraparound() -> None:
    result = heading_consensus(_obs([179.5, -179.5, 180.0]))
    assert result.trusted
    assert abs(((math.degrees(result.heading_rad) - 180.0) + 180.0) % 360.0 - 180.0) < 1e-9


# ---------------------------------------------------------------------------
# L5: the iterative lag estimate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lag_true", [2.3, 0.0, -1.5])
def test_l5_recovers_the_true_lag_including_zero_and_negative(lag_true: float) -> None:
    edges, latlons, times, shapes = _world(lag_true)
    est = estimate_gps_lag(edges, latlons, times, shapes)
    assert est.status == "estimated", est.reason
    assert abs(est.lag_m - lag_true) <= LAG_TOL_M
    assert est.history_m[0] == 0.0
    assert est.rounds == len(est.history_m) - 1


def test_l5_opposite_edge_threshold_is_the_validated_value() -> None:
    """10: the smallest count with lag-error p95 and p99 <= 0.37 m in both synthetic survey
    geometries (CLAUDE.md). Changing it needs the same two-geometry validation."""
    assert LAG_MIN_OPPOSITE_EDGES == 10


def test_l5_halving_the_grid_step_does_not_materially_change_the_estimate(monkeypatch) -> None:
    """The 0.05 m grid is a known limitation (+-0.025 m quantization, CLAUDE.md): a 0.025 m
    grid must land within one coarse step of the 0.05 m result, on a noisy world where the
    optimum is not a grid point by construction."""
    import sea_mosaic.gps_lag as gps_lag

    edges, latlons, times, shapes = _world(2.3)
    rng = np.random.default_rng(9)
    noisy = {k: (lat + rng.normal(0, 2e-6), lon + rng.normal(0, 2e-6)) for k, (lat, lon) in latlons.items()}
    coarse = estimate_gps_lag(edges, noisy, times, shapes)
    monkeypatch.setattr(gps_lag, "LAG_SEARCH_STEP_M", LAG_SEARCH_STEP_M / 2)
    fine = estimate_gps_lag(edges, noisy, times, shapes)
    assert coarse.status == fine.status == "estimated"
    assert abs(fine.lag_m - coarse.lag_m) <= LAG_SEARCH_STEP_M
    assert abs(fine.lag_m - 2.3) <= LAG_TOL_M + 0.5  # noise of ~0.2 m per fix, loose truth check


def _noisy_two_line_world(rng, n_opposite: int, lag_m: float = 2.3):
    """Two separate flight segments (eastbound line, then westbound 27 m north after a time
    gap), every same-line pair < 40 m, only n_opposite random cross-line pairs; 0.37 m GPS
    noise per axis, homographies DLT-re-fit to exact points + 0.7 px noise -- the noise
    levels CLAUDE.md records for data/."""
    import cv2

    line_b = _line((0.0, 0.0), 83.0, 12, 70.8)
    end = line_b[-1]
    cams = line_b + _line((end.east_m, end.north_m + 27.0), 263.0, 12, -109.2)
    dirs = _true_directions(line_b) + _true_directions(cams[12:])
    en = np.array([[c.east_m, c.north_m] for c in cams])

    def fitted(i, j):
        lo, hi = en[[i, j]].min(0) - 80, en[[i, j]].max(0) + 80
        e, n = np.meshgrid(np.arange(lo[0], hi[0], 5.0), np.arange(lo[1], hi[1], 5.0))
        g = np.stack([e.ravel(), n.ravel(), np.ones(e.size)])
        a, b = ((ground_to_image(c) @ g) for c in (cams[i], cams[j]))
        a, b = (a[:2] / a[2]).T, (b[:2] / b[2]).T
        inside = np.all((a >= 0) & (a < [IMAGE_SHAPE[1], IMAGE_SHAPE[0]]) & (b >= 0) & (b < [IMAGE_SHAPE[1], IMAGE_SHAPE[0]]), axis=1)
        H, _ = cv2.findHomography(a[inside], b[inside] + rng.normal(0, 0.7, (int(inside.sum()), 2)), 0)
        return H / H[2, 2]

    close = [(i, j) for i in range(24) for j in range(i + 1, 24) if np.linalg.norm(en[i] - en[j]) < MAX_PAIR_DISTANCE_M]
    same = [(i, j) for i, j in close if (i < 12) == (j < 12)]
    cross = [(i, j) for i, j in close if (i < 12) != (j < 12)]
    chosen = same + [cross[k] for k in rng.choice(len(cross), n_opposite, replace=False)]
    edges = [HeadingEdge(i, j, fitted(i, j), IMAGE_SHAPE, 1.0) for i, j in chosen]
    gps = en + lag_m * np.array(dirs) + rng.normal(0, 0.37, en.shape)
    latlons = {k: latlon_from_en(*gps[k]) for k in range(24)}
    times = {k: INTERVAL_S * (k if k < 12 else k + 100) for k in range(24)}
    return edges, latlons, times, {k: IMAGE_SHAPE for k in range(24)}


def test_l5_every_opposite_direction_edge_informs_the_estimate(monkeypatch) -> None:
    """The objective must use every trusted observation (Huber sum), not a median of
    per-node medians, which only moves once opposite-direction edges are the majority at
    a node. With the identifiability gate lifted and only 4 cross-line edges, 12 seeded
    noisy trials: the Huber objective stays within 0.60 m, the median objective lost the
    lag entirely twice (3.85 m, 5.75 m) on these very trials (measured). 1.5 m sits well
    between the two."""
    import sea_mosaic.gps_lag as gps_lag

    monkeypatch.setattr(gps_lag, "LAG_MIN_OPPOSITE_EDGES", 0)
    rng = np.random.default_rng(104)
    errors = []
    for _ in range(12):
        est = estimate_gps_lag(*_noisy_two_line_world(rng, 4))
        assert est.status == "estimated", est.reason
        errors.append(abs(est.lag_m - 2.3))
    assert max(errors) <= 1.5, sorted(errors)


def test_l5_is_robust_to_false_edges() -> None:
    edges, latlons, times, shapes = _world(2.3)
    rng = np.random.default_rng(3)
    true_pairs = [(e.src_index, e.dst_index) for e in edges]
    extra = [HeadingEdge(i, j, _random_homography(rng), IMAGE_SHAPE, 1.0) for i, j in true_pairs[::4]]
    est = estimate_gps_lag(edges + extra, latlons, times, shapes)
    assert est.status == "estimated", est.reason
    assert abs(est.lag_m - 2.3) <= LAG_TOL_M


def test_l5_only_false_edges_is_not_estimable_and_applies_no_correction() -> None:
    edges, latlons, times, shapes = _world(2.3)
    rng = np.random.default_rng(5)
    false_edges = [HeadingEdge(e.src_index, e.dst_index, _random_homography(rng), IMAGE_SHAPE, 1.0) for e in edges]
    est = estimate_gps_lag(false_edges, latlons, times, shapes)
    assert est.status == "not_estimable"
    assert est.reason == "no_trusted_edges"
    assert est.lag_m == 0.0
    # Not "trusted_nodes == 0": with the provisional sigma model (wide 3-sigma for
    # cross-track edges), random headings can make an isolated node look trusted -- seen
    # here: node 7, 5 of 7 random headings within 55-91 deg, sigma 8-11 deg. An edge needs
    # BOTH ends trusted, so no edge survives. Revisit with the sigma refit (CLAUDE.md).


def test_l5_a_single_straight_line_cannot_identify_the_lag() -> None:
    """Every fix shifts by the same vector, so no GPS bearing changes: L is unobservable
    and must not be reported as estimated (the heading spread is flat in L)."""
    cams = _line((0.0, 0.0), 83.0, 10, 70.8)
    edges, latlons, times, shapes = _world(2.3, cams)
    est = estimate_gps_lag(edges, latlons, times, shapes)
    assert est.status == "not_estimable"
    assert est.reason == "not_identifiable"
    assert est.lag_m == 0.0


def test_l5_optimum_on_the_search_bound_is_not_trusted(monkeypatch) -> None:
    """A lag beyond the search range puts the optimum on the grid edge: reported, not used.
    The range is narrowed to 2.0 m (true lag 2.3 m) so the optimum provably lands on the
    bound whatever the consensus gate. (With the constant 1.691 deg sigma a truly huge lag,
    e.g. 12 m, already fails earlier as not_identifiable: at L0 = 0 its cross-line
    observations are all outside the ~10.7 deg gate, leaving < 10 opposite-direction edges.)"""
    import sea_mosaic.gps_lag as gps_lag

    monkeypatch.setattr(gps_lag, "LAG_SEARCH_MAX_M", 2.0)
    edges, latlons, times, shapes = _world(2.3)
    est = estimate_gps_lag(edges, latlons, times, shapes)
    assert est.status == "not_estimable"
    assert est.reason == "search_bound"
    assert est.lag_m == 0.0
    assert est.history_m[-1] == pytest.approx(2.0)


def test_l5_running_out_of_rounds_is_not_converged(monkeypatch) -> None:
    import sea_mosaic.gps_lag as gps_lag

    monkeypatch.setattr(gps_lag, "LAG_MAX_ROUNDS", 1)  # round 1 jumps 0 -> ~2.3, no round 2
    edges, latlons, times, shapes = _world(2.3)
    est = estimate_gps_lag(edges, latlons, times, shapes)
    assert est.status == "not_estimable"
    assert est.reason == "not_converged"
    assert est.lag_m == 0.0
    assert est.rounds == 1


def test_l5_without_capture_times_nothing_is_corrected() -> None:
    edges, latlons, _times, shapes = _world(2.3)
    est = estimate_gps_lag(edges, latlons, None, shapes)
    assert est.status == "no_capture_times"
    assert est.lag_m == 0.0
    assert est.uncorrected_nodes == set(latlons)


def test_l5_located_nodes_without_a_travel_direction_are_reported() -> None:
    edges, latlons, times, shapes = _world(2.3)
    del times[3]  # node 3 keeps its GPS but loses its capture time
    est = estimate_gps_lag(edges, latlons, times, shapes)
    assert est.uncorrected_nodes == {3}


# ---------------------------------------------------------------------------
# L6: reading capture times (EXIF only)
# ---------------------------------------------------------------------------


def test_l6_load_exif_capture_times_uses_exif_only(tmp_path) -> None:
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures" / "dji_smoke"
    paths = {
        0: fixtures / "DJI_20230127131426_0352_W.JPG",
        1: fixtures / "DJI_20230127131429_0353_W.JPG",
        2: fixtures / "exif_stripped_xmp_only.jpg",  # has xmp:CreateDate, no EXIF time
    }
    times = load_exif_capture_times(paths)
    assert set(times) == {0, 1}
    assert times[1] - times[0] == 3.0
