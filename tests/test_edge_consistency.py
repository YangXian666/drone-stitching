"""Tests for sea_mosaic.edge_consistency: the second line of defence for edges.

An edge's homography is only trusted if it agrees with GPS, checked in a fixed order:
no_gps -> too_short -> untrusted_node -> heading_disagrees -> length -> orientation_reversing
-> rotation (CLAUDE.md's 邊一致性檢查：參數定案). Each rejection reason is triggered in
isolation by composing a TRUE homography with an affine map A about the src image centre c
that changes exactly one measured quantity:
  length      A scales about c: the dst-centre displacement grows, its direction and the
              local rotation at c stay.
  mirror      A reflects about the displacement axis: displacement unchanged, Jacobian det < 0.
  rotation    A shears along the displacement axis: it fixes c AND the point where the dst
              centre lands (both on that axis), so both images' measured travel directions stay
              and only the local rotation at c changes, by -atan(s / 2) for shear s.
  heading     A rotates about c: the displacement direction measured in the src image turns.
Expected values come from the synthetic pinhole camera and these hand constructions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sea_mosaic.edge_consistency import (
    LENGTH_TOL_LOG,
    PPM_MODE_WINDOW_LOG,
    REJECTION_ORDER,
    ROTATION_TOL_DEG,
    check_edge_consistency,
)
from sea_mosaic.types import PairResult
from synthetic_camera import FOCAL_PX, IMAGE_CENTRE, IMAGE_SHAPE, SyntheticCamera, latlon_from_en, plane_homography

ALTITUDE = 100.0
PPM = FOCAL_PX / ALTITUDE  # exact ground scale of an untilted nadir camera
STEP_M = 13.4
MAX_PAIR_DISTANCE_M = 40.0
C = np.append(IMAGE_CENTRE, 1.0)


# ---------------------------------------------------------------------------
# Synthetic serpentine world
# ---------------------------------------------------------------------------


def _line(start, course_deg, n, yaw_deg):
    c = math.radians(course_deg)
    return [SyntheticCamera(start[0] + k * STEP_M * math.sin(c), start[1] + k * STEP_M * math.cos(c), ALTITUDE, yaw_deg)
            for k in range(n)]


def _cameras() -> dict[int, SyntheticCamera]:
    """Eastbound line, two-image U-turn, westbound line 27 m north: every node has >= 3
    candidate edges (< 40 m), so every node can reach a consensus."""
    b = _line((0.0, 0.0), 83.0, 8, 70.8)
    end = b[-1]
    turn = [SyntheticCamera(end.east_m + 8.0, end.north_m + 8.0, ALTITUDE, 30.8),
            SyntheticCamera(end.east_m + 6.0, end.north_m + 19.0, ALTITUDE, -29.2)]
    c = _line((end.east_m, end.north_m + 27.0), 263.0, 8, -109.2)
    return dict(enumerate(b + turn + c))


def _pair(cams, i, j, H=None) -> PairResult:
    H = plane_homography(cams[i], cams[j]) if H is None else H
    pts = np.array([[1000.0, 1000.0], [3000.0, 1000.0], [2000.0, 2000.0], [1200.0, 2500.0], [2800.0, 2600.0]])
    return PairResult(i, j, pts, pts, np.ones(len(pts), dtype=bool), H)


def _world(cams=None):
    cams = cams if cams is not None else _cameras()
    keys = sorted(cams)
    pairs = [_pair(cams, i, j) for n, i in enumerate(keys) for j in keys[n + 1:]
             if math.hypot(cams[i].east_m - cams[j].east_m, cams[i].north_m - cams[j].north_m) < MAX_PAIR_DISTANCE_M]
    latlons = {k: latlon_from_en(c.east_m, c.north_m) for k, c in cams.items()}
    return cams, pairs, latlons, {k: IMAGE_SHAPE for k in cams}


def _displacement_axis(H) -> np.ndarray:
    m = np.linalg.inv(H) @ C
    v = m[:2] / m[2] - IMAGE_CENTRE
    return v / np.linalg.norm(v)


def _about_centre(linear: np.ndarray) -> np.ndarray:
    """Affine map p -> c + L (p - c) as a 3x3 matrix."""
    A = np.eye(3)
    A[:2, :2] = linear
    A[:2, 2] = IMAGE_CENTRE - linear @ IMAGE_CENTRE
    return A


def _basis(u):
    return np.column_stack([u, [-u[1], u[0]]])  # columns: along the displacement, perpendicular


def _scaled_length(H, factor):
    """inv(H') c - c = factor * (inv(H) c - c): H' = H @ A, inv(A) scales by factor about c."""
    return H @ _about_centre(np.eye(2) / factor)


def _mirrored(H):
    B = _basis(_displacement_axis(H))
    return H @ _about_centre(B @ np.diag([1.0, -1.0]) @ B.T)


def _sheared(H, rotation_deg):
    """Local rotation at c changed by rotation_deg, both measured travel directions kept."""
    s = -2.0 * math.tan(math.radians(rotation_deg))
    B = _basis(_displacement_axis(H))
    return H @ _about_centre(B @ np.array([[1.0, s], [0.0, 1.0]]) @ B.T)


def _turned(H, deg):
    t = math.radians(deg)
    return H @ _about_centre(np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]))


def _replace(pairs, key, H):
    return [PairResult(p.src_index, p.dst_index, p.src_points, p.dst_points, p.inlier_mask, H)
            if (p.src_index, p.dst_index) == key else p for p in pairs]


def _random_H(rng):
    t = rng.uniform(-np.pi, np.pi)
    return np.array([[math.cos(t), -math.sin(t), rng.uniform(-3000, 3000)],
                     [math.sin(t), math.cos(t), rng.uniform(-3000, 3000)], [0.0, 0.0, 1.0]])


def _key(p):
    return (p.src_index, p.dst_index)


# ---------------------------------------------------------------------------
# E0: constants and the documented order
# ---------------------------------------------------------------------------


def test_e0_parameters_are_the_finalized_values() -> None:
    """CLAUDE.md's 邊一致性檢查：參數定案. Changing any needs the same validation."""
    assert (PPM_MODE_WINDOW_LOG, LENGTH_TOL_LOG, ROTATION_TOL_DEG) == (0.08, 0.15, 12.45)
    assert REJECTION_ORDER == (
        "no_gps", "too_short", "untrusted_node", "heading_disagrees", "length", "orientation_reversing", "rotation"
    )


# ---------------------------------------------------------------------------
# E1: exact data
# ---------------------------------------------------------------------------


def test_e1_exact_world_accepts_every_edge_with_the_exact_scale() -> None:
    cams, pairs, latlons, shapes = _world()
    check = check_edge_consistency(pairs, latlons, shapes)
    assert check.rejected == {}
    assert [_key(p) for p in check.accepted] == [_key(p) for p in pairs]  # order kept
    assert check.ppm_mode == pytest.approx(PPM, rel=1e-3)
    assert check.ppm_support == len(pairs)
    for k, cam in cams.items():  # consensus heading = compass yaw in the north-up pixel frame
        assert check.consensus[k].trusted
        assert math.degrees((check.consensus[k].heading_rad - math.radians(cam.yaw_deg) + math.pi) % (2 * math.pi) - math.pi) \
            == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# E2: each rejection reason in isolation
# ---------------------------------------------------------------------------


def test_e2_no_gps() -> None:
    cams, pairs, latlons, shapes = _world()
    del latlons[3]
    check = check_edge_consistency(pairs, latlons, shapes)
    touching = {_key(p) for p in pairs if 3 in _key(p)}
    assert check.rejected == {k: "no_gps" for k in touching}


def test_e2_too_short() -> None:
    cams = _cameras()
    cams[99] = SyntheticCamera(cams[4].east_m + 3.0, cams[4].north_m, ALTITUDE, 70.8)  # 3 m from node 4
    _cams, pairs, latlons, shapes = _world(cams)
    check = check_edge_consistency(pairs, latlons, shapes)
    assert check.rejected[(4, 99)] == "too_short"


def test_e2_untrusted_node() -> None:
    cams, pairs, latlons, shapes = _world()
    rng = np.random.default_rng(1)
    mine = [p for p in pairs if 3 in _key(p)]
    for p in mine[: len(mine) // 2 + 1]:  # a majority of node 3's edges are false
        pairs = _replace(pairs, _key(p), _random_H(rng))
    check = check_edge_consistency(pairs, latlons, shapes)
    assert not check.consensus[3].trusted
    assert all(check.rejected[_key(p)] == "untrusted_node" for p in mine)


def test_e2_heading_disagrees() -> None:
    cams, pairs, latlons, shapes = _world()
    key = (1, 2)
    pairs = _replace(pairs, key, _turned(dict((_key(p), p) for p in pairs)[key].homography, 30.0))
    check = check_edge_consistency(pairs, latlons, shapes)
    assert check.consensus[1].trusted and check.consensus[2].trusted
    assert check.rejected == {key: "heading_disagrees"}


def test_e2_length() -> None:
    cams, pairs, latlons, shapes = _world()
    key = (1, 2)
    pairs = _replace(pairs, key, _scaled_length(dict((_key(p), p) for p in pairs)[key].homography, 1.3))
    check = check_edge_consistency(pairs, latlons, shapes)
    assert check.rejected == {key: "length"}


def test_e2_orientation_reversing() -> None:
    cams, pairs, latlons, shapes = _world()
    key = (1, 2)
    pairs = _replace(pairs, key, _mirrored(dict((_key(p), p) for p in pairs)[key].homography))
    check = check_edge_consistency(pairs, latlons, shapes)
    assert check.rejected == {key: "orientation_reversing"}


def test_e2_rotation() -> None:
    cams, pairs, latlons, shapes = _world()
    key = (1, 2)
    pairs = _replace(pairs, key, _sheared(dict((_key(p), p) for p in pairs)[key].homography, 20.0))
    check = check_edge_consistency(pairs, latlons, shapes)
    assert check.rejected == {key: "rotation"}


# ---------------------------------------------------------------------------
# E3: boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log_ratio, accepted", [(0.1499, True), (0.1501, False), (-0.1499, True), (-0.1501, False)])
def test_e3_length_boundary(log_ratio: float, accepted: bool) -> None:
    cams, pairs, latlons, shapes = _world()
    key = (1, 2)
    pairs = _replace(pairs, key, _scaled_length(dict((_key(p), p) for p in pairs)[key].homography, math.exp(log_ratio)))
    check = check_edge_consistency(pairs, latlons, shapes)
    assert (key not in check.rejected) is accepted


@pytest.mark.parametrize("rotation_deg, accepted", [(12.4, True), (12.5, False), (-12.4, True), (-12.5, False)])
def test_e3_rotation_boundary(rotation_deg: float, accepted: bool) -> None:
    cams, pairs, latlons, shapes = _world()
    key = (1, 2)
    pairs = _replace(pairs, key, _sheared(dict((_key(p), p) for p in pairs)[key].homography, rotation_deg))
    check = check_edge_consistency(pairs, latlons, shapes)
    assert (key not in check.rejected) is accepted


# ---------------------------------------------------------------------------
# E4: the ppm mode is taken over heading survivors only
# ---------------------------------------------------------------------------


def test_e4_ppm_mode_counts_only_heading_survivors() -> None:
    """Heading-rejected edges keep their TRUE length (turned, not scaled), so they sit inside
    the mode window: counting them would raise ppm_support above the survivors' count."""
    cams, pairs, latlons, shapes = _world()
    turned = [(1, 2), (5, 6), (12, 13)]
    by_key = {_key(p): p for p in pairs}
    for key in turned:
        pairs = _replace(pairs, key, _turned(by_key[key].homography, 30.0))
    check = check_edge_consistency(pairs, latlons, shapes)
    assert {k: check.rejected[k] for k in turned} == {k: "heading_disagrees" for k in turned}
    assert check.ppm_support == len(pairs) - len(turned)


# ---------------------------------------------------------------------------
# E5: the first failing check in REJECTION_ORDER is the reason recorded
# ---------------------------------------------------------------------------


def test_e5_reason_is_the_first_failing_check_in_order() -> None:
    cams = _cameras()
    cams[99] = SyntheticCamera(cams[4].east_m + 3.0, cams[4].north_m, ALTITUDE, 70.8)
    _cams, pairs, latlons, shapes = _world(cams)
    by_key = {_key(p): p for p in pairs}
    # (1, 2): heading, length and rotation all wrong -> heading_disagrees
    bad = _sheared(_scaled_length(_turned(by_key[(1, 2)].homography, 30.0), 1.3), 20.0)
    # (5, 6): length and rotation wrong, heading right -> length
    worse_len = _sheared(_scaled_length(by_key[(5, 6)].homography, 1.3), 20.0)
    # (12, 13): mirrored AND sheared -> orientation_reversing
    mirror_rot = _sheared(_mirrored(by_key[(12, 13)].homography), 20.0)
    for key, H in (((1, 2), bad), ((5, 6), worse_len), ((12, 13), mirror_rot)):
        pairs = _replace(pairs, key, H)
    del latlons[99]  # (4, 99) is both short and without GPS -> no_gps
    check = check_edge_consistency(pairs, latlons, shapes)
    assert check.rejected[(1, 2)] == "heading_disagrees"
    assert check.rejected[(5, 6)] == "length"
    assert check.rejected[(12, 13)] == "orientation_reversing"
    assert check.rejected[(4, 99)] == "no_gps"


# ---------------------------------------------------------------------------
# E6 / E7: all false, a quarter false
# ---------------------------------------------------------------------------


def test_e6_only_false_edges_rejects_everything() -> None:
    cams, pairs, latlons, shapes = _world()
    rng = np.random.default_rng(2)
    pairs = [PairResult(p.src_index, p.dst_index, p.src_points, p.dst_points, p.inlier_mask, _random_H(rng)) for p in pairs]
    check = check_edge_consistency(pairs, latlons, shapes)
    assert check.accepted == []
    assert set(check.rejected) == {_key(p) for p in pairs}
    assert math.isnan(check.ppm_mode)


def test_e7_a_quarter_false_edges_all_rejected_true_edges_all_kept() -> None:
    cams, pairs, latlons, shapes = _world()
    rng = np.random.default_rng(3)
    false = {_key(p) for p in pairs[::4]}
    pairs = [PairResult(p.src_index, p.dst_index, p.src_points, p.dst_points, p.inlier_mask, _random_H(rng))
             if _key(p) in false else p for p in pairs]
    check = check_edge_consistency(pairs, latlons, shapes)
    assert set(check.rejected) == false
    assert {_key(p) for p in check.accepted} == {_key(p) for p in pairs} - false
