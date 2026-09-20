"""Unit tests for sea_mosaic.geo.projection. Uses real GPS values confirmed during the
data/smoke/ EXIF/XMP metadata investigation (image 0352 as origin, 0353 for deltas)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sea_mosaic.geo.projection import R_EARTH_M, geodetic_to_local_xy, project_gps_positions


# --- geodetic_to_local_xy ------------------------------------------------------------


def test_geodetic_to_local_xy_zero_at_origin():
    lat0, lon0 = 36.4277901, -5.1262460

    xy = geodetic_to_local_xy(lat0, lon0, lat0, lon0)

    assert isinstance(xy, np.ndarray)
    assert xy.shape == (2,)
    assert xy == pytest.approx([0.0, 0.0], abs=1e-9)


def test_geodetic_to_local_xy_known_north_offset():
    lat0, lon0 = 36.4277901, -5.1262460

    xy = geodetic_to_local_xy(lat0 + 0.001, lon0, lat0, lon0)

    expected_north_m = math.radians(0.001) * R_EARTH_M  # ~111.1949 m
    assert xy.shape == (2,)
    assert xy[0] == pytest.approx(0.0, abs=1e-6)  # east component ~0
    assert xy[1] == pytest.approx(expected_north_m, rel=1e-9)


def test_geodetic_to_local_xy_known_east_offset():
    lat0, lon0 = 36.4277901, -5.1262460

    xy = geodetic_to_local_xy(lat0, lon0 + 0.001, lat0, lon0)

    expected_east_m = math.radians(0.001) * R_EARTH_M * math.cos(math.radians(lat0))  # ~89.4681 m
    assert xy.shape == (2,)
    assert xy[0] == pytest.approx(expected_east_m, rel=1e-9)
    assert xy[1] == pytest.approx(0.0, abs=1e-6)


def test_geodetic_to_local_xy_combined_offset_is_additive():
    """A combined lat+lon offset must equal the independent sum of the two isolated-axis
    offsets — guards against an implementation that only handles single-axis deltas
    correctly (e.g. an accidental cross term or an east/north swap that only manifests
    when both axes move at once)."""
    lat0, lon0 = 36.4277901, -5.1262460
    dlat, dlon = 0.002, 0.0015

    xy = geodetic_to_local_xy(lat0 + dlat, lon0 + dlon, lat0, lon0)

    expected_north_m = math.radians(dlat) * R_EARTH_M
    expected_east_m = math.radians(dlon) * R_EARTH_M * math.cos(math.radians(lat0))
    assert xy == pytest.approx([expected_east_m, expected_north_m], rel=1e-9)
    # ~134.2021 m east, ~222.3899 m north


# --- project_gps_positions ------------------------------------------------------------


def test_project_gps_positions_origin_is_zeroed():
    gps_positions = {
        0: np.array([36.4277901, -5.1262460, 150.001, 99.978]),
        1: np.array([36.4279185, -5.1262886, 150.003, 99.980]),
    }

    projected = project_gps_positions(gps_positions, origin_index=0)

    assert set(projected.keys()) == {0, 1}
    for value in projected.values():
        assert isinstance(value, np.ndarray)
        assert value.shape == (2,)
    assert projected[0] == pytest.approx([0.0, 0.0], abs=1e-9)
    # hand-computed from the real 0352->0353 delta (dlat=0.0001284, dlon=-0.0000426)
    assert projected[1] == pytest.approx([-3.8113406988565917, 14.27742858051022], rel=1e-6)


def test_project_gps_positions_explicit_non_zero_origin_index():
    """Choosing image 1 as the origin instead of 0 must zero image 1 and move image 0
    to the negated relative position (sign-flip check, not just a single fixed origin)."""
    gps_positions = {
        0: np.array([36.4277901, -5.1262460, 150.001, 99.978]),
        1: np.array([36.4279185, -5.1262886, 150.003, 99.980]),
    }

    projected = project_gps_positions(gps_positions, origin_index=1)

    assert projected[1] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert projected[0][0] != pytest.approx(0.0, abs=1e-6)
    assert projected[0][1] != pytest.approx(0.0, abs=1e-6)


def test_project_gps_positions_default_origin_is_smallest_key():
    """With origin_index=None, the function's own dependency-free default is the
    smallest key present — not any pipeline-level 'reference_index' concept, since this
    module doesn't import PipelineConfig."""
    gps_positions = {
        5: np.array([36.4277901, -5.1262460, 150.001, 99.978]),
        2: np.array([36.4279185, -5.1262886, 150.003, 99.980]),
    }

    projected = project_gps_positions(gps_positions)

    assert projected[2] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert projected[5] != pytest.approx([0.0, 0.0], abs=1e-6)
