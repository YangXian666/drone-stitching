"""Unit tests for sea_mosaic.io_utils GPS extraction. EXIF and XMP sign-conversion
paths are tested independently (per the DMS+Ref vs. pre-signed-decimal trap found
during the data/smoke/ metadata investigation) before the end-to-end real-file tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sea_mosaic.io_utils import (
    _exif_dms_to_decimal_degrees,
    _xmp_signed_decimal,
    load_gimbal_yaw,
    load_gps_position,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dji_smoke"


# --- EXIF sign-convention path (DMS rational + Ref), independent of the XMP path ------


@pytest.mark.parametrize(
    "dms, ref, expected",
    [
        (((36, 1), (25, 1), (400444, 10000)), "N", 36.42779011111111),
        (((36, 1), (25, 1), (400444, 10000)), "S", -36.42779011111111),
        (((5, 1), (7, 1), (344856, 10000)), "E", 5.126245999999999),
        (((5, 1), (7, 1), (344856, 10000)), "W", -5.126245999999999),
    ],
)
def test_exif_dms_to_decimal_degrees_sign_conventions(dms, ref, expected):
    assert _exif_dms_to_decimal_degrees(dms, ref) == pytest.approx(expected, rel=1e-9)


def test_exif_dms_to_decimal_degrees_zero_is_unaffected_by_ref():
    """0 DMS must stay 0 regardless of hemisphere ref — guards against a sign
    implementation that multiplies by -1 unconditionally instead of branching on ref."""
    assert _exif_dms_to_decimal_degrees(((0, 1), (0, 1), (0, 1)), "S") == pytest.approx(0.0, abs=1e-12)
    assert _exif_dms_to_decimal_degrees(((0, 1), (0, 1), (0, 1)), "N") == pytest.approx(0.0, abs=1e-12)


# --- XMP sign-convention path (pre-signed decimal string), independent of the EXIF path -


@pytest.mark.parametrize(
    "value_str, expected",
    [
        ("+36.4277901", 36.4277901),
        ("-36.4277901", -36.4277901),
        ("-5.1262460", -5.1262460),
        ("+5.1262460", 5.1262460),
    ],
)
def test_xmp_signed_decimal_sign_conventions(value_str, expected):
    assert _xmp_signed_decimal(value_str) == pytest.approx(expected, rel=1e-9)


def test_xmp_signed_decimal_no_explicit_sign_defaults_positive():
    """A value string with no leading '+' or '-' (not observed in this dataset, but not
    guaranteed absent from October's data) must parse as positive, not raise."""
    assert _xmp_signed_decimal("36.4277901") == pytest.approx(36.4277901, rel=1e-9)


# --- End-to-end: real file, EXIF path (primary source) --------------------------------


def test_load_gps_position_real_exif_file():
    result = load_gps_position(FIXTURES / "DJI_20230127131426_0352_W.JPG")

    assert result is not None
    assert result.shape == (4,)
    lat, lon, abs_alt, rel_alt = result
    assert lat == pytest.approx(36.4277901, abs=1e-6)
    assert lon == pytest.approx(-5.1262460, abs=1e-6)
    assert abs_alt == pytest.approx(150.001, abs=1e-3)
    assert rel_alt == pytest.approx(99.978, abs=1e-3)


def test_load_gps_position_second_real_file_matches_survey_delta():
    """Cross-check against the independently-verified 0352->0353 delta, not just a
    single hardcoded file."""
    result = load_gps_position(FIXTURES / "DJI_20230127131429_0353_W.JPG")

    assert result is not None
    lat, lon, _abs_alt, _rel_alt = result
    assert lat == pytest.approx(36.4279185, abs=1e-6)
    assert lon == pytest.approx(-5.1262886, abs=1e-6)


# --- End-to-end: EXIF missing, must fall back to XMP -----------------------------------


def test_load_gps_position_exif_missing_falls_back_to_xmp():
    """exif_stripped_xmp_only.jpg has no Exif\\x00\\x00 APP1 segment at all (stripped),
    only the intact XMP block — this is the only fixture that actually forces
    load_gps_position through its XMP fallback path, rather than just having XMP
    available-but-unused because EXIF already satisfied the read."""
    result = load_gps_position(FIXTURES / "exif_stripped_xmp_only.jpg")

    assert result is not None
    lat, lon, _abs_alt, _rel_alt = result
    assert lat == pytest.approx(36.4277901, abs=1e-6)
    assert lon == pytest.approx(-5.1262460, abs=1e-6)


# --- End-to-end: no GPS at all ---------------------------------------------------------


def test_load_gps_position_no_gps_returns_none():
    result = load_gps_position(FIXTURES / "no_gps.jpg")

    assert result is None


# --- load_gimbal_yaw: XMP-only (no EXIF equivalent for gimbal attitude) ----------------


def test_load_gimbal_yaw_real_file():
    result = load_gimbal_yaw(FIXTURES / "DJI_20230127131426_0352_W.JPG")

    assert result == pytest.approx(39.00, abs=1e-6)


def test_load_gimbal_yaw_second_real_file_is_negative():
    """Cross-check a second real file with a different sign, not just a single
    hardcoded positive value."""
    result = load_gimbal_yaw(FIXTURES / "DJI_20230127131429_0353_W.JPG")

    assert result == pytest.approx(-23.20, abs=1e-6)


def test_load_gimbal_yaw_no_xmp_returns_none():
    """no_gps.jpg has no XMP packet at all (not just a missing GPS fix), so there is no
    GimbalYawDegree attribute to read either."""
    result = load_gimbal_yaw(FIXTURES / "no_gps.jpg")

    assert result is None
