"""Image loading and GPS metadata extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image
from PIL.ExifTags import GPSTAGS

_XMP_PACKET_RE = re.compile(rb"<\?xpacket begin=.*?<\?xpacket end=.*?\?>", re.DOTALL)
_XMP_DRONE_DJI_ATTR_RE = re.compile(r'drone-dji:(\w+)="([^"]*)"')


def load_image(path: Path) -> np.ndarray:
    """Load a single image from disk as an array."""
    ...


def load_images(paths: list[Path]) -> dict[int, np.ndarray]:
    """Load a sequence of images, keyed by their index in the input list."""
    ...


def _exif_dms_to_decimal_degrees(
    dms: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    ref: str,
) -> float:
    """Convert an EXIF GPS DMS rational triplet + Ref ('N'/'S'/'E'/'W') to signed
    decimal degrees. N/E positive, S/W negative."""
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]
    decimal_degrees = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def _xmp_signed_decimal(value_str: str) -> float:
    """Parse an already-signed XMP decimal-degree string (e.g. '-5.1262460') directly.
    Must NOT reuse _exif_dms_to_decimal_degrees's Ref-based sign logic — XMP drone-dji
    GPS attributes are pre-signed, unlike EXIF's DMS+Ref pair."""
    return float(value_str)


def _read_exif_gps(path: Path) -> dict[str, Any] | None:
    """Read the EXIF GPS IFD via Pillow, keyed by tag name; None if there is no GPS
    IFD at all (e.g. no EXIF, or EXIF present but without a GPS block)."""
    with Image.open(path) as img:
        exif = img.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if not gps_ifd:
        return None
    return {GPSTAGS.get(tag_id, tag_id): value for tag_id, value in gps_ifd.items()}


def _to_dms_tuple(raw: Any) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert Pillow's tuple of 3 IFDRational GPS components to (numerator,
    denominator) pairs, matching _exif_dms_to_decimal_degrees's expected input shape."""
    return tuple((int(component.numerator), int(component.denominator)) for component in raw)


def _read_xmp_drone_dji_fields(path: Path) -> dict[str, str]:
    """Extract the embedded XMP packet's drone-dji:* attributes as {name: raw_value_str},
    or {} if there is no XMP packet in the file at all."""
    data = Path(path).read_bytes()
    match = _XMP_PACKET_RE.search(data)
    if match is None:
        return {}
    xmp_text = match.group(0).decode("utf-8", errors="ignore")
    return dict(_XMP_DRONE_DJI_ATTR_RE.findall(xmp_text))


def load_gps_position(path: Path) -> np.ndarray | None:
    """Extract [lat_deg, lon_deg, absolute_altitude_m, relative_altitude_m] from EXIF
    GPS IFD (primary) or XMP drone-dji fields (fallback); return None if no GPS fix is
    present in the image at all. Full EXIF/XMP parsing is a separate roadmap item (see
    CLAUDE.md)."""
    exif_gps = _read_exif_gps(path)
    xmp_fields = _read_xmp_drone_dji_fields(path)

    lat: float | None = None
    lon: float | None = None

    if exif_gps is not None:
        raw_lat = exif_gps.get("GPSLatitude")
        raw_lon = exif_gps.get("GPSLongitude")
        lat_ref = exif_gps.get("GPSLatitudeRef")
        lon_ref = exif_gps.get("GPSLongitudeRef")
        if raw_lat is not None and lat_ref is not None:
            lat = _exif_dms_to_decimal_degrees(_to_dms_tuple(raw_lat), lat_ref)
        if raw_lon is not None and lon_ref is not None:
            lon = _exif_dms_to_decimal_degrees(_to_dms_tuple(raw_lon), lon_ref)

    if lat is None or lon is None:
        xmp_lat = xmp_fields.get("GpsLatitude")
        xmp_lon = xmp_fields.get("GpsLongitude")
        if xmp_lat is not None and xmp_lon is not None:
            lat = _xmp_signed_decimal(xmp_lat)
            lon = _xmp_signed_decimal(xmp_lon)

    if lat is None or lon is None:
        return None

    abs_alt = np.nan
    if "AbsoluteAltitude" in xmp_fields:
        abs_alt = _xmp_signed_decimal(xmp_fields["AbsoluteAltitude"])
    elif exif_gps is not None and exif_gps.get("GPSAltitude") is not None:
        abs_alt = float(exif_gps["GPSAltitude"])
        if exif_gps.get("GPSAltitudeRef") in (1, b"\x01"):
            abs_alt = -abs_alt

    rel_alt = np.nan
    if "RelativeAltitude" in xmp_fields:
        rel_alt = _xmp_signed_decimal(xmp_fields["RelativeAltitude"])

    return np.array([lat, lon, abs_alt, rel_alt], dtype=np.float64)


def load_gimbal_yaw(path: Path) -> float | None:
    """Extract GimbalYawDegree from the embedded XMP drone-dji metadata; None if the XMP
    packet has no such attribute at all (e.g. no XMP packet, or one without gimbal
    fields). DJI-specific XMP field with no EXIF equivalent, unlike load_gps_position's
    EXIF-primary/XMP-fallback GPS fix — there is no second source to fall back to here."""
    xmp_fields = _read_xmp_drone_dji_fields(path)
    if "GimbalYawDegree" not in xmp_fields:
        return None
    return _xmp_signed_decimal(xmp_fields["GimbalYawDegree"])
