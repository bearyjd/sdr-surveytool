"""Minimal NMEA 0183 parsing for the u-blox M8N: checksum validation and GGA
sentences (position, altitude, fix quality). Pure functions, no I/O."""

from __future__ import annotations

from ingest.gps_fix import GpsFix

# Talker IDs vary with constellation: GP (GPS), GN (combined), GL, GA, BD.
_GGA_SUFFIX = "GGA"


def checksum_ok(sentence: str) -> bool:
    """True if '$....*HH' carries a correct XOR checksum. A sentence without a
    checksum is rejected: a serial line corrupted mid-sentence must not yield a
    plausible-looking wrong position."""
    sentence = sentence.strip()
    if not sentence.startswith("$") or "*" not in sentence:
        return False
    body, _, given = sentence[1:].rpartition("*")
    if len(given) != 2:
        return False
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(given, 16)
    except ValueError:
        return False


def _coord(value: str, hemisphere: str, degree_digits: int) -> float | None:
    if not value or hemisphere not in ("N", "S", "E", "W"):
        return None
    try:
        degrees = int(value[:degree_digits])
        minutes = float(value[degree_digits:])
    except ValueError:
        return None
    if not 0 <= minutes < 60:
        return None
    result = degrees + minutes / 60.0
    return -result if hemisphere in ("S", "W") else result


def parse_gga(sentence: str) -> GpsFix | None:
    """GpsFix from a valid GGA sentence, or None if the line is not GGA, fails
    its checksum, or reports no fix (quality 0 / empty position)."""
    if not checksum_ok(sentence):
        return None
    fields = sentence.strip()[1:].rpartition("*")[0].split(",")
    if len(fields) < 10 or not fields[0].endswith(_GGA_SUFFIX):
        return None
    try:
        quality = int(fields[6])
    except ValueError:
        return None
    if quality <= 0:
        return None
    lat = _coord(fields[2], fields[3], 2)
    lon = _coord(fields[4], fields[5], 3)
    if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
        return None
    try:
        altitude = float(fields[9]) if fields[9] else None
    except ValueError:
        altitude = None
    return GpsFix(lat=lat, lon=lon, altitude=altitude, fix_quality=quality)
