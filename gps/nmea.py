"""Minimal NMEA 0183 parser: GGA and RMC sentences -> GpsFix."""
from __future__ import annotations

from functools import reduce

from ingest.gps_fix import GpsFix


def _checksum(body: str) -> int:
    return reduce(lambda acc, ch: acc ^ ord(ch), body, 0)


def with_checksum(body: str) -> str:
    """Build a full sentence from the text between '$' and '*'."""
    return f"${body}*{_checksum(body):02X}"


def _coord(value: str, hemi: str, deg_digits: int) -> float:
    degrees = int(value[:deg_digits])
    minutes = float(value[deg_digits:])
    result = degrees + minutes / 60.0
    if hemi in ("S", "W"):
        return -result
    if hemi in ("N", "E"):
        return result
    raise ValueError(f"bad hemisphere {hemi!r}")


def parse_nmea(sentence: str) -> GpsFix | None:
    """Return a GpsFix for a valid GGA/RMC sentence carrying a fix, else None.

    Sentences with a missing/bad checksum, other types, malformed fields, or
    no fix (GGA quality 0, RMC status V) all return None.
    """
    sentence = sentence.strip()
    if not sentence.startswith("$") or "*" not in sentence:
        return None
    body, _, given = sentence[1:].partition("*")
    try:
        if int(given, 16) != _checksum(body):
            return None
    except ValueError:
        return None
    fields = body.split(",")
    kind = fields[0][2:]
    try:
        if kind == "GGA" and len(fields) >= 10:
            quality = int(fields[6] or 0)
            if quality == 0:
                return None
            altitude = float(fields[9]) if fields[9] else None
            return GpsFix(
                lat=_coord(fields[2], fields[3], 2),
                lon=_coord(fields[4], fields[5], 3),
                altitude=altitude,
                fix_quality=quality,
            )
        if kind == "RMC" and len(fields) >= 7:
            if fields[2] != "A":
                return None
            return GpsFix(
                lat=_coord(fields[3], fields[4], 2),
                lon=_coord(fields[5], fields[6], 3),
                altitude=None,
                fix_quality=1,
            )
    except (ValueError, IndexError):
        return None
    return None
