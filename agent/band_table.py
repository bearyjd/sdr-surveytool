# agent/band_table.py
"""The curated US band table (agent/data/band_table_us.json) and grounded matching.

Deliberately small: about twenty well-known allocations an urban drive
survey will hit, each with an eCFR citation and the verbatim text that
confirmed its edges. A match is *grounded* when the signal's center lies
inside the band AND its occupied bandwidth is plausible for that band.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_BAND_TABLE = Path(__file__).resolve().parent / "data" / "band_table_us.json"
MIN_HZ = 47e6
MAX_HZ = 6e9


class BandEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    start_hz: float
    end_hz: float
    service: str = Field(min_length=1)
    typical_signals: tuple[str, ...] = Field(min_length=1)
    expected_obw_hz: tuple[float, float]
    citation: str = Field(pattern=r"^47 CFR \d")
    source: str = Field(min_length=1)  # verbatim eCFR text that confirmed the edges

    @model_validator(mode="after")
    def _sane(self) -> BandEntry:
        if not MIN_HZ <= self.start_hz < self.end_hz <= MAX_HZ:
            raise ValueError(f"{self.id}: edges must satisfy {MIN_HZ:g} <= start < end <= {MAX_HZ:g}")
        low, high = self.expected_obw_hz
        if not 0 <= low < high <= MAX_HZ:
            raise ValueError(f"{self.id}: expected_obw_hz must be 0 <= min < max")
        return self


class BandTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    region: str
    verified: str
    entries: tuple[BandEntry, ...] = Field(min_length=1)


@dataclass(frozen=True)
class BandMatch:
    entry: BandEntry
    grounded: bool


def load_band_table(path: Path = DEFAULT_BAND_TABLE) -> BandTable:
    table = BandTable.model_validate(json.loads(path.read_text()))
    ids = [entry.id for entry in table.entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate band ids")
    return table


def match_bands(entries: tuple[BandEntry, ...], center_hz: float, obw_hz: float) -> list[BandMatch]:
    """Every entry the signal's occupied span [center - obw/2, center + obw/2]
    overlaps, marked grounded or not, lowest start first."""
    low, high = center_hz - obw_hz / 2, center_hz + obw_hz / 2
    return [
        BandMatch(
            entry=entry,
            grounded=entry.start_hz <= center_hz <= entry.end_hz
            and entry.expected_obw_hz[0] <= obw_hz <= entry.expected_obw_hz[1],
        )
        for entry in sorted(entries, key=lambda entry: entry.start_hz)
        if entry.start_hz <= high and entry.end_hz >= low
    ]
