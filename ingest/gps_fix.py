from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GpsFix:
    lat: float
    lon: float
    altitude: float | None
    fix_quality: int


class GpsFixProvider(Protocol):
    def current_fix(self) -> GpsFix | None: ...


class StaticGpsFixProvider:
    """Test/dev stand-in until the real u-blox M8N reader service (gps/)
    exists. Always returns the fix it was constructed with."""

    def __init__(self, fix: GpsFix) -> None:
        self._fix = fix

    def current_fix(self) -> GpsFix | None:
        return self._fix
