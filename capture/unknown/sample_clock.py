# capture/unknown/sample_clock.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SampleClock:
    """Maps absolute stream sample indices to UTC datetimes and back.

    All capture timing (trigger timestamps, cooldown windows, SigMF capture
    datetimes) is derived from sample counts, never from the wall clock. The
    wall clock is read once, when a stream starts, to produce `anchor`, so
    timing is deterministic under test and immune to scheduler jitter.
    """

    anchor: datetime  # UTC time of stream sample index 0
    sample_rate: float

    def time_at(self, sample_index: int) -> datetime:
        return self.anchor + timedelta(seconds=sample_index / self.sample_rate)

    def first_index_at_or_after(self, when: datetime) -> int:
        return math.ceil((when - self.anchor).total_seconds() * self.sample_rate)
