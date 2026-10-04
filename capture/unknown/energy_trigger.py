# capture/unknown/energy_trigger.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from capture.unknown.sample_clock import SampleClock


@dataclass(frozen=True)
class TriggerConfig:
    threshold_dbfs: float  # absolute: noise floor (dBFS) + trigger margin (dB)
    cooldown: timedelta


@dataclass(frozen=True)
class TriggerEvent:
    sample_index: int  # absolute stream index of the first above-threshold sample
    time: datetime  # sample-derived UTC time of that sample
    center_freq_hz: float


def find_trigger(
    power: np.ndarray,
    start_index: int,
    clock: SampleClock,
    center_freq_hz: float,
    config: TriggerConfig,
    last_trigger_at: Mapping[float, datetime],
) -> TriggerEvent | None:
    """Return the first sample of `power` at or above the threshold that is
    outside the cooldown window of the last trigger at `center_freq_hz`.

    `power` is linear moving-average |x|^2 with full scale = 1.0 (0 dBFS);
    `power[0]` is stream sample `start_index`. Cooldown is kept as absolute
    time, not sample index, so it survives a flowgraph rebuild that restarts
    sample numbering. Level-triggered: a signal still above threshold when
    its cooldown expires triggers again at the first re-armed sample.
    """
    above = np.flatnonzero(power >= 10.0 ** (config.threshold_dbfs / 10.0))
    last = last_trigger_at.get(center_freq_hz)
    if last is not None:
        rearm_offset = clock.first_index_at_or_after(last + config.cooldown) - start_index
        above = above[above >= rearm_offset]
    if above.size == 0:
        return None
    sample_index = start_index + int(above[0])
    return TriggerEvent(
        sample_index=sample_index,
        time=clock.time_at(sample_index),
        center_freq_hz=center_freq_hz,
    )


def record_trigger(
    last_trigger_at: Mapping[float, datetime], event: TriggerEvent
) -> dict[float, datetime]:
    """Return a new cooldown map with `event` as the latest trigger at its frequency."""
    return {**last_trigger_at, event.center_freq_hz: event.time}
