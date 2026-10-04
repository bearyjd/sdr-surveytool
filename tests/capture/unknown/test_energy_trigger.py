# tests/capture/unknown/test_energy_trigger.py
from datetime import datetime, timedelta, timezone

import numpy as np

from capture.unknown.energy_trigger import (
    TriggerConfig,
    TriggerEvent,
    find_trigger,
    record_trigger,
)
from capture.unknown.sample_clock import SampleClock

# 100 kHz keeps every sample time an exact whole number of microseconds,
# which is timedelta's resolution, so time <-> index round trips are exact.
FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
CLOCK = SampleClock(anchor=ANCHOR, sample_rate=FS)
FREQ = 915e6
CONFIG = TriggerConfig(threshold_dbfs=-30.0, cooldown=timedelta(seconds=1))
QUIET = 1e-4  # -40 dBFS
LOUD = 1e-2  # -20 dBFS


def _power(n: int, loud: slice | None = None) -> np.ndarray:
    power = np.full(n, QUIET, dtype=np.float32)
    if loud is not None:
        power[loud] = LOUD
    return power


def test_sample_clock_maps_index_to_time_and_back():
    assert CLOCK.time_at(150_000) == ANCHOR + timedelta(seconds=1.5)
    assert CLOCK.first_index_at_or_after(ANCHOR + timedelta(seconds=1.5)) == 150_000


def test_no_trigger_when_everything_is_below_threshold():
    assert find_trigger(_power(1000), 0, CLOCK, FREQ, CONFIG, {}) is None


def test_trigger_reports_absolute_index_and_sample_derived_time():
    event = find_trigger(_power(1000, slice(400, 500)), 5_000, CLOCK, FREQ, CONFIG, {})
    assert event == TriggerEvent(
        sample_index=5_400,
        time=ANCHOR + timedelta(seconds=0.054),
        center_freq_hz=FREQ,
    )


def test_power_exactly_at_threshold_triggers():
    power = _power(10)
    power[3] = 10 ** (-30.0 / 10)
    event = find_trigger(power, 0, CLOCK, FREQ, CONFIG, {})
    assert event is not None and event.sample_index == 3


def test_cooldown_suppresses_trigger_at_same_frequency():
    last = {FREQ: CLOCK.time_at(0)}
    # Loud samples at 0.5 s, well inside the 1 s cooldown.
    assert find_trigger(_power(1000, slice(0, 1000)), 50_000, CLOCK, FREQ, CONFIG, last) is None


def test_cooldown_is_tracked_per_center_frequency():
    last = {2.4e9: CLOCK.time_at(0)}
    event = find_trigger(_power(1000, slice(10, 20)), 50_000, CLOCK, FREQ, CONFIG, last)
    assert event is not None and event.sample_index == 50_010


def test_rearms_exactly_when_cooldown_expires():
    last = {FREQ: CLOCK.time_at(0)}
    # Cooldown expires at index 100_000; the chunk covers 99_990..100_009, all loud.
    event = find_trigger(_power(20, slice(0, 20)), 99_990, CLOCK, FREQ, CONFIG, last)
    assert event is not None and event.sample_index == 100_000


def test_continuous_signal_retriggers_once_cooldown_expires():
    """Level-triggered by design: an emitter that never goes quiet is
    re-sampled once per cooldown window instead of recorded once and lost."""
    last = {FREQ: CLOCK.time_at(0)}
    event = find_trigger(_power(200_000, slice(0, 200_000)), 0, CLOCK, FREQ, CONFIG, last)
    assert event is not None and event.sample_index == 100_000


def test_cooldown_survives_a_stream_restart_with_a_new_anchor():
    """Sample indices restart at 0 whenever the flowgraph is rebuilt, so the
    cooldown is kept as absolute time: a trigger 0.2 s before the restart
    still suppresses one 0.3 s after it (0.5 s < 1 s cooldown)."""
    restart_clock = SampleClock(anchor=ANCHOR + timedelta(seconds=0.2), sample_rate=FS)
    last = {FREQ: ANCHOR}
    assert find_trigger(_power(1000, slice(0, 1000)), 30_000, restart_clock, FREQ, CONFIG, last) is None
    event = find_trigger(_power(1000, slice(0, 1000)), 80_000, restart_clock, FREQ, CONFIG, last)
    assert event is not None and event.sample_index == 80_000


def test_record_trigger_returns_new_map_without_mutating_input():
    original = {2.4e9: ANCHOR}
    event = TriggerEvent(sample_index=7, time=CLOCK.time_at(7), center_freq_hz=FREQ)
    updated = record_trigger(original, event)
    assert updated == {2.4e9: ANCHOR, FREQ: CLOCK.time_at(7)}
    assert original == {2.4e9: ANCHOR}
