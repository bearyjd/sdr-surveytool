# tests/capture/unknown/test_snippet_assembler.py
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from capture.unknown.energy_trigger import TriggerConfig
from capture.unknown.sample_clock import SampleClock
from capture.unknown.snippet_assembler import CapturedSnippet, SnippetAssembler

FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
FREQ = 915e6
CONFIG = TriggerConfig(threshold_dbfs=-30.0, cooldown=timedelta(seconds=1))
PRE = 100
POST = 400


def _assembler(
    clock: SampleClock | None = None,
    last_trigger_at: dict | None = None,
    pre: int = PRE,
) -> SnippetAssembler:
    return SnippetAssembler(
        clock=clock or SampleClock(anchor=ANCHOR, sample_rate=FS),
        center_freq_hz=FREQ,
        config=CONFIG,
        pre_trigger_samples=pre,
        post_trigger_samples=POST,
        last_trigger_at=last_trigger_at or {},
    )


def _signal(n: int, bursts: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Quiet ramp (unique values, so slices are distinguishable) with loud
    bursts. Power is |iq|^2 directly: the assembler only cares that power[i]
    describes iq[i], not how it was averaged."""
    iq = (0.01 * np.exp(1j * np.arange(n) / 10.0)).astype(np.complex64)
    for start, stop in bursts:
        iq[start:stop] *= 10.0
    return iq, (np.abs(iq) ** 2).astype(np.float32)


def _feed(assembler: SnippetAssembler, iq, power, chunk: int) -> list[CapturedSnippet]:
    """Feed in fixed-size chunks through ONE reused buffer pair, scribbled
    over after every call -- exactly what GNU Radio does to work()'s input
    buffers, so any sample the assembler keeps without copying is corrupted."""
    iq_buf = np.empty(chunk, dtype=np.complex64)
    power_buf = np.empty(chunk, dtype=np.float32)
    snippets = []
    for start in range(0, len(iq), chunk):
        n = min(chunk, len(iq) - start)
        iq_buf[:n] = iq[start : start + n]
        power_buf[:n] = power[start : start + n]
        snippets += assembler.process(iq_buf[:n], power_buf[:n], start)
        iq_buf[:] = np.nan
        power_buf[:] = np.nan
    return snippets


def test_quiet_input_produces_no_snippet():
    iq, power = _signal(5_000, [])
    assert _feed(_assembler(), iq, power, 1024) == []


@pytest.mark.parametrize("chunk", [1, 7, 64, 333, 5_000])
def test_snippet_is_exact_slice_around_trigger_for_any_chunking(chunk):
    """GNU Radio hands work() variable-size chunks; a burst straddling chunk
    boundaries must yield the identical snippet as one big chunk."""
    iq, power = _signal(5_000, [(1_000, 1_200)])
    snippets = _feed(_assembler(), iq, power, chunk)
    assert len(snippets) == 1
    snippet = snippets[0]
    assert snippet.trigger.sample_index == 1_000
    assert snippet.start_index == 1_000 - PRE
    assert snippet.start_time == ANCHOR + timedelta(seconds=900 / FS)
    np.testing.assert_array_equal(snippet.iq, iq[900:1_400])
    np.testing.assert_array_equal(snippet.power, power[900:1_400])
    assert snippet.iq.dtype == np.complex64 and snippet.power.dtype == np.float32


def test_trigger_within_first_pre_samples_truncates_pre_trigger_history():
    iq, power = _signal(5_000, [(30, 200)])
    (snippet,) = _feed(_assembler(), iq, power, 16)
    assert snippet.start_index == 0
    assert len(snippet.iq) == 30 + POST
    np.testing.assert_array_equal(snippet.iq, iq[: 30 + POST])


def test_burst_too_close_to_end_of_stream_yields_nothing():
    """A capture still collecting when input stops is dropped, never emitted
    half-written."""
    iq, power = _signal(5_000, [(4_800, 5_000)])
    assert _feed(_assembler(), iq, power, 512) == []


def test_burst_inside_cooldown_is_ignored_and_later_burst_captured():
    # Cooldown 1 s = 100_000 samples after the trigger at 1_000.
    iq, power = _signal(130_000, [(1_000, 1_200), (50_000, 50_200), (120_000, 120_200)])
    assembler = _assembler()
    snippets = _feed(assembler, iq, power, 4096)
    assert [s.trigger.sample_index for s in snippets] == [1_000, 120_000]
    assert assembler.last_trigger_at == {FREQ: ANCHOR + timedelta(seconds=1.2)}


def test_cooldown_carried_into_a_rebuilt_assembler_suppresses_retrigger():
    first = _assembler()
    iq, power = _signal(2_000, [(1_000, 1_200)])
    assert len(_feed(first, iq, power, 512)) == 1

    # Flowgraph rebuilt 0.1 s later: indices restart at 0 under a new anchor.
    rebuilt = _assembler(
        clock=SampleClock(anchor=ANCHOR + timedelta(seconds=0.1), sample_rate=FS),
        last_trigger_at=first.last_trigger_at,
    )
    iq, power = _signal(5_000, [(1_000, 1_200)])
    assert _feed(rebuilt, iq, power, 512) == []


def test_zero_pre_trigger_samples_starts_snippet_at_trigger():
    iq, power = _signal(5_000, [(1_000, 1_200)])
    (snippet,) = _feed(_assembler(pre=0), iq, power, 100)
    assert snippet.start_index == 1_000
    np.testing.assert_array_equal(snippet.iq, iq[1_000:1_400])


def test_rejects_an_empty_post_trigger_window():
    """post=0 used to hang process(): the capture could never fill, so the
    read position never advanced."""
    with pytest.raises(ValueError, match="post_trigger_samples"):
        SnippetAssembler(
            clock=SampleClock(anchor=ANCHOR, sample_rate=FS),
            center_freq_hz=FREQ,
            config=CONFIG,
            pre_trigger_samples=PRE,
            post_trigger_samples=0,
            last_trigger_at={},
        )
