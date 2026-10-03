"""_open_session against the real GNU Radio scheduler; only the SDR source
itself (_build_soapy_source) is replaced, by a finite vector source."""
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("gnuradio")

from gnuradio import blocks, gr  # noqa: E402

from capture.unknown import service  # noqa: E402
from capture.unknown.service import CaptureSettings  # noqa: E402

FS = 100_000.0


def _settings(tmp_path, **overrides) -> CaptureSettings:
    return CaptureSettings(
        **{
            "center_freq_hz": 915e6,
            "sample_rate": FS,
            "noise_floor_dbfs": -40.0,
            "staging_dir": tmp_path,
            "stall_seconds": 0.2,
            # The vector source replays 2 s of samples in milliseconds, so
            # sample time races ahead of wall time; that's not drift here.
            "max_clock_drift_seconds": 60.0,
            **overrides,
        }
    )


def _iq_with_burst_at(start: int, n: int = 200_000) -> np.ndarray:
    rng = np.random.default_rng(5)
    iq = np.sqrt(1e-4 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    iq[start : start + 20_000] += 0.1
    return iq.astype(np.complex64)


@pytest.fixture
def top_block_calls(monkeypatch):
    calls = []
    for name in ("start", "stop", "wait"):
        real = getattr(gr.top_block, name)

        def spy(self, *args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(gr.top_block, name, spy)
    return calls


def test_open_session_streams_snippets_then_stalls_and_tears_down(tmp_path, monkeypatch, top_block_calls):
    opened_at = []

    def fake_source(settings):
        opened_at.append(datetime.now(timezone.utc))
        return blocks.vector_source_c(_iq_with_burst_at(50_000), False), FS

    monkeypatch.setattr(service, "_build_soapy_source", fake_source)
    settings = _settings(tmp_path)

    with service._open_session(settings, {}, Counter()) as snippets:
        snippet = next(snippets)
        # The finite source runs dry, so samples stop arriving: the stall
        # watchdog must end the session rather than wait forever.
        with pytest.raises(RuntimeError, match="stalled"):
            next(snippets)

    assert 50_000 <= snippet.trigger.sample_index < 50_000 + settings.samples(settings.averaging_seconds)
    assert len(snippet.iq) == settings.samples(0.1) + settings.samples(0.9)
    # The anchor (time of sample 0) is read after the device opened.
    anchor = snippet.trigger.time - timedelta(seconds=snippet.trigger.sample_index / FS)
    assert anchor >= opened_at[0]
    assert top_block_calls == ["start", "stop", "wait"]


def test_open_session_tears_down_a_flowgraph_whose_start_fails(tmp_path, monkeypatch, top_block_calls):
    monkeypatch.setattr(
        service,
        "_build_soapy_source",
        lambda settings: (blocks.vector_source_c(np.zeros(10, np.complex64), False), FS),
    )

    def failing_start(self):
        top_block_calls.append("start")
        raise RuntimeError("start failed")

    monkeypatch.setattr(gr.top_block, "start", failing_start)
    with pytest.raises(RuntimeError, match="start failed"):
        with service._open_session(_settings(tmp_path), {}, Counter()):
            pass  # pragma: no cover
    assert top_block_calls == ["start", "stop", "wait"]


def test_source_gets_100ms_of_output_buffer_before_start(tmp_path, monkeypatch):
    """The Python tap competes for the GIL; ~100 ms of source buffer lets a
    briefly starved tap catch up instead of overflowing the SDR."""
    sources = []
    seen_at_start = []

    def fake_source(settings):
        sources.append(blocks.vector_source_c(np.zeros(1_000, np.complex64), False))
        return sources[-1], FS

    real_start = gr.top_block.start

    def recording_start(self, *args, **kwargs):
        seen_at_start.append(sources[0].min_output_buffer(0))
        return real_start(self, *args, **kwargs)

    monkeypatch.setattr(service, "_build_soapy_source", fake_source)
    monkeypatch.setattr(gr.top_block, "start", recording_start)
    with service._open_session(_settings(tmp_path), {}, Counter()) as snippets:
        with pytest.raises(RuntimeError, match="stalled"):
            next(snippets)
    assert seen_at_start == [10_000]  # 0.1 s at 100 kS/s


def test_session_times_samples_at_the_rate_the_sdr_actually_runs(tmp_path, monkeypatch, caplog):
    """Drivers round unsupported rates. Timestamps, windows and the snippet's
    own rate must follow the rate read back from the device, or sample time
    drifts from wall time by the rounding error."""
    actual = FS / 2
    opened_at = []

    def fake_source(settings):
        opened_at.append(datetime.now(timezone.utc))
        return blocks.vector_source_c(_iq_with_burst_at(50_000), False), actual

    monkeypatch.setattr(service, "_build_soapy_source", fake_source)
    with caplog.at_level(logging.WARNING):
        with service._open_session(_settings(tmp_path), {}, Counter()) as snippets:
            snippet = next(snippets)

    assert "requested" in caplog.text
    assert snippet.sample_rate == actual
    assert len(snippet.iq) == round(0.1 * actual) + round(0.9 * actual)
    anchor = snippet.trigger.time - timedelta(seconds=snippet.trigger.sample_index / actual)
    assert opened_at[0] <= anchor <= opened_at[0] + timedelta(seconds=5)


def test_cooldowns_from_the_future_are_clamped_after_a_backward_clock_step(tmp_path, monkeypatch, caplog):
    """Cooldowns persist across sessions as absolute UTC. After the wall clock
    steps back, a stored trigger time can lie in the new session's future,
    which would suppress detection for cooldown + the step."""
    anchor = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        service,
        "_build_soapy_source",
        lambda settings: (blocks.vector_source_c(_iq_with_burst_at(150_000, n=300_000), False), FS),
    )
    settings = _settings(tmp_path, cooldown_seconds=1.0)
    stored = {915e6: anchor + timedelta(minutes=10)}  # recorded before the clock stepped back

    with caplog.at_level(logging.WARNING):
        with service._open_session(settings, stored, Counter(), wall_clock=lambda: anchor) as snippets:
            snippet = next(snippets)

    # Clamped to the anchor, the 1 s cooldown expired before the 1.5 s burst.
    assert snippet.trigger.time == anchor + timedelta(seconds=snippet.trigger.sample_index / FS)
    assert 150_000 <= snippet.trigger.sample_index < 150_100
    assert "stepped back" in caplog.text
