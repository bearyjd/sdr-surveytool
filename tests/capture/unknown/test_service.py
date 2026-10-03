# tests/capture/unknown/test_service.py
import gc
import logging
import math
import os
import queue
import signal
import weakref
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sigmf
from sigmf import sigmffile

from capture.unknown import service
from capture.unknown.energy_trigger import TriggerEvent
from capture.unknown.sample_clock import SampleClock
from capture.unknown.service import CaptureSettings, process_snippet
from capture.unknown.snippet_assembler import CapturedSnippet
from capture.unknown.snippet_writer import THRESHOLD_KEY
from schema.records import ClassificationStatus, Modality, UnifiedRecord
from storage.snippet_store import DEFAULT_MAX_SNIPPET_BYTES

FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)


def _settings(staging_dir: Path, **overrides) -> CaptureSettings:
    return CaptureSettings(
        **{
            "center_freq_hz": 915e6,
            "sample_rate": FS,
            "noise_floor_dbfs": -40.0,
            "staging_dir": staging_dir,
            **overrides,
        }
    )


def _snippet(
    trigger_index: int = 60_000,
    pre: int = 10_000,
    post: int = 90_000,
    sample_rate: float = FS,
    center_freq_hz: float = 915e6,
) -> CapturedSnippet:
    n = pre + post
    iq = np.full(n, 0.01, dtype=np.complex64)
    iq[pre : pre + 20_000] = 0.1  # -20 dBFS burst right at the trigger
    power = (np.abs(iq) ** 2).astype(np.float32)
    start = trigger_index - pre
    return CapturedSnippet(
        iq=iq,
        power=power,
        start_index=start,
        start_time=ANCHOR + timedelta(seconds=start / sample_rate),
        trigger=TriggerEvent(
            sample_index=trigger_index,
            time=ANCHOR + timedelta(seconds=trigger_index / sample_rate),
            center_freq_hz=center_freq_hz,
        ),
        sample_rate=sample_rate,
    )


def test_settings_reject_a_threshold_at_or_above_full_scale(tmp_path):
    with pytest.raises(ValueError, match="full scale"):
        _settings(tmp_path, noise_floor_dbfs=-5.0, threshold_db=10.0)


def test_settings_reject_non_positive_cooldown(tmp_path):
    """A zero cooldown with a level trigger re-captures a continuous emitter
    back to back, i.e. writes to disk as fast as the radio produces samples."""
    with pytest.raises(ValueError, match="cooldown_seconds"):
        _settings(tmp_path, cooldown_seconds=0.0)


def test_settings_require_a_cooldown_of_at_least_a_second_and_a_snippet(tmp_path):
    """A sub-microsecond cooldown rounds to timedelta(0), and one shorter than
    the snippet re-arms a level trigger as soon as the capture completes."""
    for too_short in (1e-7, 0.5):
        with pytest.raises(ValueError, match="cooldown_seconds"):
            _settings(tmp_path, cooldown_seconds=too_short)
    assert _settings(tmp_path, cooldown_seconds=1.0).cooldown_seconds == 1.0  # 0.1 + 0.9 s
    with pytest.raises(ValueError, match="cooldown_seconds"):
        _settings(tmp_path, pre_trigger_seconds=1.0, post_trigger_seconds=2.0, cooldown_seconds=2.5)
    assert _settings(
        tmp_path, pre_trigger_seconds=1.0, post_trigger_seconds=2.0, cooldown_seconds=3.0
    ).cooldown_seconds == 3.0


def test_settings_reject_a_drift_threshold_inside_normal_buffering_lag(tmp_path):
    """Sample time normally lags wall time by ~100 ms of buffering; a tighter
    threshold would rebuild the radio session on every poll."""
    with pytest.raises(ValueError, match="max_clock_drift_seconds"):
        _settings(tmp_path, max_clock_drift_seconds=0.05)
    assert _settings(tmp_path, max_clock_drift_seconds=0.5).max_clock_drift_seconds == 0.5


FLOAT_SETTINGS = [
    "center_freq_hz",
    "sample_rate",
    "noise_floor_dbfs",
    "threshold_db",
    "averaging_seconds",
    "pre_trigger_seconds",
    "post_trigger_seconds",
    "cooldown_seconds",
    "gain_db",
    "stall_seconds",
    "max_clock_drift_seconds",
]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", FLOAT_SETTINGS)
def test_settings_reject_non_finite_numbers(tmp_path, field, value):
    """NaN slips past every ordering check (all comparisons are False), and
    inf turns windows and cooldowns into hangs or never-triggers."""
    with pytest.raises(ValueError, match=field):
        _settings(tmp_path, **{field: value})


def test_settings_cap_the_sample_rate_at_the_ad9361_maximum(tmp_path):
    roomy = 16 * 1024**3
    assert _settings(tmp_path, sample_rate=61.44e6, max_snippet_memory_bytes=roomy).sample_rate == 61.44e6
    with pytest.raises(ValueError, match="61.44e6"):
        _settings(tmp_path, sample_rate=61.45e6, max_snippet_memory_bytes=roomy)


@pytest.mark.parametrize("field", ["averaging_seconds", "pre_trigger_seconds", "post_trigger_seconds"])
def test_settings_cap_each_window_at_five_seconds(tmp_path, field):
    """At 61.44 MS/s a 5 s window is already 2.5 GB of cf32 in memory."""
    assert getattr(_settings(tmp_path, **{field: 5.0}), field) == 5.0
    with pytest.raises(ValueError, match=field):
        _settings(tmp_path, **{field: 5.01})


def test_the_largest_configurable_snippet_fits_the_store_size_bound():
    """Ingest rejects staged data larger than DEFAULT_MAX_SNIPPET_BYTES; the
    biggest snippet capture's settings allow must stay within it."""
    largest = service._MAX_SAMPLE_RATE * 2 * service._MAX_WINDOW_SECONDS * 8
    assert largest <= DEFAULT_MAX_SNIPPET_BYTES


def test_peak_snippet_memory_estimate_follows_the_real_buffers(tmp_path):
    """12 B/sample (cf32 iq + float32 power) for 2 queued + 1 processing + 1
    assembling snippet, 2 x 12 B x pre for the history and its copy when a
    capture begins, and 20 B/sample of measurement transients."""
    settings = _settings(tmp_path, sample_rate=20e6)  # 0.1 s pre + 0.9 s post
    assert settings.estimated_peak_memory_bytes == 12 * 20_000_000 * 4 + 24 * 2_000_000 + 20 * 20_000_000
    assert settings.estimated_peak_memory_bytes <= 2 * 1024**3  # the defaults fit a Jetson


def test_settings_fail_fast_when_snippets_would_not_fit_the_memory_budget(tmp_path):
    with pytest.raises(ValueError, match="3,942,400,000"):
        _settings(tmp_path, sample_rate=56e6)
    assert _settings(tmp_path, sample_rate=56e6, max_snippet_memory_bytes=4 * 1024**3).sample_rate == 56e6


def test_settings_reject_windows_shorter_than_one_sample(tmp_path):
    with pytest.raises(ValueError, match="at least one sample"):
        _settings(tmp_path, averaging_seconds=1e-7)


def test_process_snippet_stages_sigmf_and_builds_record(tmp_path):
    record = process_snippet(_snippet(), _settings(tmp_path / "staging"), "s1", "op1")

    data_path = Path(record.metadata.iq_snippet_path)
    assert data_path.parent == (tmp_path / "staging").resolve()
    assert data_path.is_file() and data_path.with_suffix(".sigmf-meta").is_file()
    assert record.modality is Modality.UNKNOWN
    # Sample-derived, never wall clock: anchor + 60_000 / 100 kHz.
    assert record.timestamp == ANCHOR + timedelta(seconds=0.6)
    assert record.identifier.center_freq == 915e6
    assert record.identifier.bandwidth_estimate > 0
    assert record.signal.peak_power == pytest.approx(-20.0, abs=0.01)
    assert record.signal.rssi == pytest.approx(-20.0, abs=0.01)
    assert record.signal.snr == pytest.approx(20.0, abs=0.01)
    assert record.metadata.snippet_duration_ms == 1000
    assert record.metadata.sample_rate == FS
    assert record.metadata.classification_status is ClassificationStatus.UNCLASSIFIED


def test_record_describes_the_snippet_at_its_actual_sample_rate(tmp_path):
    record = process_snippet(_snippet(sample_rate=FS / 2), _settings(tmp_path), "s1", "op1")
    assert record.metadata.sample_rate == FS / 2
    assert record.metadata.snippet_duration_ms == 2000  # 100_000 samples at 50 kS/s


def test_bandwidth_is_measured_on_the_burst_against_the_pre_trigger_noise(tmp_path, monkeypatch):
    """The pre-trigger samples are below threshold by construction: they are
    the per-bin noise reference for the post-trigger burst."""
    calls = []

    def spy(iq, sample_rate, threshold_dbfs, *, reference_iq=None):
        calls.append((iq, reference_iq))
        return service.OccupiedBandwidth(hz=20_000.0, reliable=True)

    monkeypatch.setattr(service, "occupied_bandwidth", spy)
    snippet = _snippet(trigger_index=60_000, pre=10_000, post=90_000)
    process_snippet(snippet, _settings(tmp_path), "s1", "op1")
    ((burst, reference),) = calls
    np.testing.assert_array_equal(reference, snippet.iq[:10_000])
    np.testing.assert_array_equal(burst, snippet.iq[10_000:])


def test_record_flags_an_unreliable_bandwidth_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        service,
        "occupied_bandwidth",
        lambda *args, **kwargs: service.OccupiedBandwidth(hz=95_000.0, reliable=False),
    )
    record = process_snippet(_snippet(), _settings(tmp_path), "s1", "op1")
    assert record.identifier.bandwidth_estimate == 95_000.0
    assert record.metadata.quality_flags["bandwidth_estimate_unreliable"] is True


def test_record_and_sigmf_carry_the_frequency_the_sdr_actually_tuned(tmp_path):
    tuned = 915_001_250.0
    record = process_snippet(_snippet(center_freq_hz=tuned), _settings(tmp_path), "s1", "op1")
    assert record.identifier.center_freq == tuned
    recording = sigmffile.fromfile(record.metadata.iq_snippet_path)
    assert recording.get_captures()[0][sigmf.FREQUENCY_KEY] == tuned


def test_the_sigmf_records_the_trigger_threshold_for_part_4(tmp_path):
    settings = _settings(tmp_path)
    record = process_snippet(_snippet(), settings, "s1", "op1")
    pre_trigger, _ = sigmffile.fromfile(record.metadata.iq_snippet_path).get_annotations()
    assert pre_trigger[THRESHOLD_KEY] == settings.threshold_dbfs == -30.0


def test_snippet_duration_reflects_samples_actually_captured(tmp_path):
    """A trigger in the first 0.1 s truncates the pre-trigger history."""
    record = process_snippet(_snippet(trigger_index=3_000, pre=3_000), _settings(tmp_path), "s1", "op1")
    assert record.metadata.snippet_duration_ms == 930


class _FakeTap:
    def __init__(self) -> None:
        self.samples_seen = 0
        self.error: Exception | None = None


def _drain(snippets, tap, monotonic, wall_clock=lambda: ANCHOR, dropped=None):
    return service._drain(
        snippets,
        tap,
        dropped=deque() if dropped is None else dropped,
        clock=SampleClock(anchor=ANCHOR, sample_rate=FS),
        stall_seconds=5.0,
        max_drift_seconds=2.0,
        poll_seconds=0.0,
        monotonic=monotonic,
        wall_clock=wall_clock,
    )


def test_drain_yields_snippets_then_raises_when_stream_stalls():
    snippets: queue.Queue = queue.Queue()
    snippets.put("snippet-1")
    tap = _FakeTap()
    times = iter([0.0, 1.0, 3.0, 9.0])
    drained = _drain(snippets, tap, monotonic=lambda: next(times))

    assert next(drained) == "snippet-1"
    tap.samples_seen = 4096  # progress at t=1.0
    with pytest.raises(RuntimeError, match="stalled"):
        next(drained)  # t=3.0 still within 5 s of progress; t=9.0 is not


def test_drain_reraises_a_tap_failure():
    tap = _FakeTap()
    tap.error = ValueError("assembler bug")
    drained = _drain(queue.Queue(), tap, monotonic=lambda: 0.0)
    with pytest.raises(RuntimeError, match="tap failed") as excinfo:
        next(drained)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_a_measurement_failure_writes_nothing(tmp_path, monkeypatch, caplog):
    """All DSP and normalization runs before anything is written, so a
    failure there can't leave a staged pair behind for nobody to adopt."""
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)

    def broken_bandwidth(*args, **kwargs):
        raise FloatingPointError("injected DSP failure")

    monkeypatch.setattr(service, "occupied_bandwidth", broken_bandwidth)
    emitter = _FakeEmitter("/unused.sock")
    drops: Counter = Counter()
    with caplog.at_level(logging.ERROR):
        service._stage_and_emit(_snippet(), _settings(staging, min_free_bytes=0), emitter, "s1", "op1", drops)

    assert list(staging.iterdir()) == []
    assert emitter.records == []
    assert drops == Counter({"processing_error": 1})
    assert "injected DSP failure" in caplog.text


def test_a_write_failure_still_emits_the_measured_detection(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)

    def broken_write(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(service, "write_sigmf_snippet", broken_write)
    emitter = _FakeEmitter("/unused.sock")
    drops: Counter = Counter()
    service._stage_and_emit(_snippet(), _settings(staging, min_free_bytes=0), emitter, "s1", "op1", drops)

    (record,) = emitter.records
    assert record.metadata.iq_snippet_path is None
    assert record.metadata.quality_flags["snippet_dropped"] == "processing_error"
    assert record.signal.peak_power == pytest.approx(-20.0, abs=0.01)
    assert list(staging.iterdir()) == []
    assert drops == Counter({"processing_error": 1})


def _with_nan_burst(snippet: CapturedSnippet, start: int, count: int) -> CapturedSnippet:
    """DMA or driver corruption: a run of NaN samples (and the moving-average
    power they poison)."""
    iq, power = snippet.iq.copy(), snippet.power.copy()
    iq[start : start + count] = np.nan + 1j * np.nan
    power[start : start + count] = np.nan
    return CapturedSnippet(
        iq=iq,
        power=power,
        start_index=snippet.start_index,
        start_time=snippet.start_time,
        trigger=snippet.trigger,
        sample_rate=snippet.sample_rate,
    )


def test_non_finite_samples_are_left_out_of_measurements_and_flagged(tmp_path):
    """NaN propagated into power and SNR, serialized as JSON null, and failed
    ingest's validation: the whole detection vanished."""
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    emitter = _FakeEmitter("/unused.sock")
    snippet = _with_nan_burst(_snippet(), start=12_000, count=500)
    service._stage_and_emit(snippet, _settings(staging, min_free_bytes=0), emitter, "s1", "op1", Counter())

    (record,) = emitter.records
    assert record.metadata.quality_flags["non_finite_samples"] == 500
    assert record.signal.peak_power == pytest.approx(-20.0, abs=0.01)
    assert record.signal.rssi == pytest.approx(-20.0, abs=0.01)
    assert math.isfinite(record.identifier.bandwidth_estimate)
    assert record.metadata.iq_snippet_path is not None  # the raw IQ is still kept
    assert UnifiedRecord.model_validate_json(record.model_dump_json()) == record


def test_a_snippet_with_no_finite_samples_is_dropped_not_emitted(tmp_path, caplog):
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    emitter = _FakeEmitter("/unused.sock")
    drops: Counter = Counter()
    snippet = _with_nan_burst(_snippet(), start=0, count=100_000)
    with caplog.at_level(logging.ERROR):
        service._stage_and_emit(snippet, _settings(staging, min_free_bytes=0), emitter, "s1", "op1", drops)
    assert emitter.records == []
    assert list(staging.iterdir()) == []
    assert drops == Counter({"non_finite": 1})
    assert "100000 of 100000" in caplog.text


def test_queue_full_summary_ignores_non_finite_power():
    snippets: queue.Queue = queue.Queue(maxsize=1)
    dropped: deque = deque(maxlen=4)
    offer = service._offer_or_drop(snippets, Counter(), dropped, threshold_dbfs=-30.0)
    offer(_snippet())
    offer(_with_nan_burst(_snippet(), start=12_000, count=500))
    (summary,) = dropped
    assert summary.peak_power_dbfs == pytest.approx(-20.0, abs=0.01)
    assert summary.mean_power_dbfs == pytest.approx(-20.0, abs=0.01)


def test_missing_staging_dir_drops_only_the_snippet_not_the_session(tmp_path, caplog):
    """If staging vanishes mid-survey, an exception escaping here would end
    (and rebuild) the radio session on every snippet. Log it, keep the
    detection without its IQ, and carry on."""
    emitter = _FakeEmitter("/unused.sock")
    drops: Counter = Counter()
    with caplog.at_level(logging.ERROR):
        service._stage_and_emit(_snippet(), _settings(tmp_path / "gone"), emitter, "s1", "op1", drops)

    (record,) = emitter.records
    assert record.metadata.iq_snippet_path is None
    assert record.metadata.quality_flags["snippet_dropped"] == "staging_unavailable"
    assert drops == Counter({"staging_unavailable": 1})
    assert "staging" in caplog.text


def test_full_snippet_queue_drops_the_iq_but_keeps_a_summary_without_blocking():
    """The tap calls this on the GNU Radio scheduler thread, which must never
    block (the SDR would overflow): a full queue drops the snippet's IQ but
    keeps a cheap summary of the detection for the service thread."""
    snippets: queue.Queue = queue.Queue(maxsize=2)
    dropped: deque = deque(maxlen=4)
    drops: Counter = Counter()
    offer = service._offer_or_drop(snippets, drops, dropped, threshold_dbfs=-30.0)
    for _ in range(3):
        offer(_snippet())
    assert snippets.qsize() == 2
    assert drops == Counter({"queue_full": 1})
    (summary,) = dropped
    assert summary.trigger == _snippet().trigger
    assert summary.sample_rate == FS
    assert summary.peak_power_dbfs == pytest.approx(-20.0, abs=0.01)
    assert summary.mean_power_dbfs == pytest.approx(-20.0, abs=0.01)
    assert summary.duration_ms == 1000


def test_drain_hands_dropped_detections_to_the_service_thread():
    dropped: deque = deque()
    summary = service._DroppedDetection(
        trigger=_snippet().trigger,
        sample_rate=FS,
        peak_power_dbfs=-20.0,
        mean_power_dbfs=-21.0,
        duration_ms=1000,
    )
    dropped.append(summary)
    drained = _drain(queue.Queue(), _FakeTap(), monotonic=lambda: 0.0, dropped=dropped)
    assert next(drained) is summary


def test_drain_delivers_completed_snippets_before_reraising_a_tap_failure():
    snippets: queue.Queue = queue.Queue()
    snippets.put("snippet-1")
    snippets.put("snippet-2")
    tap = _FakeTap()
    tap.error = ValueError("assembler bug")
    drained = _drain(snippets, tap, monotonic=lambda: 0.0)
    assert [next(drained), next(drained)] == ["snippet-1", "snippet-2"]
    with pytest.raises(RuntimeError, match="tap failed"):
        next(drained)


def test_drain_delivers_completed_snippets_before_reporting_a_stall():
    snippets: queue.Queue = queue.Queue()
    for name in ("snippet-1", "snippet-2", "snippet-3"):
        snippets.put(name)
    times = iter([0.0, 9.0, 9.0, 9.0, 9.0])
    drained = _drain(snippets, _FakeTap(), monotonic=lambda: next(times))
    assert list(_take_until_error(drained)) == ["snippet-1", "snippet-2", "snippet-3"]


@pytest.mark.parametrize("wall_offset_s", [2.5, -2.5])
def test_drain_ends_the_session_when_sample_time_drifts_from_wall_time(wall_offset_s):
    """Dropped samples (SDR overflow) put sample time behind wall time, and an
    NTP/GPS clock step jumps wall time either way. Past the threshold the
    session ends, and the rebuild re-anchors the sample clock."""
    tap = _FakeTap()
    tap.samples_seen = 100_000
    progress = iter([100_000, 200_000])  # the stream advances to anchor + 2.0 s

    def monotonic():
        tap.samples_seen = next(progress)
        return 0.0

    walls = iter([ANCHOR + timedelta(seconds=2.0 + wall_offset_s)])
    drained = _drain(queue.Queue(), tap, monotonic=monotonic, wall_clock=lambda: next(walls))
    with pytest.raises(RuntimeError, match="off the wall clock"):
        next(drained)


def test_frozen_stream_is_reported_as_a_stall_not_as_clock_drift():
    """With no samples arriving, sample time stands still while wall time
    moves on. That is the stall watchdog's case (including a slow stream
    start), not a clock step, so drift is only judged on progress."""
    ticks = iter([0.0, 1.0, 2.5, 4.0, 6.0])
    now = [0.0]

    def monotonic():
        now[0] = next(ticks)
        return now[0]

    drained = _drain(
        queue.Queue(),
        _FakeTap(),
        monotonic=monotonic,
        wall_clock=lambda: ANCHOR + timedelta(seconds=now[0]),
    )
    with pytest.raises(RuntimeError, match="stalled"):
        next(drained)


def _take_until_error(drained):
    with pytest.raises(RuntimeError, match="stalled"):
        while True:
            yield next(drained)


class _FakeEmitter:
    def __init__(self, socket_path: str) -> None:
        self.records = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        pass

    def emit(self, record) -> None:
        self.records.append(record)


def test_run_survives_radio_failures_and_keeps_cooldown_across_rebuilds(tmp_path, monkeypatch):
    """Radio missing twice (backoff 1 s, 2 s), then a session that captures
    one snippet before its stream stalls (backoff resets to 1 s because the
    radio did open), then the 4th session must receive that snippet's
    trigger time so the rebuilt flowgraph doesn't immediately retrigger."""
    snippet = _snippet()
    sessions = []

    @contextmanager
    def fake_open_session(settings, last_trigger_at, drops):
        sessions.append(dict(last_trigger_at))
        if len(sessions) <= 2:
            raise RuntimeError("SoapySDR::Device::make() no match")
        if len(sessions) == 4:
            raise KeyboardInterrupt  # ends the test; not an Exception, so not retried

        def stalled():
            yield snippet
            raise RuntimeError("SDR stream stalled")

        yield stalled()

    sleeps = []
    emitters = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 10:  # raised inside run()'s except block, so it escapes
            raise AssertionError(f"run() kept retrying: {sleeps}")

    monkeypatch.setattr(service, "_open_session", fake_open_session)
    monkeypatch.setattr(service.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        service, "RecordEmitter", lambda path: emitters.append(_FakeEmitter(path)) or emitters[-1]
    )

    with pytest.raises(KeyboardInterrupt):
        service.run(_settings(tmp_path, min_free_bytes=0), "/unused.sock", "s1", "op1")

    assert sleeps == [1.0, 2.0, 1.0]
    assert sessions[3] == {915e6: snippet.trigger.time}
    assert len(emitters[0].records) == 1


def test_quick_drift_rebuilds_escalate_the_backoff_instead_of_looping(tmp_path, monkeypatch):
    """Sustained overflow makes every new session drift within seconds. A
    session that drifts within 60 s of opening must not reset the backoff,
    while a drift after a long healthy session still restarts promptly."""
    sessions = []

    @contextmanager
    def fake_open_session(settings, last_trigger_at, drops):
        sessions.append(len(sessions) + 1)
        if len(sessions) == 4:
            raise KeyboardInterrupt

        def drifting():
            raise service._ClockDrift("Sample clock is +3.0s off the wall clock")
            yield  # pragma: no cover

        yield drifting()

    # monotonic() at: open 1, drift 1 (100 s later), open 2, drift 2 (1 s),
    # open 3, drift 3 (1 s).
    ticks = iter([0.0, 100.0, 100.0, 101.0, 101.0, 102.0])
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 10:
            raise AssertionError(f"run() kept retrying: {sleeps}")

    monkeypatch.setattr(service, "_open_session", fake_open_session)
    monkeypatch.setattr(service, "RecordEmitter", _FakeEmitter)
    monkeypatch.setattr(service.time, "sleep", fake_sleep)
    monkeypatch.setattr(service.time, "monotonic", lambda: next(ticks))
    with pytest.raises(KeyboardInterrupt):
        service.run(_settings(tmp_path, min_free_bytes=0), "/unused.sock", "s1", "op1")
    assert sleeps == [1.0, 1.0, 2.0]


def test_a_long_healthy_session_clears_an_escalated_backoff(tmp_path, monkeypatch):
    """Two quick drift rebuilds escalate the backoff. A third session that then
    runs healthily for two hours before stalling must retry after the initial
    backoff, not the escalated one."""
    sessions = []

    @contextmanager
    def fake_open_session(settings, last_trigger_at, drops):
        sessions.append(len(sessions) + 1)
        if len(sessions) == 4:
            raise KeyboardInterrupt

        def failing():
            if len(sessions) < 3:
                raise service._ClockDrift("Sample clock is +3.0s off the wall clock")
            raise RuntimeError("SDR stream stalled")
            yield  # pragma: no cover

        yield failing()

    # monotonic() at: open 1, drift 1 (1 s), open 2, drift 2 (1 s),
    # open 3, stall 3 (2 h later).
    ticks = iter([0.0, 1.0, 1.0, 2.0, 2.0, 7202.0])
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 10:
            raise AssertionError(f"run() kept retrying: {sleeps}")

    monkeypatch.setattr(service, "_open_session", fake_open_session)
    monkeypatch.setattr(service, "RecordEmitter", _FakeEmitter)
    monkeypatch.setattr(service.time, "sleep", fake_sleep)
    monkeypatch.setattr(service.time, "monotonic", lambda: next(ticks))
    with pytest.raises(KeyboardInterrupt):
        service.run(_settings(tmp_path, min_free_bytes=0), "/unused.sock", "s1", "op1")
    assert sleeps == [1.0, 2.0, 1.0]


def test_run_emits_a_queue_full_detection_and_keeps_its_cooldown(tmp_path, monkeypatch):
    trigger = _snippet().trigger
    summary = service._DroppedDetection(
        trigger=trigger, sample_rate=FS, peak_power_dbfs=-18.0, mean_power_dbfs=-20.0, duration_ms=1000
    )
    sessions = []

    @contextmanager
    def fake_open_session(settings, last_trigger_at, drops):
        sessions.append(dict(last_trigger_at))
        if len(sessions) == 2:
            raise KeyboardInterrupt

        def one_dropped_detection():
            yield summary
            raise RuntimeError("SDR stream stalled")

        yield one_dropped_detection()

    emitters = []
    monkeypatch.setattr(service, "_open_session", fake_open_session)
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(service, "RecordEmitter", lambda path: emitters.append(_FakeEmitter(path)) or emitters[-1])
    with pytest.raises(KeyboardInterrupt):
        service.run(_settings(tmp_path, min_free_bytes=0), "/unused.sock", "s1", "op1")

    (record,) = emitters[0].records
    assert record.timestamp == trigger.time
    assert record.metadata.iq_snippet_path is None
    assert record.metadata.quality_flags["snippet_dropped"] == "queue_full"
    assert record.identifier.bandwidth_estimate is None
    assert (record.signal.peak_power, record.signal.rssi) == (-18.0, -20.0)
    assert sessions[1] == {915e6: trigger.time}


def test_settings_reject_a_relative_staging_dir(tmp_path):
    """Capture and ingest resolve relative paths against their own working
    directories; if those differ every snippet is rejected as outside_staging."""
    with pytest.raises(ValueError, match="absolute"):
        _settings(Path("data/snippet-staging"))


def test_run_secures_and_logs_the_staging_dir_at_startup(tmp_path, monkeypatch, caplog):
    seen = []

    @contextmanager
    def fake_open_session(settings, last_trigger_at, drops):
        seen.append(settings.staging_dir)
        raise KeyboardInterrupt
        yield  # pragma: no cover

    monkeypatch.setattr(service, "_open_session", fake_open_session)
    monkeypatch.setattr(service, "RecordEmitter", _FakeEmitter)
    with caplog.at_level(logging.INFO), pytest.raises(KeyboardInterrupt):
        service.run(_settings(tmp_path / "staging"), "/unused.sock", "s1", "op1")

    staging = (tmp_path / "staging").resolve()
    assert seen == [staging]
    assert staging.stat().st_mode & 0o077 == 0
    assert str(staging) in caplog.text


def test_a_failed_emit_drops_only_that_snippet(tmp_path, caplog):
    class BrokenEmitter:
        def emit(self, record) -> None:
            raise OSError("ingest socket gone")

    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)  # run() creates it at startup
    drops: Counter = Counter()
    with caplog.at_level(logging.ERROR):
        service._stage_and_emit(
            _snippet(), _settings(staging, min_free_bytes=0), BrokenEmitter(), "s1", "op1", drops
        )
    assert "dropping it" in caplog.text
    # The record never reached ingest, so nothing will ever adopt the staged
    # pair: it must not be left behind to fill the disk.
    assert list(staging.iterdir()) == []
    assert drops == Counter({"emit_failed": 1})


def test_low_disk_drops_only_the_snippet_and_still_emits_the_detection(tmp_path, monkeypatch, caplog):
    """Staging, the snippet store and the database share one disk, and a
    continuous emitter writes ~19 GB/h at the default cooldown: below the
    free-space floor the IQ is never written, but the detection and its
    measurements are still emitted, flagged (mirroring rejected snippets)."""
    monkeypatch.setattr(service.shutil, "disk_usage", lambda path: SimpleNamespace(free=3 * 1024**3))
    emitter = _FakeEmitter("/unused.sock")
    drops: Counter = Counter()
    settings = _settings(tmp_path / "staging", min_free_bytes=3 * 1024**3)

    with caplog.at_level(logging.WARNING):
        service._stage_and_emit(_snippet(), settings, emitter, "s1", "op1", drops)

    (record,) = emitter.records
    assert record.metadata.iq_snippet_path is None
    assert record.metadata.quality_flags == {"power_units": "dBFS", "snippet_dropped": "low_disk"}
    assert record.signal.peak_power == pytest.approx(-20.0, abs=0.01)
    assert not (tmp_path / "staging").exists()
    assert drops == Counter({"low_disk": 1})
    assert "free" in caplog.text


def test_cli_requires_noise_floor_and_builds_settings():
    with pytest.raises(SystemExit):
        service._parse_args(["--survey-id", "s", "--operator-id", "o", "--center-freq", "915e6"])

    settings, args = service._parse_args(
        [
            "--survey-id", "s",
            "--operator-id", "o",
            "--center-freq", "915e6",
            "--noise-floor-dbfs", "-60",
        ]
    )
    assert settings.center_freq_hz == 915e6
    assert settings.sample_rate == 20e6
    assert settings.threshold_dbfs == -50.0
    assert settings.staging_dir == Path("/var/lib/sdr-surveytool/snippet-staging")
    assert settings.min_free_bytes == 2 * 1024**3
    assert settings.max_clock_drift_seconds == 2.0
    assert settings.max_snippet_memory_bytes == 2 * 1024**3
    assert args.socket_path == "/tmp/sdr-ingest.sock"


def test_cli_rejects_a_relative_staging_dir_with_usage_error():
    with pytest.raises(SystemExit):
        service._parse_args(
            [
                "--survey-id", "s",
                "--operator-id", "o",
                "--center-freq", "915e6",
                "--noise-floor-dbfs", "-60",
                "--staging-dir", "data/snippet-staging",
            ]
        )


def test_cli_rejects_invalid_settings_with_usage_error():
    with pytest.raises(SystemExit):
        service._parse_args(
            [
                "--survey-id", "s",
                "--operator-id", "o",
                "--center-freq", "915e6",
                "--noise-floor-dbfs", "-5",
            ]
        )


class _UnhandledSigterm(Exception):
    pass


def test_sigterm_shuts_down_cleanly_like_ctrl_c(monkeypatch, caplog):
    """systemd/docker stop send SIGTERM; it must unwind like Ctrl-C (flowgraph
    stop/wait, emitter close, a clean log line), not kill the process."""

    def guard(signum, frame):  # stands in for the default action: killing pytest
        raise _UnhandledSigterm

    def run_until_sigterm(**kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        pytest.fail("SIGTERM did not interrupt run()")

    original = signal.signal(signal.SIGTERM, guard)
    monkeypatch.setattr(service, "run", run_until_sigterm)
    try:
        with caplog.at_level(logging.INFO):
            service.main(
                ["--survey-id", "s", "--operator-id", "o", "--center-freq", "915e6", "--noise-floor-dbfs", "-60"]
            )
    finally:
        signal.signal(signal.SIGTERM, original)
    assert "Shutting down" in caplog.text


def test_run_releases_each_snippet_before_waiting_for_the_next(tmp_path, monkeypatch):
    """A 1 s snippet can be ~670 MB in memory; holding the previous one while
    the radio waits (up to a cooldown) for the next doubles peak memory."""
    refs = []
    alive_while_waiting = []

    def new_snippet():
        snippet = _snippet()
        refs.append(weakref.ref(snippet))
        return snippet

    @contextmanager
    def fake_open_session(settings, last_trigger_at, drops):
        def snippets():
            yield new_snippet()
            gc.collect()
            alive_while_waiting.append(refs[0]() is not None)
            raise KeyboardInterrupt

        yield snippets()

    def no_retry(seconds):  # raised inside run()'s except block, so it escapes
        raise AssertionError("run() retried; the fake session failed unexpectedly")

    monkeypatch.setattr(service, "_open_session", fake_open_session)
    monkeypatch.setattr(service, "RecordEmitter", _FakeEmitter)
    monkeypatch.setattr(service.time, "sleep", no_retry)
    with pytest.raises(KeyboardInterrupt):
        service.run(_settings(tmp_path, min_free_bytes=0), "/unused.sock", "s1", "op1")
    assert alive_while_waiting == [False]


def test_drain_does_not_hold_a_delivered_snippet_while_polling():
    snippets: queue.Queue = queue.Queue()
    snippets.put(_snippet())
    delivered = []
    alive_while_polling = []

    def monotonic():
        if delivered:
            gc.collect()
            alive_while_polling.append(delivered[0]() is not None)
            return 9.0  # past the stall threshold: ends the generator
        return 0.0

    drained = _drain(snippets, _FakeTap(), monotonic=monotonic)
    delivered.append(weakref.ref(next(drained)))
    with pytest.raises(RuntimeError, match="stalled"):
        next(drained)
    assert alive_while_polling == [False]
