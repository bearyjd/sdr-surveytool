# tests/capture/unknown/test_service.py
import logging
import queue
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from capture.unknown import service
from capture.unknown.energy_trigger import TriggerEvent
from capture.unknown.service import CaptureSettings, process_snippet
from capture.unknown.snippet_assembler import CapturedSnippet
from schema.records import ClassificationStatus, Modality

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


def _snippet(trigger_index: int = 60_000, pre: int = 10_000, post: int = 90_000) -> CapturedSnippet:
    n = pre + post
    iq = np.full(n, 0.01, dtype=np.complex64)
    iq[pre : pre + 20_000] = 0.1  # -20 dBFS burst right at the trigger
    power = (np.abs(iq) ** 2).astype(np.float32)
    start = trigger_index - pre
    return CapturedSnippet(
        iq=iq,
        power=power,
        start_index=start,
        start_time=ANCHOR + timedelta(seconds=start / FS),
        trigger=TriggerEvent(
            sample_index=trigger_index,
            time=ANCHOR + timedelta(seconds=trigger_index / FS),
            center_freq_hz=915e6,
        ),
    )


def test_settings_reject_a_threshold_at_or_above_full_scale(tmp_path):
    with pytest.raises(ValueError, match="full scale"):
        _settings(tmp_path, noise_floor_dbfs=-5.0, threshold_db=10.0)


def test_settings_reject_non_positive_cooldown(tmp_path):
    """A zero cooldown with a level trigger re-captures a continuous emitter
    back to back, i.e. writes to disk as fast as the radio produces samples."""
    with pytest.raises(ValueError, match="cooldown_seconds"):
        _settings(tmp_path, cooldown_seconds=0.0)


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


def test_snippet_duration_reflects_samples_actually_captured(tmp_path):
    """A trigger in the first 0.1 s truncates the pre-trigger history."""
    record = process_snippet(_snippet(trigger_index=3_000, pre=3_000), _settings(tmp_path), "s1", "op1")
    assert record.metadata.snippet_duration_ms == 930


class _FakeTap:
    def __init__(self) -> None:
        self.samples_seen = 0
        self.error: Exception | None = None


def test_drain_yields_snippets_then_raises_when_stream_stalls():
    snippets: queue.Queue = queue.Queue()
    snippets.put("snippet-1")
    tap = _FakeTap()
    times = iter([0.0, 1.0, 3.0, 9.0])
    drained = service._drain(snippets, tap, stall_seconds=5.0, poll_seconds=0.0, monotonic=lambda: next(times))

    assert next(drained) == "snippet-1"
    tap.samples_seen = 4096  # progress at t=1.0
    with pytest.raises(RuntimeError, match="stalled"):
        next(drained)  # t=3.0 still within 5 s of progress; t=9.0 is not


def test_drain_reraises_a_tap_failure():
    tap = _FakeTap()
    tap.error = ValueError("assembler bug")
    drained = service._drain(queue.Queue(), tap, stall_seconds=5.0, poll_seconds=0.0, monotonic=lambda: 0.0)
    with pytest.raises(RuntimeError, match="tap failed") as excinfo:
        next(drained)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_full_snippet_queue_drops_and_counts_without_blocking():
    """The tap calls this on the GNU Radio scheduler thread, which must never
    block (the SDR would overflow): a full queue drops the snippet instead."""
    snippets: queue.Queue = queue.Queue(maxsize=2)
    drops: Counter = Counter()
    offer = service._offer_or_drop(snippets, drops)
    for _ in range(3):
        offer(_snippet())
    assert snippets.qsize() == 2
    assert drops == Counter({"queue_full": 1})


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


def test_a_failed_emit_drops_only_that_snippet(tmp_path, caplog):
    class BrokenEmitter:
        def emit(self, record) -> None:
            raise OSError("ingest socket gone")

    with caplog.at_level(logging.ERROR):
        service._stage_and_emit(
            _snippet(), _settings(tmp_path, min_free_bytes=0), BrokenEmitter(), "s1", "op1", Counter()
        )
    assert "dropping it" in caplog.text


def test_snippet_is_dropped_before_writing_when_disk_is_nearly_full(tmp_path, monkeypatch, caplog):
    """Staging, the snippet store and the database share one disk, and a
    continuous emitter writes ~19 GB/h at the default cooldown: below the
    free-space floor the snippet is dropped, never written."""
    monkeypatch.setattr(service.shutil, "disk_usage", lambda path: SimpleNamespace(free=3 * 1024**3))
    emitter = _FakeEmitter("/unused.sock")
    drops: Counter = Counter()
    settings = _settings(tmp_path / "staging", min_free_bytes=3 * 1024**3)

    with caplog.at_level(logging.WARNING):
        service._stage_and_emit(_snippet(), settings, emitter, "s1", "op1", drops)

    assert emitter.records == []
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
    assert settings.staging_dir == Path("data/snippet-staging")
    assert settings.min_free_bytes == 2 * 1024**3
    assert args.socket_path == "/tmp/sdr-ingest.sock"


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
