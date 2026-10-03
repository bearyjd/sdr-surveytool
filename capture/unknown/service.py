# capture/unknown/service.py
from __future__ import annotations

import argparse
import logging
import math
import queue
import shutil
import signal
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from capture.common.emitter import RecordEmitter
from capture.unknown.energy_trigger import TriggerConfig, TriggerEvent, record_trigger
from capture.unknown.normalizer import SnippetCaptureEvent, normalize_snippet_event
from capture.unknown.sample_clock import SampleClock
from capture.unknown.snippet_assembler import CapturedSnippet, SnippetAssembler
from capture.unknown.snippet_writer import write_sigmf_snippet
from dsp.spectral import mean_burst_power_dbfs, occupied_bandwidth_hz, peak_power_dbfs
from schema.records import UnifiedRecord
from storage.snippet_store import ensure_private_dir

if TYPE_CHECKING:
    from gnuradio import gr

logger = logging.getLogger(__name__)

# On consecutive failures to open the radio the sleep doubles from 1 s up to
# this ceiling, matching capture/wifi and capture/bluetooth.
_INITIAL_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0
_POLL_SECONDS = 0.5
# Completed snippets cross from the GNU Radio scheduler thread to the service
# thread through a queue this small: a stalled consumer (slow disk, wedged
# ingest socket) must not pile up ~670 MB snippets in memory.
_SNIPPET_QUEUE_MAXSIZE = 2
# Summaries of snippets dropped for a full queue (see _DroppedDetection).
_DROPPED_DETECTIONS_MAXLEN = 64
# The Python tap competes for the GIL, so give the SDR source ~100 ms of
# output buffer: a briefly starved tap then catches up instead of the SDR
# overflowing. Verified on GNU Radio 3.10.12 up to 5.6M items (100 ms at
# 56 MS/s, 42.7 MiB of complex64).
_SOURCE_BUFFER_SECONDS = 0.1
_MIN_CLOCK_DRIFT_SECONDS = 0.5
_MAX_SAMPLE_RATE = 61.44e6  # AD9361 (bladeRF xA9) maximum
_MAX_WINDOW_SECONDS = 5.0  # cap on each of the averaging, pre- and post-trigger windows
_FLOAT_SETTINGS = (
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
)
# A session that ends in a drift rebuild sooner than this after opening
# doesn't reset the backoff (see run()).
_QUICK_DRIFT_SECONDS = 60.0


@dataclass(frozen=True)
class CaptureSettings:
    center_freq_hz: float
    sample_rate: float
    noise_floor_dbfs: float  # operator-measured; no hardware-validated default exists
    staging_dir: Path
    threshold_db: float = 10.0  # trigger margin above the noise floor
    averaging_seconds: float = 0.001
    pre_trigger_seconds: float = 0.1
    post_trigger_seconds: float = 0.9  # snippet = pre + post = 1.0 s
    cooldown_seconds: float = 30.0
    gain_db: float = 30.0
    device: str = "driver=bladerf"
    device_args: str = ""
    # Staging, the snippet store and the database share one disk; below this
    # much free space snippets are dropped instead of written.
    min_free_bytes: int = 2 * 1024**3
    # No new samples for this long means the SDR stream is dead (unplugged,
    # wedged driver, or its source returned WORK_DONE): rebuild the session.
    stall_seconds: float = 5.0
    # Sample time is anchor + index / rate. Dropped samples (SDR overflow)
    # make it lag wall time, and NTP/GPS steps move wall time; past this
    # divergence the session is rebuilt, which re-anchors it.
    max_clock_drift_seconds: float = 2.0
    # Fail fast at startup if snippet buffers could outgrow this (see
    # estimated_peak_memory_bytes); 2 GiB suits an 8 GB Jetson Orin Nano.
    max_snippet_memory_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        # First: NaN passes every ordering check below (comparisons are False).
        for name in _FLOAT_SETTINGS:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
        if not 0 < self.sample_rate <= _MAX_SAMPLE_RATE:
            raise ValueError("sample_rate must be in (0, 61.44e6] samples/s (the AD9361 maximum)")
        if self.center_freq_hz <= 0:
            raise ValueError("center_freq_hz must be positive")
        for name in ("averaging_seconds", "pre_trigger_seconds", "post_trigger_seconds"):
            if getattr(self, name) > _MAX_WINDOW_SECONDS:
                raise ValueError(f"{name} must be <= {_MAX_WINDOW_SECONDS:g} s")
        if self.threshold_dbfs >= 0.0:
            raise ValueError(
                f"Trigger threshold {self.threshold_dbfs} dBFS is at or above full "
                "scale (0 dBFS) and could never fire; lower the noise floor or margin"
            )
        if self.pre_trigger_seconds < 0:
            raise ValueError("pre_trigger_seconds must be >= 0")
        min_cooldown = max(1.0, self.pre_trigger_seconds + self.post_trigger_seconds)
        if self.cooldown_seconds < min_cooldown:
            raise ValueError(
                f"cooldown_seconds must be >= {min_cooldown:g} (1 s, and at least one "
                "snippet): the trigger is level-triggered, so a shorter cooldown "
                "re-captures a continuous emitter back to back"
            )
        if self.min_free_bytes < 0:
            raise ValueError("min_free_bytes must be >= 0")
        if self.stall_seconds <= 0:
            raise ValueError("stall_seconds must be > 0")
        if self.max_clock_drift_seconds < _MIN_CLOCK_DRIFT_SECONDS:
            raise ValueError(
                f"max_clock_drift_seconds must be >= {_MIN_CLOCK_DRIFT_SECONDS}: sample "
                "time normally lags wall time by ~100 ms of buffering, and a tighter "
                "threshold would rebuild the radio session on every poll"
            )
        if self.samples(self.averaging_seconds) < 1 or self.samples(self.post_trigger_seconds) < 1:
            raise ValueError(
                "averaging_seconds and post_trigger_seconds must each span at least one sample"
            )
        peak = self.estimated_peak_memory_bytes
        if peak > self.max_snippet_memory_bytes:
            raise ValueError(
                f"Estimated peak snippet memory is {peak:,} bytes, over "
                f"max_snippet_memory_bytes {self.max_snippet_memory_bytes:,}; lower the "
                "sample rate or the pre/post windows, or raise --max-snippet-memory-bytes"
            )

    @property
    def estimated_peak_memory_bytes(self) -> int:
        """Worst-case snippet memory, from the actual buffers: 12 B/sample
        (cf32 iq + float32 power) for each queued snippet, the one being
        processed and the one being assembled; the pre-trigger history plus
        its concatenated copy when a capture begins; and up to 10 B/sample of
        measurement transients (a boolean mask and the above-threshold copy,
        on the service thread and, for queue-full summaries, the GR thread)."""
        pre = self.samples(self.pre_trigger_seconds)
        snippet = pre + self.samples(self.post_trigger_seconds)
        return 12 * snippet * (_SNIPPET_QUEUE_MAXSIZE + 2) + 2 * 12 * pre + 10 * snippet

    @property
    def threshold_dbfs(self) -> float:
        return self.noise_floor_dbfs + self.threshold_db

    def samples(self, seconds: float) -> int:
        return round(seconds * self.sample_rate)


@dataclass(frozen=True)
class _DroppedDetection:
    """What survives of a snippet whose IQ was dropped because the writer fell
    behind: enough for a snippet-less record, computed on the GNU Radio
    thread (measured ~13 ms per drop at 20 MS/s, ~46 ms at 56 MS/s, inside
    the 100 ms source buffer). No bandwidth estimate: that needs the IQ."""

    trigger: TriggerEvent
    sample_rate: float
    peak_power_dbfs: float
    mean_power_dbfs: float
    duration_ms: int


def process_snippet(
    snippet: CapturedSnippet,
    settings: CaptureSettings,
    survey_id: str,
    operator_id: str,
) -> UnifiedRecord:
    """Stage `snippet` as SigMF and build its UnifiedRecord. All times are
    sample-derived (see SampleClock) at the snippet's own sample rate, which
    is the rate read back from the SDR, not necessarily the requested one.
    The duration is what was actually captured, which is shorter than
    pre + post when the trigger came within pre_trigger_seconds of the
    stream starting. Every measurement and the normalization run before
    anything is written, and the writer removes its own files on failure, so
    a failure here never leaves a staged file behind."""
    measured = _snippet_record(snippet, settings, None, survey_id, operator_id)
    return _with_snippet_path(measured, _write_snippet(snippet, settings))


def _write_snippet(snippet: CapturedSnippet, settings: CaptureSettings) -> str:
    data_path = write_sigmf_snippet(
        snippet.iq,
        settings.staging_dir,
        snippet.sample_rate,
        settings.center_freq_hz,
        snippet.start_time,
    )
    return str(data_path)


def _with_snippet_path(record: UnifiedRecord, snippet_path: str) -> UnifiedRecord:
    metadata = record.metadata.model_copy(update={"iq_snippet_path": snippet_path})
    return record.model_copy(update={"metadata": metadata})


def _snippet_record(
    snippet: CapturedSnippet,
    settings: CaptureSettings,
    snippet_path: str | None,
    survey_id: str,
    operator_id: str,
) -> UnifiedRecord:
    event = SnippetCaptureEvent(
        timestamp=snippet.trigger.time,
        center_freq_hz=settings.center_freq_hz,
        sample_rate=snippet.sample_rate,
        bandwidth_estimate_hz=occupied_bandwidth_hz(
            snippet.iq, snippet.sample_rate, settings.threshold_dbfs
        ),
        peak_power_dbfs=peak_power_dbfs(snippet.power),
        mean_power_dbfs=mean_burst_power_dbfs(snippet.power, settings.threshold_dbfs),
        noise_floor_dbfs=settings.noise_floor_dbfs,
        snippet_path=snippet_path,
        snippet_duration_ms=round(len(snippet.iq) * 1000 / snippet.sample_rate),
    )
    return normalize_snippet_event(event, survey_id, operator_id)


def run(
    settings: CaptureSettings, socket_path: str, survey_id: str, operator_id: str
) -> None:
    """Capture forever: one radio session at a time, rebuilt on failure.

    A failing session (no SDR attached, driver error, stalled stream, tap
    failure) is logged and retried rather than killing the capture process:
    field surveys must survive transient faults unattended. Cooldown state
    is kept here, across sessions, as absolute trigger times.
    """
    # The staging dir is the shared contract with ingest's snippet store, so
    # it is checked with the store's own rule (owner-only, this uid).
    staging_dir = ensure_private_dir(settings.staging_dir)
    settings = replace(settings, staging_dir=staging_dir)
    logger.info("Staging unknown-signal snippets in %s", staging_dir)
    backoff = _INITIAL_BACKOFF_SECONDS
    escalate = False  # the previous session drifted soon after opening
    last_trigger_at: Mapping[float, datetime] = {}
    drops: Counter[str] = Counter()  # dropped snippets by cause, for the logs
    with RecordEmitter(socket_path) as emitter:
        while True:
            opened_at: float | None = None
            try:
                with _open_session(settings, last_trigger_at, drops) as snippets:
                    opened_at = time.monotonic()
                    if not escalate:
                        backoff = _INITIAL_BACKOFF_SECONDS  # the radio opened
                    for item in snippets:
                        last_trigger_at = record_trigger(last_trigger_at, item.trigger)
                        if isinstance(item, _DroppedDetection):
                            _emit_without_snippet(
                                _dropped_detection_record(item, settings, survey_id, operator_id),
                                emitter,
                                "queue_full",
                                item.trigger.sample_index,
                            )
                        else:
                            _stage_and_emit(item, settings, emitter, survey_id, operator_id, drops)
                        # Up to ~670 MB: don't hold it while waiting for the next.
                        del item
            except Exception as failure:
                lasted = None if opened_at is None else time.monotonic() - opened_at
                if lasted is not None and lasted >= _QUICK_DRIFT_SECONDS:
                    # A long healthy session clears any earlier escalation.
                    backoff = _INITIAL_BACKOFF_SECONDS
                # Sustained overflow makes every new session drift within
                # seconds: count a quick drift rebuild like a failure to open,
                # so the backoff escalates instead of rebuilding in a loop.
                escalate = (
                    isinstance(failure, _ClockDrift)
                    and lasted is not None
                    and lasted < _QUICK_DRIFT_SECONDS
                )
                logger.exception(
                    "Unknown-signal capture session failed; retrying in %.1fs", backoff
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)


def _stage_and_emit(
    snippet: CapturedSnippet,
    settings: CaptureSettings,
    emitter: RecordEmitter,
    survey_id: str,
    operator_id: str,
    drops: Counter[str],
) -> None:
    """A full disk, a failed write or a dead ingest socket loses this one
    snippet, not the radio session. The snippet is measured first; whenever
    the IQ then can't be kept (low disk, staging unavailable, write failure)
    the measured detection is still emitted, flagged. A snippet whose record
    could not be emitted has its staged pair deleted: nothing would adopt it."""
    sample_index = snippet.trigger.sample_index
    try:
        measured = _snippet_record(snippet, settings, None, survey_id, operator_id)
    except Exception:
        drops["processing_error"] += 1
        logger.exception("Failed to measure snippet triggered at sample %d; dropping it", sample_index)
        return
    reason = _reason_not_to_write(snippet, settings, drops)
    if reason is not None:
        _emit_without_snippet(measured, emitter, reason, sample_index)
        return
    try:
        record = _with_snippet_path(measured, _write_snippet(snippet, settings))
    except Exception:
        drops["processing_error"] += 1
        logger.exception(
            "Failed to write snippet triggered at sample %d; keeping its detection", sample_index
        )
        _emit_without_snippet(measured, emitter, "processing_error", sample_index)
        return
    try:
        emitter.emit(record)
    except Exception:
        drops["emit_failed"] += 1
        _discard_staged(record.metadata.iq_snippet_path)
        logger.exception(
            "Failed to emit snippet triggered at sample %d; dropping it and "
            "deleting its staged files",
            sample_index,
        )


def _reason_not_to_write(
    snippet: CapturedSnippet, settings: CaptureSettings, drops: Counter[str]
) -> str | None:
    """Why the IQ must not be written right now, if anything (logged and counted)."""
    try:
        free = shutil.disk_usage(settings.staging_dir).free
    except OSError:
        drops["staging_unavailable"] += 1
        logger.exception(
            "Cannot check free space in staging dir %s; dropping the snippet "
            "triggered at sample %d but keeping its detection",
            settings.staging_dir,
            snippet.trigger.sample_index,
        )
        return "staging_unavailable"
    if free - snippet.iq.nbytes < settings.min_free_bytes:
        drops["low_disk"] += 1
        logger.warning(
            "Only %d bytes free in %s (floor %d); dropping snippet triggered at "
            "sample %d (%d dropped for low disk so far)",
            free,
            settings.staging_dir,
            settings.min_free_bytes,
            snippet.trigger.sample_index,
            drops["low_disk"],
        )
        return "low_disk"
    return None


def _emit_without_snippet(
    measured: UnifiedRecord, emitter: RecordEmitter, reason: str, sample_index: int
) -> None:
    """The detection and its measurements survive when the IQ can't be kept:
    emit the record with no snippet path, flagged, like ingest does for a
    rejected snippet."""
    flags = {**measured.metadata.quality_flags, "snippet_dropped": reason}
    metadata = measured.metadata.model_copy(update={"quality_flags": flags})
    try:
        emitter.emit(measured.model_copy(update={"metadata": metadata}))
    except Exception:
        logger.exception(
            "Failed to emit the snippet-less record triggered at sample %d", sample_index
        )


def _dropped_detection_record(
    item: _DroppedDetection, settings: CaptureSettings, survey_id: str, operator_id: str
) -> UnifiedRecord:
    event = SnippetCaptureEvent(
        timestamp=item.trigger.time,
        center_freq_hz=settings.center_freq_hz,
        sample_rate=item.sample_rate,
        bandwidth_estimate_hz=None,
        peak_power_dbfs=item.peak_power_dbfs,
        mean_power_dbfs=item.mean_power_dbfs,
        noise_floor_dbfs=settings.noise_floor_dbfs,
        snippet_path=None,
        snippet_duration_ms=item.duration_ms,
    )
    return normalize_snippet_event(event, survey_id, operator_id)


def _discard_staged(data_path: str | None) -> None:
    if data_path is None:
        return
    data = Path(data_path)
    for staged in (data, data.with_suffix(".sigmf-meta")):
        staged.unlink(missing_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_to_anchor(
    last_trigger_at: Mapping[float, datetime], anchor: datetime
) -> dict[float, datetime]:
    """After a backward wall-clock step (NTP/GPS), a trigger time stored by an
    earlier session can lie in the new session's future, which would
    suppress detection for cooldown + the step. Clamp those to the anchor."""
    future = [when for when in last_trigger_at.values() if when > anchor]
    if future:
        logger.warning(
            "Wall clock stepped back: clamping %d cooldown(s) up to %s down to the new "
            "anchor %s",
            len(future),
            max(future).isoformat(),
            anchor.isoformat(),
        )
    return {freq: min(when, anchor) for freq, when in last_trigger_at.items()}


def _offer_or_drop(
    snippets: queue.Queue,
    drops: Counter[str],
    dropped: deque[_DroppedDetection],
    threshold_dbfs: float,
) -> Callable[[CapturedSnippet], None]:
    """The tap's on_snippet callback. It runs on the GNU Radio scheduler
    thread, which must never block (the SDR would overflow), so a full queue
    drops the snippet's IQ instead of waiting, keeping only a cheap summary
    of the detection on `dropped` for the service thread to emit."""

    def offer(snippet: CapturedSnippet) -> None:
        try:
            snippets.put_nowait(snippet)
        except queue.Full:
            drops["queue_full"] += 1
            if len(dropped) == dropped.maxlen:
                drops["queue_full_summary_lost"] += 1
            dropped.append(
                _DroppedDetection(
                    trigger=snippet.trigger,
                    sample_rate=snippet.sample_rate,
                    peak_power_dbfs=peak_power_dbfs(snippet.power),
                    mean_power_dbfs=mean_burst_power_dbfs(snippet.power, threshold_dbfs),
                    duration_ms=round(len(snippet.iq) * 1000 / snippet.sample_rate),
                )
            )
            logger.warning(
                "Snippet queue full; dropping snippet triggered at sample %d "
                "(%d dropped for a full queue so far)",
                snippet.trigger.sample_index,
                drops["queue_full"],
            )

    return offer


@contextmanager
def _open_session(
    settings: CaptureSettings,
    last_trigger_at: Mapping[float, datetime],
    drops: Counter[str],
    wall_clock: Callable[[], datetime] = _utc_now,
) -> Iterator[Iterator[CapturedSnippet | _DroppedDetection]]:
    """Open the SDR, start the flowgraph, and yield an iterator of completed
    snippets; always stops the flowgraph on exit. Needs GNU Radio; its tests
    run it against the real scheduler with only _build_soapy_source replaced
    (see tests/capture/unknown/test_session.py)."""
    # Deferred: GNU Radio is a system (dnf) package, not a pip dependency,
    # and the rest of this module must import without it.
    from capture.unknown.flowgraph import build_flowgraph

    # Open the device first: opening and tuning a bladeRF can take seconds, and
    # the anchor must be read as close as possible to sample 0, i.e. right
    # before start(), or every timestamp would be early by the open time.
    source, actual_rate = _build_soapy_source(settings)
    if actual_rate != settings.sample_rate:
        logger.warning(
            "SDR runs at %.9g samples/s, not the requested %.9g; timing uses the actual rate",
            actual_rate,
            settings.sample_rate,
        )
        # Every sample count and the clock follow the rate the samples
        # actually arrive at, or sample time drifts from wall time.
        settings = replace(settings, sample_rate=actual_rate)
    source.set_min_output_buffer(settings.samples(_SOURCE_BUFFER_SECONDS))
    clock = SampleClock(anchor=wall_clock(), sample_rate=actual_rate)
    last_trigger_at = _clamp_to_anchor(last_trigger_at, clock.anchor)
    assembler = SnippetAssembler(
        clock=clock,
        center_freq_hz=settings.center_freq_hz,
        config=TriggerConfig(
            threshold_dbfs=settings.threshold_dbfs,
            cooldown=timedelta(seconds=settings.cooldown_seconds),
        ),
        pre_trigger_samples=settings.samples(settings.pre_trigger_seconds),
        post_trigger_samples=settings.samples(settings.post_trigger_seconds),
        last_trigger_at=last_trigger_at,
    )
    snippets: queue.Queue[CapturedSnippet] = queue.Queue(maxsize=_SNIPPET_QUEUE_MAXSIZE)
    dropped: deque[_DroppedDetection] = deque(maxlen=_DROPPED_DETECTIONS_MAXLEN)
    flowgraph = build_flowgraph(
        source,
        settings.samples(settings.averaging_seconds),
        assembler,
        _offer_or_drop(snippets, drops, dropped, settings.threshold_dbfs),
    )
    try:
        # Inside the try: a start() that fails part-way still gets torn down.
        flowgraph.top_block.start()
        yield _drain(
            snippets,
            flowgraph.tap,
            dropped=dropped,
            clock=clock,
            stall_seconds=settings.stall_seconds,
            max_drift_seconds=settings.max_clock_drift_seconds,
            wall_clock=wall_clock,
        )
    finally:
        flowgraph.top_block.stop()
        flowgraph.top_block.wait()


class _ClockDrift(RuntimeError):
    """Sample time diverged from wall time; the session must be rebuilt."""


class _TapHealth(Protocol):
    """What _drain polls on the flowgraph's SnippetTap."""

    samples_seen: int
    error: Exception | None


def _drain(
    snippets: queue.Queue,
    tap: _TapHealth,
    dropped: deque[_DroppedDetection],
    clock: SampleClock,
    stall_seconds: float,
    max_drift_seconds: float,
    poll_seconds: float = _POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] = _utc_now,
) -> Iterator[CapturedSnippet | _DroppedDetection]:
    """Yield snippets as the flowgraph completes them, plus summaries of any
    the tap had to drop (see _offer_or_drop); raise once the flowgraph dies.

    `tap` is the flowgraph's SnippetTap. A GNU Radio flowgraph fails silently from Python's point of
    view, so health is polled: a tap error is re-raised, samples_seen
    frozen for stall_seconds means the SDR stopped streaming, and sample time
    (clock.time_at(samples_seen)) more than max_drift_seconds from wall time
    means samples were dropped or the wall clock stepped. The monotonic and
    wall clocks here are health checks only; record timing never reads them.
    Snippets the flowgraph completed before it died are still delivered
    before the failure is raised.
    """
    last_seen = tap.samples_seen
    last_progress = monotonic()
    while True:
        try:
            # Yielded without binding a local, so this generator doesn't keep
            # the last snippet alive while it polls for the next one.
            yield snippets.get(timeout=poll_seconds)
        except queue.Empty:
            pass
        yield from _pop_all(dropped)
        failure = None
        drift_failure = None
        if tap.error is not None:
            failure = "Snippet tap failed; flowgraph stopped"
        else:
            now = monotonic()
            if tap.samples_seen != last_seen:
                last_seen, last_progress = tap.samples_seen, now
                # Judged only on progress: with no samples arriving, sample
                # time stands still and the stall check owns that case.
                drift = (wall_clock() - clock.time_at(last_seen)).total_seconds()
                if abs(drift) > max_drift_seconds:
                    drift_failure = (
                        f"Sample clock is {drift:+.1f}s off the wall clock (dropped "
                        "samples or a clock step); rebuilding to re-anchor"
                    )
            elif now - last_progress > stall_seconds:
                failure = f"SDR stream stalled: no samples for {stall_seconds:g}s"
        if failure is not None:
            yield from _take_all(snippets)
            yield from _pop_all(dropped)
            raise RuntimeError(failure) from tap.error
        if drift_failure is not None:
            yield from _take_all(snippets)
            yield from _pop_all(dropped)
            raise _ClockDrift(drift_failure)


def _pop_all(dropped: deque[_DroppedDetection]) -> Iterator[_DroppedDetection]:
    while True:
        try:
            yield dropped.popleft()
        except IndexError:
            return


def _take_all(snippets: queue.Queue) -> Iterator[CapturedSnippet]:
    while True:
        try:
            yield snippets.get_nowait()
        except queue.Empty:
            return


def _build_soapy_source(settings: CaptureSettings) -> tuple[gr.basic_block, float]:
    """gr-soapy source for the configured device, and the sample rate read
    back from it (drivers round unsupported rates). Never called by tests (no
    SDR hardware); API verified by introspection against GNU Radio 3.10.12 and
    its bundled soapy_bladerf_source.block.yml template. Without a device or
    driver module, soapy.source raises RuntimeError('SoapySDR::Device::make()
    no match'), which run() logs and retries with backoff."""
    from gnuradio import soapy

    # pyright can't see into GNU Radio's pybind11 modules (no stubs).
    source = soapy.source(  # pyright: ignore[reportAttributeAccessIssue]
        settings.device, "fc32", 1, settings.device_args, "", [""], [""]
    )
    source.set_sample_rate(0, settings.sample_rate)
    source.set_frequency(0, settings.center_freq_hz)
    # Manual gain: a dBFS trigger threshold is only meaningful at fixed gain.
    source.set_gain_mode(0, False)
    source.set_gain(0, settings.gain_db)
    return source, source.get_sample_rate(0)


def _parse_args(argv: list[str] | None = None) -> tuple[CaptureSettings, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Unknown-signal energy-triggered IQ capture -> ingest queue"
    )
    parser.add_argument("--socket-path", default="/tmp/sdr-ingest.sock")
    parser.add_argument("--survey-id", required=True)
    parser.add_argument("--operator-id", required=True)
    _add_radio_args(parser)
    _add_trigger_args(parser)
    _add_safety_args(parser)
    args = parser.parse_args(argv)
    try:
        settings = _settings_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return settings, args


def _add_radio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--center-freq", type=float, required=True, help="Hz")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=20e6,
        help="Samples/s (= instantaneous bandwidth). The bladeRF xA9 reaches "
        "~56e6, but the Python capture chain sustained only ~58e6 on a fast "
        "x86 desktop; profile the Jetson before raising this.",
    )
    parser.add_argument("--gain-db", type=float, default=30.0)
    parser.add_argument("--device", default="driver=bladerf")
    parser.add_argument("--device-args", default="")


def _add_trigger_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--noise-floor-dbfs",
        type=float,
        required=True,
        help="Measured noise floor in dBFS at this gain and frequency. Required: "
        "there is no hardware-validated default, and a wrong guess either "
        "triggers on noise every cooldown or never triggers at all.",
    )
    parser.add_argument("--threshold-db", type=float, default=10.0)
    parser.add_argument("--averaging-ms", type=float, default=1.0)
    parser.add_argument("--pre-trigger-s", type=float, default=0.1)
    parser.add_argument("--post-trigger-s", type=float, default=0.9)
    parser.add_argument("--cooldown-s", type=float, default=30.0)


def _add_safety_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--staging-dir",
        default="data/snippet-staging",
        help="Must match ingest's --snippet-staging-dir, on the same filesystem "
        "as its --snippet-store-dir, owned by the uid both services run as.",
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=2 * 1024**3,
        help="Drop snippets instead of writing them when the staging disk would "
        "fall below this many free bytes (staging, store and database share it).",
    )
    parser.add_argument(
        "--max-snippet-memory-bytes",
        type=int,
        default=2 * 1024**3,
        help="Refuse to start if snippet buffers could need more memory than this "
        "at the configured sample rate and windows.",
    )
    parser.add_argument(
        "--max-clock-drift-s",
        type=float,
        default=2.0,
        help="Rebuild the radio session (re-anchoring sample time) when sample "
        "time and wall time diverge by more than this.",
    )


def _settings_from_args(args: argparse.Namespace) -> CaptureSettings:
    return CaptureSettings(
        center_freq_hz=args.center_freq,
        sample_rate=args.sample_rate,
        noise_floor_dbfs=args.noise_floor_dbfs,
        staging_dir=Path(args.staging_dir),
        threshold_db=args.threshold_db,
        averaging_seconds=args.averaging_ms / 1000.0,
        pre_trigger_seconds=args.pre_trigger_s,
        post_trigger_seconds=args.post_trigger_s,
        cooldown_seconds=args.cooldown_s,
        gain_db=args.gain_db,
        device=args.device,
        device_args=args.device_args,
        min_free_bytes=args.min_free_bytes,
        max_clock_drift_seconds=args.max_clock_drift_s,
        max_snippet_memory_bytes=args.max_snippet_memory_bytes,
    )


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (`sdr-capture-unknown`)."""
    settings, args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    # systemd and docker stop with SIGTERM: take the Ctrl-C path, which
    # unwinds through the flowgraph's stop()/wait() and closes the emitter.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        run(
            settings=settings,
            socket_path=args.socket_path,
            survey_id=args.survey_id,
            operator_id=args.operator_id,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down unknown-signal capture")
