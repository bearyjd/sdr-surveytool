# capture/unknown/service.py
from __future__ import annotations

import argparse
import logging
import queue
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from capture.common.emitter import RecordEmitter
from capture.unknown.energy_trigger import TriggerConfig, record_trigger
from capture.unknown.normalizer import SnippetCaptureEvent, normalize_snippet_event
from capture.unknown.sample_clock import SampleClock
from capture.unknown.snippet_assembler import CapturedSnippet, SnippetAssembler
from capture.unknown.snippet_writer import write_sigmf_snippet
from dsp.spectral import mean_burst_power_dbfs, occupied_bandwidth_hz, peak_power_dbfs
from schema.records import UnifiedRecord
from storage.snippet_store import ensure_private_dir

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

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.center_freq_hz <= 0:
            raise ValueError("sample_rate and center_freq_hz must be positive")
        if self.threshold_dbfs >= 0.0:
            raise ValueError(
                f"Trigger threshold {self.threshold_dbfs} dBFS is at or above full "
                "scale (0 dBFS) and could never fire; lower the noise floor or margin"
            )
        if self.pre_trigger_seconds < 0:
            raise ValueError("pre_trigger_seconds must be >= 0")
        if self.cooldown_seconds <= 0:
            raise ValueError(
                "cooldown_seconds must be > 0: the trigger is level-triggered, so a "
                "zero cooldown re-captures a continuous emitter back to back"
            )
        if self.min_free_bytes < 0:
            raise ValueError("min_free_bytes must be >= 0")
        if self.stall_seconds <= 0 or self.max_clock_drift_seconds <= 0:
            raise ValueError("stall_seconds and max_clock_drift_seconds must be > 0")
        if self.samples(self.averaging_seconds) < 1 or self.samples(self.post_trigger_seconds) < 1:
            raise ValueError(
                "averaging_seconds and post_trigger_seconds must each span at least one sample"
            )

    @property
    def threshold_dbfs(self) -> float:
        return self.noise_floor_dbfs + self.threshold_db

    def samples(self, seconds: float) -> int:
        return round(seconds * self.sample_rate)


def process_snippet(
    snippet: CapturedSnippet,
    settings: CaptureSettings,
    survey_id: str,
    operator_id: str,
) -> UnifiedRecord:
    """Stage `snippet` as SigMF and build its UnifiedRecord. All times are
    sample-derived (see SampleClock); the duration is what was actually
    captured, which is shorter than pre + post when the trigger came within
    pre_trigger_seconds of the stream starting."""
    data_path = write_sigmf_snippet(
        snippet.iq,
        settings.staging_dir,
        settings.sample_rate,
        settings.center_freq_hz,
        snippet.start_time,
    )
    event = SnippetCaptureEvent(
        timestamp=snippet.trigger.time,
        center_freq_hz=settings.center_freq_hz,
        sample_rate=settings.sample_rate,
        bandwidth_estimate_hz=occupied_bandwidth_hz(
            snippet.iq, settings.sample_rate, settings.threshold_dbfs
        ),
        peak_power_dbfs=peak_power_dbfs(snippet.power),
        mean_power_dbfs=mean_burst_power_dbfs(snippet.power, settings.threshold_dbfs),
        noise_floor_dbfs=settings.noise_floor_dbfs,
        snippet_path=str(data_path),
        snippet_duration_ms=round(len(snippet.iq) * 1000 / settings.sample_rate),
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
    last_trigger_at: Mapping[float, datetime] = {}
    drops: Counter[str] = Counter()  # dropped snippets by cause, for the logs
    with RecordEmitter(socket_path) as emitter:
        while True:
            try:
                with _open_session(settings, last_trigger_at, drops) as snippets:
                    backoff = _INITIAL_BACKOFF_SECONDS  # the radio opened
                    for snippet in snippets:
                        last_trigger_at = record_trigger(last_trigger_at, snippet.trigger)
                        _stage_and_emit(snippet, settings, emitter, survey_id, operator_id, drops)
            except Exception:
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
    """A full disk or a dead ingest socket loses this one snippet, not the
    radio session. A snippet whose record could not be emitted has its staged
    pair deleted: nothing would ever adopt it."""
    free = shutil.disk_usage(settings.staging_dir).free
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
        return
    try:
        record = process_snippet(snippet, settings, survey_id, operator_id)
    except Exception:
        drops["stage_failed"] += 1
        logger.exception(
            "Failed to stage snippet triggered at sample %d; dropping it",
            snippet.trigger.sample_index,
        )
        return
    try:
        emitter.emit(record)
    except Exception:
        drops["emit_failed"] += 1
        _discard_staged(record.metadata.iq_snippet_path)
        logger.exception(
            "Failed to emit snippet triggered at sample %d; dropping it and "
            "deleting its staged files",
            snippet.trigger.sample_index,
        )


def _discard_staged(data_path: str | None) -> None:
    if data_path is None:
        return
    data = Path(data_path)
    for staged in (data, data.with_suffix(".sigmf-meta")):
        staged.unlink(missing_ok=True)


def _offer_or_drop(
    snippets: queue.Queue, drops: Counter[str]
) -> Callable[[CapturedSnippet], None]:
    """The tap's on_snippet callback. It runs on the GNU Radio scheduler
    thread, which must never block (the SDR would overflow), so a full queue
    drops the snippet and counts it instead of waiting."""

    def offer(snippet: CapturedSnippet) -> None:
        try:
            snippets.put_nowait(snippet)
        except queue.Full:
            drops["queue_full"] += 1
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
) -> Iterator[Iterator[CapturedSnippet]]:
    """Open the SDR, start the flowgraph, and yield an iterator of completed
    snippets; always stops the flowgraph on exit. Needs GNU Radio and a real
    SDR, so tests replace it (see tests/capture/unknown/test_service.py)."""
    # Deferred: GNU Radio is a system (dnf) package, not a pip dependency,
    # and the rest of this module must import without it.
    from capture.unknown.flowgraph import build_flowgraph

    # Open the device first: opening and tuning a bladeRF can take seconds, and
    # the anchor must be read as close as possible to sample 0, i.e. right
    # before start(), or every timestamp would be early by the open time.
    source = _build_soapy_source(settings)
    clock = SampleClock(anchor=datetime.now(timezone.utc), sample_rate=settings.sample_rate)
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
    flowgraph = build_flowgraph(
        source,
        settings.samples(settings.averaging_seconds),
        assembler,
        _offer_or_drop(snippets, drops),
    )
    try:
        # Inside the try: a start() that fails part-way still gets torn down.
        flowgraph.top_block.start()
        yield _drain(
            snippets,
            flowgraph.tap,
            clock=clock,
            stall_seconds=settings.stall_seconds,
            max_drift_seconds=settings.max_clock_drift_seconds,
        )
    finally:
        flowgraph.top_block.stop()
        flowgraph.top_block.wait()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _drain(
    snippets: queue.Queue,
    tap,
    clock: SampleClock,
    stall_seconds: float,
    max_drift_seconds: float,
    poll_seconds: float = _POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] = _utc_now,
) -> Iterator[CapturedSnippet]:
    """Yield snippets as the flowgraph completes them; raise once it dies.

    `tap` is the flowgraph's SnippetTap (anything with samples_seen and
    error). A GNU Radio flowgraph fails silently from Python's point of
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
            snippet = snippets.get(timeout=poll_seconds)
        except queue.Empty:
            pass
        else:
            yield snippet
        failure = None
        if tap.error is not None:
            failure = "Snippet tap failed; flowgraph stopped"
        else:
            now = monotonic()
            if tap.samples_seen != last_seen:
                last_seen, last_progress = tap.samples_seen, now
            elif now - last_progress > stall_seconds:
                failure = f"SDR stream stalled: no samples for {stall_seconds:g}s"
            if failure is None:
                drift = (wall_clock() - clock.time_at(tap.samples_seen)).total_seconds()
                if abs(drift) > max_drift_seconds:
                    failure = (
                        f"Sample clock is {drift:+.1f}s off the wall clock (dropped "
                        "samples or a clock step); rebuilding to re-anchor"
                    )
        if failure is not None:
            yield from _take_all(snippets)
            raise RuntimeError(failure) from tap.error


def _take_all(snippets: queue.Queue) -> Iterator[CapturedSnippet]:
    while True:
        try:
            yield snippets.get_nowait()
        except queue.Empty:
            return


def _build_soapy_source(settings: CaptureSettings):
    """gr-soapy source for the configured device. Never called by tests (no
    SDR hardware); API verified by introspection against GNU Radio 3.10.12 and
    its bundled soapy_bladerf_source.block.yml template. Without a device or
    driver module, soapy.source raises RuntimeError('SoapySDR::Device::make()
    no match'), which run() logs and retries with backoff."""
    from gnuradio import soapy

    source = soapy.source(settings.device, "fc32", 1, settings.device_args, "", [""], [""])
    source.set_sample_rate(0, settings.sample_rate)
    source.set_frequency(0, settings.center_freq_hz)
    # Manual gain: a dBFS trigger threshold is only meaningful at fixed gain.
    source.set_gain_mode(0, False)
    source.set_gain(0, settings.gain_db)
    return source


def _parse_args(argv: list[str] | None = None) -> tuple[CaptureSettings, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Unknown-signal energy-triggered IQ capture -> ingest queue"
    )
    parser.add_argument("--socket-path", default="/tmp/sdr-ingest.sock")
    parser.add_argument("--survey-id", required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--center-freq", type=float, required=True, help="Hz")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=20e6,
        help="Samples/s (= instantaneous bandwidth). The bladeRF xA9 reaches "
        "~56e6, but the Python capture chain sustained only ~58e6 on a fast "
        "x86 desktop; profile the Jetson before raising this.",
    )
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
    parser.add_argument("--gain-db", type=float, default=30.0)
    parser.add_argument("--device", default="driver=bladerf")
    parser.add_argument("--device-args", default="")
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=2 * 1024**3,
        help="Drop snippets instead of writing them when the staging disk would "
        "fall below this many free bytes (staging, store and database share it).",
    )
    parser.add_argument(
        "--max-clock-drift-s",
        type=float,
        default=2.0,
        help="Rebuild the radio session (re-anchoring sample time) when sample "
        "time and wall time diverge by more than this.",
    )
    parser.add_argument(
        "--staging-dir",
        default="data/snippet-staging",
        help="Must match ingest's --snippet-staging-dir.",
    )
    args = parser.parse_args(argv)
    try:
        settings = CaptureSettings(
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
        )
    except ValueError as exc:
        parser.error(str(exc))
    return settings, args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (`sdr-capture-unknown`)."""
    settings, args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run(
        settings=settings,
        socket_path=args.socket_path,
        survey_id=args.survey_id,
        operator_id=args.operator_id,
    )
