# tests/capture/unknown/test_flowgraph_integration.py
"""End-to-end closure proof for unknown-signal capture, no SDR hardware:

synthetic IQ file -> GNU Radio file_source -> |x|^2 -> moving average ->
SnippetTap/SnippetAssembler -> SigMF staging -> UnifiedRecord ->
RecordEmitter -> QueueServer -> IngestService (adopts the snippet into the
LocalSnippetStore) -> SQLite.

Bursts at 0.5 s, 2.0 s and 4.0 s with a 3 s cooldown: the 2.0 s burst falls
inside the first trigger's cooldown and must NOT produce a record. Timing
is sample-derived, so the whole 5.5 s of signal runs in well under a second.
"""
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gnuradio")

from gnuradio import blocks, gr  # noqa: E402
from sigmf import sigmffile  # noqa: E402

from capture.common.emitter import RecordEmitter  # noqa: E402
from capture.unknown.energy_trigger import TriggerConfig  # noqa: E402
from capture.unknown.flowgraph import CaptureFlowgraph, build_flowgraph  # noqa: E402
from capture.unknown.sample_clock import SampleClock  # noqa: E402
from capture.unknown.service import CaptureSettings, process_snippet  # noqa: E402
from capture.unknown.snippet_assembler import SnippetAssembler  # noqa: E402
from ingest.gps_fix import GpsFix, StaticGpsFixProvider  # noqa: E402
from ingest.queue_server import QueueServer  # noqa: E402
from ingest.service import IngestService  # noqa: E402
from storage.db import init_db, make_engine, make_session_factory  # noqa: E402
from storage.models import SurveyRecord  # noqa: E402
from storage.snippet_store import LocalSnippetStore  # noqa: E402

FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
NOISE_POWER = 1e-4  # -40 dBFS
BURST_POWER = 1e-2  # -20 dBFS: 20 dB SNR
BURST_BANDWIDTH_HZ = 20_000.0
BURST_SECONDS = 0.2
BURST_STARTS_S = [0.5, 2.0, 4.0]
TOTAL_SECONDS = 5.5
FIX = GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)


def _synthetic_iq() -> np.ndarray:
    rng = np.random.default_rng(11)
    n = round(TOTAL_SECONDS * FS)
    scale = math.sqrt(NOISE_POWER / 2)
    iq = scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    burst_len = round(BURST_SECONDS * FS)
    freqs = np.fft.fftfreq(burst_len, 1 / FS)
    for start_s in BURST_STARTS_S:
        spectrum = np.fft.fft(rng.standard_normal(burst_len) + 1j * rng.standard_normal(burst_len))
        spectrum[np.abs(freqs - 10_000.0) > BURST_BANDWIDTH_HZ / 2] = 0
        burst = np.fft.ifft(spectrum)
        burst *= math.sqrt(BURST_POWER / np.mean(np.abs(burst) ** 2))
        start = round(start_s * FS)
        iq[start : start + burst_len] += burst
    return iq.astype(np.complex64)


def _run_bounded(flowgraph: CaptureFlowgraph, timeout: float = 30.0) -> None:
    waiter = threading.Thread(target=flowgraph.top_block.wait, daemon=True)
    flowgraph.top_block.start()
    waiter.start()
    waiter.join(timeout)
    if waiter.is_alive():
        flowgraph.top_block.stop()
        waiter.join(5)
        pytest.fail(f"flowgraph did not finish within {timeout}s")


def test_synthetic_bursts_become_stored_unknown_records(tmp_path):
    staging, store_root = tmp_path / "staging", tmp_path / "snippets"
    settings = CaptureSettings(
        center_freq_hz=915e6,
        sample_rate=FS,
        noise_floor_dbfs=-40.0,
        staging_dir=staging,
        threshold_db=10.0,
        averaging_seconds=0.001,
        pre_trigger_seconds=0.1,
        post_trigger_seconds=0.9,
        cooldown_seconds=3.0,
    )
    iq_file = tmp_path / "synthetic.cf32"
    _synthetic_iq().tofile(iq_file)

    # 1. Real GNU Radio scheduler over the synthetic file.
    assembler = SnippetAssembler(
        clock=SampleClock(anchor=ANCHOR, sample_rate=FS),
        center_freq_hz=settings.center_freq_hz,
        config=TriggerConfig(
            threshold_dbfs=settings.threshold_dbfs,
            cooldown=timedelta(seconds=settings.cooldown_seconds),
        ),
        pre_trigger_samples=settings.samples(settings.pre_trigger_seconds),
        post_trigger_samples=settings.samples(settings.post_trigger_seconds),
        last_trigger_at={},
    )
    snippets = []
    source = blocks.file_source(gr.sizeof_gr_complex, str(iq_file), False)
    flowgraph = build_flowgraph(source, settings.samples(settings.averaging_seconds), assembler, snippets.append)
    _run_bounded(flowgraph)
    assert flowgraph.tap.error is None
    assert [round(s.trigger.sample_index / FS, 1) for s in snippets] == [0.5, 4.0]

    # 2. Capture -> queue -> ingest -> storage.
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)
    service = IngestService(
        server,
        session_factory,
        StaticGpsFixProvider(FIX),
        snippet_store=LocalSnippetStore(staging, store_root),
    )
    try:
        with RecordEmitter(socket_path) as emitter:
            for snippet in snippets:
                emitter.emit(process_snippet(snippet, settings, "survey-1", "op-1"))
        processed = [service.process_one(timeout=2) for _ in snippets]
    finally:
        server.stop()

    # Capture never left anything behind in staging; ingest took ownership.
    assert list(staging.iterdir()) == []
    with session_factory() as session:
        rows = session.query(SurveyRecord).order_by(SurveyRecord.id).all()
    engine.dispose()  # close the pooled in-memory connection; no ResourceWarning at GC
    assert len(rows) == 2
    for row, record, start_s in zip(rows, processed, [0.5, 4.0]):
        assert row.modality == "unknown"
        # Sample-derived: within one 1 ms averaging window of the burst onset.
        offset = (record.timestamp - ANCHOR).total_seconds() - start_s
        assert 0.0 <= offset < 0.001
        assert row.identifier["center_freq"] == 915e6
        assert row.identifier["bandwidth_estimate"] == pytest.approx(BURST_BANDWIDTH_HZ, rel=0.15)
        assert row.signal["rssi"] == pytest.approx(-20.0, abs=1.0)
        assert row.signal["snr"] == pytest.approx(20.0, abs=1.0)
        assert -20.0 < row.signal["peak_power"] < -15.0
        assert row.metadata_["snippet_duration_ms"] == 1000
        assert row.metadata_["sample_rate"] == FS
        assert row.metadata_["classification_status"] == "unclassified"
        assert row.lat == FIX.lat and row.gps_fix_quality == FIX.fix_quality

        stored = Path(row.metadata_["iq_snippet_path"])
        assert stored.parent == store_root.resolve()
        recording = sigmffile.fromfile(str(stored))
        samples = recording.read_samples()
        assert len(samples) == round(1.0 * FS)
        assert recording.get_captures()[0]["core:frequency"] == 915e6
