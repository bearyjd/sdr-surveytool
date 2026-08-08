import threading
import time
from datetime import datetime, timezone

from capture.common.emitter import RecordEmitter
from ingest.queue_server import QueueServer
from schema.records import Identifier, Modality, Signal, UnifiedRecord


def _record(bssid: str) -> UnifiedRecord:
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=1.0,
        lon=2.0,
        survey_id="s",
        operator_id="o",
        modality=Modality.WIFI,
        identifier=Identifier(bssid=bssid),
        signal=Signal(rssi=-40.0),
    )


def test_queue_server_delivers_records_sent_by_emitter(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_record("AA:BB:CC:DD:EE:01"))
            emitter.emit(_record("AA:BB:CC:DD:EE:02"))

        first = server.get(timeout=2)
        second = server.get(timeout=2)
        assert first.identifier.bssid == "AA:BB:CC:DD:EE:01"
        assert second.identifier.bssid == "AA:BB:CC:DD:EE:02"
    finally:
        server.stop()


def test_queue_server_drops_invalid_json_without_crashing(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        import socket as socket_module

        sock = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        sock.connect(socket_path)
        sock.sendall(b"not valid json\n")
        sock.close()

        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_record("AA:BB:CC:DD:EE:03"))

        record = server.get(timeout=2)
        assert record.identifier.bssid == "AA:BB:CC:DD:EE:03"
    finally:
        server.stop()


def test_queue_server_restart_clears_stop_flag(tmp_path):
    """Regression test for C1: start() must clear _stop so restart works."""
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)

    # First cycle
    server.start()
    with RecordEmitter(socket_path) as emitter:
        emitter.emit(_record("AA:BB:CC:DD:EE:11"))
    record = server.get(timeout=2)
    assert record.identifier.bssid == "AA:BB:CC:DD:EE:11"
    server.stop()

    # Second cycle — should work if _stop is cleared in start()
    server.start()
    with RecordEmitter(socket_path) as emitter:
        emitter.emit(_record("AA:BB:CC:DD:EE:12"))
    record = server.get(timeout=2)
    assert record.identifier.bssid == "AA:BB:CC:DD:EE:12"
    server.stop()


def test_emitter_concurrent_emits_are_atomic(tmp_path):
    """Regression test for C2: concurrent emit() calls must not interleave."""
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        emitter = RecordEmitter(socket_path)
        emitter.connect()

        # Emit from multiple threads concurrently
        def emit_record(bssid: str):
            emitter.emit(_record(bssid))

        threads = []
        for i in range(5):
            t = threading.Thread(target=emit_record, args=(f"AA:BB:CC:DD:EE:{i:02X}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        emitter.close()

        # Collect all records
        records = []
        for _ in range(5):
            records.append(server.get(timeout=2))

        # Verify all records were delivered and distinct
        bssids = sorted([r.identifier.bssid for r in records])
        expected = sorted([f"AA:BB:CC:DD:EE:{i:02X}" for i in range(5)])
        assert bssids == expected
    finally:
        server.stop()


def test_queue_server_stop_closes_handler_threads_promptly(tmp_path):
    """Regression test for I1: stop() must close handler threads with timeout."""
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        import socket as socket_module

        # Connect and keep connection open
        sock = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        sock.connect(socket_path)

        # Measure time to stop
        start = time.time()
        server.stop()
        elapsed = time.time() - start

        # stop() should not hang; should complete in < 5 seconds (well below
        # any timeout that would occur without proper join())
        assert elapsed < 5.0, f"stop() took {elapsed}s, suggests handler thread not joined"

        sock.close()
    except Exception:
        server.stop()
        raise
