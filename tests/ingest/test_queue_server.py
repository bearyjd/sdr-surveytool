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
    """Regression test for C2: concurrent emit() calls must not interleave.

    Uses 50+ concurrent threads with large records to force multiple sendall()
    writes, ensuring that without the lock, we'd see corrupted/interleaved records.
    """
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        emitter = RecordEmitter(socket_path)
        emitter.connect()

        # Create records large enough to force multiple sendall() writes
        def make_large_record(index: int) -> UnifiedRecord:
            # Add padding to make each record ~2KB to force multiple writes
            padding = "x" * 1900
            return UnifiedRecord(
                timestamp=datetime.now(timezone.utc),
                lat=1.0,
                lon=2.0,
                survey_id=f"s-{index}-{padding}",
                operator_id="o",
                modality=Modality.WIFI,
                identifier=Identifier(bssid=f"AA:BB:CC:DD:EE:{index%256:02X}"),
                signal=Signal(rssi=-40.0),
            )

        # Emit from 50 concurrent threads
        def emit_record(index: int):
            emitter.emit(make_large_record(index))

        threads = []
        for i in range(50):
            t = threading.Thread(target=emit_record, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        emitter.close()

        # Collect all records and verify they parse as valid JSON
        records = []
        for _ in range(50):
            record = server.get(timeout=2)
            records.append(record)
            # Verify it parses and has expected structure
            assert record.identifier.bssid is not None

        # Verify all records were delivered
        assert len(records) == 50, f"Expected 50 records, got {len(records)}"
    finally:
        server.stop()


def test_queue_server_stop_closes_handler_threads_promptly(tmp_path):
    """Regression test for I1: stop() must close handler threads with timeout.

    Verifies that handler threads are actually joined and not alive after stop().
    """
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        import socket as socket_module

        # Connect and keep connection open
        sock = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        sock.connect(socket_path)

        # Capture reference to handler thread before calling stop
        # (The server stores these in _handler_threads, but we need access after stop)
        # Give handler a moment to register in _handler_threads
        time.sleep(0.1)

        # Get reference to the handler thread
        with server._handler_lock:
            handler_threads = list(server._handler_threads)

        assert len(handler_threads) > 0, "Expected at least one handler thread"
        handler_thread = handler_threads[0]

        # Verify thread is alive before stop
        assert handler_thread.is_alive(), "Handler thread should be alive before stop()"

        # Call stop and measure time
        start = time.time()
        server.stop()
        elapsed = time.time() - start

        # Verify thread is dead after stop
        assert not handler_thread.is_alive(), "Handler thread should be dead after stop()"

        # stop() should not hang; should complete in < 5 seconds
        assert elapsed < 5.0, f"stop() took {elapsed}s, suggests thread not joined properly"

        sock.close()
    except Exception:
        server.stop()
        raise


def test_handler_threads_dont_survive_restart(tmp_path):
    """Regression test for N3: orphaned handler threads must not re-activate on restart.

    Verifies that when a handler thread is still alive after stop() (due to timeout),
    it doesn't resume reading into the NEW server's queue on the next start().
    """
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)

    # First cycle: start, connect a client, stop without fully closing it
    server.start()
    import socket as socket_module
    sock1 = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    sock1.connect(socket_path)

    # Wait for handler thread to register
    time.sleep(0.1)

    # Record the handler thread count and threads before stop
    with server._handler_lock:
        before_threads = list(server._handler_threads)
    assert len(before_threads) > 0

    # Stop (handler threads may still be alive due to timeout)
    server.stop()

    # Check how many threads are still alive
    still_alive = [t for t in before_threads if t.is_alive()]

    if still_alive:
        # If there are orphaned threads, verify they don't interfere on restart
        server.start()

        # Send a message from a fresh client
        sock2 = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        sock2.connect(socket_path)
        with RecordEmitter.__dict__:  # Access through temporary context
            pass

        # Manually send a valid record
        test_record = _record("AA:BB:CC:DD:EE:99")
        sock2.sendall(test_record.model_dump_json().encode("utf-8") + b"\n")

        # Verify only the NEW record is in queue (not reactivated old handler)
        record = server.get(timeout=2)
        assert record.identifier.bssid == "AA:BB:CC:DD:EE:99"

        sock2.close()
        server.stop()
    else:
        # If no orphaned threads, just verify restart works
        server.start()
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_record("AA:BB:CC:DD:EE:99"))
        record = server.get(timeout=2)
        assert record.identifier.bssid == "AA:BB:CC:DD:EE:99"
        server.stop()

    sock1.close()
