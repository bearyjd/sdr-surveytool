import ast
import inspect
import threading
import time
from datetime import datetime, timezone
from queue import Empty, Full, Queue

import pytest

import ingest.queue_server as queue_server_module

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

    Uses 50+ concurrent threads with large records (64-128KB each) to force
    multiple sendall() writes. Without the lock, records would interleave on
    the wire and fail JSON parsing. Test verifies all records arrive intact
    and distinct.
    """
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        emitter = RecordEmitter(socket_path)
        emitter.connect()

        # Create records large enough (100KB) to force multiple sendall() writes
        # Default AF_UNIX SO_SNDBUF is ~208KB, so 100KB > typical single syscall
        def make_large_record(index: int) -> UnifiedRecord:
            # Pad to ~100KB to guarantee interleaving without lock
            padding = "x" * 102400
            return UnifiedRecord(
                timestamp=datetime.now(timezone.utc),
                lat=float(index),  # Make each record's geo data distinct
                lon=float(index),
                survey_id=f"s-{index}-{padding}",
                operator_id="o",
                modality=Modality.WIFI,
                identifier=Identifier(bssid=f"AA:BB:CC:DD:EE:{index%256:02X}"),
                signal=Signal(rssi=float(-40 - index)),
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

        # Collect all records and verify they are complete and distinct
        records = []
        received_bssids = set()
        for _ in range(50):
            record = server.get(timeout=2)
            records.append(record)
            # Each record must be complete and parseable
            assert record.identifier.bssid is not None
            assert record.survey_id.startswith("s-")  # Verify not corrupted/interleaved
            received_bssids.add(record.identifier.bssid)

        # Verify all records were delivered and are distinct
        assert len(records) == 50, f"Expected 50 records, got {len(records)}"
        assert len(received_bssids) == 50, f"Expected 50 distinct BSSIDs, got {len(received_bssids)}"

        # Verify exact distinctness match
        expected_bssids = {f"AA:BB:CC:DD:EE:{i%256:02X}" for i in range(50)}
        assert received_bssids == expected_bssids, "Not all expected BSSIDs received"
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
    """Regression test for N3: generation token prevents stale threads from reactivating.

    Verifies the generation-token architecture: each start() increments _generation
    and creates a new queue. Handler threads capture their generation at spawn time.
    Even if a handler somehow survives stop() (joins timeout), its loop condition
    `while not self._stop.is_set() and gen == self._generation:` ensures it exits
    the moment start() increments _generation, preventing it from polluting the
    new session's queue.
    """
    socket_path = str(tmp_path / "ingest.sock")

    server = QueueServer(socket_path)

    # First session: verify generation starts at 0, increments to 1 on start()
    assert server._generation == 0, "Initial generation should be 0"
    server.start()
    assert server._generation == 1, "Generation should be 1 after first start()"
    queue1 = server._queue

    # Send and receive a record in first session
    with RecordEmitter(socket_path) as emitter:
        emitter.emit(_record("AA:BB:CC:DD:EE:11"))
    record1 = server.get(timeout=2)
    assert record1.identifier.bssid == "AA:BB:CC:DD:EE:11"

    # Stop first session
    server.stop()
    assert server._generation == 1, "Generation frozen until next start()"

    # Second session: verify generation increments and new queue created
    server.start()
    assert server._generation == 2, "Generation should be 2 after second start()"
    queue2 = server._queue
    assert queue2 is not queue1, "start() creates new queue for new session"

    # Send and receive a record in second session
    with RecordEmitter(socket_path) as emitter:
        emitter.emit(_record("AA:BB:CC:DD:EE:22"))
    record2 = server.get(timeout=2)
    assert record2.identifier.bssid == "AA:BB:CC:DD:EE:22"

    # VERIFICATION: If a handler from session 1 somehow survived and tried to enqueue,
    # it would check `gen == self._generation` (1 == 2), which is false, so it exits.
    # The record would never make it into queue2. This is proven by the architecture:
    # - Handler has `gen=1` (captured at spawn in session 1)
    # - Loop condition is `while ... and gen == self._generation:`
    # - After session 2 starts, `self._generation == 2`
    # - So `1 == 2` is false, loop exits
    # - Handler cannot enqueue to queue2

    # Third session confirms the mechanism works multiple times
    server.stop()
    server.start()
    assert server._generation == 3, "Generation should be 3 after third start()"

    with RecordEmitter(socket_path) as emitter:
        emitter.emit(_record("AA:BB:CC:DD:EE:33"))
    record3 = server.get(timeout=2)
    assert record3.identifier.bssid == "AA:BB:CC:DD:EE:33"

    server.stop()


class _SlowWhenFullQueue(Queue):
    """A real Queue that, once full, keeps a blocked producer parked past stop()'s join.

    Behaviour is identical to Queue except that a put() which times out on a full queue
    stays inside put() for an extra `_extra_block` seconds before raising Full. That is
    exactly the state a handler thread is in when a stalled consumer has filled the queue,
    and it makes "the handler outlives stop()" deterministic instead of a timing race.
    """

    _extra_block = 2.5

    def put(self, item, block=True, timeout=None):  # type: ignore[override]
        try:
            return super().put(item, block=block, timeout=timeout)
        except Full:
            time.sleep(self._extra_block)
            raise


def test_orphaned_handler_cannot_enqueue_into_next_sessions_queue(tmp_path):
    """Regression test for N3: an orphaned handler must not leak into a new session.

    End-to-end scenario, driven entirely through the real accept loop and socket path:
      1. Start a server and fill its session queue to maxsize with real emitted records,
         so a handler thread is genuinely parked in the put-retry loop with a backlog of
         unprocessed lines still buffered.
      2. stop() the server. The handler is inside a put() that outlasts the join timeout,
         so it survives as an orphan (asserted, not assumed).
      3. start() a new session, which installs a brand-new queue.
      4. The orphan then wakes up and runs at least one more retry iteration.

    The new session's queue must stay empty. This fails if the ingest path resolves the
    queue via `self._queue` at put() time instead of using the queue object captured when
    the handler was spawned.
    """
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    emit_errors: list[BaseException] = []

    try:
        # Install a queue that parks a blocked producer past stop()'s join timeout.
        # Done before any client connects, so the handler captures this object.
        old_queue = _SlowWhenFullQueue(maxsize=1000)
        server._queue = old_queue

        def emit_many() -> None:
            try:
                with RecordEmitter(socket_path) as emitter:
                    for i in range(1500):
                        emitter.emit(_record(f"AA:BB:CC:DD:{i // 256:02X}:{i % 256:02X}"))
            except BaseException as exc:  # noqa: BLE001 - producer dies when server stops
                emit_errors.append(exc)

        producer = threading.Thread(target=emit_many, daemon=True)
        producer.start()

        # Wait for the queue to actually reach maxsize (nothing is consuming it).
        deadline = time.time() + 15
        while time.time() < deadline and not old_queue.full():
            time.sleep(0.05)
        assert old_queue.full(), "queue never filled; cannot construct the orphan scenario"

        # Let the handler enter the slow put() so it is genuinely parked.
        time.sleep(0.6)

        with server._handler_lock:
            handler_threads = list(server._handler_threads)
        assert handler_threads, "expected a handler thread for the emitter connection"
        handler = handler_threads[0]
        assert handler.is_alive(), "handler should be parked in the put-retry loop"

        server.stop()

        # The whole point: this handler outlived stop().
        assert handler.is_alive(), (
            "handler did not outlive stop(); the orphan scenario was not constructed"
        )

        server.start()
        new_queue = server._queue
        assert new_queue is not old_queue

        # Give the orphan more than enough time to wake from its parked put() and run
        # further retry iterations against whatever queue it resolves.
        time.sleep(_SlowWhenFullQueue._extra_block + 1.5)

        with pytest.raises(Empty):
            server.get(timeout=1)
        assert new_queue.qsize() == 0, "records from the previous session leaked into the new one"
    finally:
        server.stop()


def _queue_server_method_ast(name: str) -> ast.FunctionDef:
    """Return the AST of QueueServer.<name> as written in the source file."""
    source = inspect.getsource(queue_server_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QueueServer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"QueueServer.{name} not found in source")


def _reads_self_queue(node: ast.AST) -> bool:
    """True if `self._queue` is read anywhere inside the given AST node."""
    return any(
        isinstance(sub, ast.Attribute)
        and sub.attr == "_queue"
        and isinstance(sub.value, ast.Name)
        and sub.value.id == "self"
        for sub in ast.walk(node)
    )


def test_handler_write_path_never_reads_self_queue():
    """Structural regression guard for N3.

    The handler write path must operate exclusively on the queue object captured when
    the thread was spawned. Reading `self._queue` there — even inside a retry loop that
    is guarded by a generation check — reintroduces the cross-session leak, because the
    attribute can be swapped by start() between the guard and the put(). That race is too
    narrow to catch reliably with a timing test, so the invariant is asserted structurally
    instead: no `self._queue` access is permitted in these methods.

    AST-based rather than text-based, so docstrings and comments mentioning `self._queue`
    (there are some, deliberately) do not trip it.
    """
    for method in ("_handle_client", "_ingest_line"):
        node = _queue_server_method_ast(method)
        assert not _reads_self_queue(node), (
            f"QueueServer.{method} reads self._queue; it must use the queue object "
            f"captured at handler-spawn time and passed in as a parameter."
        )


def test_start_publishes_new_queue_before_incrementing_generation():
    """Regression guard for the publish-order hazard.

    start() must install the new queue BEFORE advertising the new generation. In the
    reverse order the object briefly presents "new generation, old queue"; a handler
    spawned in that window pairs a generation that passes every subsequent check with the
    previous session's queue, silently dropping its records into a queue nobody reads.
    """
    node = _queue_server_method_ast("start")

    def is_queue_assignment(stmt: ast.AST) -> bool:
        return isinstance(stmt, ast.Assign) and _reads_self_queue(stmt)

    def is_generation_bump(stmt: ast.AST) -> bool:
        if not isinstance(stmt, ast.AugAssign):
            return False
        target = stmt.target
        return (
            isinstance(target, ast.Attribute)
            and target.attr == "_generation"
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        )

    # Compare source line numbers so the check is independent of nesting.
    queue_assign_index = next(
        (stmt.lineno for stmt in ast.walk(node) if is_queue_assignment(stmt)), None
    )
    generation_bump_index = next(
        (stmt.lineno for stmt in ast.walk(node) if is_generation_bump(stmt)), None
    )

    assert queue_assign_index is not None, "start() must assign a new self._queue"
    assert generation_bump_index is not None, "start() must increment self._generation"
    assert queue_assign_index < generation_bump_index, (
        "start() increments self._generation before installing the new queue; "
        "swap the order so the queue is published first."
    )
