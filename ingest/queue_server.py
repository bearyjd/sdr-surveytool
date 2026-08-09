from __future__ import annotations

import logging
import os
import socket
import stat
import threading
from queue import Queue, Full

from pydantic import ValidationError

from schema.records import UnifiedRecord

logger = logging.getLogger(__name__)

# Buffer limit: 1MB per connection to prevent unbounded growth
_MAX_BUFFER_SIZE = 1024 * 1024

# Bounded queue: applies backpressure to capture processes instead of growing memory.
_QUEUE_MAXSIZE = 1000


class QueueServer:
    """Unix domain socket server. Capture processes connect and send
    newline-delimited unified-schema JSON records; validated records are
    made available via `get()`. Malformed lines are dropped with a warning.

    get(timeout=T) will raise queue.Empty if no record is available after T seconds.
    get(timeout=None) blocks indefinitely. Callers MUST use a timeout to ensure
    they can be unblocked when stop() is called (stop() does not unblock blocked get() calls).

    ARCHITECTURE NOTE: Session isolation is enforced structurally, not by timing. Each
    start() installs a brand-new queue object and increments _generation. Handler threads
    capture BOTH the generation and the queue object itself at spawn time and only ever
    enqueue into their captured queue. A handler that outlives stop() therefore physically
    cannot reach a later session's queue, no matter how long it stays alive or how many
    times it retries a blocked put(). The generation check is a promptness optimisation on
    top of that guarantee, not the guarantee itself.

    RESTART SEMANTICS: start() installs a fresh queue, so any records the previous
    session's consumer had not yet read (up to _QUEUE_MAXSIZE) are discarded."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._queue: Queue[UnifiedRecord] = Queue(maxsize=_QUEUE_MAXSIZE)
        self._server_sock: socket.socket | None = None
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._handler_threads: list[threading.Thread] = []
        self._handler_lock = threading.Lock()
        self._started = False
        self._generation = 0
        # True only after this instance's own bind() succeeded; guards cleanup-time
        # socket-file removal so we never warn about a file we never created.
        self._owns_socket_file = False

    def _remove_if_socket(self) -> None:
        """Remove socket file only if it exists and is actually a socket.
        Raises RuntimeError if path exists but is not a socket."""
        if os.path.exists(self._socket_path):
            try:
                file_stat = os.stat(self._socket_path)
                if stat.S_ISSOCK(file_stat.st_mode):
                    os.remove(self._socket_path)
                else:
                    raise RuntimeError(
                        f"Socket path {self._socket_path} exists but is not a socket"
                    )
            except FileNotFoundError:
                pass

    def start(self) -> None:
        """Start the queue server. Raises RuntimeError if already started.

        Restart discards data: a fresh queue is installed, so any records the previous
        session's consumer had not yet read (up to _QUEUE_MAXSIZE) are dropped.
        """
        if self._started:
            raise RuntimeError("QueueServer.start() called while already started")
        self._stop.clear()
        # Publish ORDER MATTERS: install the new queue before advertising the new
        # generation. The reverse order leaves a window in which the object says
        # "new generation, old queue", so a handler spawned in that window would pair a
        # generation that passes every future check with the previous session's queue and
        # silently lose its records into a queue nobody reads.
        self._queue = Queue(maxsize=_QUEUE_MAXSIZE)
        self._generation += 1
        self._owns_socket_file = False

        try:
            # Validate and remove any existing socket file, with error handling (NEW-2)
            try:
                self._remove_if_socket()
            except RuntimeError as e:
                logger.warning(f"Failed to remove old socket file: {e}")
                raise

            self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server_sock.bind(self._socket_path)
            # From here on the socket file is ours, so cleanup may remove it.
            self._owns_socket_file = True
            # Restrict permissions to owner only (0o600)
            os.chmod(self._socket_path, 0o600)
            self._server_sock.listen(8)
            self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            self._accept_thread.start()

            # Only set _started to True after everything succeeded
            self._started = True
        except Exception:
            # If startup failed, reset state before re-raising (NEW-4)
            self._started = False
            if self._server_sock is not None:
                self._server_sock.close()
                self._server_sock = None
            # Only clean up the socket file if this instance actually bound it; otherwise
            # the failure came from the pre-bind check and the file is not ours to touch.
            if self._owns_socket_file:
                try:
                    self._remove_if_socket()
                except (RuntimeError, OSError) as e:
                    logger.warning(f"Could not clean up socket file on startup failure: {e}")
                self._owns_socket_file = False
            raise

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        self._server_sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except (socket.timeout, OSError):
                continue
            # Capture BOTH the generation and the queue OBJECT at spawn time. The captured
            # queue is what makes session isolation structural: this handler can only ever
            # enqueue into the queue that was live when it was created.
            gen = self._generation
            queue = self._queue
            thread = threading.Thread(
                target=self._handle_client, args=(conn, gen, queue), daemon=True
            )
            with self._handler_lock:
                # Prune finished handlers as we go; otherwise the list grows without
                # bound over a long-lived session (entries are only reaped in stop()).
                self._handler_threads = [t for t in self._handler_threads if t.is_alive()]
                self._handler_threads.append(thread)
            thread.start()

    def _handle_client(self, conn: socket.socket, gen: int, queue: Queue[UnifiedRecord]) -> None:
        """Handle a client connection for the given generation.

        `queue` is the queue object that was live when this handler was spawned. All
        records read from this connection go into that object and nowhere else, so a
        handler that outlives stop() cannot contaminate a later session's queue. The
        generation checks additionally make such a handler wind down promptly."""
        buffer = b""
        try:
            conn.settimeout(0.5)
            # Exit if generation changes (new start() was called) or stop is set
            while not self._stop.is_set() and gen == self._generation:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > _MAX_BUFFER_SIZE:
                    logger.warning(
                        f"Client buffer exceeded {_MAX_BUFFER_SIZE} bytes; closing connection"
                    )
                    break
                while b"\n" in buffer:
                    # Re-check liveness per line so a stale handler with a large backlog
                    # stops doing work promptly rather than draining the whole buffer.
                    if self._stop.is_set() or gen != self._generation:
                        return
                    line, buffer = buffer.split(b"\n", 1)
                    self._ingest_line(line, gen, queue)
        finally:
            conn.close()

    def _ingest_line(self, line: bytes, gen: int, queue: Queue[UnifiedRecord]) -> None:
        """Validate one line and enqueue it into `queue`.

        `queue` is the caller's captured queue object. This must never read
        `self._queue`: doing so would let a stale handler enqueue into a newer
        session's queue the moment start() swapped the attribute."""
        if not line.strip():
            return
        try:
            record = UnifiedRecord.model_validate_json(line)
        except ValidationError:
            logger.warning(
                f"Dropped malformed record: {line[:100].decode('utf-8', errors='replace')}"
            )
            return

        # Put with a timeout so the handler stays responsive to stop()/restart instead of
        # blocking forever on a full queue. Retrying applies backpressure to the producer.
        warned = False
        while True:
            try:
                queue.put(record, timeout=0.5)
                return
            except Full:
                if self._stop.is_set() or gen != self._generation:
                    logger.warning("Queue full and session ended; dropping record")
                    return
                if not warned:
                    # Rate-limited: warn once per record on first backpressure, not on
                    # every 0.5s retry, so sustained backpressure is visible but not spam.
                    warned = True
                    logger.warning(
                        f"Queue full (size={queue.qsize()}); handler applying backpressure"
                    )

    def get(self, timeout: float | None = None) -> UnifiedRecord:
        """Get the next record from the queue.

        Args:
            timeout: Timeout in seconds. If None, blocks indefinitely.

        Returns:
            The next UnifiedRecord.

        Raises:
            queue.Empty: If timeout expires with no record available.

        Note: Callers MUST use a timeout to ensure they can respond to stop()."""
        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        """Stop the queue server and clean up resources.

        Shutdown discards data: handler threads stop processing as soon as the stop flag
        is observed, so complete lines already sitting unparsed in a handler's read buffer
        are dropped along with anything still in flight on the socket. This mirrors the
        RESTART SEMANTICS note on start(), which covers unconsumed records in the queue.
        """
        self._stop.set()
        try:
            # Closing the listening socket lives inside the try so that a close()
            # failure can never skip the state reset in the finally block.
            if self._server_sock is not None:
                self._server_sock.close()

            # Join accept thread with timeout
            if self._accept_thread is not None:
                self._accept_thread.join(timeout=2.0)

            # Join all handler threads with timeout, only remove successfully joined ones.
            # Stale threads from a previous generation will exit due to generation check.
            with self._handler_lock:
                still_alive = []
                for thread in self._handler_threads:
                    thread.join(timeout=0.5)
                    if thread.is_alive():
                        still_alive.append(thread)
                # Only clear the ones we successfully joined
                self._handler_threads = still_alive

            # Clean up socket file only if this instance actually bound it.
            if self._owns_socket_file:
                try:
                    self._remove_if_socket()
                except (RuntimeError, OSError) as e:
                    logger.warning(f"Could not remove socket file on stop: {e}")
        finally:
            # Always reset state to allow restart, even if cleanup failed
            self._started = False
            self._accept_thread = None
            self._server_sock = None
            self._owns_socket_file = False
