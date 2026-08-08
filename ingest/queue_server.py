from __future__ import annotations

import logging
import os
import socket
import stat
import threading
from queue import Queue

from pydantic import ValidationError

from schema.records import UnifiedRecord

logger = logging.getLogger(__name__)

# Buffer limit: 1MB per connection to prevent unbounded growth
_MAX_BUFFER_SIZE = 1024 * 1024


class QueueServer:
    """Unix domain socket server. Capture processes connect and send
    newline-delimited unified-schema JSON records; validated records are
    made available via `get()`. Malformed lines are dropped with a warning.

    get(timeout=T) will raise queue.Empty if no record is available after T seconds.
    get(timeout=None) blocks indefinitely. Callers MUST use a timeout to ensure
    they can be unblocked when stop() is called (stop() does not unblock blocked get() calls)."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._queue: Queue[UnifiedRecord] = Queue(maxsize=1000)
        self._server_sock: socket.socket | None = None
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._handler_threads: list[threading.Thread] = []
        self._handler_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """Start the queue server. Raises RuntimeError if already started."""
        if self._started:
            raise RuntimeError("QueueServer.start() called while already started")
        self._started = True
        self._stop.clear()

        # Validate and remove any existing socket file
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

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self._socket_path)
        # Restrict permissions to owner only (0o600)
        os.chmod(self._socket_path, 0o600)
        self._server_sock.listen(8)
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        self._server_sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except (socket.timeout, OSError):
                continue
            thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            with self._handler_lock:
                self._handler_threads.append(thread)
            thread.start()

    def _handle_client(self, conn: socket.socket) -> None:
        buffer = b""
        try:
            conn.settimeout(0.5)
            while not self._stop.is_set():
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
                    line, buffer = buffer.split(b"\n", 1)
                    self._ingest_line(line)
        finally:
            conn.close()

    def _ingest_line(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            record = UnifiedRecord.model_validate_json(line)
        except ValidationError:
            logger.warning(
                f"Dropped malformed record: {line[:100].decode('utf-8', errors='replace')}"
            )
            return
        self._queue.put(record)

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
        """Stop the queue server and clean up resources."""
        self._stop.set()
        if self._server_sock is not None:
            self._server_sock.close()

        # Join accept thread with timeout
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)

        # Join all handler threads with timeout
        with self._handler_lock:
            for thread in self._handler_threads:
                thread.join(timeout=0.5)
            self._handler_threads.clear()

        # Clean up socket file
        if os.path.exists(self._socket_path):
            try:
                os.remove(self._socket_path)
            except OSError:
                pass

        # Reset state to allow restart
        self._started = False
        self._accept_thread = None
        self._server_sock = None
