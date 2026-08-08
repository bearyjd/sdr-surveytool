from __future__ import annotations

import logging
import socket
import threading

from schema.records import UnifiedRecord

logger = logging.getLogger(__name__)


class RecordEmitter:
    """Connects to the ingest queue's Unix domain socket and sends unified
    records as newline-delimited JSON.

    Thread-safe: concurrent calls to emit() are serialized via an internal lock,
    ensuring that records are never interleaved on the wire."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)

    def emit(self, record: UnifiedRecord) -> None:
        """Send one record. If the connection has died (typically because the
        ingest process restarted), reconnect once and re-send. A second failure
        propagates to the caller, whose per-cycle error handling logs it and
        retries on the next poll rather than dying."""
        if self._sock is None:
            raise RuntimeError("RecordEmitter.connect() must be called before emit()")
        payload = (record.model_dump_json() + "\n").encode("utf-8")
        with self._lock:
            try:
                self._sock.sendall(payload)
            except OSError as exc:
                # BrokenPipeError/ConnectionResetError are OSError subclasses.
                logger.warning(
                    "Emit failed (%s); reconnecting to %s and retrying once",
                    exc,
                    self._socket_path,
                )
                self._reconnect_locked().sendall(payload)

    def _reconnect_locked(self) -> socket.socket:
        """Re-establish the socket and return it. Caller must hold self._lock."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self.connect()
        if self._sock is None:  # pragma: no cover - connect() sets it or raises
            raise OSError(f"Failed to reconnect to {self._socket_path}")
        return self._sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> RecordEmitter:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
