from __future__ import annotations

import os
import socket
import threading
from queue import Queue

from pydantic import ValidationError

from schema.records import UnifiedRecord


class QueueServer:
    """Unix domain socket server. Capture processes connect and send
    newline-delimited unified-schema JSON records; validated records are
    made available via `get()`. Malformed lines are dropped, not raised."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._queue: Queue[UnifiedRecord] = Queue()
        self._server_sock: socket.socket | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if os.path.exists(self._socket_path):
            os.remove(self._socket_path)
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self._socket_path)
        self._server_sock.listen(8)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        self._server_sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket) -> None:
        buffer = b""
        with conn:
            while not self._stop.is_set():
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    self._ingest_line(line)

    def _ingest_line(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            record = UnifiedRecord.model_validate_json(line)
        except ValidationError:
            return
        self._queue.put(record)

    def get(self, timeout: float | None = None) -> UnifiedRecord:
        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._server_sock is not None:
            self._server_sock.close()
        if os.path.exists(self._socket_path):
            os.remove(self._socket_path)
