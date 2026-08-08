from __future__ import annotations

import socket

from schema.records import UnifiedRecord


class RecordEmitter:
    """Connects to the ingest queue's Unix domain socket and sends unified
    records as newline-delimited JSON."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)

    def emit(self, record: UnifiedRecord) -> None:
        if self._sock is None:
            raise RuntimeError("RecordEmitter.connect() must be called before emit()")
        line = record.model_dump_json() + "\n"
        self._sock.sendall(line.encode("utf-8"))

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> RecordEmitter:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
