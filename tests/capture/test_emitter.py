import json
import logging
import os
import socket
import struct
import threading

from datetime import datetime, timezone

import pytest

from capture.common.emitter import RecordEmitter
from schema.records import Identifier, Modality, Signal, UnifiedRecord


def _record() -> UnifiedRecord:
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=1.0,
        lon=2.0,
        survey_id="s",
        operator_id="o",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
        signal=Signal(rssi=-40.0),
    )


def test_emit_reconnects_after_peer_restart(tmp_path):
    """If the ingest process restarts, the emitter's socket dies and sendall
    raises. emit() must reconnect once and deliver the record rather than
    letting the capture process fall over."""
    socket_path = str(tmp_path / "ingest.sock")

    def serve_one(sock: socket.socket, sink: dict, key: str) -> None:
        conn, _ = sock.accept()
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        sink[key] = data.decode("utf-8")
        # Abrupt close (SO_LINGER 0) so the peer sees a reset, mimicking a
        # crashed/restarted ingest rather than a graceful shutdown.
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()

    received: dict[str, str] = {}

    first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    first.bind(socket_path)
    first.listen(1)
    thread = threading.Thread(target=serve_one, args=(first, received, "first"), daemon=True)
    thread.start()

    emitter = RecordEmitter(socket_path)
    emitter.connect()
    try:
        emitter.emit(_record())
        thread.join(timeout=2)
        assert "first" in received

        # "Restart" the ingest side on the same path.
        first.close()
        os.unlink(socket_path)
        second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        second.bind(socket_path)
        second.listen(1)
        thread2 = threading.Thread(
            target=serve_one, args=(second, received, "second"), daemon=True
        )
        thread2.start()

        # The emitter still holds the dead socket; this must reconnect and land.
        emitter.emit(_record())
        thread2.join(timeout=2)
        second.close()

        assert "second" in received, "record was not delivered after reconnect"
        assert json.loads(received["second"])["identifier"]["bssid"] == "AA:BB:CC:DD:EE:FF"
    finally:
        emitter.close()


def test_emit_sends_record_as_ndjson_line(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    received = {}

    def accept_one():
        try:
            conn, _ = server.accept()
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    received["error"] = "Connection closed before newline received"
                    break
                data += chunk
            if "error" not in received:
                received["line"] = data.decode("utf-8")
            conn.close()
        except Exception as e:
            received["error"] = str(e)

    thread = threading.Thread(target=accept_one, daemon=True)
    thread.start()

    record = UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=1.0,
        lon=2.0,
        survey_id="s",
        operator_id="o",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
        signal=Signal(rssi=-40.0),
    )

    with RecordEmitter(socket_path) as emitter:
        emitter.emit(record)

    thread.join(timeout=2)
    server.close()

    # Assert on results in main thread for proper test failure reporting
    assert "error" not in received, f"Helper thread error: {received.get('error')}"
    assert "line" in received, "No line received from socket"
    assert received["line"].endswith("\n")
    parsed = json.loads(received["line"])
    assert parsed["identifier"]["bssid"] == "AA:BB:CC:DD:EE:FF"


def _serve_one_line(server: socket.socket, sink: dict) -> None:
    conn, _ = server.accept()
    data = b""
    while not data.endswith(b"\n"):
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    sink["line"] = data.decode("utf-8")
    conn.close()


def test_entering_while_ingest_is_down_logs_instead_of_raising(tmp_path, caplog):
    """Capture processes may start before ingest: a missing socket at
    startup is logged, not fatal (it used to kill the whole service)."""
    socket_path = str(tmp_path / "ingest.sock")
    with caplog.at_level(logging.WARNING):
        with RecordEmitter(socket_path):
            pass
    assert socket_path in caplog.text


def test_emit_connects_lazily_once_ingest_comes_up(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    received: dict = {}
    with RecordEmitter(socket_path) as emitter:
        # Still down: emit raises, so the caller's per-cycle handling
        # (log, drop, delete the staged snippet) takes over as before.
        with pytest.raises(OSError):
            emitter.emit(_record())

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)
        thread = threading.Thread(target=_serve_one_line, args=(server, received), daemon=True)
        thread.start()
        emitter.emit(_record())
        thread.join(timeout=2)
        server.close()

    assert json.loads(received["line"])["identifier"]["bssid"] == "AA:BB:CC:DD:EE:FF"
