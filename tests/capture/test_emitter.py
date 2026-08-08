import json
import socket
import threading

from datetime import datetime, timezone

from capture.common.emitter import RecordEmitter
from schema.records import Identifier, Modality, Signal, UnifiedRecord


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
