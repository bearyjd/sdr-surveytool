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
        conn, _ = server.accept()
        data = b""
        while not data.endswith(b"\n"):
            data += conn.recv(4096)
        received["line"] = data.decode("utf-8")
        conn.close()

    thread = threading.Thread(target=accept_one)
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

    assert received["line"].endswith("\n")
    parsed = json.loads(received["line"])
    assert parsed["identifier"]["bssid"] == "AA:BB:CC:DD:EE:FF"
