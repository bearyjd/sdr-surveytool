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
