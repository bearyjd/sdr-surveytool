"""End-to-end pipeline tests.

Every other test exercises one seam with a hand-built, already-well-formed
UnifiedRecord. These drive a realistic vendor payload through the whole chain
-- normalizer -> RecordEmitter -> QueueServer -> IngestService -> SQLite --
which is the only way a schema mismatch at the normalizer boundary (e.g. the
Kismet "6HT40" channel string) shows up as a test failure rather than a
crashed capture process in the field.
"""

from capture.bluetooth.normalizer import normalize_ble_advertisement
from capture.common.emitter import RecordEmitter
from capture.wifi.normalizer import normalize_kismet_device
from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord

# A realistic Kismet phy80211 device, with an HT-style channel string of the
# form real 802.11n APs report.
KISMET_DEVICE = {
    "kismet.device.base.macaddr": "AA:BB:CC:DD:EE:FF",
    "kismet.device.base.channel": "6HT40",
    "kismet.device.base.last_time": 1750000000,
    "kismet.device.base.signal": {"kismet.common.signal.last_signal": -55},
    "dot11.device": {
        "dot11.device.advertised_ssid_map": {
            "0": {
                "dot11.advertisedssid.ssid": "TestNet",
                "dot11.advertisedssid.crypt_string": "WPA2",
            }
        }
    },
}

FIX = GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)


def _pipeline(tmp_path):
    """Build a live queue server + in-memory DB + ingest service."""
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    service = IngestService(server, session_factory, StaticGpsFixProvider(FIX))
    return socket_path, server, session_factory, service


def test_wifi_record_flows_from_kismet_json_to_database(tmp_path):
    socket_path, server, session_factory, service = _pipeline(tmp_path)
    try:
        record = normalize_kismet_device(
            KISMET_DEVICE, survey_id="survey-1", operator_id="op-1"
        )
        assert record is not None
        # The HT suffix must have been stripped by the normalizer, not by the
        # schema -- an unparsed "6HT40" would have raised before reaching here.
        assert record.identifier.channel == 6

        with RecordEmitter(socket_path) as emitter:
            emitter.emit(record)

        processed = service.process_one(timeout=2)
        assert processed.metadata.sample_count_in_grid_cell == 1

        with session_factory() as session:
            rows = session.query(SurveyRecord).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.modality == "wifi"
            assert row.identifier["bssid"] == "AA:BB:CC:DD:EE:FF"
            assert row.identifier["ssid"] == "TestNet"
            assert row.identifier["channel"] == 6
            assert row.signal["rssi"] == -55.0
            assert row.metadata_["encryption_type_if_broadcast_visible"] == "WPA2"
            # GPS enrichment happened at ingest, not capture.
            assert row.lat == FIX.lat
            assert row.lon == FIX.lon
            assert row.altitude == FIX.altitude
            assert row.gps_fix_quality == FIX.fix_quality
            assert row.metadata_["sample_count_in_grid_cell"] == 1
    finally:
        server.stop()


def test_bluetooth_record_flows_from_advertisement_to_database(tmp_path):
    socket_path, server, session_factory, service = _pipeline(tmp_path)
    try:
        record = normalize_ble_advertisement(
            address="11:22:33:44:55:66",
            device_name="Fitness Tracker",
            rssi=-72.0,
            survey_id="survey-1",
            operator_id="op-1",
        )

        with RecordEmitter(socket_path) as emitter:
            emitter.emit(record)

        processed = service.process_one(timeout=2)
        assert processed.metadata.sample_count_in_grid_cell == 1

        with session_factory() as session:
            rows = session.query(SurveyRecord).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.modality == "bluetooth"
            assert row.identifier["bt_mac"] == "11:22:33:44:55:66"
            assert row.identifier["device_name"] == "Fitness Tracker"
            assert row.signal["rssi"] == -72.0
            assert row.lat == FIX.lat
            assert row.lon == FIX.lon
            assert row.gps_fix_quality == FIX.fix_quality
            assert row.metadata_["sample_count_in_grid_cell"] == 1
    finally:
        server.stop()
