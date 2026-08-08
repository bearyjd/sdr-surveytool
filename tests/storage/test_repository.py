from datetime import datetime, timezone

from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.repository import save_record


def test_save_record_persists_all_fields():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    record = UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=47.6,
        lon=-122.3,
        survey_id="s1",
        operator_id="op1",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF", ssid="Net", channel=6),
        signal=Signal(rssi=-50.0),
    )

    with session_factory() as session:
        saved = save_record(session, record)
        assert saved.id is not None
        assert saved.survey_id == "s1"
        assert saved.modality == "wifi"
        assert saved.identifier["bssid"] == "AA:BB:CC:DD:EE:FF"
        assert saved.signal["rssi"] == -50.0
