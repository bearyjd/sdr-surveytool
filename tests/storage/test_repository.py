from datetime import datetime, timezone

from sqlalchemy import select

from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord
from storage.repository import save_record


def test_save_record_persists_all_fields():
    """Verify save_record persists all 12 columns with correct values."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    timestamp_utc = datetime.now(timezone.utc)
    record = UnifiedRecord(
        timestamp=timestamp_utc,
        lat=47.6,
        lon=-122.3,
        altitude=100.5,
        gps_fix_quality=3,
        survey_id="s1",
        operator_id="op1",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF", ssid="Net", channel=6),
        signal=Signal(rssi=-50.0),
    )

    with session_factory() as session:
        saved = save_record(session, record)
        # Assert all 12 columns
        assert saved.id is not None
        # Timestamp is stored as UTC-naive
        assert saved.timestamp == timestamp_utc.astimezone(timezone.utc).replace(tzinfo=None)
        assert saved.lat == 47.6
        assert saved.lon == -122.3
        assert saved.altitude == 100.5
        assert saved.gps_fix_quality == 3
        assert saved.survey_id == "s1"
        assert saved.operator_id == "op1"
        assert saved.modality == "wifi"
        assert saved.identifier["bssid"] == "AA:BB:CC:DD:EE:FF"
        assert saved.signal["rssi"] == -50.0
        # Verify correct accessor: .metadata_, not .metadata
        assert saved.metadata_["sample_count_in_grid_cell"] == 0


def test_json_fields_store_all_keys_with_explicit_nulls():
    """Verify JSON fields store all schema keys with explicit null values, not excluded."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    # Create a WiFi record with many None fields in identifier
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
        # Verify Bluetooth-specific fields exist with explicit null values, not excluded.
        # WiFi record should still have bt_mac key, just with null value.
        assert "bt_mac" in saved.identifier
        assert saved.identifier["bt_mac"] is None
        # Verify WiFi-specific fields are present
        assert saved.identifier["bssid"] == "AA:BB:CC:DD:EE:FF"
        assert saved.identifier["ssid"] == "Net"


def test_save_record_persists_across_sessions():
    """Verify records persist across separate session instances."""
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

    saved_id = None
    with session_factory() as session:
        saved = save_record(session, record)
        saved_id = saved.id

    # Open a NEW session and read the record back
    with session_factory() as session2:
        retrieved = session2.execute(select(SurveyRecord).where(SurveyRecord.id == saved_id)).scalar_one()
        assert retrieved.id == saved_id
        assert retrieved.survey_id == "s1"
        assert retrieved.modality == "wifi"
        assert retrieved.identifier["bssid"] == "AA:BB:CC:DD:EE:FF"
        assert retrieved.signal["rssi"] == -50.0
