import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord
from storage.repository import save_record


def test_save_record_persists_all_fields():
    """Verify save_record persists all 12 columns with correct values."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    # Use UTC timestamp
    timestamp_utc = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
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
        # Timestamp is stored as UTC-naive (literal value, not formula)
        assert saved.timestamp == datetime(2026, 8, 8, 12, 0, 0)
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


def test_timestamp_normalization_with_tz_aware_input():
    """Verify aware timestamps are converted to UTC and stored as naive."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    # Create timestamp in UTC+5 timezone: 2026-08-08T12:00:00+05:00
    # This should be stored as 2026-08-08T07:00:00 (UTC-naive, 5 hours earlier)
    tz_plus5 = timezone(timedelta(hours=5))
    timestamp_aware = datetime(2026, 8, 8, 12, 0, 0, tzinfo=tz_plus5)
    record = UnifiedRecord(
        timestamp=timestamp_aware,
        lat=47.6,
        lon=-122.3,
        survey_id="s1",
        operator_id="op1",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
        signal=Signal(rssi=-50.0),
    )

    with session_factory() as session:
        saved = save_record(session, record)
        # Must be stored as 07:00 UTC (naive), not 12:00
        assert saved.timestamp == datetime(2026, 8, 8, 7, 0, 0)


def test_timestamp_normalization_with_naive_input(monkeypatch):
    """Verify naive timestamps are treated as UTC and stored unchanged.

    Uses monkeypatch to pin timezone to America/Los_Angeles so the test
    is deterministic and fails under the buggy (old) formula on any CI runner,
    not just non-UTC ones.
    """
    # Pin the timezone to America/Los_Angeles for deterministic testing
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        engine = make_engine("sqlite:///:memory:")
        init_db(engine)
        session_factory = make_session_factory(engine)

        # Create naive timestamp: 2026-08-08T12:00:00 (no tzinfo)
        # Under the BUGGY formula, this would be treated as local time (Los Angeles)
        # and converted to UTC, storing as 19:00 (7 hours later).
        # Under the CORRECT formula, naive is treated as already UTC and stored unchanged.
        timestamp_naive = datetime(2026, 8, 8, 12, 0, 0)
        record = UnifiedRecord(
            timestamp=timestamp_naive,
            lat=47.6,
            lon=-122.3,
            survey_id="s1",
            operator_id="op1",
            modality=Modality.WIFI,
            identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
            signal=Signal(rssi=-50.0),
        )

        with session_factory() as session:
            saved = save_record(session, record)
            # Must remain exactly as provided (12:00, not 19:00)
            assert saved.timestamp == datetime(2026, 8, 8, 12, 0, 0)
    finally:
        # Restore default timezone
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()


def test_json_fields_store_all_keys_with_explicit_nulls():
    """Verify JSON fields store all schema keys with explicit null values, not excluded."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    # Create a WiFi record with many None fields
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
        # Verify identifier has all keys, even those with null values
        assert "bt_mac" in saved.identifier
        assert saved.identifier["bt_mac"] is None
        assert saved.identifier["bssid"] == "AA:BB:CC:DD:EE:FF"
        assert saved.identifier["ssid"] == "Net"

        # Verify signal has all keys with null for missing fields
        assert "rsrq" in saved.signal
        assert saved.signal["rsrq"] is None
        assert saved.signal["rssi"] == -50.0

        # Verify metadata_ has all keys with null for missing fields
        assert "tag" in saved.metadata_
        assert saved.metadata_["tag"] is None
        assert saved.metadata_["sample_count_in_grid_cell"] == 0


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


def test_save_record_rolls_back_and_session_stays_usable():
    """Verify save_record catches commit errors and rolls back without poisoning the session.

    The except block in save_record must be executed to prevent PendingRollbackError
    from poisoning the session for subsequent calls.
    """
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    # Create a shared state to control when the mock raises
    mock_state = {"should_fail": False}

    class MockCommit:
        def __init__(self, original_commit):
            self.original_commit = original_commit

        def __call__(self):
            if mock_state["should_fail"]:
                raise IntegrityError("Simulated constraint violation", "", "")
            return self.original_commit()

    with session_factory() as session:
        # Patch commit from the start
        original_commit = session.commit
        session.commit = MockCommit(original_commit)

        # Save first record successfully
        record1 = UnifiedRecord(
            timestamp=datetime.now(timezone.utc),
            lat=47.6,
            lon=-122.3,
            survey_id="s1",
            operator_id="op1",
            modality=Modality.WIFI,
            identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
            signal=Signal(rssi=-50.0),
        )
        saved1 = save_record(session, record1)
        assert saved1.id is not None

        # Now enable failure mode and try to save a second record
        record2 = UnifiedRecord(
            timestamp=datetime.now(timezone.utc),
            lat=48.0,
            lon=-123.0,
            survey_id="s2",
            operator_id="op2",
            modality=Modality.WIFI,
            identifier=Identifier(ssid="Net2"),
            signal=Signal(rssi=-60.0),
        )
        mock_state["should_fail"] = True

        # This should raise IntegrityError
        try:
            save_record(session, record2)
            assert False, "Expected IntegrityError to be raised"
        except IntegrityError:
            # Expected: the except block in save_record should have rolled back
            pass

        # Disable failure mode
        mock_state["should_fail"] = False

        # Now verify the session is still usable by saving a different valid record
        record3 = UnifiedRecord(
            timestamp=datetime.now(timezone.utc),
            lat=49.0,
            lon=-124.0,
            survey_id="s3",
            operator_id="op3",
            modality=Modality.WIFI,
            identifier=Identifier(bssid="FF:EE:DD:CC:BB:AA"),
            signal=Signal(rssi=-70.0),
        )
        saved3 = save_record(session, record3)
        assert saved3.id is not None
        assert saved3.survey_id == "s3"

    # Verify persisted records (only successful ones should be in DB)
    with session_factory() as session2:
        all_records = session2.execute(select(SurveyRecord)).scalars().all()
        # Should have 2 records: first save succeeded, second failed (rolled back), third succeeded
        assert len(all_records) == 2
        survey_ids = {r.survey_id for r in all_records}
        assert "s1" in survey_ids
        assert "s3" in survey_ids
        assert "s2" not in survey_ids
