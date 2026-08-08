from datetime import datetime, timezone

from capture.common.emitter import RecordEmitter
from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord


def _record(
    lat: float = 0.0,
    lon: float = 0.0,
    gps_fix_quality: int | None = None,
) -> UnifiedRecord:
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=lat,
        lon=lon,
        gps_fix_quality=gps_fix_quality,
        survey_id="s",
        operator_id="o",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
        signal=Signal(rssi=-40.0),
    )


def test_process_one_attaches_gps_and_persists(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider)

    try:
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_record(lat=0.0, lon=0.0, gps_fix_quality=None))

        processed = service.process_one(timeout=2)
        assert processed.lat == 47.6062
        assert processed.gps_fix_quality == 4
        assert processed.metadata.sample_count_in_grid_cell == 1

        with session_factory() as session:
            rows = session.query(SurveyRecord).all()
            assert len(rows) == 1
            assert rows[0].lat == 47.6062
    finally:
        server.stop()


def test_process_one_increments_grid_density_for_repeated_cell(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider)

    try:
        with RecordEmitter(socket_path) as emitter:
            for _ in range(2):
                emitter.emit(_record(lat=0.0, lon=0.0, gps_fix_quality=None))

        first = service.process_one(timeout=2)
        second = service.process_one(timeout=2)
        assert first.metadata.sample_count_in_grid_cell == 1
        assert second.metadata.sample_count_in_grid_cell == 2

        with session_factory() as session:
            rows = (
                session.query(SurveyRecord)
                .order_by(SurveyRecord.id)
                .all()
            )
            assert len(rows) == 2
            assert rows[0].metadata_["sample_count_in_grid_cell"] == 1
            assert rows[1].metadata_["sample_count_in_grid_cell"] == 2
    finally:
        server.stop()


def test_process_one_leaves_existing_gps_fix_untouched(tmp_path):
    """A record that already reports a usable (quality > 0) fix and real
    coordinates must not be overwritten by the ingest host's own GPS."""
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider)

    try:
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_record(lat=10.0, lon=20.0, gps_fix_quality=1))

        processed = service.process_one(timeout=2)
        assert processed.lat == 10.0
        assert processed.lon == 20.0
        assert processed.gps_fix_quality == 1
    finally:
        server.stop()


def test_process_one_persists_record_when_no_gps_fix_available(tmp_path):
    """When the GPS provider has no fix at all, the record should still be
    persisted using its own (placeholder) coordinates rather than crashing."""
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    gps_provider = StaticGpsFixProvider(None)
    service = IngestService(server, session_factory, gps_provider)

    try:
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_record(lat=0.0, lon=0.0, gps_fix_quality=None))

        processed = service.process_one(timeout=2)
        assert processed.lat == 0.0
        assert processed.lon == 0.0
        assert processed.gps_fix_quality is None

        with session_factory() as session:
            rows = session.query(SurveyRecord).all()
            assert len(rows) == 1
    finally:
        server.stop()


def test_attach_grid_density_keys_counts_per_cell(tmp_path):
    """Two records in distinct grid cells must be counted independently;
    a regression to a single global counter would make both come back as 1
    and 2 respectively regardless of location, so we assert on distinct
    cells directly rather than relying on a shared GPS fix."""
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider)

    try:
        record_a = _record(lat=10.0, lon=20.0, gps_fix_quality=1)
        record_b = _record(lat=30.0, lon=40.0, gps_fix_quality=1)

        result_a, key_a = service._attach_grid_density(record_a)
        result_b, key_b = service._attach_grid_density(record_b)
        result_a_again, key_a_again = service._attach_grid_density(record_a)

        assert key_a != key_b
        assert key_a == key_a_again
        assert result_a.metadata.sample_count_in_grid_cell == 1
        assert result_b.metadata.sample_count_in_grid_cell == 1
        assert result_a_again.metadata.sample_count_in_grid_cell == 2
    finally:
        server.stop()
