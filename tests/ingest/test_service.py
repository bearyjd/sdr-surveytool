from datetime import datetime, timezone

from capture.common.emitter import RecordEmitter
from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord


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
            record = UnifiedRecord(
                timestamp=datetime.now(timezone.utc),
                lat=0.0,
                lon=0.0,
                gps_fix_quality=None,
                survey_id="s",
                operator_id="o",
                modality=Modality.WIFI,
                identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
                signal=Signal(rssi=-40.0),
            )
            emitter.emit(record)

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
                emitter.emit(
                    UnifiedRecord(
                        timestamp=datetime.now(timezone.utc),
                        lat=0.0,
                        lon=0.0,
                        gps_fix_quality=None,
                        survey_id="s",
                        operator_id="o",
                        modality=Modality.WIFI,
                        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
                        signal=Signal(rssi=-40.0),
                    )
                )

        first = service.process_one(timeout=2)
        second = service.process_one(timeout=2)
        assert first.metadata.sample_count_in_grid_cell == 1
        assert second.metadata.sample_count_in_grid_cell == 2
    finally:
        server.stop()
