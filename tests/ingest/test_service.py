from datetime import datetime, timezone

import pytest

from capture.common.emitter import RecordEmitter
from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord
from storage.repository import save_record as real_save_record


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


def test_process_one_enriches_record_with_no_fix_quality_zero(tmp_path):
    """Regression test for the quality=0 sentinel bug: NMEA GGA quality 0
    means "no fix" (invalid), not "already has a fix." A record reporting
    quality=0 at placeholder coordinates must still be enriched from the
    ingest host's own GPS provider.

    Under the old buggy guard (`gps_fix_quality is not None: return record`)
    this record would short-circuit and come back UNCHANGED at (0.0, 0.0)
    with quality still 0 -- this test fails against that code."""
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
            emitter.emit(_record(lat=0.0, lon=0.0, gps_fix_quality=0))

        processed = service.process_one(timeout=2)
        assert processed.lat == 47.6062
        assert processed.lon == -122.3321
        assert processed.gps_fix_quality == 4
    finally:
        server.stop()


def test_process_one_preserves_real_coordinates_without_quality(tmp_path):
    """Regression test for the inverse sentinel bug: a record with genuine,
    non-placeholder coordinates but no reported gps_fix_quality must NOT be
    silently overwritten by the ingest host's own GPS fix.

    Under the old buggy code (which overwrote purely based on
    `gps_fix_quality is None`, with no placeholder check) this record would
    come back with lat/lon replaced by the static fix (47.6062/-122.3321)
    -- this test fails against that code."""
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
            emitter.emit(_record(lat=10.0, lon=20.0, gps_fix_quality=None))

        processed = service.process_one(timeout=2)
        assert processed.lat == 10.0
        assert processed.lon == 20.0
        assert processed.gps_fix_quality is None
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


def test_process_one_reverts_grid_count_on_persist_failure(tmp_path, monkeypatch):
    """Regression test for the grid-count ordering bug: if save_record
    raises, process_one must re-raise (not swallow the error) AND must not
    leave the grid-density counter inflated for a record that was never
    actually persisted.

    Under the old buggy code (increment before save, no compensation on
    failure), a subsequent successful record in the SAME grid cell would
    come back with sample_count_in_grid_cell == 2 instead of 1, and the
    failed record's data would be silently lost with no way to tell the
    counter was already wrong -- this test fails against that code."""
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
            emitter.emit(_record(lat=10.0, lon=20.0, gps_fix_quality=1))

        call_count = {"n": 0}

        def _flaky_save_record(session, record):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated DB failure")
            return real_save_record(session, record)

        monkeypatch.setattr("ingest.service.save_record", _flaky_save_record)

        with pytest.raises(RuntimeError, match="simulated DB failure"):
            service.process_one(timeout=2)

        second = service.process_one(timeout=2)
        assert second.metadata.sample_count_in_grid_cell == 1

        with session_factory() as session:
            rows = session.query(SurveyRecord).all()
            assert len(rows) == 1
    finally:
        server.stop()


def test_attach_grid_density_keys_counts_per_cell():
    """Two records in distinct grid cells must be counted independently;
    a regression to a single global counter would make both come back as 1
    and 2 respectively regardless of location, so we assert on distinct
    cells directly rather than relying on a shared GPS fix. This calls
    _attach_grid_density directly (no queue/DB involved), so no
    QueueServer or engine is needed."""
    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(None, None, gps_provider)

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
