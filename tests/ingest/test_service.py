from datetime import datetime, timezone
from pathlib import Path

import pytest

from capture.common.emitter import RecordEmitter
from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from schema.records import Identifier, Metadata, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord
from storage.repository import save_record as real_save_record
from storage.snippet_store import LocalSnippetStore


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


def test_grid_counts_seeded_from_existing_rows_on_construction(tmp_path):
    """Regression test for the restart bug: _grid_counts used to start empty,
    so an ingest restart mid-survey reset every cell to 1 even though the DB
    already held records there, making sample_count_in_grid_cell non-monotonic
    and useless for coverage-gap analysis.

    Two rows are seeded into one cell before the service is constructed; the
    next record processed into that same cell must be the 3rd sample, not the
    1st -- this test fails against the un-seeded code."""
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    # Pre-existing survey data, as if written before an ingest restart.
    with session_factory() as session:
        for _ in range(2):
            real_save_record(session, _record(lat=47.6062, lon=-122.3321, gps_fix_quality=4))

    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider)

    try:
        with RecordEmitter(socket_path) as emitter:
            # Placeholder coords, so ingest enriches it into the same cell.
            emitter.emit(_record(lat=0.0, lon=0.0, gps_fix_quality=None))

        processed = service.process_one(timeout=2)
        assert processed.metadata.sample_count_in_grid_cell == 3
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


def _stage_snippet(staging_dir, stem: str = "20261003T120000123456Z_915000000Hz_0123abcd") -> str:
    staging_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    (staging_dir / f"{stem}.sigmf-data").write_bytes(b"\x00" * 16)
    (staging_dir / f"{stem}.sigmf-meta").write_text("{}")
    return str((staging_dir / f"{stem}.sigmf-data").resolve())


def _snippet_record(snippet_path: str) -> UnifiedRecord:
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=10.0,
        lon=20.0,
        gps_fix_quality=1,
        survey_id="s",
        operator_id="o",
        modality=Modality.UNKNOWN,
        identifier=Identifier(center_freq=915e6, bandwidth_estimate=20_000.0),
        signal=Signal(rssi=-20.0, peak_power=-17.0),
        metadata=Metadata(iq_snippet_path=snippet_path),
    )


def _snippet_pipeline(tmp_path, snippet_store):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)
    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider, snippet_store=snippet_store)
    return socket_path, server, session_factory, service


def test_process_one_adopts_staged_snippet_and_persists_final_path(tmp_path):
    staging, store_root = tmp_path / "staging", tmp_path / "snippets"
    staged = _stage_snippet(staging)
    socket_path, server, session_factory, service = _snippet_pipeline(
        tmp_path, LocalSnippetStore(staging, store_root)
    )
    try:
        original = _snippet_record(staged)
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(original)

        processed = service.process_one(timeout=2)

        final = str((store_root / "20261003T120000123456Z_915000000Hz_0123abcd.sigmf-data").resolve())
        assert processed.metadata.iq_snippet_path == final
        assert (store_root / "20261003T120000123456Z_915000000Hz_0123abcd.sigmf-meta").is_file()
        assert list(staging.iterdir()) == []
        # The emitted record object is never mutated; ingest works on copies.
        assert original.metadata.iq_snippet_path == staged
        with session_factory() as session:
            rows = session.query(SurveyRecord).all()
            assert len(rows) == 1
            assert rows[0].metadata_["iq_snippet_path"] == final
    finally:
        server.stop()


def test_process_one_persists_record_without_a_rejected_snippet(tmp_path, caplog):
    """A rejected snippet loses only the snippet: the detection itself is
    still persisted, flagged, and process_one returns normally (so the ingest
    loop doesn't back off as it would on a real failure)."""
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    outside = _stage_snippet(tmp_path / "elsewhere")
    socket_path, server, session_factory, service = _snippet_pipeline(
        tmp_path, LocalSnippetStore(staging, tmp_path / "snippets")
    )
    try:
        original = _snippet_record(outside)
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(original)
            emitter.emit(_record(lat=10.0, lon=20.0, gps_fix_quality=1))

        with caplog.at_level("WARNING"):
            rejected = service.process_one(timeout=2)
        second = service.process_one(timeout=2)

        assert rejected.metadata.iq_snippet_path is None
        assert rejected.metadata.quality_flags["snippet_rejected"] == "outside_staging"
        assert repr(outside) in caplog.text
        assert original.metadata.iq_snippet_path == outside
        assert Path(outside).exists()
        assert second.metadata.sample_count_in_grid_cell == 2
        with session_factory() as session:
            rows = session.query(SurveyRecord).order_by(SurveyRecord.id).all()
            assert len(rows) == 2
            assert rows[0].metadata_["iq_snippet_path"] is None
            assert rows[0].metadata_["quality_flags"]["snippet_rejected"] == "outside_staging"
    finally:
        server.stop()


def test_process_one_flags_snippet_record_when_no_store_configured(tmp_path):
    staged = _stage_snippet(tmp_path / "staging")
    socket_path, server, session_factory, service = _snippet_pipeline(tmp_path, None)
    try:
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_snippet_record(staged))

        processed = service.process_one(timeout=2)

        assert processed.metadata.iq_snippet_path is None
        assert processed.metadata.quality_flags["snippet_rejected"] == "no_snippet_store"
        with session_factory() as session:
            assert session.query(SurveyRecord).count() == 1
    finally:
        server.stop()
