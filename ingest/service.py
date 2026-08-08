from __future__ import annotations

from collections import Counter

from sqlalchemy.orm import sessionmaker

from ingest.gps_fix import GpsFixProvider
from ingest.grid import grid_cell_key
from ingest.queue_server import QueueServer
from schema.records import UnifiedRecord
from storage.repository import save_record


class IngestService:
    """Consumes validated records from a QueueServer, attaches the nearest
    GPS fix when the record didn't already carry one, tracks per-grid-cell
    sample density, and persists via storage.repository.save_record."""

    def __init__(
        self,
        queue_server: QueueServer,
        session_factory: sessionmaker,
        gps_provider: GpsFixProvider,
    ) -> None:
        self._queue_server = queue_server
        self._session_factory = session_factory
        self._gps_provider = gps_provider
        self._grid_counts: Counter[str] = Counter()

    def process_one(self, timeout: float | None = None) -> UnifiedRecord:
        record = self._queue_server.get(timeout=timeout)
        record = self._attach_gps_if_missing(record)
        record = self._attach_grid_density(record)
        with self._session_factory() as session:
            save_record(session, record)
        return record

    def _attach_gps_if_missing(self, record: UnifiedRecord) -> UnifiedRecord:
        if record.gps_fix_quality is not None:
            return record
        fix = self._gps_provider.current_fix()
        if fix is None:
            return record
        return record.model_copy(
            update={
                "lat": fix.lat,
                "lon": fix.lon,
                "altitude": fix.altitude,
                "gps_fix_quality": fix.fix_quality,
            }
        )

    def _attach_grid_density(self, record: UnifiedRecord) -> UnifiedRecord:
        key = grid_cell_key(record.lat, record.lon)
        self._grid_counts[key] += 1
        updated_metadata = record.metadata.model_copy(
            update={"sample_count_in_grid_cell": self._grid_counts[key]}
        )
        return record.model_copy(update={"metadata": updated_metadata})
