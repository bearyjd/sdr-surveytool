from __future__ import annotations

import logging
import threading
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ingest.gps_fix import GpsFixProvider
from ingest.grid import grid_cell_key
from ingest.queue_server import QueueServer
from schema.records import UnifiedRecord
from storage.models import SurveyRecord
from storage.repository import save_record
from storage.snippet_store import SnippetRejected, SnippetStore

logger = logging.getLogger(__name__)

# NMEA GGA convention: fix quality 0 means "no fix" (invalid); 1+ means a real fix.
_NO_FIX_QUALITY = 0

# Placeholder convention for "no GPS of my own" records: (lat=0.0, lon=0.0, gps_fix_quality=None).
# Confirmed by Task 6's WiFi normalizer (capture/wifi/normalizer.py) and
# Task 7's Bluetooth normalizer (capture/bluetooth/normalizer.py).
_PLACEHOLDER_LAT = 0.0
_PLACEHOLDER_LON = 0.0


class IngestService:
    """Consumes validated records from a QueueServer, adopts any staged IQ
    snippet into the snippet store, attaches the nearest GPS fix when the
    record didn't already carry a usable one, tracks per-grid-cell sample
    density, and persists via storage.repository.save_record.

    Thread-safety: process_one() itself may be called from any single
    thread, but internal grid-density bookkeeping (_grid_counts) is
    protected by a lock so that concurrent callers do not corrupt counts.
    Persistence and GPS-provider calls are NOT synchronized beyond that;
    callers wanting genuinely parallel ingestion should use one
    IngestService per worker or add additional coordination.
    """

    def __init__(
        self,
        queue_server: QueueServer,
        session_factory: sessionmaker | None,
        gps_provider: GpsFixProvider,
        snippet_store: SnippetStore | None = None,
    ) -> None:
        self._queue_server = queue_server
        self._session_factory = session_factory
        self._gps_provider = gps_provider
        self._snippet_store = snippet_store
        # Seeded from the DB so sample_count_in_grid_cell stays monotonic across
        # ingest restarts mid-survey; without this every cell restarts at 1 and
        # the persisted density values are useless for coverage-gap analysis.
        # KNOWN LIMITATION: this Counter is never evicted, so a multi-day survey
        # covering a very large area grows it without bound (one entry per ~11m
        # cell visited). Acceptable for field-tool scale; revisit if it bites.
        self._grid_counts: Counter[str] = self._load_grid_counts(session_factory)
        self._grid_lock = threading.Lock()

    @staticmethod
    def _load_grid_counts(session_factory: sessionmaker | None) -> Counter[str]:
        """Rebuild per-cell sample counts from rows already in the database.

        One query at startup. A None session_factory (used by unit tests that
        exercise the in-memory bookkeeping only) yields an empty counter.
        """
        if session_factory is None:
            return Counter()
        with session_factory() as session:
            # yield_per streams rows instead of materializing the whole
            # result set at once, so a multi-day survey's restart doesn't
            # spike memory pulling every persisted row before ingest can
            # accept its first record. Doesn't address unbounded Counter
            # growth (see KNOWN LIMITATION above) -- that's a separate,
            # deliberately deferred concern.
            rows = session.execute(
                select(SurveyRecord.lat, SurveyRecord.lon)
            ).yield_per(5000)
            return Counter(grid_cell_key(lat, lon) for lat, lon in rows)

    def process_one(self, timeout: float = 1.0) -> UnifiedRecord:
        """Pull one record off the queue, enrich it, and persist it.

        Raises queue.Empty if no record arrives within `timeout` seconds.
        This is a normal idle tick, not an error -- callers running a
        polling loop should treat it that way, e.g.:

            while running:
                try:
                    service.process_one(timeout=1.0)
                except queue.Empty:
                    continue

        `timeout` defaults to a finite value (rather than None) because
        QueueServer.get()'s contract requires a timeout for callers to be
        able to unblock promptly when the server is stopped; a caller that
        never passes a timeout and relies on the default must still be able
        to wake up periodically to observe a shutdown signal.
        """
        record = self._queue_server.get(timeout=timeout)
        # Adopt before the grid-density bump: an I/O error raises here with
        # nothing to roll back (a rejected snippet doesn't raise; the record is
        # kept without it). If save_record later fails, the adopted pair is
        # discarded again rather than left unreferenced in the store.
        staged = record.metadata.iq_snippet_path
        record = self._adopt_snippet_if_present(record)
        adopted = record.metadata.iq_snippet_path if staged is not None else None
        record = self._attach_gps_if_missing(record)
        record, grid_key = self._attach_grid_density(record)
        try:
            with self._session_factory() as session:
                save_record(session, record)
        except Exception:
            # Persistence failed: the record was never stored, so the grid
            # count we optimistically bumped must be rolled back to stay
            # consistent with what's actually in the DB. Log loudly instead
            # of dropping the record silently.
            with self._grid_lock:
                self._grid_counts[grid_key] -= 1
            if adopted is not None:
                self._discard_adopted(adopted)
            logger.error(
                "Failed to persist record for grid cell %s; record dropped "
                "and grid count reverted",
                grid_key,
                exc_info=True,
            )
            raise
        return record

    def _adopt_snippet_if_present(self, record: UnifiedRecord) -> UnifiedRecord:
        """Move a capture-staged SigMF pair into the snippet store and point
        the record at its final location. Capture never writes to storage
        itself (design doc section 6); this is where snippets cross over."""
        staged = record.metadata.iq_snippet_path
        if staged is None:
            return record
        if self._snippet_store is None:
            return self._without_snippet(
                record, "no_snippet_store", "ingest has no snippet store configured"
            )
        try:
            final = self._snippet_store.adopt(staged)
        except SnippetRejected as rejection:
            return self._without_snippet(record, rejection.reason, str(rejection))
        updated_metadata = record.metadata.model_copy(update={"iq_snippet_path": final})
        return record.model_copy(update={"metadata": updated_metadata})

    def _discard_adopted(self, stored_path: str) -> None:
        """The record referencing this pair was never persisted: remove the
        pair instead of leaving it orphaned in the store."""
        assert self._snippet_store is not None  # only adopted when a store exists
        try:
            self._snippet_store.discard(stored_path)
        except Exception:
            logger.error("Could not discard orphaned snippet %r", stored_path, exc_info=True)
        else:
            logger.warning("Discarded snippet %r: its record could not be persisted", stored_path)

    def _without_snippet(self, record: UnifiedRecord, reason: str, detail: str) -> UnifiedRecord:
        """A rejected snippet loses only the snippet: the detection is still
        persisted, with iq_snippet_path cleared and the reason flagged."""
        logger.warning(
            "Rejected snippet %r (%s): %s; persisting the record without it",
            record.metadata.iq_snippet_path,
            reason,
            detail,
        )
        updated_metadata = record.metadata.model_copy(
            update={
                "iq_snippet_path": None,
                "quality_flags": {**record.metadata.quality_flags, "snippet_rejected": reason},
            }
        )
        return record.model_copy(update={"metadata": updated_metadata})

    def _has_usable_fix(self, record: UnifiedRecord) -> bool:
        """True if the record already reports a real GPS fix.

        Quality 0 (NMEA GGA "no fix") and None (no quality reported) both
        count as "not usable" -- either way the record still needs
        enrichment from this service's own GPS provider.
        """
        return (
            record.gps_fix_quality is not None
            and record.gps_fix_quality > _NO_FIX_QUALITY
        )

    def _has_placeholder_coords(self, record: UnifiedRecord) -> bool:
        return record.lat == _PLACEHOLDER_LAT and record.lon == _PLACEHOLDER_LON

    def _attach_gps_if_missing(self, record: UnifiedRecord) -> UnifiedRecord:
        if self._has_usable_fix(record):
            return record
        if not self._has_placeholder_coords(record):
            # The record has no usable fix quality but already carries
            # real-looking coordinates from its capture source. Overwriting
            # them with this host's own GPS fix would silently discard a
            # genuine position, so leave it alone and just warn.
            logger.warning(
                "Record has non-placeholder coordinates (lat=%s, lon=%s) but "
                "no usable GPS fix quality (%s); leaving coordinates as-is "
                "instead of overwriting with the ingest host's own fix",
                record.lat,
                record.lon,
                record.gps_fix_quality,
            )
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

    def _attach_grid_density(self, record: UnifiedRecord) -> tuple[UnifiedRecord, str]:
        key = grid_cell_key(record.lat, record.lon)
        with self._grid_lock:
            self._grid_counts[key] += 1
            count = self._grid_counts[key]
        updated_metadata = record.metadata.model_copy(
            update={"sample_count_in_grid_cell": count}
        )
        return record.model_copy(update={"metadata": updated_metadata}), key
