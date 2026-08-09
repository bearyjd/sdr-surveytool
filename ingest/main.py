"""Long-lived ingest process: owns the queue server, the database, and the
enrichment loop that capture processes feed via the Unix domain socket."""

from __future__ import annotations

import argparse
import logging
import queue
import time

from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from storage.db import init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDR survey ingest service")
    parser.add_argument(
        "--socket-path",
        default="/tmp/sdr-ingest.sock",
        help="Unix domain socket that capture processes connect to.",
    )
    parser.add_argument(
        "--database-url",
        default="sqlite:///survey.db",
        help="SQLAlchemy database URL.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for a record before re-checking for shutdown.",
    )
    # STAND-IN: the real u-blox GPS reader service (gps/) does not exist yet per
    # the plan, so ingest is wired to a fixed fix supplied on the command line.
    # Swap StaticGpsFixProvider for the real provider once gps/ lands.
    parser.add_argument("--gps-lat", type=float, default=0.0)
    parser.add_argument("--gps-lon", type=float, default=0.0)
    parser.add_argument("--gps-altitude", type=float, default=None)
    parser.add_argument(
        "--gps-fix-quality",
        type=int,
        default=None,
        help="NMEA fix quality to stamp on every record. Required -- no real "
        "GPS provider exists yet, so there is no safe default. A default of "
        "1 (a valid fix) would silently mark every record as having a real "
        "GPS fix at whatever --gps-lat/--gps-lon happen to be, which is "
        "indistinguishable from genuine data once persisted. Pass 0 "
        "explicitly to mark records as fix-less rather than fabricating one.",
    )
    args = parser.parse_args(argv)
    if args.gps_fix_quality is None:
        parser.error(
            "--gps-fix-quality is required until a real GPS provider exists "
            "(pass 0 to explicitly mark records as fix-less, not a real fix)"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    engine = make_engine(args.database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)

    gps_provider = StaticGpsFixProvider(
        GpsFix(
            lat=args.gps_lat,
            lon=args.gps_lon,
            altitude=args.gps_altitude,
            fix_quality=args.gps_fix_quality,
        )
    )

    server = QueueServer(args.socket_path)
    server.start()
    service = IngestService(server, session_factory, gps_provider)
    logger.info("Ingest listening on %s -> %s", args.socket_path, args.database_url)

    try:
        while True:
            try:
                service.process_one(timeout=args.poll_timeout)
            except queue.Empty:
                # Normal idle tick, not an error.
                continue
            except Exception:
                # A single bad record (e.g. a transient DB failure) must not
                # take down a survey that may be hours into the field. The
                # bounded sleep keeps a persistent failure (e.g. the DB is
                # down) from becoming a tight retry loop that floods the log
                # with tracebacks; it doesn't touch the queue.Empty happy path.
                logger.exception("Failed to process record; continuing")
                time.sleep(min(args.poll_timeout, 1.0))
    except KeyboardInterrupt:
        logger.info("Shutting down ingest")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
