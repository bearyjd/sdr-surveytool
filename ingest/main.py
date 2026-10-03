"""Long-lived ingest process: owns the queue server, the database, and the
enrichment loop that capture processes feed via the Unix domain socket."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import time
from pathlib import Path

from sqlalchemy.engine import make_url

from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from storage.db import init_db, make_engine, make_session_factory, redacted_url
from storage.snippet_store import DEFAULT_MAX_SNIPPET_BYTES, LocalSnippetStore, require_absolute

logger = logging.getLogger(__name__)

DATABASE_URL_ENV = "SURVEYTOOL_DATABASE_URL"
# The name LoadCredential= gives it: systemd puts the file in $CREDENTIALS_DIRECTORY.
DATABASE_URL_CREDENTIAL = "database_url"
_DEFAULT_DATABASE_URL = "sqlite:///survey.db"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDR survey ingest service")
    parser.add_argument(
        "--socket-path",
        default="/tmp/sdr-ingest.sock",
        help="Unix domain socket that capture processes connect to.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=f"SQLAlchemy database URL. Never put a password here: the command line is "
        f"readable by every local user. Use {DATABASE_URL_ENV} or the systemd credential "
        f"{DATABASE_URL_CREDENTIAL} instead (default: {_DEFAULT_DATABASE_URL}).",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for a record before re-checking for shutdown.",
    )
    parser.add_argument(
        "--snippet-staging-dir",
        default=None,
        help="Opt in to storing unknown-signal IQ snippets (with --snippet-store-dir): "
        "the absolute directory capture stages SigMF snippets in, e.g. "
        "/var/lib/sdr-surveytool/snippet-staging. Must match the capture side's "
        "--staging-dir. Without it, unknown-signal records are kept without IQ.",
    )
    parser.add_argument(
        "--snippet-store-dir",
        default=None,
        help="Opt in to storing IQ snippets (with --snippet-staging-dir): the absolute "
        "directory ingest moves adopted SigMF snippets into, e.g. "
        "/var/lib/sdr-surveytool/snippets.",
    )
    parser.add_argument(
        "--max-snippet-bytes",
        type=int,
        default=DEFAULT_MAX_SNIPPET_BYTES,
        help="Reject staged snippets larger than this (default: the largest "
        "snippet any capture configuration can write).",
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
    snippet_dirs = (args.snippet_staging_dir, args.snippet_store_dir)
    if (snippet_dirs[0] is None) != (snippet_dirs[1] is None):
        parser.error(
            "--snippet-staging-dir and --snippet-store-dir enable the snippet store "
            "together; pass both or neither"
        )
    for directory in snippet_dirs:
        if directory is not None:
            try:
                require_absolute(directory)
            except ValueError as exc:
                parser.error(str(exc))
    if args.gps_fix_quality is None:
        parser.error(
            "--gps-fix-quality is required until a real GPS provider exists "
            "(pass 0 to explicitly mark records as fix-less, not a real fix)"
        )
    return args


def database_url(args: argparse.Namespace) -> str:
    """--database-url, else $SURVEYTOOL_DATABASE_URL, else the systemd
    credential database_url (LoadCredential=), else the local SQLite file.
    A password belongs in the environment or the credential: the command
    line is readable by every local user (ps, /proc/<pid>/cmdline), so one
    there still works but is warned about."""
    if args.database_url is not None:
        url = make_url(args.database_url)
        if url.password is not None or "password" in url.query:
            logger.warning(
                "--database-url carries a password, which is visible to every local user "
                "(ps, /proc/<pid>/cmdline); pass the URL in %s or the systemd credential %s",
                DATABASE_URL_ENV,
                DATABASE_URL_CREDENTIAL,
            )
        return args.database_url
    from_env = os.environ.get(DATABASE_URL_ENV)
    if from_env:
        return from_env
    credentials = os.environ.get("CREDENTIALS_DIRECTORY")
    if credentials:
        try:
            from_credential = (Path(credentials) / DATABASE_URL_CREDENTIAL).read_text().strip()
        except FileNotFoundError:
            from_credential = ""
        if from_credential:
            return from_credential
    return _DEFAULT_DATABASE_URL


def _open_snippet_store(args: argparse.Namespace) -> LocalSnippetStore | None:
    """The snippet store if the operator opted in (both snippet dirs given),
    else None: ingest then runs exactly as without unknown-signal capture,
    and records carrying an iq_snippet_path are kept without it, flagged
    no_snippet_store. When opted in, create and check the dirs (absolute,
    0700, owned by this uid, hard-linkable), failing fast on a
    misconfiguration before ingest accepts any record, and log them."""
    if args.snippet_staging_dir is None:
        logger.info(
            "Snippet store disabled (no --snippet-staging-dir/--snippet-store-dir): "
            "unknown-signal records are kept without IQ"
        )
        return None
    store = LocalSnippetStore(
        args.snippet_staging_dir, args.snippet_store_dir, max_snippet_bytes=args.max_snippet_bytes
    )
    logger.info("Adopting snippets from %s into %s", store.staging_dir, store.root_dir)
    return store


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    url = database_url(args)
    engine = make_engine(url)
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

    snippet_store = _open_snippet_store(args)
    server = QueueServer(args.socket_path)
    server.start()
    service = IngestService(server, session_factory, gps_provider, snippet_store=snippet_store)
    logger.info("Ingest listening on %s -> %s", args.socket_path, redacted_url(url))

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
