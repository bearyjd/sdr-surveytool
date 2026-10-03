# agent/db_gateway.py
"""The agent's only database access (design: "Structural boundary").

SQL against the boundary objects installed by storage/sql/agent_boundary.sql
and nothing else: the public.agent_pending_unknown view (read) and the
public.classify_unknown function (write). `connect_gateway` refuses to
return a gateway unless the connected role is confined by that boundary.
The only module in agent/ allowed to import sqlalchemy or psycopg.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from schema.records import ClassificationStatus

DEFAULT_AGENT_ROLE = "surveytool_agent"
DEFAULT_STATEMENT_TIMEOUT_S = 30.0

_NOT_PENDING = "P0002"  # classify_unknown's no_data_found
_TIMED_OUT = {"57014", "55P03"}  # query_canceled (statement_timeout), lock_not_available
# Held for the agent's lifetime on its own session: one agent per database.
_SINGLETON_LOCK = 0x5344_5241_4745_4E54 & 0x7FFF_FFFF_FFFF_FFFF  # "SDRAGENT", a positive bigint
_CLASSIFY = "public.classify_unknown(integer, text, text, double precision, text)"
_PREDICATE = "public.agent_is_pending_unknown(text, json)"

# The view returns its numbers as text (agent_boundary.sql); these bounds
# say what a plausible value is. Anything else makes the record malformed.
_NUMBER = re.compile(r"-?\d+(\.\d+)?([eE][+-]?\d+)?")
_BOUNDS = {
    "sample_rate": (1.0, 1e10),
    "center_freq": (1.0, 1e12),
    "peak_power": (-400.0, 100.0),  # dBFS
    "snippet_duration_ms": (0.0, 2**31 - 1),
}
_MAX_PATH_CHARS = 4096

# The self-check is an allowlist, run against the LOGIN (session_user): a
# login whose role setting switches it to another role at connect would
# otherwise be judged by that role, and could RESET ROLE back. The session
# must run as the login itself, with no role settings. The login may be a
# member of nothing but itself and the agent role; neither may carry a
# dangerous attribute, any privilege on survey_records, the right to create
# objects anywhere, TEMPORARY on any database, or a large-object writer;
# and the only SECURITY DEFINER function it may execute (outside
# extensions) is classify_unknown.
# Each query returns one row per problem: a priority (attributes and role
# tricks first, memberships last) and the problem as text.
_PROBLEMS = text(
    """
    SELECT 0, 'the session runs as ' || current_user || ', not as the login ' || session_user
     WHERE current_user <> session_user
    UNION ALL
    SELECT 0, session_user || ' has role settings ('
           || pg_catalog.array_to_string(s.setconfig, ', ') || ')'
      FROM pg_catalog.pg_db_role_setting AS s
      JOIN pg_catalog.pg_roles AS r ON r.oid = s.setrole
     WHERE r.rolname = session_user
    UNION ALL
    SELECT 5, 'member of ' || r.rolname
      FROM pg_catalog.pg_roles AS r
     WHERE pg_catalog.pg_has_role(session_user, r.oid, 'MEMBER')
       AND r.rolname NOT IN (session_user, :agent_role)
    UNION ALL
    SELECT 1, r.rolname || ' has '
           || concat_ws(', ',
                        CASE WHEN r.rolsuper THEN 'SUPERUSER' END,
                        CASE WHEN r.rolreplication THEN 'REPLICATION' END,
                        CASE WHEN r.rolbypassrls THEN 'BYPASSRLS' END,
                        CASE WHEN r.rolcreaterole THEN 'CREATEROLE' END,
                        CASE WHEN r.rolcreatedb THEN 'CREATEDB' END)
      FROM pg_catalog.pg_roles AS r
     WHERE r.rolname IN (session_user, :agent_role)
       AND (r.rolsuper OR r.rolreplication OR r.rolbypassrls OR r.rolcreaterole OR r.rolcreatedb)
    UNION ALL
    SELECT 2, r.rolname || ' has a privilege on survey_records'
      FROM pg_catalog.pg_roles AS r
     WHERE r.rolname IN (session_user, :agent_role)
       AND (pg_catalog.has_table_privilege(
                r.oid, pg_catalog.to_regclass('public.survey_records'),
                'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER')
            OR pg_catalog.has_any_column_privilege(
                r.oid, pg_catalog.to_regclass('public.survey_records'),
                'SELECT, INSERT, UPDATE, REFERENCES'))
    UNION ALL
    SELECT 3, r.rolname || ' can create objects in schema ' || n.nspname
      FROM pg_catalog.pg_roles AS r, pg_catalog.pg_namespace AS n
     WHERE r.rolname IN (session_user, :agent_role)
       AND pg_catalog.has_schema_privilege(r.oid, n.oid, 'CREATE')
    UNION ALL
    SELECT 3, r.rolname || ' has CREATE on database ' || pg_catalog.current_database()
      FROM pg_catalog.pg_roles AS r
     WHERE r.rolname IN (session_user, :agent_role)
       AND pg_catalog.has_database_privilege(r.oid, pg_catalog.current_database(), 'CREATE')
    UNION ALL
    SELECT 3, r.rolname || ' has TEMPORARY on database ' || d.datname
      FROM pg_catalog.pg_roles AS r, pg_catalog.pg_database AS d
     WHERE r.rolname IN (session_user, :agent_role)
       AND pg_catalog.has_database_privilege(r.oid, d.oid, 'TEMPORARY')
    UNION ALL
    SELECT 4, 'can execute large-object writer ' || w.fn::pg_catalog.regprocedure::text
      FROM (VALUES (pg_catalog.to_regprocedure('pg_catalog.lo_create(oid)')),
                   (pg_catalog.to_regprocedure('pg_catalog.lo_creat(integer)')),
                   (pg_catalog.to_regprocedure('pg_catalog.lo_from_bytea(oid, bytea)')),
                   (pg_catalog.to_regprocedure('pg_catalog.lo_import(text)')),
                   (pg_catalog.to_regprocedure('pg_catalog.lo_import(text, oid)')),
                   (pg_catalog.to_regprocedure('pg_catalog.lo_open(oid, integer)')),
                   (pg_catalog.to_regprocedure('pg_catalog.lo_put(oid, bigint, bytea)'))) AS w (fn)
     WHERE pg_catalog.has_function_privilege(session_user, w.fn, 'EXECUTE')
    UNION ALL
    SELECT 1, 'unexpected overload ' || p.oid::pg_catalog.regprocedure::text
      FROM pg_catalog.pg_proc AS p
      JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
     WHERE n.nspname = 'public'
       AND p.proname IN ('agent_is_pending_unknown', 'classify_unknown')
       AND p.oid IS DISTINCT FROM pg_catalog.to_regprocedure(:predicate)
       AND p.oid IS DISTINCT FROM pg_catalog.to_regprocedure(:classify)
    UNION ALL
    SELECT 1, 'agent_pending_unknown depends on ' || d.refobjid::pg_catalog.regprocedure::text
              || ', not only on ' || :predicate
      FROM pg_catalog.pg_rewrite AS rw
      JOIN pg_catalog.pg_depend AS d
        ON d.classid = 'pg_catalog.pg_rewrite'::pg_catalog.regclass AND d.objid = rw.oid
     WHERE rw.ev_class = pg_catalog.to_regclass('public.agent_pending_unknown')
       AND d.refclassid = 'pg_catalog.pg_proc'::pg_catalog.regclass
       AND d.refobjid IS DISTINCT FROM pg_catalog.to_regprocedure(:predicate)
    UNION ALL
    SELECT 1, 'agent_pending_unknown does not filter through ' || :predicate
     WHERE pg_catalog.to_regclass('public.agent_pending_unknown') IS NOT NULL
       AND NOT EXISTS (
           SELECT 1
             FROM pg_catalog.pg_rewrite AS rw
             JOIN pg_catalog.pg_depend AS d
               ON d.classid = 'pg_catalog.pg_rewrite'::pg_catalog.regclass AND d.objid = rw.oid
            WHERE rw.ev_class = pg_catalog.to_regclass('public.agent_pending_unknown')
              AND d.refobjid = pg_catalog.to_regprocedure(:predicate))
    UNION ALL
    SELECT 4, 'can execute SECURITY DEFINER function ' || p.oid::pg_catalog.regprocedure::text
      FROM pg_catalog.pg_proc AS p
     WHERE p.prosecdef
       AND pg_catalog.has_function_privilege(session_user, p.oid, 'EXECUTE')
       AND p.oid IS DISTINCT FROM pg_catalog.to_regprocedure(:classify)
       AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_depend AS d
                        WHERE d.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
                          AND d.objid = p.oid AND d.deptype = 'e')
    """
)
# The boundary's two entry points must be usable by the login without SET ROLE.
_BOUNDARY_ACCESS = text(
    """
    SELECT pg_catalog.pg_has_role(session_user, r.oid, 'USAGE'),
           coalesce(pg_catalog.has_table_privilege(
               session_user, pg_catalog.to_regclass('public.agent_pending_unknown'), 'SELECT'), false),
           coalesce(pg_catalog.has_function_privilege(
               session_user, pg_catalog.to_regprocedure(:classify), 'EXECUTE'), false)
      FROM pg_catalog.pg_roles AS r
     WHERE r.rolname = :agent_role
    """
)
_COLUMNS = """id, iq_snippet_path, sample_rate, center_freq, peak_power, snippet_duration_ms,
           snippet_rejected, snippet_dropped"""
_FETCH = text(
    f"""
    SELECT {_COLUMNS}
      FROM public.agent_pending_unknown
     WHERE id > :after_id
     ORDER BY id
     LIMIT :limit
    """
)
_NEWEST_WITH_SNIPPET = text(
    f"""
    SELECT {_COLUMNS}
      FROM public.agent_pending_unknown
     WHERE id > :after_id AND iq_snippet_path IS NOT NULL
     ORDER BY id DESC
     LIMIT :limit
    """
)
_BY_IDS = text(
    f"""
    SELECT {_COLUMNS}
      FROM public.agent_pending_unknown
     WHERE id = ANY(CAST(:ids AS integer[]))
     ORDER BY id
    """
)
_TRY_LOCK = text("SELECT pg_catalog.pg_try_advisory_lock(:key)")
_UNLOCK = text("SELECT pg_catalog.pg_advisory_unlock(:key)")
_SUBMIT = text(
    """
    SELECT public.classify_unknown(
        CAST(:record_id AS integer),
        CAST(:status AS text),
        CAST(:tag AS text),
        CAST(:confidence AS double precision),
        CAST(:reasoning AS text))
    """
)


class BoundaryViolation(RuntimeError):
    """The connected role is not confined by the agent database boundary."""


class AgentAlreadyRunning(RuntimeError):
    """Another agent holds the single-instance advisory lock on this database."""


class RecordNotPending(RuntimeError):
    """No pending unknown row has this id any more: for example, a human
    tagged it while the LLM call was in flight. The human's tag stands."""


class SubmitRejected(RuntimeError):
    """The database refused the write's arguments (SQLSTATE class 22, such as
    classify_unknown's 22023): a bug in the agent, not a record judgment."""


class SubmitTimedOut(RuntimeError):
    """The write hit statement_timeout or a lock timeout; the row stays pending."""


@dataclass(frozen=True)
class PendingRecord:
    """One row of public.agent_pending_unknown. Every value is DB data and
    therefore untrusted (the snippet path above all). iq_snippet_path is
    None when ingest rejected the snippet or capture dropped it; the
    matching quality flag, if any, is in snippet_rejected/snippet_dropped.
    malformed says why a value failed parse_pending_row's checks (that
    value is then None): the record is closed out, never analysed."""

    id: int
    iq_snippet_path: str | None
    sample_rate: float | None
    center_freq: float | None
    peak_power: float | None
    snippet_duration_ms: int | None
    snippet_rejected: str | None = None
    snippet_dropped: str | None = None
    malformed: str | None = None


def _number(name: str, raw: str | None, problems: list[str]) -> float | None:
    if raw is None:
        return None
    shown = repr(raw[:32])
    if not _NUMBER.fullmatch(raw):
        problems.append(f"{name} {shown} is not a number")
        return None
    value = float(raw)
    low, high = _BOUNDS[name]
    if not (math.isfinite(value) and low <= value <= high):
        problems.append(f"{name} {shown} is outside [{low:.12g}, {high:.12g}]")
        return None
    return value


def parse_pending_row(row: Sequence) -> PendingRecord:
    """Parse and bound-check one view row (numbers arrive as text). Never
    raises for bad values: they set PendingRecord.malformed instead."""
    record_id, path, sample_rate, center_freq, peak_power, duration, rejected, dropped = row
    problems: list[str] = []
    if path is not None and (len(path) > _MAX_PATH_CHARS or "\x00" in path):
        problems.append(f"iq_snippet_path is longer than {_MAX_PATH_CHARS} characters or holds NUL")
        path = None
    duration_ms = _number("snippet_duration_ms", duration, problems)
    if duration_ms is not None and not duration_ms.is_integer():
        problems.append(f"snippet_duration_ms {duration[:32]!r} is not an integer")
        duration_ms = None
    return PendingRecord(
        id=record_id,
        iq_snippet_path=path,
        sample_rate=_number("sample_rate", sample_rate, problems),
        center_freq=_number("center_freq", center_freq, problems),
        peak_power=_number("peak_power", peak_power, problems),
        snippet_duration_ms=None if duration_ms is None else int(duration_ms),
        snippet_rejected=rejected,
        snippet_dropped=dropped,
        malformed="; ".join(problems) or None,
    )


class AgentGateway:
    def __init__(self, engine: Engine, lock: Connection | None = None) -> None:
        self._engine = engine
        self._lock = lock

    def fetch_pending(self, after_id: int, limit: int) -> list[PendingRecord]:
        """Pending unknown rows with id > after_id, lowest id first."""
        with self._engine.connect() as conn:
            rows = conn.execute(_FETCH, {"after_id": after_id, "limit": limit}).all()
        return [parse_pending_row(row) for row in rows]

    def fetch_by_ids(self, ids: Sequence[int]) -> list[PendingRecord]:
        """Those of `ids` that are still pending, lowest id first: the
        agent's deferred records, retried whatever the cursor."""
        with self._engine.connect() as conn:
            rows = conn.execute(_BY_IDS, {"ids": list(ids)}).all()
        return [parse_pending_row(row) for row in rows]

    def fetch_newest_with_snippet(self, after_id: int, limit: int) -> list[PendingRecord]:
        """The newest pending rows that carry a snippet path, newest first:
        the startup probe of the snippet-store mount."""
        with self._engine.connect() as conn:
            rows = conn.execute(_NEWEST_WITH_SNIPPET, {"after_id": after_id, "limit": limit}).all()
        return [parse_pending_row(row) for row in rows]

    def submit_classification(
        self,
        record_id: int,
        status: ClassificationStatus,
        tag: str | None,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Write the four classification keys through classify_unknown.

        Raises RecordNotPending if the row stopped being pending,
        SubmitTimedOut on a statement or lock timeout, and SubmitRejected
        when the arguments are refused. NUL characters, which PostgreSQL
        text cannot hold, are removed from the reasoning first.
        """
        params = {
            "record_id": record_id,
            "status": status.value,
            "tag": tag,
            "confidence": confidence,
            "reasoning": reasoning.replace("\x00", ""),
        }
        try:
            with self._engine.begin() as conn:
                conn.execute(_SUBMIT, params)
        except DBAPIError as exc:
            driver_error = exc.orig if isinstance(exc.orig, psycopg.Error) else None
            sqlstate = None if driver_error is None else driver_error.sqlstate
            if sqlstate == _NOT_PENDING:
                raise RecordNotPending(f"Record {record_id} is no longer pending") from exc
            if sqlstate in _TIMED_OUT:
                raise SubmitTimedOut(f"Writing record {record_id} timed out ({sqlstate})") from exc
            if isinstance(driver_error, psycopg.DataError):
                raise SubmitRejected(f"The database rejected the write for record {record_id}: {driver_error}") from exc
            raise

    def close(self) -> None:
        if self._lock is not None:
            self._lock.execute(_UNLOCK, {"key": _SINGLETON_LOCK})
            self._lock.close()
        self._engine.dispose()


def verify_boundary(engine: Engine, agent_role: str) -> None:
    """Raise BoundaryViolation unless the connected role is confined."""
    params = {"agent_role": agent_role, "classify": _CLASSIFY, "predicate": _PREDICATE}
    with engine.connect() as conn:
        problems = [problem for _, problem in sorted(conn.execute(_PROBLEMS, params).tuples().all())]
        access = conn.execute(_BOUNDARY_ACCESS, params).first()
    if problems:
        raise BoundaryViolation(
            "Refusing to run: the database role is not confined to the agent boundary: "
            + "; ".join(problems[:10])
            + f". Connect as a login role that is only IN ROLE {agent_role}, with no role "
            "settings and no TEMPORARY on any database (agent/README.md), and re-run "
            "sdr-agent-boundary."
        )
    if access is None or not all(access):
        raise BoundaryViolation(
            f"Refusing to run: the database role does not inherit {agent_role}, or the "
            "boundary (agent_pending_unknown, classify_unknown) is not installed. "
            "Run sdr-agent-boundary as an administrator first."
        )


def connect_gateway(
    database_url: str,
    agent_role: str = DEFAULT_AGENT_ROLE,
    statement_timeout_s: float = DEFAULT_STATEMENT_TIMEOUT_S,
) -> AgentGateway:
    """Connect, verify the boundary, and return the gateway. PostgreSQL only."""
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("The agent needs a PostgreSQL URL; its boundary only exists there")
    engine = create_engine(
        url.set(drivername="postgresql+psycopg"),
        connect_args={"options": f"-c statement_timeout={round(statement_timeout_s * 1000)}"},
        # One session for the lock, one for the work: CONNECTION LIMIT 2.
        pool_size=2,
        max_overflow=0,
    )
    try:
        verify_boundary(engine, agent_role)
        lock = _take_singleton_lock(engine)
    except BaseException:
        engine.dispose()
        raise
    return AgentGateway(engine, lock)


def _take_singleton_lock(engine: Engine) -> Connection:
    """A session advisory lock held until close(): a second agent on the same
    database would double the spend and race for the same records."""
    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        if not conn.execute(_TRY_LOCK, {"key": _SINGLETON_LOCK}).scalar_one():
            raise AgentAlreadyRunning(
                "Another sdr-agent is running against this database (it holds advisory lock "
                f"{_SINGLETON_LOCK}); refusing to start a second one."
            )
    except BaseException:
        conn.close()
        raise
    return conn
