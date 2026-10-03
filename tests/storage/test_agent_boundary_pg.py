# tests/storage/test_agent_boundary_pg.py
"""The agent's database boundary on real PostgreSQL (never SQLite).

Skipped unless SURVEYTOOL_TEST_PG_URL is set to a superuser URL on a
THROWAWAY cluster (never the compose dev volume). Each run creates its own
database and uniquely suffixed roles -- roles are cluster-global -- and drops
all of them afterwards.
"""

import json
import logging
import os
import secrets
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import psycopg
import pytest
from anthropic.types import Message
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.pool import NullPool

from agent.band_table import load_band_table
from agent.db_gateway import (
    AgentAlreadyRunning,
    BoundaryViolation,
    PendingRecord,
    RecordNotPending,
    SubmitRejected,
    SubmitTimedOut,
    connect_gateway,
)
from agent import service as agent_service
from agent.service import AgentSettings, ClassificationAgent, SystemicFault
from capture.unknown.snippet_writer import write_sigmf_snippet
from dsp import synthetic
from schema.records import (
    ClassificationStatus,
    Identifier,
    Metadata,
    Modality,
    Signal,
    UnifiedRecord,
)
from storage.agent_boundary import install_agent_boundary
from storage.agent_boundary import main as boundary_main
from storage.db import init_db, make_session_factory
from storage.repository import save_record
from storage.snippet_store import LocalSnippetStore

ADMIN_URL = os.environ.get("SURVEYTOOL_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    not ADMIN_URL, reason="SURVEYTOOL_TEST_PG_URL (throwaway PostgreSQL admin URL) not set"
)

INSUFFICIENT_PRIVILEGE = "42501"
INVALID_PARAMETER = "22023"


@dataclass(frozen=True)
class Boundary:
    root_url: URL  # superuser, maintenance database
    admin_url: URL  # superuser, this run's database
    database: str
    agent_url: URL  # a login role IN ROLE agent_role, nothing else
    agent_role: str
    owner_role: str
    suffix: str
    password: str


def _engine(url: URL, **kwargs) -> Engine:
    return create_engine(url, poolclass=NullPool, **kwargs)


def _url(url: URL) -> str:
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def boundary():
    suffix = secrets.token_hex(4)
    database = f"sdr_agent_test_{suffix}"
    agent_role, owner_role = f"sdr_agent_{suffix}", f"sdr_owner_{suffix}"
    login = f"sdr_login_{suffix}"
    password = secrets.token_hex(16)
    root_url = make_url(ADMIN_URL).set(drivername="postgresql+psycopg")
    admin_url = root_url.set(database=database)
    root = _engine(root_url, isolation_level="AUTOCOMMIT")
    with root.connect() as conn:
        # A default cluster: PUBLIC keeps TEMPORARY on postgres and the
        # templates, which the self-check only warns about.
        conn.exec_driver_sql(f"CREATE DATABASE {database}")
    try:
        admin = _engine(admin_url)
        init_db(admin)
        install_agent_boundary(admin, agent_role, owner_role)
        install_agent_boundary(admin, agent_role, owner_role)  # idempotent
        admin.dispose()
        with root.connect() as conn:
            conn.exec_driver_sql(
                f"CREATE ROLE {login} LOGIN PASSWORD '{password}' IN ROLE {agent_role}"
            )
        yield Boundary(
            root_url=root_url,
            admin_url=admin_url,
            database=database,
            agent_url=admin_url.set(username=login, password=password),
            agent_role=agent_role,
            owner_role=owner_role,
            suffix=suffix,
            password=password,
        )
    finally:
        with root.connect() as conn:
            conn.exec_driver_sql(f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
            leftovers = conn.execute(
                text("SELECT rolname FROM pg_catalog.pg_roles WHERE rolname LIKE :pattern"),
                {"pattern": f"sdr\\_%\\_{suffix}"},
            ).scalars().all()
            for role in sorted(leftovers, key=lambda name: name in (agent_role, owner_role)):
                conn.exec_driver_sql(f"DROP ROLE {role}")
            remaining = conn.execute(
                text("SELECT count(*) FROM pg_catalog.pg_roles WHERE rolname LIKE :pattern"),
                {"pattern": f"sdr\\_%\\_{suffix}"},
            ).scalar_one()
        root.dispose()
        assert remaining == 0


def _insert(
    boundary: Boundary,
    modality: Modality,
    status: ClassificationStatus | None,
    path: str | None = "/srv/snippets/a.sigmf-data",
    sample_rate: float = 100_000.0,
    flags: dict | None = None,
) -> int:
    record = UnifiedRecord(
        timestamp=datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc),
        lat=47.6062,
        lon=-122.3321,
        survey_id="survey-secret",
        operator_id="operator-secret",
        modality=modality,
        identifier=Identifier(center_freq=915e6, bandwidth_estimate=20_000.0),
        signal=Signal(rssi=-20.0, snr=20.0, peak_power=-17.5),
        metadata=Metadata(
            quality_flags={"power_units": "dBFS", **(flags or {})},
            iq_snippet_path=path,
            snippet_duration_ms=1000,
            sample_rate=sample_rate,
            classification_status=status,
            tag="human-tag" if status is ClassificationStatus.MANUALLY_TAGGED else None,
        ),
    )
    admin = _engine(boundary.admin_url)
    try:
        with make_session_factory(admin)() as session:
            return save_record(session, record).id
    finally:
        admin.dispose()


def _admin_scalar(boundary: Boundary, sql: str, **params):
    admin = _engine(boundary.admin_url)
    try:
        with admin.begin() as conn:
            return conn.execute(text(sql), params).scalar()
    finally:
        admin.dispose()


def _metadata(boundary: Boundary, record_id: int) -> dict:
    return json.loads(
        _admin_scalar(
            boundary, "SELECT metadata::text FROM survey_records WHERE id = :id", id=record_id
        )
    )


def _watermark(boundary: Boundary) -> int:
    return _admin_scalar(boundary, "SELECT coalesce(max(id), 0) FROM survey_records")


def _sqlstate(excinfo) -> str | None:
    return getattr(excinfo.value.orig, "sqlstate", None)


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * FROM public.survey_records",
        "INSERT INTO public.survey_records (timestamp, lat, lon, survey_id, operator_id, "
        "modality, identifier, signal, metadata) VALUES (now(), 0, 0, 's', 'o', 'unknown', "
        "'{}', '{}', '{}')",
        "UPDATE public.survey_records SET metadata = '{}'",
        "DELETE FROM public.survey_records",
        "TRUNCATE public.survey_records",
    ],
)
def test_agent_role_has_no_direct_access_to_survey_records(boundary, statement):
    agent = _engine(boundary.agent_url)
    try:
        with pytest.raises(DBAPIError) as excinfo, agent.begin() as conn:
            conn.execute(text(statement))
        assert _sqlstate(excinfo) == INSUFFICIENT_PRIVILEGE
    finally:
        agent.dispose()


def test_view_shows_only_pending_unknown_rows(boundary):
    watermark = _watermark(boundary)
    pending = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED)
    no_status = _insert(boundary, Modality.UNKNOWN, None)
    rejected = _insert(boundary, Modality.UNKNOWN, None, path=None, flags={"snippet_rejected": "bad_size"})
    dropped = _insert(boundary, Modality.UNKNOWN, None, path=None, flags={"snippet_dropped": "low_disk"})
    _insert(boundary, Modality.WIFI, None)
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED)
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.AUTO_CLASSIFIED)
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.NEEDS_REVIEW)

    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        rows = gateway.fetch_pending(after_id=watermark, limit=100)
        assert [row.id for row in rows] == [pending, no_status, rejected, dropped]
        assert rows[0] == PendingRecord(
            id=pending,
            iq_snippet_path="/srv/snippets/a.sigmf-data",
            sample_rate=100_000.0,
            center_freq=915e6,
            peak_power=-17.5,
            snippet_duration_ms=1000,
        )
        # Snippet-less rows stay visible, flagged, so the agent can close them out.
        assert (rows[2].iq_snippet_path, rows[2].snippet_rejected, rows[2].snippet_dropped) == (None, "bad_size", None)
        assert (rows[3].iq_snippet_path, rows[3].snippet_rejected, rows[3].snippet_dropped) == (None, None, "low_disk")
        assert [row.id for row in gateway.fetch_pending(after_id=rejected, limit=100)] == [dropped]
        assert [row.id for row in gateway.fetch_pending(after_id=watermark, limit=1)] == [pending]
    finally:
        gateway.close()


def test_view_projects_only_the_agent_columns(boundary):
    agent = _engine(boundary.agent_url)
    try:
        with agent.connect() as conn:
            columns = list(conn.execute(text("SELECT * FROM public.agent_pending_unknown LIMIT 0")).keys())
        assert columns == [
            "id",
            "iq_snippet_path",
            "sample_rate",
            "center_freq",
            "peak_power",
            "snippet_duration_ms",
            "snippet_rejected",
            "snippet_dropped",
        ]
    finally:
        agent.dispose()


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE FUNCTION public.probe(t text) RETURNS boolean LANGUAGE sql RETURN true",
        "CREATE FUNCTION pg_temp.probe(t text) RETURNS boolean LANGUAGE sql RETURN true",
        "CREATE TEMPORARY TABLE probe (x integer)",
        "CREATE SCHEMA probe",
    ],
)
def test_agent_cannot_create_objects_to_probe_the_view(boundary, statement):
    agent = _engine(boundary.agent_url)
    try:
        with pytest.raises(DBAPIError) as excinfo, agent.begin() as conn:
            conn.execute(text(statement))
        assert _sqlstate(excinfo) == INSUFFICIENT_PRIVILEGE
    finally:
        agent.dispose()


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT lo_create(0)",
        "SELECT lo_creat(-1)",
        "SELECT lo_from_bytea(0, 'x'::bytea)",
        "SELECT lo_open(1, 131072)",
        "SELECT lo_put(1, 0, 'x'::bytea)",
        "SELECT lo_import('/etc/hostname')",
        "SELECT lo_import('/etc/hostname', 0)",
    ],
)
def test_agent_cannot_write_large_objects(boundary, statement):
    """Large objects are a write path that needs no table privilege:
    PUBLIC can execute these functions until the boundary revokes them."""
    agent = _engine(boundary.agent_url)
    try:
        with pytest.raises(DBAPIError) as excinfo, agent.begin() as conn:
            conn.execute(text(statement))
        assert _sqlstate(excinfo) == INSUFFICIENT_PRIVILEGE
    finally:
        agent.dispose()


def test_leaky_function_sees_only_rows_the_view_shows(boundary):
    """The agent cannot create a function (previous test), so an
    administrator plants the classic leaky probe for it: near-zero cost, so
    the planner would run it before the view's own filter, and it reports
    every value it is passed. The manually tagged unknown row survives the
    view's index condition (modality = 'unknown'), so without
    security_barrier its path is leaked here; with it, it never is."""
    watermark = _watermark(boundary)
    visible = _insert(boundary, Modality.UNKNOWN, None, path="/srv/snippets/visible.sigmf-data")
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED, path="HIDDEN-manual")
    _insert(boundary, Modality.WIFI, None, path="HIDDEN-wifi")
    leak = f"public.leak_{boundary.suffix}"
    admin = _engine(boundary.admin_url)
    agent = _engine(boundary.agent_url)
    try:
        with admin.begin() as conn:
            conn.exec_driver_sql(
                f"CREATE FUNCTION {leak}(t text) RETURNS boolean LANGUAGE plpgsql "
                "COST 0.0000001 AS $$BEGIN RAISE NOTICE USING MESSAGE = 'saw ' || t; "
                "RETURN true; END$$"
            )
            conn.exec_driver_sql(f"GRANT EXECUTE ON FUNCTION {leak}(text) TO {boundary.agent_role}")
        notices: list[str] = []
        with agent.connect() as conn:
            conn.connection.driver_connection.add_notice_handler(
                lambda diagnostic: notices.append(diagnostic.message_primary)
            )
            ids = conn.execute(
                text(
                    f"SELECT id FROM public.agent_pending_unknown "
                    f"WHERE id > :watermark AND {leak}(iq_snippet_path)"
                ),
                {"watermark": watermark},
            ).scalars().all()
        assert ids == [visible]
        assert notices == ["saw /srv/snippets/visible.sigmf-data"]
    finally:
        with admin.begin() as conn:
            conn.exec_driver_sql(f"DROP FUNCTION IF EXISTS {leak}(text)")
        admin.dispose()
        agent.dispose()


def test_classify_unknown_changes_only_the_four_classification_keys(boundary):
    record_id = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED)
    before = _metadata(boundary, record_id)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        gateway.submit_classification(
            record_id, ClassificationStatus.AUTO_CLASSIFIED, "ism_902_928:lora", 0.9, "Because."
        )
    finally:
        gateway.close()
    # Compared as parsed JSON: the jsonb round trip reorders keys.
    assert _metadata(boundary, record_id) == {
        **before,
        "classification_status": "auto_classified",
        "tag": "ism_902_928:lora",
        "confidence": 0.9,
        "reasoning": "Because.",
    }


def test_needs_review_may_carry_a_null_tag(boundary):
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        gateway.submit_classification(
            record_id, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "Model output failed validation."
        )
    finally:
        gateway.close()
    metadata = _metadata(boundary, record_id)
    assert metadata["classification_status"] == "needs_review"
    assert metadata["tag"] is None and metadata["confidence"] == 0.0


@pytest.mark.parametrize(
    "modality, status",
    [
        (Modality.WIFI, None),
        (Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED),
        (Modality.UNKNOWN, ClassificationStatus.AUTO_CLASSIFIED),
    ],
)
def test_classify_unknown_refuses_rows_that_are_not_pending_unknown(boundary, modality, status):
    record_id = _insert(boundary, modality, status)
    before = _metadata(boundary, record_id)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        with pytest.raises(RecordNotPending):
            gateway.submit_classification(
                record_id, ClassificationStatus.AUTO_CLASSIFIED, "x", 0.9, "r"
            )
    finally:
        gateway.close()
    assert _metadata(boundary, record_id) == before


@pytest.mark.parametrize(
    "status, tag, confidence, reasoning",
    [
        ("manually_tagged", "x", 0.5, "r"),
        ("unclassified", "x", 0.5, "r"),
        ("bogus", "x", 0.5, "r"),
        ("auto_classified", "x", 1.5, "r"),
        ("auto_classified", "x", -0.1, "r"),
        ("auto_classified", "x", float("nan"), "r"),
        ("auto_classified", "x", None, "r"),
        ("auto_classified", "Bad", 0.5, "r"),
        ("auto_classified", "abc\n", 0.5, "r"),
        ("auto_classified", "a" * 65, 0.5, "r"),
        ("auto_classified", "-leading-dash", 0.5, "r"),
        ("auto_classified", None, 0.5, "r"),
        ("needs_review", None, 0.5, "x" * 4001),
        ("needs_review", None, 0.5, None),
    ],
)
def test_classify_unknown_rejects_bad_arguments(boundary, status, tag, confidence, reasoning):
    """Called directly, bypassing the gateway's types: the function itself
    is the boundary."""
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    before = _metadata(boundary, record_id)
    agent = _engine(boundary.agent_url)
    try:
        with pytest.raises(DBAPIError) as excinfo, agent.begin() as conn:
            conn.execute(
                text(
                    "SELECT public.classify_unknown(CAST(:id AS integer), CAST(:status AS text), "
                    "CAST(:tag AS text), CAST(:confidence AS double precision), "
                    "CAST(:reasoning AS text))"
                ),
                {"id": record_id, "status": status, "tag": tag, "confidence": confidence, "reasoning": reasoning},
            )
        assert _sqlstate(excinfo) == INVALID_PARAMETER
    finally:
        agent.dispose()
    assert _metadata(boundary, record_id) == before


def test_human_tag_between_fetch_and_submit_wins(boundary):
    watermark = _watermark(boundary)
    record_id = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        assert [row.id for row in gateway.fetch_pending(watermark, 10)] == [record_id]
        _admin_scalar(
            boundary,
            "UPDATE survey_records SET metadata = (metadata::jsonb || "
            "'{\"classification_status\": \"manually_tagged\", \"tag\": \"human\"}')::json "
            "WHERE id = :id RETURNING id",
            id=record_id,
        )
        with pytest.raises(RecordNotPending):
            gateway.submit_classification(
                record_id, ClassificationStatus.AUTO_CLASSIFIED, "agent", 0.95, "r"
            )
    finally:
        gateway.close()
    metadata = _metadata(boundary, record_id)
    assert (metadata["classification_status"], metadata["tag"]) == ("manually_tagged", "human")


def test_human_tag_committed_while_agent_waits_on_the_row_lock_wins(boundary):
    """The real race: the human's UPDATE holds the row lock, the agent's
    classify_unknown blocks on it, then the human commits. Under READ
    COMMITTED PostgreSQL re-checks the agent's WHERE clause against the
    committed row, so the pending predicate fails and nothing is overwritten."""
    record_id = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    admin = _engine(boundary.admin_url)
    outcome: list[BaseException | None] = []

    def submit() -> None:
        try:
            gateway.submit_classification(
                record_id, ClassificationStatus.AUTO_CLASSIFIED, "agent", 0.95, "r"
            )
            outcome.append(None)
        except BaseException as exc:  # handed to the main thread
            outcome.append(exc)

    try:
        with admin.connect() as human:
            human.execute(
                text(
                    "UPDATE survey_records SET metadata = (metadata::jsonb || "
                    "'{\"classification_status\": \"manually_tagged\", \"tag\": \"human\"}')::json "
                    "WHERE id = :id"
                ),
                {"id": record_id},
            )
            agent_thread = threading.Thread(target=submit, daemon=True)
            agent_thread.start()
            deadline = time.monotonic() + 10
            while not _admin_scalar(
                boundary,
                "SELECT count(*) FROM pg_catalog.pg_stat_activity "
                "WHERE datname = current_database() "
                "AND cardinality(pg_catalog.pg_blocking_pids(pid)) > 0",
            ):
                assert time.monotonic() < deadline, "agent never blocked on the row lock"
                time.sleep(0.01)
            human.commit()
        agent_thread.join(10)
        assert not agent_thread.is_alive()
    finally:
        gateway.close()
        admin.dispose()
    assert len(outcome) == 1 and isinstance(outcome[0], RecordNotPending)
    metadata = _metadata(boundary, record_id)
    assert (metadata["classification_status"], metadata["tag"]) == ("manually_tagged", "human")


def test_a_reinstall_drops_grants_added_to_the_boundary_objects(boundary):
    """CREATE OR REPLACE keeps an object's ACL. A grant someone added to the
    view or the functions must not survive the next install."""
    stranger = f"sdr_stranger_grants_{boundary.suffix}"
    classify = "public.classify_unknown(integer, text, text, double precision, text)"
    pending = "public.agent_is_pending_unknown(text, json)"
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.exec_driver_sql(f"CREATE ROLE {stranger} NOLOGIN")
            conn.exec_driver_sql(f"GRANT SELECT ON public.agent_pending_unknown TO {stranger}")
            conn.exec_driver_sql(f"GRANT EXECUTE ON FUNCTION {classify} TO {stranger}")
            conn.exec_driver_sql(f"GRANT EXECUTE ON FUNCTION {pending} TO PUBLIC")
        install_agent_boundary(admin, boundary.agent_role, boundary.owner_role)
        with admin.connect() as conn:
            assert conn.execute(
                text(
                    "SELECT has_table_privilege(:r, 'public.agent_pending_unknown', 'SELECT'),"
                    " has_function_privilege(:r, :classify, 'EXECUTE'),"
                    " has_function_privilege(:r, :pending, 'EXECUTE')"
                ),
                {"r": stranger, "classify": classify, "pending": pending},
            ).one() == (False, False, False)
    finally:
        with admin.connect() as conn:
            conn.exec_driver_sql(f"DROP OWNED BY {stranger}")
            conn.exec_driver_sql(f"DROP ROLE IF EXISTS {stranger}")
        admin.dispose()


@pytest.mark.parametrize(
    "setup, as_owner, problem",
    [
        (["CREATE ROLE {role} LOGIN"], False, "can log in"),
        (["CREATE ROLE {role} NOLOGIN", "CREATE TABLE public.sdr_owned_{suffix} (x integer)",
          "ALTER TABLE public.sdr_owned_{suffix} OWNER TO {role}"], True, "owns objects"),
    ],
)
def test_the_installer_refuses_an_existing_role_that_could_widen_the_boundary(boundary, setup, as_owner, problem):
    """A role that can log in, or that already owns something, must not
    become the agent or owner role: the boundary would inherit its powers."""
    role = f"sdr_existing_{boundary.suffix}"
    names = {"role": role, "suffix": boundary.suffix}
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            for statement in setup:
                conn.exec_driver_sql(statement.format(**names))
        roles = (boundary.agent_role, role) if as_owner else (role, boundary.owner_role)
        with pytest.raises(psycopg.errors.InvalidParameterValue, match=f"{role} {problem}"):
            install_agent_boundary(admin, *roles)
    finally:
        with admin.connect() as conn:
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS public.sdr_owned_{boundary.suffix}")
            conn.exec_driver_sql(f"DROP ROLE IF EXISTS {role}")
        admin.dispose()


PLANT = (
    "CREATE FUNCTION public.agent_is_pending_unknown(p_modality varchar, p_metadata json)"
    " RETURNS boolean LANGUAGE sql IMMUTABLE RETURN true"
)


def test_a_planted_predicate_overload_cannot_widen_the_view(boundary):
    """The reviewer's plant: modality is varchar, so a (varchar, json)
    overload -- created by anyone with CREATE on public, as on pre-15
    clusters -- was the exact match. The view bound to it, showed every row,
    and the agent overwrote a manually_tagged one. The install now drops
    every overload by name, and the calls cast to the exact signature."""
    watermark = _watermark(boundary)
    tagged = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED)
    pending = _insert(boundary, Modality.UNKNOWN, None)
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.exec_driver_sql(PLANT)
        install_agent_boundary(admin, boundary.agent_role, boundary.owner_role)
        with admin.connect() as conn:
            overloads = conn.execute(
                text("SELECT count(*) FROM pg_catalog.pg_proc WHERE proname = 'agent_is_pending_unknown'")
            ).scalar_one()
        assert overloads == 1
        gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
        try:
            assert [r.id for r in gateway.fetch_pending(watermark, 100)] == [pending]
            with pytest.raises(RecordNotPending):
                gateway.submit_classification(tagged, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "x")
        finally:
            gateway.close()
        assert _metadata(boundary, tagged)["classification_status"] == "manually_tagged"
    finally:
        with admin.connect() as conn:
            conn.exec_driver_sql("DROP FUNCTION IF EXISTS public.agent_is_pending_unknown(varchar, json)")
        admin.dispose()


def test_self_check_refuses_an_overload_planted_after_the_install(boundary):
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.exec_driver_sql(PLANT)
        with pytest.raises(BoundaryViolation, match=r"unexpected overload agent_is_pending_unknown\(character varying,json\)"):
            connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    finally:
        with admin.connect() as conn:
            conn.exec_driver_sql("DROP FUNCTION IF EXISTS public.agent_is_pending_unknown(varchar, json)")
        admin.dispose()


def test_self_check_refuses_a_view_bound_to_another_function(boundary):
    """The view must depend on exactly the installed predicate."""
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.exec_driver_sql(
                "CREATE FUNCTION public.sdr_always_true(p text) RETURNS boolean LANGUAGE sql IMMUTABLE RETURN true"
            )
            conn.exec_driver_sql(
                "CREATE OR REPLACE VIEW public.agent_pending_unknown WITH (security_barrier = true) AS "
                "SELECT r.id, r.metadata ->> 'iq_snippet_path' AS iq_snippet_path, "
                "r.metadata ->> 'sample_rate' AS sample_rate, r.identifier ->> 'center_freq' AS center_freq, "
                "r.signal ->> 'peak_power' AS peak_power, r.metadata ->> 'snippet_duration_ms' AS snippet_duration_ms, "
                "r.metadata -> 'quality_flags' ->> 'snippet_rejected' AS snippet_rejected, "
                "r.metadata -> 'quality_flags' ->> 'snippet_dropped' AS snippet_dropped "
                "FROM public.survey_records AS r WHERE public.sdr_always_true(r.modality::text)"
            )
        with pytest.raises(BoundaryViolation, match="agent_pending_unknown depends on sdr_always_true"):
            connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    finally:
        install_agent_boundary(admin, boundary.agent_role, boundary.owner_role)
        with admin.connect() as conn:
            conn.exec_driver_sql("DROP FUNCTION IF EXISTS public.sdr_always_true(text)")
        admin.dispose()


def test_the_installer_refuses_an_owner_that_belongs_to_another_role(boundary):
    """classify_unknown runs as the owner (SECURITY DEFINER): an owner that
    inherits a broader role would lend it to every write."""
    owner = f"sdr_member_owner_{boundary.suffix}"
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.exec_driver_sql(f"CREATE ROLE {owner} NOLOGIN")
            conn.exec_driver_sql(f"GRANT pg_write_all_data TO {owner}")
        with pytest.raises(psycopg.errors.InvalidParameterValue, match=f"{owner} is a member of pg_write_all_data"):
            install_agent_boundary(admin, boundary.agent_role, owner)
    finally:
        with admin.connect() as conn:
            conn.exec_driver_sql(f"DROP ROLE IF EXISTS {owner}")
        admin.dispose()


def test_the_installer_makes_the_owner_noinherit(boundary):
    assert _admin_scalar(
        boundary, "SELECT rolinherit FROM pg_catalog.pg_roles WHERE rolname = :r", r=boundary.owner_role
    ) is False


@pytest.mark.parametrize(
    "grant, revoke, problem",
    [
        ("GRANT DELETE ON public.survey_records TO {owner}", "REVOKE DELETE ON public.survey_records FROM {owner}",
         "the definer {owner} can do more on survey_records than SELECT and UPDATE (metadata)"),
        ("GRANT UPDATE (modality) ON public.survey_records TO {owner}",
         "REVOKE UPDATE (modality) ON public.survey_records FROM {owner}",
         "the definer {owner} can do more on survey_records than SELECT and UPDATE (metadata)"),
        ("GRANT pg_read_all_data TO {owner}", "REVOKE pg_read_all_data FROM {owner}",
         "the definer {owner} is a member of pg_read_all_data"),
    ],
)
def test_self_check_refuses_a_definer_with_more_than_it_needs(boundary, grant, revoke, problem):
    names = {"owner": boundary.owner_role}
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.exec_driver_sql(grant.format(**names))
        with pytest.raises(BoundaryViolation, match="not confined") as excinfo:
            connect_gateway(_url(boundary.agent_url), boundary.agent_role)
        assert problem.format(**names) in str(excinfo.value)
    finally:
        with admin.connect() as conn:
            conn.exec_driver_sql(revoke.format(**names))
        admin.dispose()


_ACL_SNAPSHOT = """
    SELECT (SELECT datacl::text FROM pg_database WHERE datname = current_database()),
           (SELECT nspacl::text FROM pg_namespace WHERE nspname = 'public'),
           (SELECT relacl::text FROM pg_class WHERE oid = to_regclass('public.survey_records')),
           (SELECT string_agg(proname || coalesce(proacl::text, '-'), ',' ORDER BY oid)
              FROM pg_proc WHERE starts_with(proname, 'lo_') OR starts_with(proname, 'agent_')
                                 OR proname = 'classify_unknown'),
           (SELECT string_agg(rolname, ',' ORDER BY rolname) FROM pg_roles),
           (SELECT count(*) FROM pg_class WHERE relname = 'agent_pending_unknown')
"""


@pytest.fixture
def fresh_database(boundary):
    """A database the boundary was never installed in, with survey_records."""
    name = f"sdr_preview_{boundary.suffix}"
    root = _engine(boundary.root_url, isolation_level="AUTOCOMMIT")
    with root.connect() as conn:
        conn.exec_driver_sql(f"CREATE DATABASE {name}")
    url = boundary.root_url.set(database=name)
    admin = _engine(url)
    init_db(admin)
    admin.dispose()
    try:
        yield url
    finally:
        with root.connect() as conn:
            conn.exec_driver_sql(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        root.dispose()


def test_the_installer_previews_by_default_and_changes_nothing(boundary, fresh_database, monkeypatch, capsys):
    """Codex H3: revoking PUBLIC's privileges can break other roles that
    relied on them. By default the installer only lists what --apply would
    revoke, and who relies on it, and the database is left as it was."""
    relying = f"sdr_relies_{boundary.suffix}"
    agent_role, owner_role = f"sdr_preview_agent_{boundary.suffix}", f"sdr_preview_owner_{boundary.suffix}"
    root = _engine(boundary.root_url, isolation_level="AUTOCOMMIT")
    admin = _engine(fresh_database, isolation_level="AUTOCOMMIT")
    try:
        with root.connect() as conn:
            conn.exec_driver_sql(f"CREATE ROLE {relying} LOGIN")
        with admin.connect() as conn:
            conn.exec_driver_sql("GRANT SELECT ON public.survey_records TO PUBLIC")
            before = conn.exec_driver_sql(_ACL_SNAPSHOT).one()
        monkeypatch.setenv("SURVEYTOOL_ADMIN_DATABASE_URL", _url(fresh_database))
        boundary_main(["--agent-role", agent_role, "--owner-role", owner_role])
        preview = capsys.readouterr().out
        with admin.connect() as conn:
            assert conn.exec_driver_sql(_ACL_SNAPSHOT).one() == before
        assert "Nothing was changed; re-run with --apply" in preview
        assert f"REVOKE TEMPORARY ON DATABASE {fresh_database.database} FROM PUBLIC" in preview
        assert "REVOKE EXECUTE ON FUNCTION lo_create(oid) FROM PUBLIC" in preview
        (select_line,) = [line for line in preview.splitlines() if "REVOKE SELECT ON TABLE public.survey_records" in line]
        assert "relying on PUBLIC for it:" in select_line and relying in select_line
        boundary_main(["--agent-role", agent_role, "--owner-role", owner_role, "--apply"])
        with admin.connect() as conn:
            assert conn.exec_driver_sql(
                "SELECT has_table_privilege('public', 'public.survey_records', 'SELECT')"
            ).scalar_one() is False
    finally:
        admin.dispose()
        with root.connect() as conn:
            conn.exec_driver_sql(f"DROP DATABASE IF EXISTS {fresh_database.database} WITH (FORCE)")
            for role in (relying, agent_role, owner_role):
                conn.exec_driver_sql(f"DROP ROLE IF EXISTS {role}")
        root.dispose()


def test_self_check_rejects_a_superuser_url(boundary):
    with pytest.raises(BoundaryViolation, match="not confined.*has SUPERUSER"):
        connect_gateway(_url(boundary.admin_url), boundary.agent_role)


# (setup statements run as the administrator, extra cleanup, expected problem).
# {login} is a fresh login role IN ROLE the agent role; every case must be refused.
BYPASSES = {
    "owner role without inherit": (["GRANT {owner_role} TO {login} WITH INHERIT FALSE"], [], "member of {owner_role}"),
    "column privilege": (["GRANT UPDATE (metadata) ON public.survey_records TO {login}"], [], "privilege on survey_records"),
    "table privilege": (["GRANT SELECT ON public.survey_records TO {login}"], [], "privilege on survey_records"),
    "createrole": (["ALTER ROLE {login} CREATEROLE"], [], "has CREATEROLE"),
    "bypassrls": (["ALTER ROLE {login} BYPASSRLS"], [], "has BYPASSRLS"),
    "replication": (["ALTER ROLE {login} REPLICATION"], [], "has REPLICATION"),
    "createdb": (["ALTER ROLE {login} CREATEDB"], [], "has CREATEDB"),
    "agent role createdb": (["ALTER ROLE {agent_role} CREATEDB"], ["ALTER ROLE {agent_role} NOCREATEDB"], "{agent_role} has CREATEDB"),
    "pg_write_all_data": (["GRANT pg_write_all_data TO {login}"], [], "member of pg_write_all_data"),
    "pg_read_all_data": (["GRANT pg_read_all_data TO {login}"], [], "member of pg_read_all_data"),
    "pg_execute_server_program": (["GRANT pg_execute_server_program TO {login}"], [], "member of pg_execute_server_program"),
    "pg_write_server_files": (["GRANT pg_write_server_files TO {login}"], [], "member of pg_write_server_files"),
    "pg_read_server_files": (["GRANT pg_read_server_files TO {login}"], [], "member of pg_read_server_files"),
    "pg_signal_backend": (["GRANT pg_signal_backend TO {login}"], [], "member of pg_signal_backend"),
    "pg_maintain": (["GRANT pg_maintain TO {login}"], [], "member of pg_maintain"),
    "temporary through a helper without inherit": (
        [
            "CREATE ROLE sdr_helper_{suffix} NOLOGIN",
            "GRANT TEMPORARY ON DATABASE {database} TO sdr_helper_{suffix}",
            "GRANT sdr_helper_{suffix} TO {login} WITH INHERIT FALSE",
        ],
        ["DROP OWNED BY sdr_helper_{suffix}", "DROP ROLE IF EXISTS sdr_helper_{suffix}"],
        "member of sdr_helper_{suffix}",
    ),
    "temporary": (["GRANT TEMPORARY ON DATABASE {database} TO {login}"], [], "{login} has TEMPORARY on database {database}"),
    "create on the database": (["GRANT CREATE ON DATABASE {database} TO {login}"], [], "{login} has CREATE on database {database}"),
    # The reviewer's bypass: a superuser login whose role setting switches it
    # to the agent role at connect, so current_user looks confined.
    "superuser switched to the agent role": (
        ["ALTER ROLE {login} SUPERUSER", "ALTER ROLE {login} SET role = {agent_role}"],
        [],
        "{login} has SUPERUSER",
    ),
    "session switched to the agent role": (
        ["ALTER ROLE {login} IN DATABASE {database} SET role = {agent_role}"],
        [],
        "the session runs as {agent_role}, not as the login {login}",
    ),
    "role setting": (["ALTER ROLE {login} SET work_mem = '64MB'"], [], "{login} has role settings (work_mem=64MB)"),
    "create on public": (["GRANT CREATE ON SCHEMA public TO {login}"], [], "can create objects in schema public"),
    "large-object writer": (
        ["GRANT EXECUTE ON FUNCTION lo_from_bytea(oid, bytea) TO {login}"],
        ["REVOKE EXECUTE ON FUNCTION lo_from_bytea(oid, bytea) FROM {login}"],
        "can execute large-object writer lo_from_bytea(oid,bytea)",
    ),
    "create on another schema": (
        ["CREATE SCHEMA sdr_extra_{suffix}", "GRANT CREATE ON SCHEMA sdr_extra_{suffix} TO {login}"],
        ["DROP SCHEMA IF EXISTS sdr_extra_{suffix} CASCADE"],
        "can create objects in schema sdr_extra_{suffix}",
    ),
    "security definer function": (
        ["CREATE FUNCTION public.sdr_definer_{suffix}() RETURNS integer LANGUAGE sql SECURITY DEFINER RETURN 1"],
        ["DROP FUNCTION IF EXISTS public.sdr_definer_{suffix}()"],
        "can execute SECURITY DEFINER function sdr_definer_{suffix}()",
    ),
}


@pytest.mark.parametrize("case", BYPASSES)
def test_self_check_refuses_a_login_that_is_not_confined(boundary, case):
    setup, cleanup, problem = BYPASSES[case]
    login = f"sdr_bypass_{boundary.suffix}"
    names = {
        "login": login,
        "owner_role": boundary.owner_role,
        "agent_role": boundary.agent_role,
        "database": boundary.database,
        "suffix": boundary.suffix,
    }
    root = _engine(boundary.root_url, isolation_level="AUTOCOMMIT")
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with root.connect() as conn:
            if case == "pg_maintain" and not conn.execute(
                text("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'pg_maintain'")
            ).first():
                pytest.skip("pg_maintain exists from PostgreSQL 17")
            conn.exec_driver_sql(
                f"CREATE ROLE {login} LOGIN PASSWORD '{boundary.password}' IN ROLE {boundary.agent_role}"
            )
        with admin.connect() as conn:
            for statement in setup:
                conn.exec_driver_sql(statement.format(**names))
        with pytest.raises(BoundaryViolation, match="not confined") as excinfo:
            connect_gateway(
                _url(boundary.admin_url.set(username=login, password=boundary.password)),
                boundary.agent_role,
            )
        assert problem.format(**names) in str(excinfo.value)
    finally:
        with admin.connect() as conn:
            for statement in cleanup:
                conn.exec_driver_sql(statement.format(**names))
            if conn.execute(text("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :r"), {"r": login}).first():
                conn.exec_driver_sql(f"DROP OWNED BY {login}")
        with root.connect() as conn:
            conn.exec_driver_sql(f"DROP ROLE IF EXISTS {login}")
        admin.dispose()
        root.dispose()


def test_a_default_cluster_starts_the_agent_with_a_warning(boundary, caplog):
    """PUBLIC keeps TEMPORARY on postgres and the template databases on a
    default install. The installer only touches the survey database, so the
    self-check fails only there and warns about the others."""
    assert _admin_scalar(
        boundary, "SELECT has_database_privilege('public', 'postgres', 'TEMPORARY')"
    ), "the throwaway cluster is expected to be a default install"
    with caplog.at_level("WARNING", logger="agent.db_gateway"):
        connect_gateway(_url(boundary.agent_url), boundary.agent_role).close()
    assert "TEMPORARY on other databases (postgres" in caplog.text
    assert "agent/README.md" in caplog.text


def test_self_check_rejects_a_login_outside_the_agent_role(boundary):
    login = f"sdr_stranger_{boundary.suffix}"
    root = _engine(boundary.root_url, isolation_level="AUTOCOMMIT")
    try:
        with root.connect() as conn:
            conn.exec_driver_sql(f"CREATE ROLE {login} LOGIN PASSWORD '{boundary.password}'")
        with pytest.raises(BoundaryViolation, match="does not inherit"):
            connect_gateway(
                _url(boundary.admin_url.set(username=login, password=boundary.password)),
                boundary.agent_role,
            )
    finally:
        with root.connect() as conn:
            conn.exec_driver_sql(f"DROP ROLE IF EXISTS {login}")
        root.dispose()


def test_a_write_the_database_refuses_is_submit_rejected(boundary):
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        with pytest.raises(SubmitRejected, match=f"record {record_id}"):
            gateway.submit_classification(record_id, ClassificationStatus.AUTO_CLASSIFIED, "x", float("nan"), "r")
    finally:
        gateway.close()
    assert _metadata(boundary, record_id)["classification_status"] is None


def test_nul_characters_are_stripped_from_the_reasoning(boundary):
    """PostgreSQL text cannot hold NUL; model output can."""
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        gateway.submit_classification(record_id, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "a\x00b")
    finally:
        gateway.close()
    assert _metadata(boundary, record_id)["reasoning"] == "ab"


def test_a_write_blocked_past_the_statement_timeout_is_submit_timed_out(boundary):
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role, statement_timeout_s=0.5)
    admin = _engine(boundary.admin_url)
    try:
        with admin.connect() as human:
            human.execute(text("SELECT 1 FROM survey_records WHERE id = :id FOR UPDATE"), {"id": record_id})
            with pytest.raises(SubmitTimedOut, match="57014"):
                gateway.submit_classification(record_id, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "r")
            human.rollback()
    finally:
        gateway.close()
        admin.dispose()
    assert _metadata(boundary, record_id)["classification_status"] is None


def test_newest_pending_snippets_for_the_startup_probe(boundary):
    watermark = _watermark(boundary)
    older = _insert(boundary, Modality.UNKNOWN, None, path="/srv/snippets/older.sigmf-data")
    _insert(boundary, Modality.UNKNOWN, None, path=None, flags={"snippet_dropped": "low_disk"})
    newer = _insert(boundary, Modality.UNKNOWN, None, path="/srv/snippets/newer.sigmf-data")
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED, path="/srv/snippets/tagged.sigmf-data")
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        assert [r.id for r in gateway.fetch_newest_with_snippet(watermark, 5)] == [newer, older]
        assert [r.id for r in gateway.fetch_newest_with_snippet(watermark, 1)] == [newer]
    finally:
        gateway.close()


def _set_metadata(boundary: Boundary, record_id: int, key: str, raw_json: str) -> None:
    """Put a raw JSON value into a row's metadata, as a buggy or hostile
    writer could, bypassing the pydantic model."""
    _admin_scalar(
        boundary,
        "UPDATE survey_records SET metadata = jsonb_set(metadata::jsonb, ARRAY[:key], CAST(:raw AS jsonb))::json"
        " WHERE id = :id RETURNING id",
        key=key,
        raw=raw_json,
        id=record_id,
    )


def test_malformed_rows_never_fail_the_fetch(boundary):
    """The view returns the numbers as text; a value no cast could take
    (3e9 ms, a string sample rate) marks that one record instead of failing
    every fetch."""
    watermark = _watermark(boundary)
    huge = _insert(boundary, Modality.UNKNOWN, None)
    _set_metadata(boundary, huge, "snippet_duration_ms", "3e9")
    word = _insert(boundary, Modality.UNKNOWN, None)
    _set_metadata(boundary, word, "sample_rate", '"bad"')
    good = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        records = {r.id: r for r in gateway.fetch_pending(watermark, 10)}
    finally:
        gateway.close()
    assert set(records) == {huge, word, good}
    # jsonb normalizes 3e9 on the way in.
    assert "snippet_duration_ms '3000000000' is outside [0, 2147483647]" in records[huge].malformed
    assert "sample_rate 'bad' is not a number" in records[word].malformed
    assert records[good].malformed is None and records[good].snippet_duration_ms == 1000


def test_a_second_agent_refuses_to_start_while_one_runs(boundary):
    """Two replicas would each spend the token budget and race for the same
    records: the first holds a session advisory lock until it closes."""
    first = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        with pytest.raises(AgentAlreadyRunning, match="Another sdr-agent is running against this database"):
            connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    finally:
        first.close()
    connect_gateway(_url(boundary.agent_url), boundary.agent_role).close()  # free again


def _kill_the_lock_session(boundary: Boundary) -> None:
    """Terminate whichever backend holds the agent's advisory lock."""
    killed = _admin_scalar(
        boundary,
        "SELECT count(pg_terminate_backend(pid)) FROM pg_locks"
        " WHERE locktype = 'advisory' AND classid = 1396986433 AND objid = 1195724372 AND granted",
    )
    assert killed == 1


def test_a_lost_lock_is_noticed_and_halts_the_agent(boundary, tmp_path):
    """The reviewer's probe: the lock session was killed, the agent went on
    unlocked, a second agent started, and close() raised. The lock is
    checked before every batch; losing it is systemic."""
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        assert gateway.lock_held()
        _kill_the_lock_session(boundary)
        assert not gateway.lock_held()
        agent = ClassificationAgent(gateway, None, AgentSettings(snippet_root=tmp_path), load_band_table().entries)
        with pytest.raises(SystemicFault, match="single-instance lock was lost"):
            agent.run_batch()
        connect_gateway(_url(boundary.agent_url), boundary.agent_role).close()  # free for a successor
    finally:
        gateway.close()  # tolerant of the dead lock session


def test_the_startup_refusal_says_how_to_find_the_lock_holder(boundary):
    first = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        with pytest.raises(AgentAlreadyRunning) as excinfo:
            connect_gateway(_url(boundary.agent_url), boundary.agent_role)
        assert "pg_stat_activity" in str(excinfo.value) and "objid = 1195724372" in str(excinfo.value)
    finally:
        first.close()


def test_deferred_records_are_fetched_by_id_while_pending(boundary):
    first = _insert(boundary, Modality.UNKNOWN, None)
    tagged = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED)
    last = _insert(boundary, Modality.UNKNOWN, None)
    wifi = _insert(boundary, Modality.WIFI, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        assert [r.id for r in gateway.fetch_by_ids([last, tagged, wifi, first, 10**9])] == [first, last]
        assert gateway.fetch_by_ids([]) == []
    finally:
        gateway.close()


def _poison(boundary: Boundary, record_id: int, column: str) -> None:
    """Append a JSON \\u0000 escape to a row's json column, as a writer
    without the schema's NUL check could (json, unlike jsonb, stores it)."""
    _admin_scalar(
        boundary,
        f"UPDATE survey_records SET {column} = (left({column}::text, -1) || ', \"poison\": \"a\\u0000b\"}}')::json"
        " WHERE id = :id RETURNING id",
        id=record_id,
    )


def test_a_row_holding_a_nul_escape_is_left_out_instead_of_failing_every_fetch(boundary):
    """The reviewer's probe: one pending row with \\u0000 anywhere in its
    metadata (or identifier, or signal) made every ->> on it, and so every
    view fetch, fail with 22P05. Such rows are left out of the view (they
    need manual cleanup), and classify_unknown treats them as not pending."""
    watermark = _watermark(boundary)
    good = _insert(boundary, Modality.UNKNOWN, None)
    poisoned = {column: _insert(boundary, Modality.UNKNOWN, None) for column in ("metadata", "identifier", "signal")}
    for column, record_id in poisoned.items():
        _poison(boundary, record_id, column)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        assert [r.id for r in gateway.fetch_pending(watermark, 100)] == [good]
        assert gateway.fetch_by_ids(sorted(poisoned.values())) == []
        for record_id in poisoned.values():
            with pytest.raises(RecordNotPending):
                gateway.submit_classification(record_id, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "x")
    finally:
        gateway.close()


@pytest.mark.parametrize("metadata", ['["unclassified"]', '"unclassified"', "42", "null"])
def test_a_record_whose_metadata_is_not_an_object_is_never_pending(boundary, metadata):
    """Codex M7: a scalar or array metadata read as unclassified (->> gives
    NULL), and jsonb || object made it an array, so the record was
    re-classified forever. Only an object can be pending or written."""
    watermark = _watermark(boundary)
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    _admin_scalar(
        boundary, "UPDATE survey_records SET metadata = CAST(:m AS json) WHERE id = :id RETURNING id",
        m=metadata, id=record_id,
    )
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        assert record_id not in [r.id for r in gateway.fetch_pending(watermark, 100)]
        with pytest.raises(RecordNotPending):
            gateway.submit_classification(record_id, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "x")
    finally:
        gateway.close()
    assert _admin_scalar(boundary, "SELECT metadata::text FROM survey_records WHERE id = :id", id=record_id) == metadata


@pytest.mark.parametrize(
    "value, poisoned",
    [
        (r'"\u0000"', True),  # a real NUL escape
        (r'"a\\\u0000"', True),  # an escaped backslash, then a real NUL escape
        (r'"\\u0000"', False),  # a legit backslash followed by the text u0000
        (r'"a\\\\u0000"', False),  # two escaped backslashes, then text
    ],
)
def test_only_a_real_nul_escape_hides_a_row(boundary, value, poisoned):
    """Security L1: matching the raw text \\u0000 also hid a legit string
    holding a backslash followed by u0000. A NUL escape is one preceded by an
    even number of backslashes."""
    watermark = _watermark(boundary)
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    _admin_scalar(
        boundary,
        "UPDATE survey_records SET metadata = (left(metadata::text, -1) || ', \"note\": ' || :v || '}')::json"
        " WHERE id = :id RETURNING id",
        v=value, id=record_id,
    )
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        visible = record_id in [r.id for r in gateway.fetch_pending(watermark, 100)]
    finally:
        gateway.close()
    assert visible is not poisoned


def test_agent_classifies_a_real_row_through_the_boundary(boundary, tmp_path):
    """The whole agent against real PostgreSQL: a stored step-4 snippet, a
    pending row, the agent login role, a fake LLM; the row ends up
    auto_classified and nothing else in it changes. A malformed row before
    it and a pending row whose snippet capture dropped are closed out as
    needs_review."""
    rng = np.random.default_rng(42)
    n = 1 << 18
    iq = synthetic.noise(rng, n, 1e-5) + synthetic.gate(
        synthetic.band_limited(rng, n, 1e6, 125e3, 200e3, 1e-3), 1e6, [(0.05, 0.1)]
    )
    staging, store_root = tmp_path / "staging", tmp_path / "snippets"
    staged = write_sigmf_snippet(
        iq, staging, 1e6, 915e6, datetime(2026, 10, 3, 12, tzinfo=timezone.utc), trigger_offset=50_000
    )
    stored = LocalSnippetStore(staging, store_root).adopt(str(staged))
    watermark = _watermark(boundary)
    malformed_id = _insert(boundary, Modality.UNKNOWN, None, path=stored, sample_rate=1e6)
    _set_metadata(boundary, malformed_id, "snippet_duration_ms", "3e9")
    record_id = _insert(
        boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED, path=stored, sample_rate=1e6
    )
    before = _metadata(boundary, record_id)
    dropped_id = _insert(
        boundary, Modality.UNKNOWN, None, path=None, sample_rate=1e6, flags={"snippet_dropped": "low_disk"}
    )
    answer = {"tag": "ism_902_928:lora", "confidence": 0.92, "reasoning": "125 kHz at 915.2 MHz."}

    class FakeLlm:
        messages = None

        def create(self, **kwargs):
            return Message.model_validate(
                {
                    "id": "m", "type": "message", "role": "assistant", "model": kwargs["model"],
                    "stop_reason": "tool_use", "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [{"type": "tool_use", "id": "t", "name": "record_classification", "input": answer}],
                }
            )

    llm = FakeLlm()
    llm.messages = llm
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        agent = ClassificationAgent(
            gateway,
            llm,
            AgentSettings(snippet_root=store_root),
            load_band_table().entries,
            start_after_id=watermark,  # rows other tests left pending are not this test's
        )
        agent.check_snippet_root()
        assert agent.run_batch() == 3
    finally:
        gateway.close()
    malformed = _metadata(boundary, malformed_id)
    assert (malformed["classification_status"], malformed["tag"]) == ("needs_review", None)
    assert malformed["reasoning"].startswith("malformed_record: snippet_duration_ms '3000000000'")
    assert _metadata(boundary, record_id) == {
        **before,
        "classification_status": "auto_classified",
        "tag": "ism_902_928:lora",
        "confidence": 0.92,
        "reasoning": "125 kHz at 915.2 MHz.",
    }
    dropped = _metadata(boundary, dropped_id)
    assert (dropped["classification_status"], dropped["tag"], dropped["confidence"]) == ("needs_review", None, 0.0)
    assert "snippet_dropped = 'low_disk'" in dropped["reasoning"]


@pytest.mark.parametrize("right_password", [True, False], ids=["starts", "refused"])
def test_agent_startup_never_logs_the_database_password(boundary, tmp_path, monkeypatch, caplog, capsys, right_password):
    """A full start (connect, self-check, lock, startup probe, stop) and a
    refused login: the password reaches neither the logs nor the exit."""
    password = boundary.password if right_password else f"wrong{boundary.password}"
    monkeypatch.setenv("SURVEYTOOL_AGENT_DATABASE_URL", _url(boundary.agent_url.set(password=password)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    stopped = threading.Event()
    stopped.set()
    monkeypatch.setattr(agent_service, "install_stop_signal", lambda: stopped)
    argv = ["--agent-role", boundary.agent_role, "--snippet-store-dir", str(tmp_path), "--start-after-id", str(2**62)]
    with caplog.at_level(logging.DEBUG):
        if right_password:
            agent_service.main(argv)
            ended = ""
        else:
            with pytest.raises(OperationalError, match="password authentication failed") as excinfo:
                agent_service.main(argv)
            ended = "".join(traceback.format_exception(excinfo.value))
    assert password not in caplog.text + capsys.readouterr().err + ended


def test_installer_never_prints_the_admin_password(boundary, monkeypatch, capsys):
    password = f"wrong{boundary.password}"
    monkeypatch.setenv("SURVEYTOOL_ADMIN_DATABASE_URL", _url(boundary.admin_url.set(password=password)))
    with pytest.raises((OperationalError, SystemExit)) as excinfo:
        boundary_main([])
    captured = capsys.readouterr()
    assert "password authentication failed" in str(excinfo.value)
    assert password not in captured.out + captured.err + "".join(traceback.format_exception(excinfo.value))
