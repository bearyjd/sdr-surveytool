# tests/agent/test_db_gateway.py
"""Gateway checks that need no PostgreSQL (the self-check, view and
function are exercised for real in tests/storage/test_agent_boundary_pg.py)."""

import logging
import traceback

import pytest

import psycopg
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, OperationalError

from agent import service
from agent.db_gateway import (
    AgentGateway,
    PendingRecord,
    SubmitUnwritable,
    connect_gateway,
    parse_pending_row,
    require_tls_for_remote,
)
from schema.records import ClassificationStatus


@pytest.fixture(autouse=True)
def _no_libpq_environment(monkeypatch):
    """libpq reads these when the URL leaves a parameter out; the runner's
    own must not decide what these tests see."""
    for name in ("PGHOST", "PGHOSTADDR", "PGSSLMODE", "PGSERVICE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("url", ["sqlite:///survey.db", "sqlite:///:memory:", "mysql://u:p@localhost/db"])
def test_refuses_non_postgres_urls(url):
    with pytest.raises(ValueError, match="PostgreSQL"):
        connect_gateway(url)


def _row(**values):
    """A view row as PostgreSQL returns it: every number arrives as text."""
    row = {
        "id": 7,
        "iq_snippet_path": "/srv/snippets/a.sigmf-data",
        "sample_rate": "1000000.0",
        "center_freq": "915000000",
        "peak_power": "-17.5",
        "snippet_duration_ms": "262",
        "snippet_rejected": None,
        "snippet_dropped": None,
    }
    return tuple({**row, **values}.values())


def test_a_well_formed_row_parses_into_numbers():
    record = parse_pending_row(_row())
    assert record == PendingRecord(7, "/srv/snippets/a.sigmf-data", 1e6, 915e6, -17.5, 262)
    assert record.malformed is None
    nulls = parse_pending_row(_row(sample_rate=None, center_freq=None, peak_power=None, snippet_duration_ms=None))
    assert (nulls.sample_rate, nulls.snippet_duration_ms, nulls.malformed) == (None, None, None)


@pytest.mark.parametrize(
    "field, raw, problem",
    [
        ("snippet_duration_ms", "3e9", "snippet_duration_ms '3e9' is outside"),
        ("snippet_duration_ms", "1.5", "snippet_duration_ms '1.5' is not an integer"),
        ("sample_rate", "bad", "sample_rate 'bad' is not a number"),
        ("sample_rate", "NaN", "sample_rate 'NaN' is not a number"),
        ("sample_rate", "1_000", "sample_rate '1_000' is not a number"),
        ("sample_rate", "-1e6", "sample_rate '-1e6' is outside"),
        ("center_freq", "1e400", "center_freq '1e400' is outside"),
        ("peak_power", "true", "peak_power 'true' is not a number"),
    ],
)
def test_a_malformed_value_marks_the_record_instead_of_failing_the_fetch(field, raw, problem):
    record = parse_pending_row(_row(**{field: raw}))
    assert record.id == 7 and problem in record.malformed
    assert getattr(record, field) is None


def test_an_oversized_or_nul_bearing_path_is_malformed():
    assert "iq_snippet_path" in parse_pending_row(_row(iq_snippet_path="/" + "a" * 5000)).malformed
    assert "iq_snippet_path" in parse_pending_row(_row(iq_snippet_path="/srv/a\x00b")).malformed



def test_an_untranslatable_character_on_submit_is_about_the_record():
    """22P05 (a NUL escape PostgreSQL cannot convert) is about that row's
    data: SubmitUnwritable, which the agent skips, never a halt."""
    orig = psycopg.errors.UntranslatableCharacter("unsupported Unicode escape sequence")

    class _Engine:
        def begin(self):
            raise DBAPIError("SELECT public.classify_unknown(...)", {}, orig)

    gateway = AgentGateway(_Engine())  # type: ignore[arg-type]
    with pytest.raises(SubmitUnwritable, match="Record 7"):
        gateway.submit_classification(7, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "x")


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://agent@db.example.com/surveytool",
        "postgresql://agent@10.0.0.5/surveytool?sslmode=disable",
        "postgresql://agent@10.0.0.5/surveytool?sslmode=allow",
        "postgresql://agent@10.0.0.5/surveytool?sslmode=prefer",
    ],
)
def test_a_remote_database_requires_tls(url, monkeypatch):
    """Codex M8: the agent's password and every record would cross the
    network in clear; prefer silently falls back to it."""
    monkeypatch.delenv("PGSSLMODE", raising=False)
    with pytest.raises(ValueError, match="sslmode"):
        connect_gateway(url)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://agent@db.example.com/surveytool?sslmode=verify-full",
        "postgresql://agent@db.example.com/surveytool?sslmode=verify-ca",
        "postgresql://agent@db.example.com/surveytool?sslmode=require",
        "postgresql://agent@localhost/surveytool",
        "postgresql://agent@127.0.0.1/surveytool?sslmode=disable",
        "postgresql://agent@[::1]/surveytool",
        "postgresql://agent@/surveytool?host=/var/run/postgresql",
    ],
)
def test_tls_settings_that_are_accepted(url, monkeypatch):
    monkeypatch.delenv("PGSSLMODE", raising=False)
    require_tls_for_remote(make_url(url))


def test_pgsslmode_counts_when_the_url_sets_none(monkeypatch):
    monkeypatch.setenv("PGSSLMODE", "verify-full")
    require_tls_for_remote(make_url("postgresql://agent@db.example.com/surveytool"))


@pytest.mark.parametrize(
    ("url", "env"),
    [
        # hostaddr is where libpq connects; host is then only the name it checks.
        ("postgresql://agent@localhost/surveytool?hostaddr=10.0.0.5", {}),
        ("postgresql://agent@localhost/surveytool", {"PGHOSTADDR": "10.0.0.5"}),
        # A query host overrides the URL's own host.
        ("postgresql://agent@localhost/surveytool?host=db.example.com", {}),
        # libpq tries every host of a list.
        ("postgresql://agent@/surveytool?host=/var/run/postgresql,db.example.com", {}),
        ("postgresql://agent@/surveytool?host=/var/run/postgresql&host=db.example.com:5432", {}),
        # A URL without a host connects to PGHOST.
        ("postgresql://agent@/surveytool", {"PGHOST": "db.example.com"}),
        ("postgresql://agent@/surveytool", {"PGHOSTADDR": "10.0.0.5"}),
        # The URL's sslmode wins over PGSSLMODE.
        ("postgresql://agent@db.example.com/surveytool?sslmode=disable", {"PGSSLMODE": "verify-full"}),
    ],
)
def test_tls_is_judged_on_the_parameters_libpq_connects_with(url, env, monkeypatch):
    """Codex: the check looked at the URL's host alone, while libpq connects
    to hostaddr, to every host of a list, and to PGHOST/PGHOSTADDR when the
    URL names none."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="is not local: set sslmode=verify-full"):
        require_tls_for_remote(make_url(url))


@pytest.mark.parametrize(
    ("url", "env"),
    [
        ("postgresql://agent@localhost/surveytool?service=surveytool", {}),
        ("postgresql://agent@localhost/surveytool?sslmode=verify-full", {"PGSERVICE": "surveytool"}),
    ],
)
def test_a_connection_service_is_refused(url, env, monkeypatch):
    """A pg_service.conf entry can set host, hostaddr and sslmode behind the
    URL's back, wherever PGSERVICEFILE or PGSYSCONFDIR point."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="service"):
        require_tls_for_remote(make_url(url))


@pytest.mark.parametrize(
    ("url", "env"),
    [
        ("postgresql://agent@/surveytool", {"PGHOST": "/var/run/postgresql"}),
        ("postgresql://agent@localhost/surveytool?hostaddr=127.0.0.1", {}),
        ("postgresql://agent@/surveytool?host=/var/run/postgresql,localhost", {}),
        ("postgresql://agent@/surveytool?host=@pgsock", {}),
        ("postgresql://agent@db.example.com/surveytool?hostaddr=10.0.0.5&sslmode=verify-full", {}),
        ("postgresql://agent@/surveytool?host=/var/run/postgresql,db.example.com&sslmode=require", {}),
        ("postgresql://agent@/surveytool", {"PGHOST": "db.example.com", "PGSSLMODE": "verify-full"}),
        ("postgresql://agent@db.example.com/surveytool?sslmode=require", {"PGSSLMODE": "disable"}),
    ],
)
def test_effective_settings_that_are_accepted(url, env, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    require_tls_for_remote(make_url(url))


def test_agent_startup_never_logs_the_database_password(monkeypatch, caplog, capsys, tmp_path):
    """Nothing listens on port 1, so the connection fails: neither the logs
    nor the exception that ends the process may carry the password."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    monkeypatch.setenv(
        "SURVEYTOOL_AGENT_DATABASE_URL", "postgresql://agent_login:s3cret-pw@127.0.0.1:1/surveytool"
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(OperationalError) as excinfo:
        service.main(["--snippet-store-dir", str(tmp_path)])
    told = caplog.text + capsys.readouterr().err + "".join(traceback.format_exception(excinfo.value))
    assert "127.0.0.1" in told
    assert "s3cret-pw" not in told
