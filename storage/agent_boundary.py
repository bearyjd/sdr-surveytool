# storage/agent_boundary.py
"""Installs the Part 4 agent's database boundary (storage/sql/agent_boundary.sql).

Admin-only. Run it through the `sdr-agent-boundary` script with a superuser
URL in SURVEYTOOL_ADMIN_DATABASE_URL (never argv, which any local user can
read; leave the password out of the URL and let libpq find it in ~/.pgpass
or a service file). Ingest and the agent never call it: the agent connects
as a role that could not apply it, and checks at startup that it is in force.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import psycopg
from sqlalchemy import Engine
from sqlalchemy.engine import make_url

from storage.db import init_db, make_engine

DEFAULT_AGENT_ROLE = "surveytool_agent"
DEFAULT_OWNER_ROLE = "surveytool_classifier_owner"
ADMIN_URL_ENV = "SURVEYTOOL_ADMIN_DATABASE_URL"

_SQL_PATH = Path(__file__).resolve().parent / "sql" / "agent_boundary.sql"
# Lower-case unquoted identifiers only, so a substituted name is safe both as
# an identifier and inside the file's '...' literals.
_ROLE_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")


def render_agent_boundary_sql(
    agent_role: str = DEFAULT_AGENT_ROLE, owner_role: str = DEFAULT_OWNER_ROLE
) -> str:
    """Return agent_boundary.sql with both role placeholders substituted."""
    for name in (agent_role, owner_role):
        if not _ROLE_NAME.fullmatch(name):
            raise ValueError(f"Invalid role name {name!r}: must match {_ROLE_NAME.pattern}")
    if agent_role == owner_role:
        raise ValueError("The agent role and the owner role must differ")
    return (
        _SQL_PATH.read_text()
        .replace("{{agent_role}}", agent_role)
        .replace("{{owner_role}}", owner_role)
    )


def install_agent_boundary(
    admin_engine: Engine,
    agent_role: str = DEFAULT_AGENT_ROLE,
    owner_role: str = DEFAULT_OWNER_ROLE,
) -> None:
    """Apply the boundary in one transaction. public.survey_records must exist."""
    if admin_engine.dialect.name != "postgresql":
        raise ValueError("The agent boundary needs PostgreSQL")
    sql = render_agent_boundary_sql(agent_role, owner_role)
    with admin_engine.begin() as conn:
        # Straight to the DB-API cursor with no parameters: SQLAlchemy's text()
        # would read ':word' as a bind parameter, and the driver only parses
        # '%' placeholders when parameters are passed, so the file's
        # format('%I', ...) reaches PostgreSQL untouched.
        conn.connection.cursor().execute(sql)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the survey_records table if needed, then install the "
        f"Part 4 agent database boundary. Reads a superuser PostgreSQL URL from {ADMIN_URL_ENV}."
    )
    parser.add_argument("--agent-role", default=DEFAULT_AGENT_ROLE)
    parser.add_argument("--owner-role", default=DEFAULT_OWNER_ROLE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    database_url = os.environ.get(ADMIN_URL_ENV)
    if not database_url:
        raise SystemExit(f"{ADMIN_URL_ENV} is not set (a superuser PostgreSQL URL)")
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise SystemExit(f"{ADMIN_URL_ENV} must be a PostgreSQL URL")
    engine = make_engine(url.set(drivername="postgresql+psycopg").render_as_string(hide_password=False))
    try:
        init_db(engine)
        install_agent_boundary(engine, args.agent_role, args.owner_role)
    except psycopg.Error as exc:
        raise SystemExit(f"Agent boundary not installed: {exc}") from exc
    finally:
        engine.dispose()
    print(f"Agent boundary installed: agent role {args.agent_role}, owner role {args.owner_role}")


if __name__ == "__main__":
    main()
