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
from sqlalchemy import Engine, text
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


# What the install revokes from PUBLIC, whether PUBLIC holds it now, and the
# login roles that hold it only through PUBLIC (no grant to them or to any
# role they belong to), which would lose it.
_PUBLIC_REVOKES = text(
    """
    WITH objects (kind, label, acl, owner, revoked) AS (
        SELECT 'TABLE', 'public.survey_records', coalesce(c.relacl, acldefault('r', c.relowner)), c.relowner,
               ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']
          FROM pg_catalog.pg_class AS c
         WHERE c.oid = pg_catalog.to_regclass('public.survey_records')
        UNION ALL
        SELECT 'SCHEMA', 'public', coalesce(n.nspacl, acldefault('n', n.nspowner)), n.nspowner, ARRAY['CREATE']
          FROM pg_catalog.pg_namespace AS n
         WHERE n.nspname = 'public'
        UNION ALL
        SELECT 'DATABASE', pg_catalog.quote_ident(d.datname), coalesce(d.datacl, acldefault('d', d.datdba)),
               d.datdba, ARRAY['TEMPORARY']
          FROM pg_catalog.pg_database AS d
         WHERE d.datname = pg_catalog.current_database()
        UNION ALL
        SELECT 'FUNCTION', p.oid::pg_catalog.regprocedure::text, coalesce(p.proacl, acldefault('f', p.proowner)),
               p.proowner, ARRAY['EXECUTE']
          FROM pg_catalog.pg_proc AS p
         WHERE p.oid IN (pg_catalog.to_regprocedure('pg_catalog.lo_create(oid)'),
                         pg_catalog.to_regprocedure('pg_catalog.lo_creat(integer)'),
                         pg_catalog.to_regprocedure('pg_catalog.lo_from_bytea(oid, bytea)'),
                         pg_catalog.to_regprocedure('pg_catalog.lo_import(text)'),
                         pg_catalog.to_regprocedure('pg_catalog.lo_import(text, oid)'),
                         pg_catalog.to_regprocedure('pg_catalog.lo_open(oid, integer)'),
                         pg_catalog.to_regprocedure('pg_catalog.lo_put(oid, bigint, bytea)'))
    )
    SELECT o.kind, o.label, a.privilege_type,
           (SELECT pg_catalog.string_agg(r.rolname, ', ' ORDER BY r.rolname)
              FROM pg_catalog.pg_roles AS r
             WHERE r.rolcanlogin AND NOT r.rolsuper AND r.oid <> o.owner
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.aclexplode(o.acl) AS g
                    WHERE g.grantee <> 0 AND g.privilege_type = a.privilege_type
                      AND pg_catalog.pg_has_role(r.oid, g.grantee, 'USAGE')))
      FROM objects AS o, pg_catalog.aclexplode(o.acl) AS a
     WHERE a.grantee = 0 AND a.privilege_type = ANY (o.revoked)
     ORDER BY o.kind, o.label, a.privilege_type
    """
)


def preview_agent_boundary(admin_engine: Engine) -> list[str]:
    """What install_agent_boundary would revoke from PUBLIC, one line per
    privilege PUBLIC holds now, naming the login roles that hold it only
    through PUBLIC and would lose it. Reads only; changes nothing."""
    with admin_engine.connect() as conn:
        rows = conn.execute(_PUBLIC_REVOKES).all()
        table_exists = conn.execute(text("SELECT pg_catalog.to_regclass('public.survey_records') IS NOT NULL")).scalar_one()
    lines = [
        f"REVOKE {privilege} ON {kind} {label} FROM PUBLIC"
        + (f": login roles relying on PUBLIC for it: {relying}" if relying else ": no login role relies on it")
        for kind, label, privilege, relying in rows
    ]
    if not table_exists:
        lines.append("public.survey_records does not exist yet: --apply creates it (init_db) before installing")
    return lines or ["PUBLIC holds none of the privileges the boundary revokes"]


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
        description="Preview, or with --apply install, the Part 4 agent database boundary "
        "(creating the survey_records table if needed). Reads a superuser PostgreSQL URL "
        f"from {ADMIN_URL_ENV}."
    )
    parser.add_argument("--agent-role", default=DEFAULT_AGENT_ROLE)
    parser.add_argument("--owner-role", default=DEFAULT_OWNER_ROLE)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Install. Without it, only list what would be revoked from PUBLIC and who relies on it.",
    )
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
        print("The boundary revokes these privileges from PUBLIC (agent/README.md has the regrant path):")
        for line in preview_agent_boundary(engine):
            print(f"  {line}")
        if not args.apply:
            print("Nothing was changed; re-run with --apply to install.")
            return
        init_db(engine)
        install_agent_boundary(engine, args.agent_role, args.owner_role)
    except psycopg.Error as exc:
        raise SystemExit(f"Agent boundary not installed: {exc}") from exc
    finally:
        engine.dispose()
    print(f"Agent boundary installed: agent role {args.agent_role}, owner role {args.owner_role}")


if __name__ == "__main__":
    main()
