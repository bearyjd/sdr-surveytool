# tests/storage/test_agent_boundary.py
"""Installer checks that need no PostgreSQL (the boundary itself is
exercised for real in test_agent_boundary_pg.py)."""

import pytest
from sqlalchemy import create_engine

from storage.agent_boundary import (
    DEFAULT_AGENT_ROLE,
    DEFAULT_OWNER_ROLE,
    install_agent_boundary,
    main,
    render_agent_boundary_sql,
)


def test_renders_every_placeholder():
    sql = render_agent_boundary_sql("sdr_agent_x", "sdr_owner_x")
    assert "{{" not in sql and "}}" not in sql
    assert "CREATE ROLE sdr_agent_x NOLOGIN" in sql
    assert "OWNER TO sdr_owner_x" in sql
    assert "surveytool_agent" not in sql


def test_default_role_names():
    assert (DEFAULT_AGENT_ROLE, DEFAULT_OWNER_ROLE) == ("surveytool_agent", "surveytool_classifier_owner")


@pytest.mark.parametrize(
    "agent_role, owner_role",
    [
        ("Agent", "owner"),
        ("agent; DROP TABLE survey_records", "owner"),
        ("agent", "own'er"),
        ("a-b", "owner"),
        ("", "owner"),
        ("9agent", "owner"),
        ("a" * 64, "owner"),
        ("same", "same"),
    ],
)
def test_rejects_unsafe_or_clashing_role_names(agent_role, owner_role):
    with pytest.raises(ValueError):
        render_agent_boundary_sql(agent_role, owner_role)


def test_refuses_a_non_postgres_engine():
    engine = create_engine("sqlite:///:memory:")
    with pytest.raises(ValueError, match="PostgreSQL"):
        install_agent_boundary(engine)
    engine.dispose()


def test_cli_refuses_a_non_postgres_url():
    with pytest.raises(SystemExit, match="PostgreSQL"):
        main(["--database-url", "sqlite:///survey.db"])
