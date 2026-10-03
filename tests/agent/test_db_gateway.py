# tests/agent/test_db_gateway.py
"""Gateway checks that need no PostgreSQL (the self-check, view and
function are exercised for real in tests/storage/test_agent_boundary_pg.py)."""

import pytest

from agent.db_gateway import connect_gateway


@pytest.mark.parametrize("url", ["sqlite:///survey.db", "sqlite:///:memory:", "mysql://u:p@localhost/db"])
def test_refuses_non_postgres_urls(url):
    with pytest.raises(ValueError, match="PostgreSQL"):
        connect_gateway(url)
