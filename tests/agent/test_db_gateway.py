# tests/agent/test_db_gateway.py
"""Gateway checks that need no PostgreSQL (the self-check, view and
function are exercised for real in tests/storage/test_agent_boundary_pg.py)."""

import pytest

import psycopg
from sqlalchemy.exc import DBAPIError

from agent.db_gateway import AgentGateway, PendingRecord, SubmitUnwritable, connect_gateway, parse_pending_row
from schema.records import ClassificationStatus


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
