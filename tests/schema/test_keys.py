"""Pin the JSON key contract between the Python writers and agent_boundary.sql."""

from __future__ import annotations

from pathlib import Path

import pytest

from schema import keys
from schema.records import Metadata

_SQL = (Path(__file__).resolve().parents[2] / "storage" / "sql" / "agent_boundary.sql").read_text()

_METADATA_FIELDS = (
    keys.IQ_SNIPPET_PATH,
    "sample_rate",
    "classification_status",
    "tag",
    "confidence",
    "reasoning",
)


# Exact SQL path fragments, so renaming a key or changing the nesting at one
# site fails even when the same leaf key appears elsewhere in the file.
_SQL_FRAGMENTS = (
    # agent_is_pending_unknown predicate (read)
    "p_metadata ->> 'classification_status'",
    # agent_pending_unknown view columns (read)
    f"r.metadata ->> '{keys.IQ_SNIPPET_PATH}'",
    "r.metadata ->> 'sample_rate'",
    f"r.metadata -> 'quality_flags' ->> '{keys.SNIPPET_REJECTED}'",
    f"r.metadata -> 'quality_flags' ->> '{keys.SNIPPET_DROPPED}'",
    # classify_unknown write
    "'classification_status', p_status",
    "'tag', p_tag",
    "'confidence', p_confidence",
    "'reasoning', p_reasoning",
)


@pytest.mark.parametrize("fragment", _SQL_FRAGMENTS)
def test_exact_json_path_fragment_is_in_the_agent_sql(fragment: str) -> None:
    assert fragment in _SQL


@pytest.mark.parametrize("name", _METADATA_FIELDS)
def test_key_is_a_metadata_field(name: str) -> None:
    assert name in Metadata.model_fields
