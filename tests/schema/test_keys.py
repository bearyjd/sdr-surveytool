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
_QUALITY_FLAG_KEYS = (keys.SNIPPET_REJECTED, keys.SNIPPET_DROPPED)


@pytest.mark.parametrize("name", _METADATA_FIELDS + _QUALITY_FLAG_KEYS)
def test_key_is_a_quoted_json_path_in_the_agent_view(name: str) -> None:
    assert f"'{name}'" in _SQL


@pytest.mark.parametrize("name", _METADATA_FIELDS)
def test_key_is_a_metadata_field(name: str) -> None:
    assert name in Metadata.model_fields
