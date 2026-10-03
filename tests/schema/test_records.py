import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schema.records import (
    ClassificationStatus,
    Identifier,
    Metadata,
    Modality,
    Signal,
    UnifiedRecord,
)


def _make_record(**overrides):
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        lat=47.6062,
        lon=-122.3321,
        survey_id="survey-1",
        operator_id="op-1",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF", ssid="TestNet", channel=6),
        signal=Signal(rssi=-55.0),
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


def test_unified_record_round_trips_through_json():
    record = _make_record()
    restored = UnifiedRecord.model_validate_json(record.model_dump_json())
    assert restored.identifier.bssid == "AA:BB:CC:DD:EE:FF"
    assert restored.modality is Modality.WIFI


def test_metadata_defaults_to_zero_sample_count():
    record = _make_record()
    assert record.metadata.sample_count_in_grid_cell == 0
    assert record.metadata.classification_status is None


def test_signal_requires_rssi():
    with pytest.raises(ValidationError):
        Signal()


def test_needs_review_status_round_trips_through_json():
    """Part 4 routes low-confidence or ungrounded results to needs_review."""
    record = _make_record(metadata=Metadata(classification_status="needs_review", tag=None, confidence=0.0))
    restored = UnifiedRecord.model_validate_json(record.model_dump_json())
    assert restored.metadata.classification_status is ClassificationStatus.NEEDS_REVIEW
    assert [status.value for status in ClassificationStatus] == [
        "unclassified",
        "manually_tagged",
        "auto_classified",
        "needs_review",
    ]


@pytest.mark.parametrize(
    "overrides, where",
    [
        ({"survey_id": "s\x001"}, "survey_id"),
        ({"identifier": Identifier(ssid="Hid\x00den")}, "identifier.ssid"),
        ({"metadata": Metadata(reasoning="a\x00b")}, "metadata.reasoning"),
        ({"metadata": Metadata(quality_flags={"snippet_dropped": "low\x00disk"})}, "metadata.quality_flags.snippet_dropped"),
        ({"metadata": Metadata(quality_flags={"bad\x00key": True})}, "metadata.quality_flags"),
        ({"metadata": Metadata(quality_flags={"list": ["ok", "n\x00"]})}, "metadata.quality_flags.list[1]"),
    ],
)
def test_a_nul_character_anywhere_is_rejected(overrides, where):
    """PostgreSQL text and jsonb cannot hold U+0000: one stored row with it
    would fail every read of the agent's view and every jsonb cast."""
    with pytest.raises(ValidationError, match=rf"{re.escape(where)} holds a NUL character"):
        _make_record(**overrides)
