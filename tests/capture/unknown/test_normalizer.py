# tests/capture/unknown/test_normalizer.py
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from capture.unknown.normalizer import SnippetCaptureEvent, normalize_snippet_event
from schema.records import ClassificationStatus, Modality, UnifiedRecord

EVENT = SnippetCaptureEvent(
    timestamp=datetime(2026, 10, 3, 12, 0, 0, 500000, tzinfo=timezone.utc),
    center_freq_hz=915e6,
    sample_rate=2e6,
    bandwidth_estimate_hz=125_000.0,
    peak_power_dbfs=-17.5,
    mean_power_dbfs=-20.0,
    noise_floor_dbfs=-40.0,
    snippet_path="/data/snippet-staging/x.sigmf-data",
    snippet_duration_ms=1000,
)


def test_normalize_snippet_event_maps_fields():
    record = normalize_snippet_event(EVENT, survey_id="s1", operator_id="op1")
    assert record.modality is Modality.UNKNOWN
    assert record.timestamp == EVENT.timestamp
    assert record.survey_id == "s1"
    assert record.operator_id == "op1"
    assert record.identifier.center_freq == 915e6
    assert record.identifier.bandwidth_estimate == 125_000.0
    assert record.signal.rssi == -20.0
    assert record.signal.peak_power == -17.5
    assert record.signal.snr == pytest.approx(20.0)
    assert record.metadata.iq_snippet_path == "/data/snippet-staging/x.sigmf-data"
    assert record.metadata.snippet_duration_ms == 1000
    assert record.metadata.sample_rate == 2e6
    assert record.metadata.quality_flags == {"power_units": "dBFS"}
    # Placeholder position: ingest attaches the real GPS fix.
    assert (record.lat, record.lon, record.gps_fix_quality) == (0.0, 0.0, None)


def test_record_is_queued_for_part4_classification():
    """Part 4 selects records with classification_status == unclassified;
    the schema default is None, so leaving it unset would hide every
    snippet from the classifier."""
    record = normalize_snippet_event(EVENT, survey_id="s1", operator_id="op1")
    assert record.metadata.classification_status is ClassificationStatus.UNCLASSIFIED


def test_record_survives_the_queue_json_round_trip():
    record = normalize_snippet_event(EVENT, survey_id="s1", operator_id="op1")
    assert UnifiedRecord.model_validate_json(record.model_dump_json()) == record


def test_event_without_a_snippet_normalizes_to_a_null_snippet_path():
    record = normalize_snippet_event(replace(EVENT, snippet_path=None), survey_id="s1", operator_id="op1")
    assert record.metadata.iq_snippet_path is None


def test_unreliable_bandwidth_estimate_is_flagged_not_silently_reported():
    record = normalize_snippet_event(
        replace(EVENT, bandwidth_estimate_reliable=False), survey_id="s1", operator_id="op1"
    )
    assert record.metadata.quality_flags == {"power_units": "dBFS", "bandwidth_estimate_unreliable": True}
