# capture/unknown/normalizer.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from schema.records import (
    ClassificationStatus,
    Identifier,
    Metadata,
    Modality,
    Signal,
    UnifiedRecord,
)


@dataclass(frozen=True)
class SnippetCaptureEvent:
    """Everything known about one staged snippet. Powers are dBFS
    (uncalibrated; full-scale complex sample = 0 dBFS), see
    dsp.spectral."""

    timestamp: datetime  # sample-derived UTC time of the trigger sample
    center_freq_hz: float
    sample_rate: float
    bandwidth_estimate_hz: float
    peak_power_dbfs: float
    mean_power_dbfs: float
    noise_floor_dbfs: float
    snippet_path: str | None  # staged .sigmf-data path; None if the IQ was dropped
    snippet_duration_ms: int


def normalize_snippet_event(
    event: SnippetCaptureEvent, survey_id: str, operator_id: str
) -> UnifiedRecord:
    """Converts one snippet capture event into a UnifiedRecord. lat/lon are
    left at 0.0 with gps_fix_quality=None -- ingest.service attaches the real
    fix, matching capture.wifi/bluetooth/cellular.

    Signal.rssi is required by the schema; for this modality it carries the
    mean in-burst power in dBFS (not dBm -- no RF calibration exists), and
    snr is that minus the configured noise floor. quality_flags records the
    unit so a stored row is self-describing next to dBm WiFi/BT rows.
    """
    return UnifiedRecord(
        timestamp=event.timestamp,
        lat=0.0,
        lon=0.0,
        gps_fix_quality=None,
        survey_id=survey_id,
        operator_id=operator_id,
        modality=Modality.UNKNOWN,
        identifier=Identifier(
            center_freq=event.center_freq_hz,
            bandwidth_estimate=event.bandwidth_estimate_hz,
        ),
        signal=Signal(
            rssi=event.mean_power_dbfs,
            snr=event.mean_power_dbfs - event.noise_floor_dbfs,
            peak_power=event.peak_power_dbfs,
        ),
        metadata=Metadata(
            quality_flags={"power_units": "dBFS"},
            iq_snippet_path=event.snippet_path,
            snippet_duration_ms=event.snippet_duration_ms,
            sample_rate=event.sample_rate,
            classification_status=ClassificationStatus.UNCLASSIFIED,
        ),
    )
