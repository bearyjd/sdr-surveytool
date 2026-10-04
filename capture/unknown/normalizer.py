# capture/unknown/normalizer.py
from __future__ import annotations

import math
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
    bandwidth_estimate_hz: float | None  # None when the IQ was dropped unmeasured
    peak_power_dbfs: float
    mean_power_dbfs: float
    noise_floor_dbfs: float
    snippet_path: str | None  # staged .sigmf-data path; None if the IQ was dropped
    snippet_duration_ms: int
    # False when dsp.spectral couldn't measure the bandwidth meaningfully
    # (fills > 90% of the band or wraps its edges): flagged on the record.
    bandwidth_estimate_reliable: bool = True
    # IQ samples that were NaN/inf (DMA or driver corruption), left out of
    # every measurement; flagged on the record when non-zero.
    non_finite_samples: int = 0


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

    Raises ValueError for any non-finite measurement: NaN/inf would
    serialize as JSON null and fail ingest's validation, silently losing the
    record, so the caller must measure over finite samples only.
    """
    _require_finite(event)
    flags: dict = {"power_units": "dBFS"}
    if not event.bandwidth_estimate_reliable:
        flags["bandwidth_estimate_unreliable"] = True
    if event.non_finite_samples:
        flags["non_finite_samples"] = event.non_finite_samples
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
            quality_flags=flags,
            iq_snippet_path=event.snippet_path,
            snippet_duration_ms=event.snippet_duration_ms,
            sample_rate=event.sample_rate,
            classification_status=ClassificationStatus.UNCLASSIFIED,
        ),
    )


def _require_finite(event: SnippetCaptureEvent) -> None:
    for name in (
        "center_freq_hz",
        "sample_rate",
        "bandwidth_estimate_hz",
        "peak_power_dbfs",
        "mean_power_dbfs",
        "noise_floor_dbfs",
    ):
        value = getattr(event, name)
        if value is not None and not math.isfinite(value):
            raise ValueError(
                f"Snippet event has a non-finite {name} ({value!r}); refusing to build a "
                "record ingest could not store"
            )
