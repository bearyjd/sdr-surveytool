# capture/unknown/snippet_writer.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sigmf
from sigmf import SigMFFile
from sigmf.utils import SIGMF_DATETIME_ISO8601_FMT

_RECORDER = "sdr-surveytool capture.unknown"


def write_sigmf_snippet(
    iq: np.ndarray,
    staging_dir: Path,
    sample_rate: float,
    center_freq_hz: float,
    capture_start: datetime,
) -> Path:
    """Write `iq` as a SigMF pair (`.sigmf-data` raw cf32_le + `.sigmf-meta`
    JSON) directly in `staging_dir` and return the absolute `.sigmf-data`
    path. Staging is capture-owned scratch space: ingest's snippet store
    later moves the pair into storage (only ingest writes to storage).

    `capture_start` is the sample-derived UTC time of iq[0] and becomes the
    SigMF capture's core:datetime. The basename carries a random suffix, so
    concurrent capture processes can never collide.
    """
    if capture_start.tzinfo is None:
        raise ValueError("capture_start must be timezone-aware (UTC)")
    start_utc = capture_start.astimezone(timezone.utc)
    staging_dir = Path(staging_dir).resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{start_utc:%Y%m%dT%H%M%S%fZ}_{center_freq_hz:.0f}Hz_{uuid.uuid4().hex[:8]}"
    data_path = staging_dir / f"{stem}.sigmf-data"
    # '<c8' = little-endian complex64, exactly SigMF's cf32_le.
    np.asarray(iq, dtype="<c8").tofile(data_path)

    meta = SigMFFile(
        data_file=str(data_path),
        global_info={
            sigmf.DATATYPE_KEY: "cf32_le",
            sigmf.SAMPLE_RATE_KEY: float(sample_rate),
            sigmf.RECORDER_KEY: _RECORDER,
            sigmf.DESCRIPTION_KEY: "Energy-triggered unknown-signal snippet",
        },
    )
    meta.add_capture(
        0,
        metadata={
            sigmf.FREQUENCY_KEY: float(center_freq_hz),
            sigmf.DATETIME_KEY: start_utc.strftime(SIGMF_DATETIME_ISO8601_FMT),
        },
    )
    meta.tofile(str(staging_dir / f"{stem}.sigmf-meta"))
    return data_path
