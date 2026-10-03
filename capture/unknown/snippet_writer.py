from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

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

    Atomic: each file is written under a hidden temporary name and renamed
    into place, data first and meta last, so a `.sigmf-meta` is never
    visible without its complete `.sigmf-data` (the meta is the completeness
    marker the snippet store requires). On failure, the temporaries and any
    already-renamed data file are removed. Files are created 0o600, and a
    staging directory created here is 0o700.

    `capture_start` is the sample-derived UTC time of iq[0] and becomes the
    SigMF capture's core:datetime. The basename carries a random suffix, so
    concurrent capture processes can never collide.
    """
    if capture_start.tzinfo is None:
        raise ValueError("capture_start must be timezone-aware (UTC)")
    start_utc = capture_start.astimezone(timezone.utc)
    staging_dir = Path(staging_dir).resolve()
    staging_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    stem = f"{start_utc:%Y%m%dT%H%M%S%fZ}_{center_freq_hz:.0f}Hz_{uuid.uuid4().hex[:8]}"
    data_path = staging_dir / f"{stem}.sigmf-data"
    meta_path = staging_dir / f"{stem}.sigmf-meta"
    tmp_data = staging_dir / f".{stem}.sigmf-data.tmp"
    tmp_meta = staging_dir / f".{stem}.sigmf-meta.tmp"
    leftovers = [tmp_data, tmp_meta]
    try:
        with _create_private(tmp_data, "wb") as data_file:
            # '<c8' = little-endian complex64, exactly SigMF's cf32_le.
            np.asarray(iq, dtype="<c8").tofile(data_file)
        os.replace(tmp_data, data_path)
        leftovers.append(data_path)

        # Built from the final data path: sigmf records a non-conforming data
        # filename (such as the temporary one) as core:dataset in the meta.
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
        meta.validate()
        with _create_private(tmp_meta, "w") as meta_file:
            meta.dump(meta_file, pretty=True)
        os.replace(tmp_meta, meta_path)
    except BaseException:
        for leftover in leftovers:
            leftover.unlink(missing_ok=True)
        raise
    return data_path


def _create_private(path: Path, mode: str) -> IO:
    """Create `path` exclusively, readable and writable by the owner only."""
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), mode)
