# agent/snippet_reader.py
"""Load a record's SigMF snippet: contained, capped, and without a checksum pass.

The snippet path comes from the database, so it is untrusted, and so is
everything in the .sigmf-meta file. The reader therefore:
- requires both files to resolve (symlinks followed) under the snippet-store root;
- reads at most 1 MiB of .sigmf-meta and parses it itself, then hands sigmf
  the contained data path explicitly. sigmf's fromfile() is not used: it
  follows a meta's core:dataset to any path (absolute ones included), falls
  back to archive/collection/non-SigMF converters, and leaks the meta file
  handle when the JSON is invalid;
- skips sigmf's SHA-512 pass, which would read the whole data file;
- reads at most 2 s of samples, and never more than MAX_SAMPLES;
- zeroes NaN/inf samples (DMA or driver corruption) and counts them, as step
  4 does when it measures, so a corrupt sample never crashes the analysis.

It also reads step 4's "pre_trigger" annotation: the noise reference's
length and the trigger threshold it lies below.
"""

from __future__ import annotations

import errno
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sigmf
from sigmf import sigmffile

DATA_SUFFIX = ".sigmf-data"
META_SUFFIX = ".sigmf-meta"
MAX_SECONDS = 2.0
# Absolute ceiling whatever the file claims its sample rate is: 256 MiB of
# cf32. Measured peak RSS of the whole analysis at this size: ~610 MB.
MAX_SAMPLES = 1 << 25
MIN_SAMPLES = 1024  # one coarse FFT frame (dsp.segmentation.NFFT)
_MAX_META_BYTES = 1 << 20
# I/O errors that may pass (a flaky disk, NFS, memory pressure): the record
# is retried later. Anything else (missing, permission, corrupt) is final.
_TRANSIENT_ERRNOS = frozenset({errno.EIO, errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT, errno.ESTALE, errno.ENOMEM, errno.EBUSY})
# Step 4's writer records its trigger threshold on the "pre_trigger"
# annotation under this key (capture.unknown.snippet_writer.THRESHOLD_KEY;
# agent/ cannot import capture/).
THRESHOLD_KEY = "sdr_surveytool:threshold_dbfs"


class SnippetOutsideStore(Exception):
    """The record's snippet is not a file inside the store: a judgment about
    the record (needs_review), not an I/O fault."""


class SnippetUnreadable(Exception):
    """The snippet is missing, unreadable, or malformed: about the record,
    unless `transient` (an I/O error that may pass, such as EIO), when the
    agent retries it later. `missing`: a file is not there at all, which the
    agent blames on the record only once other snippets demonstrably read
    (a stale copy of the store would make every snippet missing)."""

    def __init__(self, message: str, transient: bool = False, missing: bool = False) -> None:
        super().__init__(message)
        self.transient = transient
        self.missing = missing


@dataclass(frozen=True)
class Snippet:
    iq: np.ndarray  # complex64
    sample_rate: float  # core:sample_rate of the file
    center_freq_hz: float  # core:frequency of the file's first capture
    truncated: bool  # the file held more than the cap
    # Length of step 4's "pre_trigger" annotation (signal-free samples before
    # the trigger, the noise reference), clipped to what was read; 0 if none.
    pre_trigger_samples: int = 0
    non_finite_samples: int = 0  # NaN/inf samples, zeroed in `iq`
    # The trigger threshold the pre-trigger samples lie below, as step 4
    # recorded it; None when absent or malformed (segmentation then gates
    # the reference relative to the burst instead).
    trigger_threshold_dbfs: float | None = None


def _contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise SnippetOutsideStore(f"{path} resolves to {resolved}, outside the snippet store {root}")
    return resolved


def read_snippet(
    snippet_path: str | None,
    store_root: Path,
    record_sample_rate: float | None = None,
    record_center_hz: float | None = None,
) -> Snippet:
    """Read the snippet at `snippet_path` (a .sigmf-data path from the DB).

    The record's metadata.sample_rate and center frequency are authoritative
    (step 4 stores the SDR's read-back rate); the file's core:sample_rate and
    core:frequency must agree with them, and are used only when the record
    has none."""
    if not snippet_path:
        raise SnippetOutsideStore("The record has no snippet path")
    root = store_root.resolve()
    path = Path(snippet_path)
    if not path.is_absolute() or path.suffix != DATA_SUFFIX:
        raise SnippetOutsideStore(f"{snippet_path!r} is not an absolute {DATA_SUFFIX} path")
    data = _contained(path, root)
    meta = _contained(path.with_suffix(META_SUFFIX), root)
    try:
        with meta.open("rb") as handle:
            raw = handle.read(_MAX_META_BYTES + 1)
        if len(raw) > _MAX_META_BYTES:
            raise SnippetUnreadable(f"{meta} is larger than {_MAX_META_BYTES} bytes")
        metadata = json.loads(raw)
        if sigmf.DATASET_KEY in metadata[sigmf.SigMFFile.GLOBAL_KEY]:
            raise SnippetUnreadable(f"{meta} names a non-conforming dataset; step 4 never writes one")
        recording = sigmffile.SigMFFile(metadata=metadata, data_file=str(data), skip_checksum=True)
        return _samples(recording, data, record_sample_rate, record_center_hz)
    except SnippetUnreadable:
        raise
    except Exception as exc:  # the files are untrusted: any parse or read failure is "unreadable"
        transient = isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS
        missing = isinstance(exc, FileNotFoundError)
        raise SnippetUnreadable(f"Cannot read {data}: {exc!r}", transient, missing) from exc


def _samples(
    recording: sigmffile.SigMFFile,
    data: Path,
    record_sample_rate: float | None,
    record_center_hz: float | None,
) -> Snippet:
    datatype = recording.get_global_field(sigmf.DATATYPE_KEY)
    channels = recording.get_global_field(sigmf.NUM_CHANNELS_KEY, 1)
    captures = recording.get_captures()
    if datatype != "cf32_le" or channels != 1:
        raise SnippetUnreadable(f"{data}: expected one cf32_le channel, got {datatype} x {channels}")
    sample_rate = _positive(data, "sample rate", recording.get_global_field(sigmf.SAMPLE_RATE_KEY))
    center = _positive(data, "center frequency", captures[0].get(sigmf.FREQUENCY_KEY) if captures else None)
    for name, claimed, actual in (
        ("sample rate", record_sample_rate, sample_rate),
        ("center frequency", record_center_hz, center),
    ):
        if claimed is not None and not math.isclose(claimed, actual, rel_tol=1e-9):
            raise SnippetUnreadable(f"{data}: the record's {name} {claimed} disagrees with the file's {actual}")
    if record_sample_rate is not None:
        sample_rate = record_sample_rate
    if record_center_hz is not None:
        center = record_center_hz
    count = min(recording.sample_count, round(MAX_SECONDS * sample_rate), MAX_SAMPLES)
    if count < MIN_SAMPLES:
        raise SnippetUnreadable(f"{data}: {recording.sample_count} samples, need {MIN_SAMPLES}")
    iq = np.asarray(recording.read_samples(0, count), dtype=np.complex64)
    finite = np.isfinite(iq)
    non_finite = int(iq.size - np.count_nonzero(finite))
    if non_finite == iq.size:
        raise SnippetUnreadable(f"{data}: every sample is NaN or inf")
    pre_trigger_samples, threshold = _pre_trigger(recording, count)
    return Snippet(
        iq=np.where(finite, iq, 0).astype(np.complex64) if non_finite else iq,
        sample_rate=float(sample_rate),
        center_freq_hz=float(center),
        truncated=recording.sample_count > count,
        pre_trigger_samples=pre_trigger_samples,
        non_finite_samples=non_finite,
        trigger_threshold_dbfs=threshold,
    )


def _positive(data: Path, name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SnippetUnreadable(f"{data}: invalid {name} {value!r}")
    return float(value)


def _pre_trigger(recording: sigmffile.SigMFFile, count: int) -> tuple[int, float | None]:
    """Step 4 annotates [0, trigger) as "pre_trigger", with the trigger
    threshold. Anything else (no annotation, another start, a non-integer
    length) means no reference; a threshold that is not a finite number
    below full scale (0 dBFS, which step 4 never accepts) means none."""
    for annotation in recording.get_annotations():
        length = annotation.get(sigmf.SAMPLE_COUNT_KEY)
        if (
            annotation.get(sigmf.LABEL_KEY) == "pre_trigger"
            and annotation.get(sigmf.SAMPLE_START_KEY) == 0
            and type(length) is int
            and length > 0
        ):
            threshold = annotation.get(THRESHOLD_KEY)
            valid = type(threshold) in (int, float) and math.isfinite(threshold) and threshold < 0
            return min(length, count), float(threshold) if valid else None
    return 0, None
