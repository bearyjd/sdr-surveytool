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
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sigmf
from sigmf import sigmffile

DATA_SUFFIX = ".sigmf-data"
META_SUFFIX = ".sigmf-meta"
MAX_SECONDS = 2.0
# Absolute ceiling whatever the file claims its sample rate is: 256 MiB of
# cf32. Measured peak RSS of the read plus the whole analysis at this size,
# worst case (a region filling the band, so nothing is decimated): 584 MB.
MAX_SAMPLES = 1 << 25
MIN_SAMPLES = 1024  # one coarse FFT frame (dsp.segmentation.NFFT)
_MAX_META_BYTES = 1 << 20
# Regular files only, reached from a store-root fd one path component at a
# time without following any symlink, opened without blocking on a FIFO,
# then read a bounded chunk at a time.
_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_CHUNK = 1 << 20
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
    meta = path.with_suffix(META_SUFFIX)
    _contained(meta, root)
    try:
        relative = path.relative_to(root)  # lexically: the walk below never resolves anything
    except ValueError:
        raise SnippetOutsideStore(f"{path} is not spelled under the snippet store {root}") from None
    try:
        # One fd on the root; every open below walks from it, so a component
        # swapped for a symlink after the checks above is refused, not followed.
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            metadata = json.loads(_read_meta(root_fd, relative.with_suffix(META_SUFFIX)))
            if sigmf.DATASET_KEY in metadata[sigmf.SigMFFile.GLOBAL_KEY]:
                raise SnippetUnreadable(f"{meta} names a non-conforming dataset; step 4 never writes one")
            # Metadata only: the samples are read below, from a checked regular file.
            recording = sigmffile.SigMFFile(metadata=metadata, skip_checksum=True)
            return _samples(recording, root_fd, relative, data, record_sample_rate, record_center_hz)
        finally:
            os.close(root_fd)
    except SnippetUnreadable:
        raise
    except Exception as exc:  # the files are untrusted: any parse or read failure is "unreadable"
        transient = isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS
        missing = isinstance(exc, FileNotFoundError)
        raise SnippetUnreadable(f"Cannot read {data}: {exc!r}", transient, missing) from exc


def _open_regular(root_fd: int, relative: Path) -> tuple[int, os.stat_result]:
    """A read-only fd on `relative` under the store root, which must be a
    regular file: not a symlink, FIFO, socket or device (a FIFO would block
    the reader, and the agent, forever). Each directory on the way is opened
    from the last with O_NOFOLLOW, so none can be swapped for a symlink;
    the file is checked before opening and again on its fd."""
    dir_fd = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        name = relative.parts[-1]
        if not stat.S_ISREG(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode):
            raise SnippetUnreadable(f"{relative} is not a regular file")
        fd = os.open(name, _OPEN_FLAGS, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):  # swapped since the stat
            raise SnippetUnreadable(f"{relative} is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd, status


def _read_fd(fd: int, nbytes: int) -> np.ndarray:
    """Exactly `nbytes` from `fd` into one fresh buffer, a chunk at a time."""
    out = np.empty(nbytes, dtype=np.uint8)
    view = memoryview(out)
    got = 0
    while got < nbytes:
        chunk = os.read(fd, min(_READ_CHUNK, nbytes - got))
        if not chunk:
            raise SnippetUnreadable(f"the file ended after {got} of {nbytes} bytes")
        view[got : got + len(chunk)] = chunk
        got += len(chunk)
    return out


def _read_meta(root_fd: int, meta: Path) -> bytes:
    fd, status = _open_regular(root_fd, meta)
    try:
        if status.st_size > _MAX_META_BYTES:
            raise SnippetUnreadable(f"{meta} is larger than {_MAX_META_BYTES} bytes")
        return _read_fd(fd, status.st_size).tobytes()
    finally:
        os.close(fd)


def _samples(
    recording: sigmffile.SigMFFile,
    root_fd: int,
    relative: Path,
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
    fd, status = _open_regular(root_fd, relative)
    try:
        sample_count = status.st_size // np.dtype("<c8").itemsize
        count = min(sample_count, round(MAX_SECONDS * sample_rate), MAX_SAMPLES)
        if count < MIN_SAMPLES:
            raise SnippetUnreadable(f"{data}: {sample_count} samples, need {MIN_SAMPLES}")
        iq = _read_fd(fd, count * np.dtype("<c8").itemsize).view("<c8").astype(np.complex64, copy=False)
    finally:
        os.close(fd)
    finite = np.isfinite(iq)
    non_finite = int(iq.size - np.count_nonzero(finite))
    if non_finite == iq.size:
        raise SnippetUnreadable(f"{data}: every sample is NaN or inf")
    pre_trigger_samples, threshold = _pre_trigger(recording, count)
    return Snippet(
        iq=np.where(finite, iq, 0).astype(np.complex64) if non_finite else iq,
        sample_rate=float(sample_rate),
        center_freq_hz=float(center),
        truncated=sample_count > count,
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
