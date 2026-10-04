# storage/snippet_store.py
from __future__ import annotations

import errno
import logging
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

_DATA_SUFFIX = ".sigmf-data"
_META_SUFFIX = ".sigmf-meta"
_PRIVATE_DIR_MODE = 0o700

# The only staged names adopt() accepts: exactly what
# capture.unknown.snippet_writer writes ("<%Y%m%dT%H%M%S%f>Z_<Hz>Hz_<8 hex>").
# Bounded, so a null byte or an over-long name is rejected before any syscall.
STAGED_DATA_NAME = re.compile(
    r"\d{8}T\d{12}Z_\d{1,12}Hz_[0-9a-f]{8}\.sigmf-data", re.ASCII
)
# The largest snippet capture can be configured to write: 61.44 MS/s (the
# AD9361 maximum) x (5 s pre + 5 s post, the window caps) x 8 bytes (cf32).
DEFAULT_MAX_SNIPPET_BYTES = 4_915_200_000
_CF32_BYTES = 8
_MAX_META_BYTES = 1024 * 1024  # a capture's SigMF meta is a few hundred bytes
# adopt() retries these a few times (they pass: a signal, a full fd table)
# before treating the failure as a rejection; every other OSError is final.
_TRANSIENT_ERRNOS = frozenset({errno.EINTR, errno.EAGAIN, errno.EMFILE, errno.ENFILE})
_ADOPT_ATTEMPTS = 3
_RETRY_SLEEP_SECONDS = 0.05


def require_absolute(path: Path | str) -> Path:
    """Snippet directories must be absolute: capture and ingest resolve a
    relative path against their own working directories, and if those
    differ every snippet is rejected as outside_staging (its detection then
    persists without IQ)."""
    directory = Path(path)
    if not directory.is_absolute():
        raise ValueError(
            f"Snippet directory {str(path)!r} must be an absolute path (capture and ingest "
            "would otherwise resolve it against different working directories); use e.g. "
            "/var/lib/sdr-surveytool/snippet-staging"
        )
    return directory


def ensure_private_dir(path: Path | str) -> Path:
    """Create `path` (mode 0o700) if missing and return it absolute and resolved.

    Snippet directories hold raw IQ and are the capture -> ingest handoff, so
    capture and ingest must run as one dedicated uid: the directory must be
    owned by this process's effective uid with no group/other access. Both
    sides call this at startup so they enforce one rule.
    """
    directory = require_absolute(path)
    directory.mkdir(mode=_PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    resolved = directory.resolve()
    info = os.stat(resolved)
    euid = os.geteuid()
    if info.st_uid != euid:
        raise PermissionError(
            f"Snippet directory {str(resolved)!r} is owned by uid {info.st_uid}, not "
            f"this process's uid {euid}; capture and ingest must run as one dedicated "
            f"user (fix: chown {euid} {resolved})"
        )
    if info.st_mode & 0o077:
        raise PermissionError(
            f"Snippet directory {str(resolved)!r} has mode {stat.S_IMODE(info.st_mode):o} "
            f"and must not be group/world accessible (fix: chmod 700 {resolved})"
        )
    return resolved


class SnippetRejected(ValueError):
    """adopt() refused a staged snippet. `reason` is a short, fixed cause
    (e.g. "outside_staging") safe to store in a record's quality_flags.
    Other OSErrors (permissions, I/O) propagate unchanged."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class SnippetStore(Protocol):
    def adopt(self, staged_data_path: str) -> str:
        """Take ownership of a staged SigMF pair; return its final .sigmf-data
        path. Raises SnippetRejected for a pair that must not be adopted."""
        ...

    def discard(self, stored_data_path: str) -> None:
        """Remove a pair adopt() returned, e.g. when its record could not be
        persisted. Missing files are not an error."""
        ...


class LocalSnippetStore:
    """Local flat-file IQ snippet storage (design doc section 7), swappable for
    an S3-compatible store behind the SnippetStore protocol.

    Capture writes SigMF pairs into a staging directory; ingest -- the only
    writer to storage -- adopts them here before persisting the record that
    references them. The staged path comes off the ingest socket, so it is
    untrusted. adopt() guarantees that:

    - it only touches a `.sigmf-data` file directly inside the staging
      directory (judged by absolute path, symlinks never followed) and its
      `.sigmf-meta` sibling;
    - both are regular files with exactly one hard link;
    - each is hard-linked into the store without following symlinks, never
      replacing an existing store file, and the linked inode is verified to
      be the one that was checked, with no other links to it;
    - both are linked before either staged name is removed, and on failure
      everything this call linked is unlinked again.

    Not guaranteed across a process crash between the two links (the store
    can then hold a data file without its meta) or between the two source
    unlinks. Hard links cannot cross filesystems or mount points, so staging
    and store must share one; the constructor proves it with a real link.
    """

    def __init__(
        self,
        staging_dir: Path | str,
        root_dir: Path | str,
        max_snippet_bytes: int = DEFAULT_MAX_SNIPPET_BYTES,
    ) -> None:
        self._staging_dir = ensure_private_dir(staging_dir)
        self._root_dir = ensure_private_dir(root_dir)
        self._max_snippet_bytes = max_snippet_bytes
        _probe_hard_link(self._staging_dir, self._root_dir)

    @property
    def max_snippet_bytes(self) -> int:
        return self._max_snippet_bytes

    @property
    def staging_dir(self) -> Path:
        return self._staging_dir

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def adopt(self, staged_data_path: str) -> str:
        data = Path(os.path.abspath(staged_data_path))
        if data.parent != self._staging_dir:
            raise SnippetRejected(
                "outside_staging",
                f"Snippet {staged_data_path!r} is not directly inside the staging "
                f"directory {str(self._staging_dir)!r}",
            )
        if data.suffix != _DATA_SUFFIX:
            raise SnippetRejected(
                "not_sigmf_data", f"Snippet {staged_data_path!r} is not a {_DATA_SUFFIX} file"
            )
        if not STAGED_DATA_NAME.fullmatch(data.name):
            raise SnippetRejected(
                "bad_name", f"Snippet name {data.name!r} is not one capture writes"
            )
        attempt = 1
        while True:
            try:
                # Safe to retry: a failed _link_pair() unlinks whatever it linked.
                return self._link_pair(data)
            except OSError as error:
                if error.errno in _TRANSIENT_ERRNOS and attempt < _ADOPT_ATTEMPTS:
                    time.sleep(_RETRY_SLEEP_SECONDS * attempt)
                    attempt += 1
                    continue
                code = errno.errorcode.get(error.errno or 0, str(error.errno))
                raise SnippetRejected(
                    "os_error",
                    f"Could not adopt snippet {data.name!r}: {type(error).__name__} ({code})",
                ) from error

    def discard(self, stored_data_path: str) -> None:
        """Remove a pair this store adopted (ingest's compensation when the
        record referencing it can't be persisted). Only ever touches a
        capture-named pair directly inside the store root."""
        data = Path(os.path.abspath(stored_data_path))
        if data.parent != self._root_dir or not STAGED_DATA_NAME.fullmatch(data.name):
            raise ValueError(
                f"Refusing to discard {stored_data_path!r}: not a snippet in the store "
                f"{str(self._root_dir)!r}"
            )
        data.unlink(missing_ok=True)
        data.with_suffix(_META_SUFFIX).unlink(missing_ok=True)

    def _link_pair(self, data: Path) -> str:
        meta = data.with_suffix(_META_SUFFIX)
        data_stat = _single_regular_file(data)
        _check_plausible_size(data, data_stat.st_size, self._max_snippet_bytes)
        meta_stat = _single_regular_file(meta)
        if meta_stat.st_size == 0 or meta_stat.st_size > _MAX_META_BYTES:
            raise SnippetRejected(
                "bad_size",
                f"Staged snippet meta {meta.name!r} has an implausible size "
                f"({meta_stat.st_size} bytes)",
            )
        sources = [(data, data_stat), (meta, meta_stat)]
        linked: list[Path] = []
        try:
            for source, checked in sources:
                final = self._root_dir / source.name
                _link_without_replacing(source, final)
                linked.append(final)
                now = os.lstat(final)
                if (now.st_dev, now.st_ino) != (checked.st_dev, checked.st_ino):
                    raise SnippetRejected(
                        "changed_during_adopt",
                        f"Staged snippet file {source.name!r} was replaced while being adopted",
                    )
                # A link added after the lstat keeps the inode, so also require
                # exactly the staged name plus the one just made.
                if now.st_nlink != 2:
                    raise SnippetRejected(
                        "multiple_links",
                        f"Staged snippet file {source.name!r} gained a hard link while being adopted",
                    )
            # Durable before returning: a database row is about to reference
            # these names, and a power cut must not be able to lose them.
            _fsync_dir(self._root_dir)
        except BaseException:
            for final in linked:
                final.unlink(missing_ok=True)
            raise
        # The pair is durably stored now. Removing the staged names is best
        # effort: a vanished name or a failed unlink/fsync must not orphan the
        # stored pair or fail the adoption.
        try:
            for source, _ in sources:
                source.unlink(missing_ok=True)
            _fsync_dir(self._staging_dir)
        except OSError:
            logger.warning(
                "Adopted snippet %r but could not fully clean up staging", data.name, exc_info=True
            )
        return str(self._root_dir / data.name)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _probe_hard_link(staging_dir: Path, root_dir: Path) -> None:
    """adopt() hard-links staging -> store. That fails across filesystems and
    also across two mounts of one filesystem (same st_dev, still EXDEV), so
    try one real link at startup and fail fast, leaving nothing behind."""
    name = f".link-probe-{uuid.uuid4().hex}"
    source, target = staging_dir / name, root_dir / name
    try:
        os.close(os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        os.link(source, target, follow_symlinks=False)
    except OSError as error:
        code = errno.errorcode.get(error.errno or 0, str(error.errno))
        raise ValueError(
            f"Cannot hard-link from snippet staging dir {str(staging_dir)!r} into store "
            f"{str(root_dir)!r} ({code}): both must be on the same filesystem and mount"
        ) from error
    finally:
        source.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def _single_regular_file(path: Path) -> os.stat_result:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise SnippetRejected(
            "missing_file", f"Staged snippet file missing: {path.name!r}"
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise SnippetRejected(
            "not_regular_file", f"Staged snippet file {path.name!r} is not a regular file"
        )
    if info.st_nlink != 1:
        raise SnippetRejected(
            "multiple_links",
            f"Staged snippet file {path.name!r} has {info.st_nlink} hard links",
        )
    return info


def _check_plausible_size(data: Path, size: int, max_bytes: int) -> None:
    """A cheap cf32 sanity check (ingest never parses SigMF): non-empty, a
    whole number of 8-byte complex64 samples, no larger than any capture."""
    if size == 0 or size % _CF32_BYTES or size > max_bytes:
        raise SnippetRejected(
            "bad_size",
            f"Staged snippet file {data.name!r} has an implausible size ({size} bytes)",
        )


def _link_without_replacing(source: Path, final: Path) -> None:
    try:
        os.link(source, final, follow_symlinks=False)
    except FileExistsError:
        raise SnippetRejected(
            "already_stored", f"Snippet {final.name!r} is already stored; refusing to replace it"
        ) from None
