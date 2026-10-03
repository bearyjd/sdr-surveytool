# storage/snippet_store.py
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Protocol

_DATA_SUFFIX = ".sigmf-data"
_META_SUFFIX = ".sigmf-meta"
_PRIVATE_DIR_MODE = 0o700


def ensure_private_dir(path: Path | str) -> Path:
    """Create `path` (mode 0o700) if missing and return it absolute and resolved.

    Snippet directories hold raw IQ and are the capture -> ingest handoff, so
    capture and ingest must run as one dedicated uid: the directory must be
    owned by this process's effective uid with no group/other access. Both
    sides call this at startup so they enforce one rule.
    """
    directory = Path(path)
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
      be the one that was checked;
    - both are linked before either staged name is removed, and on failure
      everything this call linked is unlinked again.

    Not guaranteed across a process crash between the two links (the store
    can then hold a data file without its meta) or between the two source
    unlinks. Hard links cannot cross filesystems, so staging and store must
    share one; the constructor checks this.
    """

    def __init__(self, staging_dir: Path | str, root_dir: Path | str) -> None:
        self._staging_dir = ensure_private_dir(staging_dir)
        self._root_dir = ensure_private_dir(root_dir)
        if _device_of(self._staging_dir) != _device_of(self._root_dir):
            raise ValueError(
                f"Snippet staging dir {str(self._staging_dir)!r} and store "
                f"{str(self._root_dir)!r} must be on the same filesystem: adopt() "
                "hard-links snippets, which cannot cross filesystems"
            )

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
        meta = data.with_suffix(_META_SUFFIX)
        sources = [(data, _single_regular_file(data)), (meta, _single_regular_file(meta))]
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
        except BaseException:
            for final in linked:
                final.unlink(missing_ok=True)
            raise
        for source, _ in sources:
            source.unlink()
        return str(self._root_dir / data.name)


def _device_of(path: Path) -> int:
    return os.stat(path).st_dev


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


def _link_without_replacing(source: Path, final: Path) -> None:
    try:
        os.link(source, final, follow_symlinks=False)
    except FileExistsError:
        raise SnippetRejected(
            "already_stored", f"Snippet {final.name!r} is already stored; refusing to replace it"
        ) from None
