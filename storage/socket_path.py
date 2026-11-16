"""Default ingest socket location and the checks that keep it private.

A fixed path directly under /tmp is predictable and world-listable, so another
local user can pre-create or squat it. Default to the per-user runtime dir and
verify the directory is ours and closed to others before binding in it.
"""

from __future__ import annotations

import os
import stat

_SUBDIR = "sdr-surveytool"
_SOCKET_NAME = "ingest.sock"


def default_socket_path() -> str:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return os.path.join(runtime_dir, _SUBDIR, _SOCKET_NAME)
    return os.path.join("/tmp", f"{_SUBDIR}-{os.getuid()}", _SOCKET_NAME)


def ensure_private_socket_dir(socket_path: str) -> None:
    """Create the socket's directory as 0700 if missing; otherwise require that it is a
    real directory (not a symlink), owned by this user, with no group/other access."""
    directory = os.path.dirname(socket_path)
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create socket directory {directory}: {exc}") from exc
    info = os.lstat(directory)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"Socket directory {directory} is a symlink or not a directory")
    if info.st_uid != os.getuid():
        raise RuntimeError(f"Socket directory {directory} is not owned by the current user")
    if info.st_mode & 0o077:
        raise RuntimeError(
            f"Socket directory {directory} has unsafe permissions "
            f"{stat.S_IMODE(info.st_mode):o}; expected 700"
        )
