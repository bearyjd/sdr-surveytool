import os
import stat

import pytest

from schema.socket_path import default_socket_path, ensure_private_socket_dir


def test_default_uses_xdg_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert default_socket_path() == str(tmp_path / "sdr-surveytool" / "ingest.sock")


def test_default_fallback_is_per_uid_not_bare_tmp(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    path = default_socket_path()
    assert path != "/tmp/sdr-ingest.sock"
    assert f"sdr-surveytool-{os.getuid()}" in path


def test_ensure_creates_dir_0700(tmp_path):
    sock = tmp_path / "run" / "ingest.sock"
    ensure_private_socket_dir(str(sock))
    mode = stat.S_IMODE(os.stat(sock.parent).st_mode)
    assert mode == 0o700


def test_ensure_rejects_group_or_world_accessible_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir(mode=0o777)
    os.chmod(d, 0o777)
    with pytest.raises(RuntimeError, match="permissions"):
        ensure_private_socket_dir(str(d / "ingest.sock"))


def test_ensure_rejects_symlinked_dir(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(RuntimeError, match="symlink"):
        ensure_private_socket_dir(str(link / "ingest.sock"))
