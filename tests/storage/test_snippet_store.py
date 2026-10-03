# tests/storage/test_snippet_store.py
import os

import pytest

from storage import snippet_store
from storage.snippet_store import LocalSnippetStore, ensure_private_dir


def _stage(directory, stem: str = "snip") -> tuple:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = directory / f"{stem}.sigmf-data"
    meta = directory / f"{stem}.sigmf-meta"
    data.write_bytes(b"\x00\x01\x02\x03")
    meta.write_text('{"global": {}}')
    return data, meta


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "staging", tmp_path / "snippets"


def test_adopt_moves_both_files_and_returns_final_data_path(dirs):
    staging, root = dirs
    data, meta = _stage(staging)

    final = LocalSnippetStore(staging, root).adopt(str(data))

    assert final == str((root / "snip.sigmf-data").resolve())
    assert (root / "snip.sigmf-data").read_bytes() == b"\x00\x01\x02\x03"
    assert (root / "snip.sigmf-meta").read_text() == '{"global": {}}'
    assert not data.exists() and not meta.exists()


def test_adopt_rejects_a_path_outside_staging(dirs, tmp_path):
    """iq_snippet_path arrives over the ingest socket, so it is untrusted:
    ingest must never move an arbitrary file into the store."""
    staging, root = dirs
    staging.mkdir(mode=0o700)
    victim, _ = _stage(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="staging directory"):
        LocalSnippetStore(staging, root).adopt(str(victim))
    assert victim.exists()


def test_adopt_rejects_dot_dot_traversal_out_of_staging(dirs, tmp_path):
    staging, root = dirs
    staging.mkdir(mode=0o700)
    victim, _ = _stage(tmp_path / "elsewhere")
    sneaky = staging / ".." / "elsewhere" / "snip.sigmf-data"
    with pytest.raises(ValueError, match="staging directory"):
        LocalSnippetStore(staging, root).adopt(str(sneaky))
    assert victim.exists()


def test_adopt_rejects_a_symlink_escaping_staging(dirs, tmp_path):
    staging, root = dirs
    victim, _ = _stage(tmp_path / "elsewhere")
    staging.mkdir(mode=0o700)
    os.symlink(victim, staging / "link.sigmf-data")
    with pytest.raises(ValueError, match="staging directory"):
        LocalSnippetStore(staging, root).adopt(str(staging / "link.sigmf-data"))
    assert victim.exists()


@pytest.mark.parametrize("suffix", [".sigmf-meta", ".txt", ""])
def test_adopt_rejects_anything_but_a_sigmf_data_file(dirs, suffix):
    staging, root = dirs
    _stage(staging)
    other = staging / f"snip{suffix}"
    other.touch()
    with pytest.raises(ValueError, match=".sigmf-data"):
        LocalSnippetStore(staging, root).adopt(str(other))


def test_adopt_fails_when_meta_sibling_is_missing_and_moves_nothing(dirs):
    staging, root = dirs
    data, meta = _stage(staging)
    meta.unlink()
    with pytest.raises(FileNotFoundError):
        LocalSnippetStore(staging, root).adopt(str(data))
    assert data.exists()


def test_adopt_refuses_to_overwrite_an_existing_snippet(dirs):
    staging, root = dirs
    data, _ = _stage(staging)
    root.mkdir(mode=0o700)
    (root / "snip.sigmf-data").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        LocalSnippetStore(staging, root).adopt(str(data))
    assert (root / "snip.sigmf-data").read_bytes() == b"original"
    assert data.exists()


def test_ensure_private_dir_creates_an_owner_only_dir_and_returns_it_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = ensure_private_dir("data/staging")
    assert directory == (tmp_path / "data" / "staging").resolve()
    assert directory.stat().st_mode & 0o077 == 0


def test_ensure_private_dir_rejects_group_or_world_access(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    with pytest.raises(PermissionError, match="chmod 700"):
        ensure_private_dir(shared)


def test_ensure_private_dir_rejects_a_directory_owned_by_another_user(tmp_path, monkeypatch):
    real_uid = os.geteuid()
    monkeypatch.setattr(snippet_store.os, "geteuid", lambda: real_uid + 1)
    with pytest.raises(PermissionError, match="one dedicated user"):
        ensure_private_dir(tmp_path / "staging")


def test_store_resolves_creates_and_secures_both_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = LocalSnippetStore("staging", "snippets")
    assert store.staging_dir == (tmp_path / "staging").resolve()
    assert store.root_dir == (tmp_path / "snippets").resolve()
    for directory in (store.staging_dir, store.root_dir):
        assert directory.stat().st_mode & 0o077 == 0


def test_store_refuses_a_world_readable_staging_dir(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    staging.chmod(0o755)
    with pytest.raises(PermissionError, match="chmod 700"):
        LocalSnippetStore(staging, tmp_path / "snippets")
