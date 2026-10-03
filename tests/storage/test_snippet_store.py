import errno
import os
from pathlib import Path

import pytest

from storage import snippet_store
from storage.snippet_store import LocalSnippetStore, SnippetRejected, ensure_private_dir


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


def _rejected(store: LocalSnippetStore, path, reason: str) -> SnippetRejected:
    with pytest.raises(SnippetRejected) as excinfo:
        store.adopt(str(path))
    assert excinfo.value.reason == reason
    return excinfo.value


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
    store = LocalSnippetStore(staging, root)
    victim, _ = _stage(tmp_path / "elsewhere")
    _rejected(store, victim, "outside_staging")
    assert victim.exists()


def test_adopt_rejects_dot_dot_traversal_out_of_staging(dirs, tmp_path):
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    victim, _ = _stage(tmp_path / "elsewhere")
    _rejected(store, staging / ".." / "elsewhere" / "snip.sigmf-data", "outside_staging")
    assert victim.exists()


def test_adopt_rejects_a_data_symlink(dirs, tmp_path):
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    victim, _ = _stage(tmp_path / "elsewhere")
    os.symlink(victim, staging / "link.sigmf-data")
    (staging / "link.sigmf-meta").write_text("{}")
    _rejected(store, staging / "link.sigmf-data", "not_regular_file")
    assert victim.exists() and list(root.iterdir()) == []


def test_adopt_rejects_a_meta_symlink(dirs, tmp_path):
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    _, victim = _stage(tmp_path / "elsewhere")
    data, meta = _stage(staging)
    meta.unlink()
    os.symlink(victim, meta)
    _rejected(store, data, "not_regular_file")
    assert data.exists() and victim.exists() and list(root.iterdir()) == []


def test_adopt_rejects_a_file_hard_linked_from_elsewhere(dirs, tmp_path):
    """A hard link would let a staged name alias a file living outside
    staging (or let adopt() steal a file someone else still uses)."""
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    data, _ = _stage(staging)
    os.link(data, tmp_path / "alias")
    _rejected(store, data, "multiple_links")
    assert data.exists() and list(root.iterdir()) == []


@pytest.mark.parametrize("suffix", [".sigmf-meta", ".txt", ""])
def test_adopt_rejects_anything_but_a_sigmf_data_file(dirs, suffix):
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    _stage(staging)
    other = staging / f"snip{suffix}"
    other.touch()
    _rejected(store, other, "not_sigmf_data")


def test_adopt_fails_when_meta_sibling_is_missing_and_moves_nothing(dirs):
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    data, meta = _stage(staging)
    meta.unlink()
    _rejected(store, data, "missing_file")
    assert data.exists() and list(root.iterdir()) == []


def test_adopt_refuses_to_overwrite_an_existing_snippet(dirs):
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    data, _ = _stage(staging)
    (root / "snip.sigmf-data").write_bytes(b"original")
    _rejected(store, data, "already_stored")
    assert (root / "snip.sigmf-data").read_bytes() == b"original"
    assert data.exists()


def test_existing_meta_destination_rolls_back_the_data_link(dirs):
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    data, meta = _stage(staging)
    (root / "snip.sigmf-meta").write_text("original")
    _rejected(store, data, "already_stored")
    assert sorted(p.name for p in root.iterdir()) == ["snip.sigmf-meta"]
    assert (root / "snip.sigmf-meta").read_text() == "original"
    assert data.exists() and meta.exists()


def test_adopt_rejects_a_file_swapped_between_check_and_link(dirs, monkeypatch):
    """Simulated race: the data file is replaced (new inode) after it was
    checked but before it is linked. The linked inode must be verified."""
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    data, meta = _stage(staging)
    impostor = staging / "impostor"
    impostor.write_bytes(b"evil")
    real_link = os.link

    def swapping_link(src, dst, *, follow_symlinks=True):
        if str(src).endswith(".sigmf-data"):
            os.replace(impostor, src)  # a different, already-existing inode
        real_link(src, dst, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(snippet_store.os, "link", swapping_link)
    _rejected(store, data, "changed_during_adopt")
    assert list(root.iterdir()) == []
    assert data.exists() and meta.exists()


def test_failed_meta_link_rolls_back_the_data_link(dirs, monkeypatch):
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    data, meta = _stage(staging)
    real_link = os.link

    def failing_meta_link(src, dst, *, follow_symlinks=True):
        if str(src).endswith(".sigmf-meta"):
            raise OSError(errno.EIO, "I/O error")
        real_link(src, dst, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(snippet_store.os, "link", failing_meta_link)
    with pytest.raises(OSError, match="I/O error"):
        store.adopt(str(data))
    assert list(root.iterdir()) == []
    assert data.exists() and meta.exists()


def test_store_refuses_dirs_on_different_filesystems(dirs, monkeypatch):
    """adopt() hard-links, which cannot cross filesystems: fail at startup,
    not on the first snippet."""
    staging, root = dirs
    monkeypatch.setattr(
        snippet_store, "_device_of", lambda path: 1 if Path(path).name == "staging" else 2
    )
    with pytest.raises(ValueError, match="same filesystem"):
        LocalSnippetStore(staging, root)


def test_rejection_messages_quote_untrusted_names(dirs):
    """A newline in a staged name must not be able to forge a log line."""
    staging, root = dirs
    store = LocalSnippetStore(staging, root)
    error = _rejected(store, staging / "evil\nINFO forged.sigmf-data", "missing_file")
    assert "\n" not in str(error)


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
