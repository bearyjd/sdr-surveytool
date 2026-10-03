# tests/storage/test_snippet_store.py
import os

import pytest

from storage.snippet_store import LocalSnippetStore


def _stage(directory, stem: str = "snip") -> tuple:
    directory.mkdir(parents=True, exist_ok=True)
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
    staging.mkdir()
    victim, _ = _stage(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="staging directory"):
        LocalSnippetStore(staging, root).adopt(str(victim))
    assert victim.exists()


def test_adopt_rejects_dot_dot_traversal_out_of_staging(dirs, tmp_path):
    staging, root = dirs
    staging.mkdir()
    victim, _ = _stage(tmp_path / "elsewhere")
    sneaky = staging / ".." / "elsewhere" / "snip.sigmf-data"
    with pytest.raises(ValueError, match="staging directory"):
        LocalSnippetStore(staging, root).adopt(str(sneaky))
    assert victim.exists()


def test_adopt_rejects_a_symlink_escaping_staging(dirs, tmp_path):
    staging, root = dirs
    victim, _ = _stage(tmp_path / "elsewhere")
    staging.mkdir()
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
    root.mkdir()
    (root / "snip.sigmf-data").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        LocalSnippetStore(staging, root).adopt(str(data))
    assert (root / "snip.sigmf-data").read_bytes() == b"original"
    assert data.exists()
