# tests/capture/unknown/test_snippet_writer.py
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import sigmf
from sigmf import sigmffile

from capture.unknown import snippet_writer
from capture.unknown.snippet_writer import write_sigmf_snippet
from storage.snippet_store import STAGED_DATA_NAME

START = datetime(2026, 10, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)
IQ = (np.arange(1_000) * (1 - 2j) / 1_000).astype(np.complex64)


def test_writes_readable_sigmf_pair_and_returns_absolute_data_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = write_sigmf_snippet(IQ, Path("staging"), 2e6, 915e6, START)

    assert data_path.is_absolute()
    assert data_path.parent == (tmp_path / "staging").resolve()
    assert data_path.suffix == ".sigmf-data"
    assert data_path.with_suffix(".sigmf-meta").is_file()

    recording = sigmffile.fromfile(str(data_path))
    np.testing.assert_array_equal(recording.read_samples(), IQ)
    assert recording.get_global_field(sigmf.DATATYPE_KEY) == "cf32_le"
    assert recording.get_global_field(sigmf.SAMPLE_RATE_KEY) == 2e6
    capture = recording.get_captures()[0]
    assert capture[sigmf.FREQUENCY_KEY] == 915e6
    assert capture[sigmf.DATETIME_KEY] == "2026-10-03T12:00:00.123456Z"


def test_two_snippets_with_identical_start_time_do_not_collide(tmp_path):
    first = write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, START)
    second = write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, START)
    assert first != second
    assert len(list(tmp_path.glob("*.sigmf-data"))) == 2


def test_rejects_naive_capture_start(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, datetime(2026, 10, 3, 12, 0, 0))


def test_snippet_files_are_private_and_meta_names_no_temp_file(tmp_path):
    data_path = write_sigmf_snippet(IQ, tmp_path / "staging", 2e6, 915e6, START)
    meta_path = data_path.with_suffix(".sigmf-meta")
    assert data_path.stat().st_mode & 0o777 == 0o600
    assert meta_path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "staging").stat().st_mode & 0o077 == 0
    # sigmf writes core:dataset when the data file isn't the conforming
    # sibling name; the temporary name must never end up in the meta.
    assert "core:dataset" not in json.loads(meta_path.read_text())["global"]
    assert sorted(p.name for p in (tmp_path / "staging").iterdir()) == sorted(
        [data_path.name, meta_path.name]
    )


def test_failed_meta_write_leaves_no_visible_or_temporary_files(tmp_path, monkeypatch):
    def broken_dump(self, filep, pretty=True):
        raise OSError("disk full")

    monkeypatch.setattr(snippet_writer.SigMFFile, "dump", broken_dump)
    with pytest.raises(OSError, match="disk full"):
        write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, START)
    assert list(tmp_path.iterdir()) == []


def test_failed_meta_rename_never_leaves_a_half_pair(tmp_path, monkeypatch):
    real_replace = os.replace

    def replace_failing_for_meta(src, dst):
        if str(dst).endswith(".sigmf-meta"):
            raise OSError("crash before the meta rename")
        real_replace(src, dst)

    monkeypatch.setattr(snippet_writer.os, "replace", replace_failing_for_meta)
    with pytest.raises(OSError, match="meta rename"):
        write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, START)
    assert list(tmp_path.iterdir()) == []


def test_written_names_match_what_the_snippet_store_accepts(tmp_path):
    """The store only adopts names in the writer's scheme; this pins the two
    sides of that contract together."""
    data_path = write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, START)
    assert STAGED_DATA_NAME.fullmatch(data_path.name)
