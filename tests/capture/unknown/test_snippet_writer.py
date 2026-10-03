# tests/capture/unknown/test_snippet_writer.py
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import sigmf
from sigmf import sigmffile

from capture.unknown.snippet_writer import write_sigmf_snippet

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
