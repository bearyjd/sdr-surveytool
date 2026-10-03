# tests/agent/test_snippet_reader.py
import errno
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import sigmf.hashing
import sigmf.sigmffile

import agent.snippet_reader
from agent.snippet_reader import (
    THRESHOLD_KEY,
    SnippetOutsideStore,
    SnippetUnreadable,
    read_snippet,
)
from capture.unknown.snippet_writer import write_sigmf_snippet

FS = 100_000.0
FREQ = 915e6
START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path) -> Path:
    root = tmp_path / "snippets"
    root.mkdir()
    return root


def _write(directory: Path, n: int = 4096, fs: float = FS) -> Path:
    """A real step-4 snippet pair (capture.unknown's own writer)."""
    iq = (np.arange(n) % 7).astype(np.complex64)
    return write_sigmf_snippet(iq, directory, fs, FREQ, START)


def _edit_meta(data_path: Path, edit) -> None:
    meta = data_path.with_suffix(".sigmf-meta")
    content = json.loads(meta.read_text())
    edit(content)
    meta.write_text(json.dumps(content))


def test_reads_a_step4_snippet(store):
    data = _write(store)
    snippet = read_snippet(str(data), store)
    assert snippet.sample_rate == FS and snippet.center_freq_hz == FREQ
    np.testing.assert_array_equal(snippet.iq, (np.arange(4096) % 7).astype(np.complex64))
    assert snippet.iq.dtype == np.complex64 and not snippet.truncated


def test_non_finite_samples_are_zeroed_and_counted(store):
    """As step 4 measures: NaN/inf (DMA or driver corruption) never crash
    the analysis; they are zeroed, counted, and reduce confidence later."""
    iq = (np.arange(4096) % 7 + 1).astype(np.complex64)
    iq[[3, 100]] = np.nan
    iq[7] = np.inf
    data = write_sigmf_snippet(iq, store, FS, FREQ, START)
    snippet = read_snippet(str(data), store)
    assert snippet.non_finite_samples == 3
    assert np.isfinite(snippet.iq).all() and snippet.iq[3] == 0 and snippet.iq[7] == 0


def test_a_snippet_with_no_finite_sample_is_unreadable(store):
    data = write_sigmf_snippet(np.full(4096, np.nan, np.complex64), store, FS, FREQ, START)
    with pytest.raises(SnippetUnreadable, match="NaN or inf"):
        read_snippet(str(data), store)


def test_the_record_must_agree_with_its_file(store):
    data = _write(store)
    assert read_snippet(str(data), store, FS, FREQ).sample_rate == FS
    # The record's read-back rate is authoritative when the two agree.
    assert read_snippet(str(data), store, FS * (1 + 1e-12), FREQ).sample_rate == FS * (1 + 1e-12)
    with pytest.raises(SnippetUnreadable, match="sample rate"):
        read_snippet(str(data), store, 2 * FS, FREQ)
    with pytest.raises(SnippetUnreadable, match="center frequency"):
        read_snippet(str(data), store, FS, FREQ + 1e3)


def test_step4s_pre_trigger_annotation_is_the_noise_reference(store):
    iq = (np.arange(4096) % 7).astype(np.complex64)
    annotated = write_sigmf_snippet(iq, store, FS, FREQ, START, trigger_offset=1024)
    assert read_snippet(str(annotated), store).pre_trigger_samples == 1024
    assert read_snippet(str(_write(store)), store).pre_trigger_samples == 0


@pytest.mark.parametrize(
    "annotation",
    [
        {"core:label": "pre_trigger", "core:sample_start": 5, "core:sample_count": 100},
        {"core:label": "pre_trigger", "core:sample_start": 0, "core:sample_count": True},
        {"core:label": "pre_trigger", "core:sample_start": 0, "core:sample_count": "9"},
        {"core:label": "noise", "core:sample_start": 0, "core:sample_count": 100},
    ],
)
def test_a_malformed_pre_trigger_annotation_means_no_reference(store, annotation):
    data = _write(store)
    _edit_meta(data, lambda m: m.update({"annotations": [annotation]}))
    assert read_snippet(str(data), store).pre_trigger_samples == 0


def test_step4s_recorded_trigger_threshold_is_read(store):
    iq = (np.arange(4096) % 7).astype(np.complex64)
    recorded = write_sigmf_snippet(iq, store, FS, FREQ, START, trigger_offset=1024, threshold_dbfs=-37.5)
    assert read_snippet(str(recorded), store).trigger_threshold_dbfs == -37.5
    unrecorded = write_sigmf_snippet(iq, store, FS, FREQ, START, trigger_offset=1024)
    assert read_snippet(str(unrecorded), store).trigger_threshold_dbfs is None


@pytest.mark.parametrize("threshold", ["-30", True, float("nan"), float("inf"), 0.0, 100.0, None])
def test_a_malformed_threshold_means_none_not_unreadable(store, threshold):
    """The meta is untrusted: a crafted threshold (say +100 dBFS, which would
    make an always-on emitter's pre-trigger count as quiet) is ignored."""
    data = _write(store)
    annotation = {"core:label": "pre_trigger", "core:sample_start": 0, "core:sample_count": 1024}
    _edit_meta(data, lambda m: m.update({"annotations": [{**annotation, THRESHOLD_KEY: threshold}]}))
    snippet = read_snippet(str(data), store)
    assert (snippet.pre_trigger_samples, snippet.trigger_threshold_dbfs) == (1024, None)


def test_the_reference_is_clipped_to_what_was_read(store, monkeypatch):
    monkeypatch.setattr(agent.snippet_reader, "MAX_SAMPLES", 2048)
    data = write_sigmf_snippet(np.zeros(8192, np.complex64), store, FS, FREQ, START, trigger_offset=4096)
    assert read_snippet(str(data), store).pre_trigger_samples == 2048


def test_a_transient_io_error_is_flagged_transient(store, monkeypatch):
    """EIO (a flaky disk or NFS) may pass; the agent retries the record
    later. A missing or corrupt file will not, and is about the record."""
    data = _write(store)

    def eio(*args, **kwargs):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(agent.snippet_reader, "_read_fd", eio)
    with pytest.raises(SnippetUnreadable) as excinfo:
        read_snippet(str(data), store)
    assert excinfo.value.transient


def test_a_missing_or_corrupt_snippet_is_not_transient(store):
    """Only a missing file says `missing`: many in a row mean the wrong copy
    of the store is mounted, which the agent must not blame on records."""
    with pytest.raises(SnippetUnreadable) as missing:
        read_snippet(str(store / "gone.sigmf-data"), store)
    data = _write(store)
    data.with_suffix(".sigmf-meta").write_text("{not json")
    with pytest.raises(SnippetUnreadable) as corrupt:
        read_snippet(str(data), store)
    assert not missing.value.transient and not corrupt.value.transient
    assert missing.value.missing and not corrupt.value.missing


def test_never_hashes_the_data_file(store, monkeypatch):
    """sigmf's default checksum pass reads the whole file (214 ms for a
    1 s / 20 MS/s snippet, against 1.6 ms without)."""
    data = _write(store)
    calls = []
    monkeypatch.setattr(sigmf.hashing, "calculate_sha512", lambda *a, **k: calls.append(a) or "x")
    read_snippet(str(data), store)
    assert calls == []


def test_caps_at_two_seconds_of_the_files_sample_rate(store):
    data = _write(store, n=3 * 4096, fs=4096.0)  # 3 s of samples
    snippet = read_snippet(str(data), store)
    assert len(snippet.iq) == 2 * 4096 and snippet.truncated


def test_caps_at_max_samples_whatever_the_sample_rate(store, monkeypatch):
    """The file's sample rate is untrusted: at 1e12 S/s, '2 s' caps
    nothing. The absolute ceiling (2**25 samples) still does; lowered here
    so the test needs no 256 MiB file."""
    assert agent.snippet_reader.MAX_SAMPLES == 1 << 25
    monkeypatch.setattr(agent.snippet_reader, "MAX_SAMPLES", 2048)
    data = _write(store, n=8192)
    _edit_meta(data, lambda m: m["global"].update({"core:sample_rate": 1e12}))
    snippet = read_snippet(str(data), store)
    assert len(snippet.iq) == 2048 and snippet.truncated


@pytest.mark.parametrize(
    "make_path",
    [
        lambda store, outside: None,
        lambda store, outside: "",
        lambda store, outside: "relative/a.sigmf-data",
        lambda store, outside: str(outside),  # a real snippet, elsewhere
        lambda store, outside: str(store / ".." / outside.parent.name / outside.name),  # ../ traversal
        lambda store, outside: str(outside.with_suffix(".sigmf-meta")),  # wrong suffix
    ],
)
def test_rejects_paths_outside_the_store(store, tmp_path, make_path):
    outside_dir = tmp_path / "elsewhere"
    outside = _write(outside_dir)
    with pytest.raises(SnippetOutsideStore):
        read_snippet(make_path(store, outside), store)


def test_rejects_a_symlink_out_of_the_store(store, tmp_path):
    outside = _write(tmp_path / "elsewhere")
    link = store / outside.name
    link.symlink_to(outside)
    store.joinpath(outside.with_suffix(".sigmf-meta").name).symlink_to(outside.with_suffix(".sigmf-meta"))
    with pytest.raises(SnippetOutsideStore):
        read_snippet(str(link), store)


def _read_in_thread(path: Path, store: Path) -> BaseException | None:
    """read_snippet in a thread: a FIFO would block a plain open() forever."""
    outcome: list[BaseException | None] = []

    def target():
        try:
            read_snippet(str(path), store)
            outcome.append(None)
        except BaseException as exc:  # noqa: BLE001 - handed back to the test
            outcome.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive(), "read_snippet hung on a non-regular file"
    return outcome[0]


@pytest.mark.parametrize("which", [".sigmf-meta", ".sigmf-data"])
def test_a_fifo_in_the_store_is_unreadable_and_never_hangs(store, which):
    """A FIFO (or a device) planted in the store would block the reader, and
    with it the whole agent. Only regular files are read."""
    data = _write(store)
    target = data.with_suffix(which)
    target.unlink()
    os.mkfifo(target)
    outcome = _read_in_thread(data, store)
    assert isinstance(outcome, SnippetUnreadable) and "not a regular file" in str(outcome)
    assert not outcome.transient and not outcome.missing


def test_a_symlink_inside_the_store_is_unreadable(store):
    """Step 4 hard-links snippets into the store; a symlink is not one of
    its files, even when it points at one."""
    real = _write(store / "real")
    link = store / real.name
    link.symlink_to(real)
    store.joinpath(real.with_suffix(".sigmf-meta").name).symlink_to(real.with_suffix(".sigmf-meta"))
    with pytest.raises(SnippetUnreadable, match="not a regular file"):
        read_snippet(str(link), store)


def test_a_meta_naming_another_dataset_is_never_followed(store, tmp_path):
    """sigmf's fromfile() follows core:dataset (relative to the meta, or
    absolute), so a crafted meta inside the store could make it read any
    file. The reader never calls fromfile, and refuses such a meta."""
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"\1" * 8 * 4096)
    data = _write(store)
    _edit_meta(data, lambda m: m["global"].update({"core:dataset": str(secret)}))
    with pytest.raises(SnippetUnreadable, match="non-conforming dataset"):
        read_snippet(str(data), store)


def test_missing_files_are_unreadable_not_outside(store):
    data = _write(store)
    data.unlink()
    with pytest.raises(SnippetUnreadable):
        read_snippet(str(data), store)
    with pytest.raises(SnippetUnreadable):
        read_snippet(str(store / "never-written.sigmf-data"), store)


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["global"].update({"core:datatype": "ci16_le"}),
        lambda m: m["global"].update({"core:num_channels": 2}),
        lambda m: m["global"].update({"core:sample_rate": -1.0}),
        lambda m: m["global"].pop("core:sample_rate"),
        lambda m: m["captures"][0].pop("core:frequency"),
        lambda m: m.pop("global"),
    ],
)
def test_malformed_metadata_is_unreadable(store, edit):
    data = _write(store)
    _edit_meta(data, edit)
    with pytest.raises(SnippetUnreadable):
        read_snippet(str(data), store)


def test_oversized_or_garbage_meta_is_unreadable(store):
    data = _write(store)
    meta = data.with_suffix(".sigmf-meta")
    meta.write_text("{not json")
    with pytest.raises(SnippetUnreadable):
        read_snippet(str(data), store)
    meta.write_text(" " * ((1 << 20) + 1))
    with pytest.raises(SnippetUnreadable, match="larger than"):
        read_snippet(str(data), store)


def test_too_short_snippet_is_unreadable(store):
    data = _write(store, n=100)
    with pytest.raises(SnippetUnreadable, match="need 1024"):
        read_snippet(str(data), store)
