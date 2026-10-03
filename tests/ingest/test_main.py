import logging

import pytest

from ingest.main import _open_snippet_store, _parse_args
from storage.snippet_store import DEFAULT_MAX_SNIPPET_BYTES


def test_gps_fix_quality_is_required():
    with pytest.raises(SystemExit):
        _parse_args(["--gps-lat", "47.6", "--gps-lon", "-122.3"])


def test_gps_fix_quality_zero_is_accepted_explicitly():
    args = _parse_args(["--gps-fix-quality", "0"])
    assert args.gps_fix_quality == 0


def test_gps_fix_quality_passthrough():
    args = _parse_args(["--gps-fix-quality", "4", "--gps-lat", "47.6", "--gps-lon", "-122.3"])
    assert args.gps_fix_quality == 4
    assert args.gps_lat == 47.6
    assert args.gps_lon == -122.3


def test_snippet_directories_default_under_data():
    args = _parse_args(["--gps-fix-quality", "0"])
    assert args.snippet_staging_dir == "data/snippet-staging"
    assert args.snippet_store_dir == "data/snippets"


def test_snippet_directories_passthrough():
    args = _parse_args(
        [
            "--gps-fix-quality", "0",
            "--snippet-staging-dir", "/srv/staging",
            "--snippet-store-dir", "/srv/snippets",
        ]
    )
    assert args.snippet_staging_dir == "/srv/staging"
    assert args.snippet_store_dir == "/srv/snippets"


def test_snippet_store_dirs_are_resolved_secured_and_logged_at_startup(tmp_path, monkeypatch, caplog):
    """The relative defaults resolve against the working directory, which can
    differ between the capture and ingest processes: log the absolute dirs."""
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        store = _open_snippet_store(_parse_args(["--gps-fix-quality", "0"]))
    assert store.staging_dir == (tmp_path / "data" / "snippet-staging").resolve()
    assert store.root_dir == (tmp_path / "data" / "snippets").resolve()
    assert str(store.staging_dir) in caplog.text
    assert str(store.root_dir) in caplog.text


def test_snippet_size_bound_defaults_to_the_largest_capture_and_passes_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _open_snippet_store(_parse_args(["--gps-fix-quality", "0"])).max_snippet_bytes == (
        DEFAULT_MAX_SNIPPET_BYTES
    )
    store = _open_snippet_store(_parse_args(["--gps-fix-quality", "0", "--max-snippet-bytes", "1024"]))
    assert store.max_snippet_bytes == 1024
