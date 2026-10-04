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


def test_snippet_store_is_off_by_default_and_never_touches_var_lib(monkeypatch):
    """WiFi/BT-only deployments must start exactly as before: no snippet
    directory is created or checked unless the operator opts in."""

    def no_store_expected(*args, **kwargs):
        raise AssertionError("snippet store opened without opting in")

    monkeypatch.setattr("ingest.main.LocalSnippetStore", no_store_expected)
    args = _parse_args(["--gps-fix-quality", "0"])
    assert (args.snippet_staging_dir, args.snippet_store_dir) == (None, None)
    assert _open_snippet_store(args) is None


@pytest.mark.parametrize("flag", ["--snippet-staging-dir", "--snippet-store-dir"])
def test_opting_in_takes_both_snippet_directories(flag, tmp_path):
    with pytest.raises(SystemExit):
        _parse_args(["--gps-fix-quality", "0", flag, str(tmp_path / "x")])


@pytest.mark.parametrize("relative", ["staging", "store"])
def test_relative_snippet_directories_are_rejected(relative, tmp_path):
    """Capture and ingest resolve relative paths against their own working
    directories; if those differ every snippet is rejected as outside_staging."""
    staging = "data/snippet-staging" if relative == "staging" else str(tmp_path / "staging")
    store = "data/snippets" if relative == "store" else str(tmp_path / "snippets")
    with pytest.raises(SystemExit):
        _parse_args(["--gps-fix-quality", "0", "--snippet-staging-dir", staging, "--snippet-store-dir", store])


def test_an_opted_in_store_with_a_bad_dir_fails_fast(tmp_path):
    shared = tmp_path / "staging"
    shared.mkdir()
    shared.chmod(0o755)
    with pytest.raises(PermissionError, match="chmod 700"):
        _open_snippet_store(_dir_args(tmp_path))


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


def _dir_args(tmp_path, *extra):
    return _parse_args(
        [
            "--gps-fix-quality", "0",
            "--snippet-staging-dir", str(tmp_path / "staging"),
            "--snippet-store-dir", str(tmp_path / "snippets"),
            *extra,
        ]
    )


def test_snippet_store_dirs_are_secured_and_logged_at_startup(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        store = _open_snippet_store(_dir_args(tmp_path))
    assert store.staging_dir == (tmp_path / "staging").resolve()
    assert store.root_dir == (tmp_path / "snippets").resolve()
    assert str(store.staging_dir) in caplog.text
    assert str(store.root_dir) in caplog.text


def test_snippet_size_bound_defaults_to_the_largest_capture_and_passes_through(tmp_path):
    assert _open_snippet_store(_dir_args(tmp_path)).max_snippet_bytes == DEFAULT_MAX_SNIPPET_BYTES
    assert _open_snippet_store(_dir_args(tmp_path, "--max-snippet-bytes", "1024")).max_snippet_bytes == 1024
