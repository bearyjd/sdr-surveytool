# tests/ingest/test_import_resolution.py
"""Guards against silently testing another checkout's code.

Under --import-mode=importlib, tests/ingest/__init__.py (likewise
tests/storage/ and tests/capture/) gets registered as the top-level
`ingest` package. Submodules such as `ingest.service` then come from any
other finder able to supply them -- in a git worktree, the user-site
editable install pointing at the main checkout -- so the worktree's tests
would run against main's code. conftest.py pre-imports the real packages
to prevent that.
"""
from pathlib import Path

import pytest

import capture.common.emitter
import ingest.service
import storage.repository

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", [capture.common.emitter, ingest.service, storage.repository])
def test_code_under_test_comes_from_this_checkout(module):
    assert Path(module.__file__).resolve().is_relative_to(REPO_ROOT)
