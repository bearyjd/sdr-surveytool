import sys
from pathlib import Path

# Ensure the project root is in the path for pytest
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the real top-level packages before pytest collects tests/. Under
# --import-mode=importlib, tests/ingest/__init__.py (likewise tests/storage/,
# tests/capture/) would otherwise be registered as the top-level package of
# the same name, and its submodules would be resolved by whatever other
# finder can supply them -- in a git worktree, the user-site editable install
# pointing at the main checkout. See tests/ingest/test_import_resolution.py.
import capture  # noqa: E402,F401
import ingest  # noqa: E402,F401
import schema  # noqa: E402,F401
import storage  # noqa: E402,F401
import viz  # noqa: E402,F401
