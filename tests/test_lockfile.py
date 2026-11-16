"""requirements-lock.txt must pin every dependency with hashes."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "requirements-lock.txt"


def _names(specs):
    return {re.split(r"[<>=!~\[; ]", s, maxsplit=1)[0].lower().replace("_", "-") for s in specs}


def _pins():
    text = LOCK.read_text()
    pins = re.findall(r"^([A-Za-z0-9_.-]+)==\S+", text, re.M)
    return text, {p.lower().replace("_", "-") for p in pins}


def test_lockfile_exists():
    assert LOCK.is_file(), "requirements-lock.txt missing"


def test_lockfile_covers_declared_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = _names(project["dependencies"]) | _names(project["optional-dependencies"]["dev"])
    _, pinned = _pins()
    assert declared <= pinned, f"unpinned: {sorted(declared - pinned)}"


def test_every_pin_has_hashes():
    text, pinned = _pins()
    entries = re.split(r"^(?=[A-Za-z0-9_.-]+==)", text, flags=re.M)[1:]
    assert len(entries) == len(pinned)
    assert all("--hash=sha256:" in e for e in entries)
