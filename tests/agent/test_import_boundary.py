# tests/agent/test_import_boundary.py
"""A tripwire, not the boundary: the container is the boundary (no ingest
socket, the snippet store read-only, egress only to the database and
api.anthropic.com; see agent/README.md). This AST scan catches the obvious
ways the agent's code could grow a path out of it: shelling out, opening
sockets, loading code dynamically, writing files, logging to the network,
or importing capture/ingest/storage. It scans agent/ and the two packages
the agent imports, dsp/ and schema/."""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ALLOWED_MODULES = {
    "__future__", "argparse", "collections", "dataclasses", "datetime", "enum", "errno", "functools",
    "json", "logging", "math", "os", "pathlib", "re", "signal", "statistics", "time", "typing",
    "anthropic", "numpy", "pydantic", "sigmf",
    "agent", "dsp", "schema",
}
DATABASE_MODULES = {"sqlalchemy", "psycopg"}  # agent/db_gateway.py only
DATABASE_GATEWAY = "agent/db_gateway.py"
BANNED_NAMES = {"__import__", "exec", "eval", "compile", "getattr", "open", "__builtins__"}
BANNED_LOGGING = {"config", "handlers"}  # file and network log handlers
OS_ALLOWED = {"environ", "getenv"}


def violations(source: str, path: str = "agent/example.py") -> list[str]:
    allowed = ALLOWED_MODULES | (DATABASE_MODULES if path == DATABASE_GATEWAY else set())
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    found.append(f"import {alias.name}")
                elif alias.name == "os" and alias.asname is not None:
                    found.append(f"import os as {alias.asname}")
                elif alias.name.split(".")[:2] in (["logging", "config"], ["logging", "handlers"]):
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top = node.module.split(".")[0]
            if top not in allowed:
                found.append(f"from {node.module} import ...")
            elif top == "os":
                found += [f"from os import {a.name}" for a in node.names if a.name not in OS_ALLOWED]
            elif node.module in ("logging.config", "logging.handlers") or (
                node.module == "logging" and any(a.name in BANNED_LOGGING for a in node.names)
            ):
                found.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            found.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr not in OS_ALLOWED:
                found.append(f"os.{node.attr}")
            elif node.value.id == "logging" and node.attr in BANNED_LOGGING:
                found.append(f"logging.{node.attr}")
    return found


SCANNED = sorted(path for package in ("agent", "dsp", "schema") for path in (REPO / package).rglob("*.py"))


def test_the_packages_are_scanned():
    names = {str(path.relative_to(REPO)) for path in SCANNED}
    assert {"agent/service.py", "agent/db_gateway.py", "dsp/segmentation.py", "schema/records.py"} <= names


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: str(p.relative_to(REPO)))
def test_module_stays_inside_the_boundary(path):
    assert violations(path.read_text(), str(path.relative_to(REPO))) == []


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess",
        "from subprocess import run",
        "import socket",
        "import ctypes",
        "import importlib",
        "from importlib import import_module",
        "import multiprocessing",
        "import sys",
        "import requests",
        "import shutil",
        "import capture.unknown.service",
        "from ingest.queue_server import QueueServer",
        "from storage.db import make_engine",
        "__import__('subprocess')",
        "exec('x = 1')",
        "eval('1 + 1')",
        "compile('x', 'f', 'exec')",
        "open('/etc/passwd')",
        "__builtins__['exec']('x')",
        # The bypasses the plan review named:
        "import os as o\no.system('ls')",
        "import os\ngetattr(os, 'system')('ls')",
        "from logging.handlers import SysLogHandler\nSysLogHandler(address=('collector', 514))",
        "import logging.handlers\nlogging.handlers.SysLogHandler(address=('collector', 514))",
        "import logging\nlogging.config.dictConfig({})",
        "from logging import handlers",
        "import os\nos.system('ls')",
        "import os\nos.popen('ls')",
        "import os\nos.execv('/bin/sh', [])",
        "import os\nos.remove('/x')",
        "from os import system",
        # Database drivers outside the gateway:
        "import sqlalchemy",
        "from psycopg import connect",
    ],
)
def test_checker_catches(source):
    assert violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "from __future__ import annotations",
        "from collections.abc import Callable",
        "import os\nkey = os.environ.get('ANTHROPIC_API_KEY')",
        "from os import environ",
        "import logging\nlogging.getLogger(__name__).info('x')",
        "import numpy as np",
        "from pathlib import Path\nPath('x').open('rb')",
        "from agent.llm import TOOL_NAME",
        "from dsp.features import region_features",
        "from schema.records import ClassificationStatus",
    ],
)
def test_checker_allows(source):
    assert violations(source) == []


def test_database_drivers_are_allowed_only_in_the_gateway():
    source = "import psycopg\nfrom sqlalchemy import text"
    assert violations(source, DATABASE_GATEWAY) == []
    assert violations(source, "agent/service.py") != []
