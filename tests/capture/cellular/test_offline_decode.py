from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from capture.cellular.offline_scanner import run_cellsearch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CELLSEARCH_BINARY = (
    _REPO_ROOT
    / "capture/cellular/vendor/lte-cell-scanner/build/src/CellSearch"
)
_FIXTURE = _REPO_ROOT / "capture/cellular/testdata/cell301_1815.3mhz.bin"
_EXPECTED = _REPO_ROOT / "capture/cellular/testdata/cell301_1815.3mhz.expected.json"

pytestmark = pytest.mark.skipif(
    not _CELLSEARCH_BINARY.exists(),
    reason=(
        "CellSearch binary not built; run capture/cellular/build.sh "
        "(see capture/cellular/README.md)"
    ),
)


def test_offline_decode_detects_known_cell():
    expected = json.loads(_EXPECTED.read_text())

    cells = run_cellsearch(
        str(_CELLSEARCH_BINARY),
        str(_FIXTURE),
        freq_start_hz=1815300000,
        extra_env={
            "LD_LIBRARY_PATH": "/usr/local/lib:"
            + os.environ.get("LD_LIBRARY_PATH", "")
        },
    )

    assert len(cells) == 1
    cell = cells[0]
    assert cell["duplex"] == expected["duplex"]
    assert cell["cell_id"] == expected["cell_id"]
    assert cell["pss_id"] == expected["pss_id"]
    assert cell["freq_mhz"] == pytest.approx(expected["freq_mhz"], abs=0.01)
    assert cell["rx_power_db"] == pytest.approx(expected["rx_power_db"], abs=0.01)
    assert cell["freq_offset_hz"] == pytest.approx(expected["freq_offset_hz"], abs=1.0)
