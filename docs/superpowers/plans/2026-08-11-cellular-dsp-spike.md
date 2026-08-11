# Cellular DSP Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the hardware-free half of the multi-modality survey tool's Phase 1
(cellular DSP spike): prove LTE-Cell-Scanner's PSS/SSS cell search works end-to-end
against a recorded LTE IQ file, wired into this repo's `UnifiedRecord` schema, with zero
physical SDR hardware.

**Architecture:** Vendor LTE-Cell-Scanner (AGPL-3.0) as an unmodified git submodule,
build only its hardware-free `CellSearch` binary via CMake, invoke it as a subprocess
against a pre-recorded `.bin` IQ file, parse its known text stdout in Python, and
normalize the result into `schema.records.UnifiedRecord` — mirroring the existing
`capture/wifi` split (`kismet_client.py` → `normalizer.py`).

**Tech Stack:** Python 3.10+ (subprocess, regex), vendored C++ (LTE-Cell-Scanner, built
via CMake — IT++, Boost, FFTW, RTL-SDR-devel as build-time-only deps), numpy (dev-only,
fixture regeneration), pytest.

## Global Constraints

- Python >= 3.10 (matches `pyproject.toml`).
- No changes to LTE-Cell-Scanner's vendored source — CMake config flags only. Preserves
  a clean, traceable diff against upstream and avoids AGPL compile-time linking concerns.
- No MIB/SIB1-3, no RSRP/RSRQ/SINR in this plan — `CellSearch` (PSS/SSS search only)
  doesn't report them. `Signal.rsrp`/`rsrq`/`snr` stay `None`. Deferred to a separate plan
  once `LTE-Tracker`'s offline-batch feasibility is confirmed.
- No live-radio code (`capture/cellular/service.py`, gain-control layer) in this plan —
  everything here reads pre-recorded `.bin` files only.
- Verified facts below (build commands, CLI flags, output format, byte layouts) come from
  an actual from-scratch build and run in a Fedora 43 sandbox — not documentation or
  guesses. Package names/paths are Fedora/dnf-specific; other distros need equivalent
  packages under different names.

---

### Task 1: Vendor LTE-Cell-Scanner and build tooling

**Files:**
- Create: `capture/cellular/vendor/lte-cell-scanner` (git submodule, upstream
  `https://github.com/JiaoXianjun/LTE-Cell-Scanner.git`)
- Create: `capture/cellular/__init__.py`
- Create: `capture/cellular/build.sh`
- Modify: `capture/cellular/README.md`
- Modify: `pyproject.toml` (add `capture.cellular` to `[tool.setuptools].packages`)
- Modify: `.gitignore` (ignore the vendored submodule's build output directory)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a buildable `capture/cellular/vendor/lte-cell-scanner/build/src/CellSearch`
  binary once `build.sh` is run. Later tasks (3, 5) reference this exact path. `Task 3`
  also relies on `LD_LIBRARY_PATH=/usr/local/lib` being required at runtime (IT++ is
  installed to `/usr/local` and is not in the default loader search path — verified).

This task has no unit-testable logic of its own (it's vendoring + a shell build script),
so there's no TDD red/green cycle here — just the steps to add and verify the submodule
builds.

- [ ] **Step 1: Add the submodule**

```bash
git submodule add https://github.com/JiaoXianjun/LTE-Cell-Scanner.git capture/cellular/vendor/lte-cell-scanner
git submodule update --init --recursive
```

- [ ] **Step 2: Create `capture/cellular/__init__.py`**

Empty file — makes `capture.cellular` an importable package, matching
`capture/wifi/__init__.py` and `capture/bluetooth/__init__.py`.

- [ ] **Step 3: Write `capture/cellular/build.sh`**

```bash
#!/usr/bin/env bash
# Builds the vendored LTE-Cell-Scanner's CellSearch binary, hardware-free
# (no BladeRF/HackRF/OpenCL). Verified working on Fedora 43; other distros
# need equivalent packages under different names.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_DIR="$REPO_ROOT/capture/cellular/vendor/lte-cell-scanner"

if [ ! -d "$VENDOR_DIR/.git" ] && [ ! -f "$VENDOR_DIR/.git" ]; then
    echo "error: $VENDOR_DIR submodule not initialized." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

echo "== Installing system build dependencies (Fedora/dnf) =="
sudo dnf install -y rtl-sdr-devel boost-devel fftw-devel ncurses-devel \
    blas-devel lapack-devel cmake gcc-c++ make git
# rtl-sdr-devel is required even for this hardware-free build: LTE-Cell-Scanner's
# CMakeLists.txt unconditionally FIND_PACKAGE(RTLSDR REQUIRED) whenever BladeRF
# and HackRF are both disabled (verified — USE_RTLSDR is dead code, never gated
# on). No RTL-SDR hardware is touched at runtime; this is a build-time-only lib.

if ! ldconfig -p | grep -q libitpp; then
    echo "== Building IT++ from source (not packaged on Fedora) =="
    ITPP_SRC="$(mktemp -d)"
    git clone --depth 1 https://git.code.sf.net/p/itpp/git "$ITPP_SRC"
    mkdir "$ITPP_SRC/build"
    (
        cd "$ITPP_SRC/build"
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
        make -j"$(nproc)"
        sudo make install
    )
    sudo ldconfig
    rm -rf "$ITPP_SRC"
fi

echo "== Building CellSearch (hardware-free: no BladeRF/HackRF/OpenCL) =="
mkdir -p "$VENDOR_DIR/build"
(
    cd "$VENDOR_DIR/build"
    cmake -DUSE_OPENCL=0 -DUSE_BLADERF=0 -DUSE_HACKRF=0 -DCMAKE_BUILD_TYPE=Release ..
    make -j"$(nproc)"
)

echo "== Done. Binary at $VENDOR_DIR/build/src/CellSearch =="
echo "Note: IT++ installs to /usr/local, which is not on the default loader"
echo "path — set LD_LIBRARY_PATH=/usr/local/lib when running CellSearch."
```

```bash
chmod +x capture/cellular/build.sh
```

- [ ] **Step 4: Run the build script and verify the binary exists**

Run: `./capture/cellular/build.sh`
Expected: script completes, and `ls capture/cellular/vendor/lte-cell-scanner/build/src/CellSearch` shows the binary.

Sanity-check it runs at all (exit code 255 is correct here — verified real behavior of `--help`, not a failure):

```bash
LD_LIBRARY_PATH=/usr/local/lib capture/cellular/vendor/lte-cell-scanner/build/src/CellSearch --help; echo "exit: $?"
```
Expected: prints the help text, `exit: 255`.

- [ ] **Step 5: Update `capture/cellular/README.md` with build + AGPL notice**

Replace its content with:

```markdown
# capture/cellular

Passive LTE cell broadcast decode: currently PSS/SSS cell search only (Cell ID, PSS ID,
RX power, residual frequency offset). Vendors LTE-Cell-Scanner
(https://github.com/JiaoXianjun/LTE-Cell-Scanner), built unmodified via CMake.

**Hard scope boundary**: broadcast-channel decode only. No paging-channel decoding, no
RRC connection setup, nothing that identifies or tracks individual subscribers.
LTE-Cell-Scanner has no RRC/paging code at all.

**Status**: hardware-free DSP spike complete — `CellSearch` correctly detects a known
cell (Cell ID 301) from a recorded IQ file, see `testdata/`. MIB/SIB1-3 decode
(`LTE-Tracker`, a separate tool) and real xA9 hardware validation are deferred to later
plans; see `docs/superpowers/specs/2026-08-11-cellular-dsp-spike-design.md`.

## Building

```bash
git submodule update --init --recursive
./build.sh
```

Requires Fedora/dnf (or equivalent packages on another distro) — see `build.sh` for the
exact dependency list. IT++ is not packaged on Fedora and is built from source by the
script.

## License notice (AGPL-3.0)

`vendor/lte-cell-scanner/` is a git submodule of LTE-Cell-Scanner, licensed AGPL-3.0.
It is vendored unmodified and invoked as a separate subprocess (not linked into this
Python package), which avoids compile-time licensing entanglement, but AGPL-3.0's
network-use disclosure clause still applies if this tool's output ever becomes part of
a network-facing service (e.g. the planned carrier-facing analytics product) — anyone
wiring this in must offer that service's users the corresponding source, including any
modifications. No modifications exist yet; flag this before shipping such a service.
```

- [ ] **Step 6: Register the package and ignore build output**

In `pyproject.toml`, add `"capture.cellular"` to the `[tool.setuptools].packages` list
(after `"capture.bluetooth"`):

```toml
packages = [
    "schema",
    "capture",
    "capture.common",
    "capture.wifi",
    "capture.bluetooth",
    "capture.cellular",
    "ingest",
    "storage",
    "viz",
]
```

Append to `.gitignore`:

```
capture/cellular/vendor/lte-cell-scanner/build/
```

- [ ] **Step 7: Commit**

```bash
git add .gitmodules capture/cellular pyproject.toml .gitignore
git commit -m "feat: vendor LTE-Cell-Scanner and add hardware-free build tooling"
```

---

### Task 2: Generate the test fixture from the vendored regression capture

**Files:**
- Create: `capture/cellular/testdata/generate_fixture.py`
- Create: `capture/cellular/testdata/cell301_1815.3mhz.bin` (generated, then committed)
- Create: `capture/cellular/testdata/cell301_1815.3mhz.expected.json`
- Modify: `pyproject.toml` (add `numpy` to `[project.optional-dependencies].dev`)

**Interfaces:**
- Consumes: `capture/cellular/vendor/lte-cell-scanner/regression_test_signal_file/f1815.3_s19.2_bw20_0.08s_hackrf-1.bin`
  (Task 1's submodule).
- Produces: `capture/cellular/testdata/cell301_1815.3mhz.bin` — a `--loadbin`-compatible
  fixture (128-byte header + 153,600 samples of 1.92 Msps int8 IQ, 307,328 bytes total)
  that Task 5's integration test runs `CellSearch` against.
  `capture/cellular/testdata/cell301_1815.3mhz.expected.json` — the verified ground
  truth (Cell ID 301, PSS ID 1, RX power −9.44976 dB, freq offset 14302.6 Hz, FDD) that
  Task 5 asserts against.

This is a one-time generation script (not re-run at test time — the `.bin` output is
committed directly), so no TDD cycle; verify by inspecting the output.

- [ ] **Step 1: Add numpy as a dev dependency**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "numpy>=1.26",  # capture/cellular/testdata/generate_fixture.py only
]
```

- [ ] **Step 2: Write `capture/cellular/testdata/generate_fixture.py`**

```python
"""Regenerates cell301_1815.3mhz.bin from the vendored LTE-Cell-Scanner's own
bundled regression capture (19.2 Msps, HackRF, 1815.3 MHz). Not run at test
time — the .bin output here is committed directly. Re-run only if the
vendored submodule updates and its source fixture changes.

Requires numpy (dev-only, see pyproject.toml [project.optional-dependencies].dev).

LTE-Cell-Scanner's CellSearch hardcodes an internal sample rate of 1.92 Msps
(FS_LTE/16); the source recording is 19.2 Msps (10x too fast, meant for a
different tool), so this decimates before writing the --loadbin header
CellSearch's read_header_from_bin() (src/capbuf.cpp) requires.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_VENDOR_FIXTURE = (
    _HERE
    / ".."
    / "vendor"
    / "lte-cell-scanner"
    / "regression_test_signal_file"
    / "f1815.3_s19.2_bw20_0.08s_hackrf-1.bin"
).resolve()
_OUTPUT = _HERE / "cell301_1815.3mhz.bin"

# --loadbin header magic constants, verified from write_header_to_bin() /
# read_header_from_bin() in vendor/lte-cell-scanner/src/capbuf.cpp.
_HEADER_MAGIC = [
    73492.215,
    -0.7923597,
    -189978508,
    93.126712,
    -53243.129,
    0.0008123898,
    -6.0098321,
    237.09983,
]
_FREQ_HZ = 1815300000
_TARGET_SAMPLE_RATE_HZ = 1920000  # CellSearch's hardcoded internal rate (FS_LTE/16)
_DECIMATION = 10  # source is 19.2 Msps


def _decimate_iq(raw: np.ndarray) -> np.ndarray:
    """Band-limits (windowed-sinc FIR) and decimates 19.2 Msps int8 IQ down
    to 1.92 Msps, re-quantized to int8."""
    i = (raw[0::2].astype(np.float64) - 128.0) / 128.0
    q = (raw[1::2].astype(np.float64) - 128.0) / 128.0

    numtaps = 129
    cutoff = 1.0 / _DECIMATION
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    taps = np.sinc(cutoff * n) * cutoff * np.kaiser(numtaps, 8.0)
    taps /= np.sum(taps)

    i_decimated = np.convolve(i, taps, mode="same")[:: _DECIMATION]
    q_decimated = np.convolve(q, taps, mode="same")[:: _DECIMATION]

    i_bytes = np.clip(np.round(i_decimated * 128 + 128), 0, 255).astype(np.uint8)
    q_bytes = np.clip(np.round(q_decimated * 128 + 128), 0, 255).astype(np.uint8)
    out = np.empty(2 * len(i_bytes), dtype=np.uint8)
    out[0::2] = i_bytes
    out[1::2] = q_bytes
    return out


def _build_header() -> bytes:
    """128-byte --loadbin header: 8 (magic double, uint64) pairs,
    little-endian. fc/fs requested==programmed since this is derived from a
    recording, not a live retune."""
    fields = [
        _FREQ_HZ,
        _FREQ_HZ,
        _TARGET_SAMPLE_RATE_HZ,
        _TARGET_SAMPLE_RATE_HZ,
        0,
        0,
        0,
        0,
    ]
    header = bytearray()
    for magic, value in zip(_HEADER_MAGIC, fields):
        header += struct.pack("<d", magic)
        header += struct.pack("<Q", value)
    return bytes(header)


def main() -> None:
    raw = np.fromfile(_VENDOR_FIXTURE, dtype=np.uint8)
    decimated = _decimate_iq(raw)
    _OUTPUT.write_bytes(_build_header() + decimated.tobytes())
    print(f"wrote {_OUTPUT} ({_OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it and verify the output**

```bash
pip install -e ".[dev]"
python capture/cellular/testdata/generate_fixture.py
```
Expected: `wrote .../cell301_1815.3mhz.bin (307328 bytes)` — 128-byte header + 153,600
samples × 2 bytes/sample (one `CAPLENGTH`).

- [ ] **Step 4: Write `capture/cellular/testdata/cell301_1815.3mhz.expected.json`**

```json
{
  "duplex": "FDD",
  "freq_mhz": 1815.3,
  "cell_id": 301,
  "pss_id": 1,
  "rx_power_db": -9.44976,
  "freq_offset_hz": 14302.6
}
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml capture/cellular/testdata
git commit -m "test: add cellular offline-decode fixture from vendored regression capture"
```

---

### Task 3: Offline scanner subprocess wrapper

**Files:**
- Create: `capture/cellular/offline_scanner.py`
- Test: `tests/capture/cellular/test_offline_scanner.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure subprocess + regex parsing).
- Produces: `capture.cellular.offline_scanner.parse_cellsearch_output(stdout: str) ->
  list[dict]` (each dict: `duplex: str`, `freq_mhz: float`, `cell_id: int`, `pss_id:
  int`, `rx_power_db: float`, `freq_offset_hz: float`); `capture.cellular.offline_scanner.run_cellsearch(binary_path:
  str, iq_file_path: str, freq_start_hz: int, timeout_seconds: float = 30.0, extra_env:
  dict[str, str] | None = None) -> list[dict]`. Task 4's normalizer consumes one dict
  from this list at a time. Task 5's integration test calls `run_cellsearch` directly.

`parse_cellsearch_output` is pure and gets full TDD coverage below, tested against the
real verbatim stdout captured from an actual `CellSearch` run. `run_cellsearch` is a
thin subprocess wrapper with no branching logic — not unit-tested here, matching how
`KismetClient` is excluded from unit-test scope in the WiFi/BT plan.

- [ ] **Step 1: Write the failing test**

```python
# tests/capture/cellular/test_offline_scanner.py
from capture.cellular.offline_scanner import parse_cellsearch_output

# Verbatim stdout from an actual `CellSearch -s 1815300000 --loadbin ...` run
# against the fixture generated in Task 2.
REAL_STDOUT = """\
LTE CellSearch (Release) beginning. 1.0 to 1.1.0: An enhanced LTE Cell Scanner/tracker. Xianjun Jiao (putaoshu@msn.com)
  PPM: 0
  correction: 1
Use file begin with 1815.3MHz actual 1815.3MHz 1.92e+06MHz
    Search frequency: 1815.3 to 1815.3 MHz
with freq correction: 0 kHz
    Search PSS at fo: -140 to 135 kHz

Examining center frequency 1815.3 MHz ... try 0
PSS XCORR  cost 4.39247s
Hit  num peaks 1
try peak 0 tdd_flag 0
  Detected a FDD cell! At freqeuncy 1815.3MHz, try 0
    cell ID: 301
     PSS ID: 1
    RX power level: -9.44976 dB
    residual frequency offset: 14302.6 Hz
                     k_factor: 0.999992
try peak 0 tdd_flag 1
Detected the following cells:
DPX:TDD/FDD; A: #antenna ports C: CP type ; P: PHICH duration ; PR: PHICH resource type
DPX CID A      fc   freq-offset RXPWR C nRB P  PR CrystalCorrectionFactor
FDD 301 2 1815.3M         14.3k -9.45 N 100 N one 1.0000078789572046656
"""

NO_CELL_STDOUT = """\
LTE CellSearch (Release) beginning. 1.0 to 1.1.0: An enhanced LTE Cell Scanner/tracker. Xianjun Jiao (putaoshu@msn.com)
  PPM: 0
  correction: 1
Use file begin with 100.0MHz actual 100.0MHz 1.92e+06MHz
    Search frequency: 100.0 to 100.0 MHz
with freq correction: 0 kHz
    Search PSS at fo: -140 to 135 kHz

Examining center frequency 100.0 MHz ... try 0
PSS XCORR  cost 4.1s
No peaks found.
"""


def test_parse_cellsearch_output_extracts_detected_cell():
    cells = parse_cellsearch_output(REAL_STDOUT)
    assert len(cells) == 1
    cell = cells[0]
    assert cell["duplex"] == "FDD"
    assert cell["freq_mhz"] == 1815.3
    assert cell["cell_id"] == 301
    assert cell["pss_id"] == 1
    assert cell["rx_power_db"] == -9.44976
    assert cell["freq_offset_hz"] == 14302.6


def test_parse_cellsearch_output_returns_empty_list_when_no_cell_found():
    assert parse_cellsearch_output(NO_CELL_STDOUT) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/capture/cellular/test_offline_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.cellular.offline_scanner'`

- [ ] **Step 3: Write `capture/cellular/offline_scanner.py`**

```python
# capture/cellular/offline_scanner.py
from __future__ import annotations

import os
import re
import subprocess

_CELL_BLOCK = re.compile(
    r"Detected a (?P<duplex>FDD|TDD) cell! At freqeuncy (?P<freq_mhz>[\d.]+)MHz.*?"
    r"cell ID:\s*(?P<cell_id>\d+).*?"
    r"PSS ID:\s*(?P<pss_id>\d+).*?"
    r"RX power level:\s*(?P<rx_power_db>-?[\d.]+)\s*dB.*?"
    r"residual frequency offset:\s*(?P<freq_offset_hz>-?[\d.]+)\s*Hz",
    re.DOTALL,
)


def parse_cellsearch_output(stdout: str) -> list[dict]:
    """Parses CellSearch's stdout for 'Detected a <FDD|TDD> cell!' blocks,
    extracting the per-cell fields it prints inline. CellSearch (PSS/SSS
    search only) reports no RSRP/RSRQ/SINR or MIB/SIB fields — those require
    the separate, deferred LTE-Tracker tool. Note "freqeuncy" reproduces a
    real typo in CellSearch's own output text, not a mistake here."""
    cells = []
    for match in _CELL_BLOCK.finditer(stdout):
        cells.append(
            {
                "duplex": match.group("duplex"),
                "freq_mhz": float(match.group("freq_mhz")),
                "cell_id": int(match.group("cell_id")),
                "pss_id": int(match.group("pss_id")),
                "rx_power_db": float(match.group("rx_power_db")),
                "freq_offset_hz": float(match.group("freq_offset_hz")),
            }
        )
    return cells


def run_cellsearch(
    binary_path: str,
    iq_file_path: str,
    freq_start_hz: int,
    timeout_seconds: float = 30.0,
    extra_env: dict[str, str] | None = None,
) -> list[dict]:
    """Runs the vendored CellSearch binary against a pre-recorded --loadbin
    IQ file and returns parsed cell-detection dicts. Never touches a live
    radio. extra_env is merged over the current environment — used to set
    LD_LIBRARY_PATH when IT++ isn't on the default loader path."""
    env = {**os.environ, **(extra_env or {})}
    result = subprocess.run(
        [binary_path, "-s", str(freq_start_hz), "--loadbin", iq_file_path],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    return parse_cellsearch_output(result.stdout)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/capture/cellular/test_offline_scanner.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add capture/cellular/offline_scanner.py tests/capture/cellular/test_offline_scanner.py
git commit -m "feat: add CellSearch subprocess wrapper and stdout parser"
```

---

### Task 4: Normalizer

**Files:**
- Create: `capture/cellular/normalizer.py`
- Test: `tests/capture/cellular/test_normalizer.py`

**Interfaces:**
- Consumes: `schema.records.{Identifier, Modality, Signal, UnifiedRecord}` (existing);
  one dict from `capture.cellular.offline_scanner.parse_cellsearch_output` (Task 3) — a
  dict with keys `duplex: str`, `freq_mhz: float`, `cell_id: int`, `pss_id: int`,
  `rx_power_db: float`, `freq_offset_hz: float`.
- Produces: `capture.cellular.normalizer.normalize_cellsearch_result(result: dict,
  survey_id: str, operator_id: str) -> UnifiedRecord`.

- [ ] **Step 1: Write the failing test**

```python
# tests/capture/cellular/test_normalizer.py
from capture.cellular.normalizer import normalize_cellsearch_result
from schema.records import Modality

SAMPLE_RESULT = {
    "duplex": "FDD",
    "freq_mhz": 1815.3,
    "cell_id": 301,
    "pss_id": 1,
    "rx_power_db": -9.44976,
    "freq_offset_hz": 14302.6,
}


def test_normalize_cellsearch_result_maps_fields():
    record = normalize_cellsearch_result(SAMPLE_RESULT, survey_id="s1", operator_id="op1")
    assert record.modality is Modality.CELLULAR
    assert record.identifier.cell_id == "301"
    assert record.identifier.center_freq == 1815300000.0
    assert record.signal.rssi == -9.44976
    assert record.signal.rsrp is None
    assert record.survey_id == "s1"
    assert record.operator_id == "op1"
    assert record.lat == 0.0
    assert record.gps_fix_quality is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/capture/cellular/test_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.cellular.normalizer'`

- [ ] **Step 3: Write `capture/cellular/normalizer.py`**

```python
# capture/cellular/normalizer.py
from __future__ import annotations

from datetime import datetime, timezone

from schema.records import Identifier, Modality, Signal, UnifiedRecord


def normalize_cellsearch_result(
    result: dict, survey_id: str, operator_id: str
) -> UnifiedRecord:
    """Converts one parsed CellSearch detection dict (see
    capture.cellular.offline_scanner.parse_cellsearch_output) into a
    UnifiedRecord. lat/lon are left at 0.0 with gps_fix_quality=None —
    ingest.service attaches the real fix, matching capture.wifi/bluetooth.
    No RSRP/RSRQ/SINR or MIB/SIB fields: CellSearch (PSS/SSS search only)
    doesn't report them; that requires the separate, deferred LTE-Tracker
    tool. No PLMN either — CellSearch reports Cell ID only."""
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=0.0,
        lon=0.0,
        gps_fix_quality=None,
        survey_id=survey_id,
        operator_id=operator_id,
        modality=Modality.CELLULAR,
        identifier=Identifier(
            cell_id=str(result["cell_id"]),
            center_freq=result["freq_mhz"] * 1_000_000.0,
        ),
        signal=Signal(rssi=result["rx_power_db"]),
        metadata={},
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/capture/cellular/test_normalizer.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add capture/cellular/normalizer.py tests/capture/cellular/test_normalizer.py
git commit -m "feat: add cellular normalizer (CellSearch result -> UnifiedRecord)"
```

---

### Task 5: Integration test — the DSP spike closure proof

**Files:**
- Test: `tests/capture/cellular/test_offline_decode.py`

**Interfaces:**
- Consumes: `capture.cellular.offline_scanner.run_cellsearch` (Task 3); the built
  `CellSearch` binary at `capture/cellular/vendor/lte-cell-scanner/build/src/CellSearch`
  (Task 1); the fixture at `capture/cellular/testdata/cell301_1815.3mhz.bin` and its
  ground truth in `cell301_1815.3mhz.expected.json` (Task 2).
- Produces: nothing further consumes this — it's the final closure proof for this plan.

This is the one test that actually exercises the real compiled binary end-to-end. It's
skipped (not failed) if the binary hasn't been built — consistent with how
`KismetClient`/bleak's `service.run` are excluded from unit-test scope as thin I/O
wrappers in the WiFi/BT plan; the equivalent boundary here is "requires a C++ toolchain
and vendored build," not "requires live hardware," but the skip-don't-fail principle is
the same.

- [ ] **Step 1: Write the test**

```python
# tests/capture/cellular/test_offline_decode.py
from __future__ import annotations

import json
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
        extra_env={"LD_LIBRARY_PATH": "/usr/local/lib"},
    )

    assert len(cells) == 1
    cell = cells[0]
    assert cell["duplex"] == expected["duplex"]
    assert cell["cell_id"] == expected["cell_id"]
    assert cell["pss_id"] == expected["pss_id"]
    assert cell["freq_mhz"] == pytest.approx(expected["freq_mhz"], abs=0.01)
    assert cell["rx_power_db"] == pytest.approx(expected["rx_power_db"], abs=0.01)
    assert cell["freq_offset_hz"] == pytest.approx(expected["freq_offset_hz"], abs=1.0)
```

- [ ] **Step 2: Run it**

Run: `pytest tests/capture/cellular/test_offline_decode.py -v`
Expected (binary built per Task 1): PASS (1 test), runtime ~5 seconds (verified: 4.64s
user / 0.21s sys, ~420MB peak RSS for this fixture size — well inside the 30s
`timeout_seconds` default).
Expected (binary not built): SKIPPED, with the reason message above.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `pytest -v`
Expected: all previously-passing tests still pass, plus the new cellular tests (5 total:
2 in `test_offline_scanner.py`, 1 in `test_normalizer.py`, 1 in `test_offline_decode.py`
— or SKIPPED for the last one if no C++ toolchain is available in this environment).

- [ ] **Step 4: Commit**

```bash
git add tests/capture/cellular/test_offline_decode.py
git commit -m "test: add end-to-end cellular offline-decode integration test"
```

---

## What this plan deliberately does not cover

- **MIB/SIB1-3 decode** — needs `LTE-Tracker`, whose offline single-shot batch-decode
  capability is unconfirmed (it's built for continuous live tracking). Separate plan,
  gated on that investigation.
- **RSRP/RSRQ/SINR** — not reported by `CellSearch`; would come from `LTE-Tracker` too.
- **libbladeRF2 native AD9361 gain-control layer** — no live radio touched in this plan.
- **`capture/cellular/service.py`** (live-radio capture service, analogous to
  `capture/wifi/service.py`) — depends on the gain-control layer existing first.
- **Real xA9 hardware validation** (Build Sequencing §10 step 3) — gated on physical
  hardware access, out of reach in this environment.
- **Wiring cellular records into `ingest.service`/`viz.app`** — natural once a live
  capture service exists, not before (no `RecordEmitter` usage in this plan; nothing yet
  produces a continuous stream of cellular records to emit).
- **Synthetic IQ generation** — no free/open-source tool for it was found; this plan
  uses the vendored repo's own recorded regression capture instead (see design doc
  Amendment).
