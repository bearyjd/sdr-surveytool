# LTE MIB Fields via CellSearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the LTE MIB fields that the vendored `CellSearch` already decodes
(antenna ports, CP type, n_RB and the derived channel bandwidth, PHICH duration and
PHICH resource) from its existing stdout. No vendored-source patch, no `LTE-Tracker`, no
hardware.

**Architecture:** `CellSearch` reports a cell only after a CRC-checked MIB decode, and
its final "Detected the following cells:" summary table prints the MIB fields.
`capture/cellular/offline_scanner.py` already parses each cell's "Detected a … cell!"
block. This plan adds a second regex for the summary-table rows. It merges each row's
MIB fields into the matching block's dict, matched by cell ID plus frequency. A dict
with no matching row gets `None` MIB fields. The existing committed fixture and
integration test are extended, not replaced.

**Tech Stack:** Python 3.10+ (regex), pytest. The vendored LTE-Cell-Scanner
`CellSearch` binary is unchanged and still built by `capture/cellular/build.sh`.

**Spec:** `docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md`. Task 3
appends an amendment recording why this plan replaces the spec's LTE-Tracker
architecture. An earlier plan for the LTE-Tracker patch route was built, verified and
rejected. It stays in git history at commit `fc13404`
(`docs/superpowers/plans/2026-10-03-lte-mib-decode-spike.md`).

## Verified facts this plan relies on

All of these come from building and running on Fedora 43 with Python 3.14.7, submodule
pinned at `e7f71cbd4fa9f5ee5b97a58b2fb236ac13e4b8b1`.

- **CellSearch decodes the MIB.**
  - `src/CellSearch.cpp:1332` calls `decode_mib()` (`src/searcher.cpp:3840`).
  - `decode_mib()` accepts a decode only when the CRC matches
    (`searcher.cpp:3940-3952`).
  - A cell whose MIB fails is dropped (`CellSearch.cpp:1334-1338`).
  - The summary table (`CellSearch.cpp:1454-1493`) prints `DPX CID A fc freq-offset
    RXPWR C nRB P PR CrystalCorrectionFactor` for every cell that survives
    `dedup()`.
  - `decode_mib()` also computes an SFN (`searcher.cpp:3998-3999`), but nothing in
    `CellSearch.cpp` prints it.
- **Row format.** It comes from the print code at `CellSearch.cpp:1460-1493`. Fields
  are whitespace-separated, and none contains internal whitespace. The values are:
  - DPX: `FDD`/`TDD`.
  - CID: `setw(3)`.
  - A: `setw(2)`, value 1, 2 or 4.
  - fc: `setprecision(5)` MHz with an `M` suffix.
  - freq-offset: e.g. `14.3k`.
  - RXPWR.
  - C: `N`/`E`, or `U` for unknown.
  - nRB: 6, 15, 25, 50, 75 or 100.
  - P: `N`/`E`, or `U`.
  - PR: `1/6`, `1/2`, `one`, `two`, or `UNK`.
  - The `U`/`UNK` values cannot reach the table, because only MIB-decoded cells do.
- **Frequency precision differs.** The table prints fc with 5 significant digits; the
  detection block uses the default 6. CellSearch's own `dedup()` (`CellSearch.cpp:476`)
  treats same-ID detections within 1 MHz as one cell. So frequencies are matched within
  1 MHz, not exactly.
- **Real stdout available for tests.** All four are verbatim, and the last two come from
  big-file captures that are not committed:
  - **Single FDD cell:** the existing committed fixture `cell301_1815.3mhz.bin`; the
    stdout is already `REAL_STDOUT` in `tests/capture/cellular/test_offline_scanner.py`.
  - **No cell:** the existing `NO_CELL_STDOUT`.
  - **Two FDD cells on one carrier** (IDs 142 and 86): `CellSearch -s 1860000000` on a
    1.92 Msps decimation of `JiaoXianjun/LTE-Cell-Scanner-big-file` @ `0791cb33`
    `f1860_s19.2_bw20_1s_hackrf_home.bin`.
  - **One TDD cell** (ID 216): `CellSearch -s 2585000000` on the same repo's
    `f2585_s19.2_bw20_1s_hackrf.bin`.
  - Every real cell decodes to the same MIB values, `2 N 100 N one`. CellSearch can't
    produce a malformed row, so none is tested.
- **Fixture ground truth** for `cell301_1815.3mhz.bin`: 2 antenna ports, normal CP,
  100 RB (20 MHz), PHICH duration normal, PHICH resource one. The full test suite on
  this branch's base is `58 passed` with CellSearch built.

## Global Constraints

- Python >= 3.10, matching `pyproject.toml`. No new dependencies.
- No change to the vendored submodule or its pin (`e7f71cbd…`). No source patch.
  `git -C capture/cellular/vendor/lte-cell-scanner status --short` stays at the
  pre-existing `?? build/`.
- No `UnifiedRecord`/`normalizer.py` changes. MIB fields stay in the parser's dicts
  only.
- No live radio. Only pre-recorded `.bin` files are read.
- Tests that are not real CellSearch stdout may only be derived from real stdout, e.g.
  a truncated prefix. Never invent CellSearch output.
- Never run `pip install -e` from a worktree. Run tests from the repo root with
  `python -m pytest`; `pyproject.toml` sets `--import-mode=importlib`, and the root
  `conftest.py` puts the repo root on `sys.path`.

## Review Focus

- **Several cells on one carrier.** Each cell must get its own row's fields, matched by
  cell ID, never the first row. Pinned by
  `test_parse_cellsearch_output_merges_mib_fields_for_each_of_several_cells` and by the
  truncation test below (Task 1).
- **Output cut off before or inside the table** (CellSearch killed or crashed while
  printing). Cells without a row should keep their detection fields and get `None` MIB
  fields, rather than raising or borrowing another cell's row. Pinned by
  `test_parse_cellsearch_output_leaves_mib_fields_none_without_summary_row` (Task 1).
- **A TDD cell.** The `TDD` row should parse exactly like `FDD`. Pinned by
  `test_parse_cellsearch_output_merges_mib_fields_for_tdd_cell` (Task 1).
- **Header lines that look like rows.** `DPX:TDD/FDD; …` and `DPX CID A …` must never
  match the row regex, which is anchored on a leading `FDD`/`TDD`. Covered by every
  real-stdout test above, since all of them contain both header lines and assert exact
  results.
- **A detection whose table row is at a slightly different printed frequency**
  (precision 5 vs 6, or a dedup'd neighbouring scan frequency). It should still merge.
  This is handled by the 1 MHz match radius. No real stdout exercises it, because
  `--loadbin` scans a single frequency, so it is documented rather than tested.

---

### Task 1: Parse MIB fields from CellSearch's summary table

**Files:**
- Modify: `capture/cellular/offline_scanner.py` (whole file shown below)
- Test: `tests/capture/cellular/test_offline_scanner.py`

**Interfaces:**
- Consumes: CellSearch's existing stdout format (see "Verified facts").
- Produces: `capture.cellular.offline_scanner.parse_cellsearch_output(stdout: str) ->
  list[dict]` and `run_cellsearch(...)` keep the same signatures. Each dict keeps the
  existing keys `duplex: str`, `freq_mhz: float`, `cell_id: int`, `pss_id: int`,
  `rx_power_db: float` and `freq_offset_hz: float`, and gains:
  - `n_ports: int | None`, one of 1, 2, 4;
  - `cp_type: str | None`, `"normal"` or `"extended"`;
  - `n_rb_dl: int | None`, one of 6, 15, 25, 50, 75, 100;
  - `bandwidth_mhz: float | None`, one of 1.4, 3.0, 5.0, 10.0, 15.0, 20.0;
  - `phich_duration: str | None`, `"normal"` or `"extended"`;
  - `phich_resource: str | None`, one of `"1/6"`, `"1/2"`, `"one"`, `"two"`.

  All six are `None` together when the cell has no summary-table row. Task 2 asserts
  these keys against the fixture's ground truth.
  `capture/cellular/normalizer.py` reads only the old keys and is untouched.

- [ ] **Step 1: Write the failing tests**

In `tests/capture/cellular/test_offline_scanner.py`, insert the following between the
end of `NO_CELL_STDOUT` (its closing `"""`) and
`def test_parse_cellsearch_output_extracts_detected_cell():`:

```python
# Verbatim stdout from an actual `CellSearch -s 1860000000 --loadbin ...` run
# against a 1.92 Msps decimation of JiaoXianjun/LTE-Cell-Scanner-big-file
# @ 0791cb33 regression_test_signal_file/f1860_s19.2_bw20_1s_hackrf_home.bin
# (not committed): two FDD cells on one carrier.
MULTI_CELL_STDOUT = """\
LTE CellSearch (Release) beginning. 1.0 to 1.1.0: An enhanced LTE Cell Scanner/tracker. Xianjun Jiao (putaoshu@msn.com)
  PPM: 0
  correction: 1
Use file begin with 1860MHz actual 1860MHz 1.92e+06MHz
    Search frequency: 1860 to 1860 MHz
with freq correction: 0 kHz
    Search PSS at fo: -140 to 135 kHz

Examining center frequency 1860 MHz ... try 0
PSS XCORR  cost 3.09389s
Hit  num peaks 2
try peak 0 tdd_flag 0
  Detected a FDD cell! At freqeuncy 1860MHz, try 0
    cell ID: 142
     PSS ID: 1
    RX power level: -27.3579 dB
    residual frequency offset: 22901 Hz
                     k_factor: 0.999988
try peak 0 tdd_flag 1
try peak 1 tdd_flag 0
  Detected a FDD cell! At freqeuncy 1860MHz, try 0
    cell ID: 86
     PSS ID: 2
    RX power level: -28.5606 dB
    residual frequency offset: 22925 Hz
                     k_factor: 0.999988
try peak 1 tdd_flag 1
Detected the following cells:
DPX:TDD/FDD; A: #antenna ports C: CP type ; P: PHICH duration ; PR: PHICH resource type
DPX CID A      fc   freq-offset RXPWR C nRB P  PR CrystalCorrectionFactor
FDD 142 2   1860M         22.9k -27.4 N 100 N one 1.0000123125041611161
FDD  86 2   1860M         22.9k -28.6 N 100 N one 1.0000123254264181583
"""

# Verbatim stdout from an actual `CellSearch -s 2585000000 --loadbin ...` run
# against a 1.92 Msps decimation of the same repo's
# f2585_s19.2_bw20_1s_hackrf.bin (not committed): one TDD cell.
TDD_STDOUT = """\
LTE CellSearch (Release) beginning. 1.0 to 1.1.0: An enhanced LTE Cell Scanner/tracker. Xianjun Jiao (putaoshu@msn.com)
  PPM: 0
  correction: 1
Use file begin with 2585MHz actual 2585MHz 1.92e+06MHz
    Search frequency: 2585 to 2585 MHz
with freq correction: 0 kHz
    Search PSS at fo: -140 to 135 kHz

Examining center frequency 2585 MHz ... try 0
PSS XCORR  cost 3.11921s
Hit  num peaks 8
try peak 0 tdd_flag 0
try peak 0 tdd_flag 1
  Detected a TDD cell! At freqeuncy 2585MHz, try 0
    cell ID: 216
     PSS ID: 0
    RX power level: -22.9132 dB
    residual frequency offset: 34902.8 Hz
                     k_factor: 0.999986
try peak 1 tdd_flag 0
try peak 1 tdd_flag 1
try peak 2 tdd_flag 0
try peak 2 tdd_flag 1
try peak 3 tdd_flag 0
try peak 3 tdd_flag 1
try peak 4 tdd_flag 0
try peak 4 tdd_flag 1
try peak 5 tdd_flag 0
try peak 5 tdd_flag 1
try peak 6 tdd_flag 0
try peak 6 tdd_flag 1
try peak 7 tdd_flag 0
try peak 7 tdd_flag 1
Detected the following cells:
DPX:TDD/FDD; A: #antenna ports C: CP type ; P: PHICH duration ; PR: PHICH resource type
DPX CID A      fc   freq-offset RXPWR C nRB P  PR CrystalCorrectionFactor
TDD 216 2   2585M         34.9k -22.9 N 100 N one 1.0000135022399760931
"""

# MIB fields every cell above decodes to: CellSearch prints them as
# "A C nRB P PR" = "2 N 100 N one" in its summary table.
EXPECTED_MIB_FIELDS = {
    "n_ports": 2,
    "cp_type": "normal",
    "n_rb_dl": 100,
    "bandwidth_mhz": 20.0,
    "phich_duration": "normal",
    "phich_resource": "one",
}
```

Then insert these tests directly after
`test_parse_cellsearch_output_returns_empty_list_when_no_cell_found`:

```python
def test_parse_cellsearch_output_merges_mib_fields_from_summary_table():
    cells = parse_cellsearch_output(REAL_STDOUT)
    assert len(cells) == 1
    assert {key: cells[0][key] for key in EXPECTED_MIB_FIELDS} == EXPECTED_MIB_FIELDS


def test_parse_cellsearch_output_merges_mib_fields_for_tdd_cell():
    cells = parse_cellsearch_output(TDD_STDOUT)
    assert len(cells) == 1
    cell = cells[0]
    assert cell["duplex"] == "TDD"
    assert cell["cell_id"] == 216
    assert {key: cell[key] for key in EXPECTED_MIB_FIELDS} == EXPECTED_MIB_FIELDS


def test_parse_cellsearch_output_merges_mib_fields_for_each_of_several_cells():
    cells = parse_cellsearch_output(MULTI_CELL_STDOUT)
    assert [cell["cell_id"] for cell in cells] == [142, 86]
    for cell in cells:
        assert {key: cell[key] for key in EXPECTED_MIB_FIELDS} == EXPECTED_MIB_FIELDS


def test_parse_cellsearch_output_leaves_mib_fields_none_without_summary_row():
    """Output cut off after the first table row (e.g. CellSearch killed
    mid-print): the cell whose row is missing keeps its detection fields but
    gets None MIB fields, rather than borrowing another cell's row."""
    truncated = MULTI_CELL_STDOUT[: MULTI_CELL_STDOUT.index("FDD  86 2")]
    cells = parse_cellsearch_output(truncated)
    assert [cell["cell_id"] for cell in cells] == [142, 86]
    assert cells[0]["n_rb_dl"] == 100
    assert cells[1]["rx_power_db"] == -28.5606
    assert {key: cells[1][key] for key in EXPECTED_MIB_FIELDS} == dict.fromkeys(
        EXPECTED_MIB_FIELDS
    )
```

The two existing parse tests and `test_run_cellsearch_raises_when_process_fails_with_no_cells`
stay as they are.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/capture/cellular/test_offline_scanner.py -v`
Expected: 4 FAILED and 3 passed. The new tests fail with `KeyError: 'n_ports'`, except
the truncation test, which fails with `KeyError: 'n_rb_dl'`.

- [ ] **Step 3: Rewrite `capture/cellular/offline_scanner.py`**

The full new content is below. It adds `_SUMMARY_ROW`, the bandwidth/CP lookups,
`_parse_summary_rows` and `_mib_fields_for`; changes `parse_cellsearch_output` to merge;
and corrects its docstring, which wrongly said CellSearch reports no MIB fields.
`run_cellsearch` is unchanged.

```python
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

# One row of the "Detected the following cells:" summary table CellSearch
# prints last (columns: DPX CID A fc freq-offset RXPWR C nRB P PR
# CrystalCorrectionFactor). Only cells whose MIB passed CRC reach it, so A
# (antenna ports), C (CP type), nRB, P (PHICH duration) and PR (PHICH
# resource) are decoded MIB fields. The U/UNK placeholders for unknown
# values deliberately don't match.
_SUMMARY_ROW = re.compile(
    r"^(?:FDD|TDD)\s+(?P<cell_id>\d+)\s+(?P<n_ports>[124])\s+"
    r"(?P<freq_mhz>[\d.]+)M\s+\S+\s+\S+\s+"
    r"(?P<cp_type>[NE])\s+(?P<n_rb_dl>6|15|25|50|75|100)\s+"
    r"(?P<phich_duration>[NE])\s+(?P<phich_resource>1/6|1/2|one|two)\s+\S+$",
    re.MULTILINE,
)

# Channel bandwidth for each MIB downlink bandwidth (3GPP TS 36.101
# Table 5.6-1).
_BANDWIDTH_MHZ_BY_N_RB = {6: 1.4, 15: 3.0, 25: 5.0, 50: 10.0, 75: 15.0, 100: 20.0}
_NORMAL_OR_EXTENDED = {"N": "normal", "E": "extended"}

# CellSearch's own dedup() treats same-ID detections within 1 MHz as one
# cell. The table prints fc to 5 significant digits and the detection
# block to 6, so frequencies are matched within this radius, not exactly.
_SAME_CELL_MHZ = 1.0

_MIB_KEYS = (
    "n_ports",
    "cp_type",
    "n_rb_dl",
    "bandwidth_mhz",
    "phich_duration",
    "phich_resource",
)


def _parse_summary_rows(stdout: str) -> list[dict]:
    rows = []
    for match in _SUMMARY_ROW.finditer(stdout):
        n_rb_dl = int(match.group("n_rb_dl"))
        rows.append(
            {
                "cell_id": int(match.group("cell_id")),
                "freq_mhz": float(match.group("freq_mhz")),
                "n_ports": int(match.group("n_ports")),
                "cp_type": _NORMAL_OR_EXTENDED[match.group("cp_type")],
                "n_rb_dl": n_rb_dl,
                "bandwidth_mhz": _BANDWIDTH_MHZ_BY_N_RB[n_rb_dl],
                "phich_duration": _NORMAL_OR_EXTENDED[match.group("phich_duration")],
                "phich_resource": match.group("phich_resource"),
            }
        )
    return rows


def _mib_fields_for(cell: dict, rows: list[dict]) -> dict:
    """MIB fields from the summary row for the same cell (same cell ID,
    frequency within _SAME_CELL_MHZ). All None if there is no such row,
    e.g. when the output was cut off before the table."""
    for row in rows:
        if (
            row["cell_id"] == cell["cell_id"]
            and abs(row["freq_mhz"] - cell["freq_mhz"]) < _SAME_CELL_MHZ
        ):
            return {key: row[key] for key in _MIB_KEYS}
    return dict.fromkeys(_MIB_KEYS)


def parse_cellsearch_output(stdout: str) -> list[dict]:
    """Parses CellSearch's stdout for 'Detected a <FDD|TDD> cell!' blocks,
    extracting the per-cell fields it prints inline, then merges in the
    MIB fields (antenna ports, CP type, n_RB and derived bandwidth, PHICH
    duration and resource) from its final summary table. CellSearch only
    reports a cell after a CRC-checked MIB decode, but it never prints the
    SFN, and it reports no RSRP/RSRQ/SINR or SIB fields (so no PLMN). Note
    "freqeuncy" reproduces a real typo in CellSearch's own output text, not
    a mistake here."""
    rows = _parse_summary_rows(stdout)
    cells = []
    for match in _CELL_BLOCK.finditer(stdout):
        cell = {
            "duplex": match.group("duplex"),
            "freq_mhz": float(match.group("freq_mhz")),
            "cell_id": int(match.group("cell_id")),
            "pss_id": int(match.group("pss_id")),
            "rx_power_db": float(match.group("rx_power_db")),
            "freq_offset_hz": float(match.group("freq_offset_hz")),
        }
        cells.append({**cell, **_mib_fields_for(cell, rows)})
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
    cells = parse_cellsearch_output(result.stdout)
    if not cells and result.returncode != 0:
        raise RuntimeError(
            f"CellSearch exited with code {result.returncode} and no cells were "
            f"parsed from its output; stderr:\n{result.stderr}"
        )
    return cells
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/capture/cellular/test_offline_scanner.py -v`
Expected: PASS (7 tests). The existing `test_normalizer.py` is unaffected; run
`python -m pytest tests/capture/cellular -v` to confirm.

Mutation check, verified: three changes each make at least one new test fail.
- Replacing the cell-ID comparison in `_mib_fields_for` with `True` fails the
  truncation test.
- Dropping `TDD` from `_SUMMARY_ROW` fails the TDD test.
- Bypassing the bandwidth table fails 3 tests.

- [ ] **Step 5: Commit**

```bash
git add capture/cellular/offline_scanner.py tests/capture/cellular/test_offline_scanner.py
git commit -m "feat: parse MIB fields from CellSearch summary table"
```

---

### Task 2: Ground truth and integration test for the MIB fields

**Files:**
- Modify: `capture/cellular/testdata/cell301_1815.3mhz.expected.json`
- Test: `tests/capture/cellular/test_offline_decode.py`

**Interfaces:**
- Consumes:
  - Task 1's six new dict keys from `run_cellsearch`.
  - The existing built `CellSearch` binary at
    `capture/cellular/vendor/lte-cell-scanner/build/src/CellSearch`, from `build.sh`.
  - The existing fixture `cell301_1815.3mhz.bin`.
- Produces: the closure proof for this plan. Nothing further consumes it.

Task 1 already makes the parser emit these fields, so this test passes as soon as it
is written. Like the predecessor plan's integration task, it has no red step. If
`CellSearch` isn't built it is SKIPPED, as before.

- [ ] **Step 1: Replace `capture/cellular/testdata/cell301_1815.3mhz.expected.json`**

```json
{
  "duplex": "FDD",
  "freq_mhz": 1815.3,
  "cell_id": 301,
  "pss_id": 1,
  "rx_power_db": -9.44976,
  "freq_offset_hz": 14302.6,
  "n_ports": 2,
  "cp_type": "normal",
  "n_rb_dl": 100,
  "bandwidth_mhz": 20.0,
  "phich_duration": "normal",
  "phich_resource": "one"
}
```

The new values are CellSearch's own summary row for this fixture,
`FDD 301 2 1815.3M 14.3k -9.45 N 100 N one …`, which is already quoted in
`REAL_STDOUT`. Bandwidth is derived from n_RB 100.

- [ ] **Step 2: Extend `tests/capture/cellular/test_offline_decode.py`**

After the last assertion in `test_offline_decode_detects_known_cell`:

```python
    assert cell["freq_offset_hz"] == pytest.approx(expected["freq_offset_hz"], abs=1.0)
```

append:

```python
    # MIB fields, from CellSearch's CRC-checked decode via its summary table.
    assert cell["n_ports"] == expected["n_ports"]
    assert cell["cp_type"] == expected["cp_type"]
    assert cell["n_rb_dl"] == expected["n_rb_dl"]
    assert cell["bandwidth_mhz"] == expected["bandwidth_mhz"]
    assert cell["phich_duration"] == expected["phich_duration"]
    assert cell["phich_resource"] == expected["phich_resource"]
```

- [ ] **Step 3: Run it**

Run: `python -m pytest tests/capture/cellular/test_offline_decode.py -v -rs`

Expected:
- **CellSearch built:** PASS (1 test) in about 4 s.
- **Not built:** SKIPPED with
  `CellSearch binary not built; run capture/cellular/build.sh (see capture/cellular/README.md)`.
- If it fails with `KeyError: 'n_ports'`, Task 1 is missing.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest`
Expected with CellSearch built: `62 passed`, which is the 58 baseline plus the 4 new
parser tests. The 33 fastapi DeprecationWarnings are pre-existing.

- [ ] **Step 5: Commit**

```bash
git add capture/cellular/testdata/cell301_1815.3mhz.expected.json tests/capture/cellular/test_offline_decode.py
git commit -m "test: assert MIB fields in cellular offline-decode integration test"
```

---

### Task 3: Spec amendment and README updates

**Files:**
- Modify: `docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md` (append)
- Modify: `capture/cellular/README.md` (intro paragraph, Status paragraph)
- Modify: `README.md:11` (Layout line for `capture/cellular/`)

**Interfaces:**
- Consumes: Tasks 1–2 (the READMEs describe what they deliver).
- Produces: documentation only.

This task is docs only, so it has no test cycle.

- [ ] **Step 1: Append the amendment to the spec**

Append to `docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md`:

```markdown

## Amendment (2026-10-03): MIB fields come from CellSearch; LTE-Tracker route rejected

A build-and-run spike (Fedora 43, LTE-Cell-Scanner submodule @ e7f71cbd) changed this
design's conclusions. The implementation follows
`docs/superpowers/plans/2026-10-03-lte-mib-fields-cellsearch.md`.

1. **CellSearch already decodes the MIB.**
   - `src/CellSearch.cpp:1332` calls `decode_mib()` (`src/searcher.cpp:3840`), which
     accepts a decode only when the CRC matches (`searcher.cpp:3940-3952`).
   - Cells whose MIB fails are dropped (`CellSearch.cpp:1334-1338`).
   - Its final summary table (`CellSearch.cpp:1454-1493`) prints antenna ports, CP
     type, n_RB, PHICH duration and PHICH resource for every reported cell. The
     committed 80 ms fixture already shows `FDD 301 2 1815.3M ... N 100 N one`.
   - `decode_mib()` also computes the SFN (`searcher.cpp:3998-3999`), but CellSearch
     never prints it.
2. **MIB decode needs 40 ms of IQ, not 160 ms.** LTE-Tracker's `do_mib_decode()` buffers
   only slot 1, symbols 0–3 of each frame (`tracker_thread.cpp:570`). So
   `mib_fifo.size()==16` (`tracker_thread.cpp:581`) is 4 frames, one 40 ms PBCH TTI.
   CellSearch decodes the MIB from the 80 ms fixture.
3. **`f2585_s19.2_bw20_1s_hackrf.bin` is a TDD cell, and LTE-Tracker is FDD-only.**
   - The file is from the big-file repo @ 0791cb339a8e88fc531494f2e447ff03bb48ff04,
     SHA-256 ea453bba4fe4edb6c5fa458b3067e4eeb4ef77defbf6acfa3a426371b0850fba. It is a
     plain git blob, not LFS.
   - CellSearch decodes it as TDD cell 216 (2 ports, 100 RB, PHICH normal/one).
   - `tracker_thread.cpp` has no duplex or TDD handling at all. In a 60 s run, a
     patched tracker made about 25,000 MIB attempts on that cell and every one failed
     CRC.
   - Weak FDD captures also never decoded in the tracker: the two −27 dB cells in
     `f1860_s19.2_bw20_1s_hackrf_home.bin`. CellSearch decodes both.
4. **The LTE-Tracker patch route was built, verified, and rejected.**
   - A 25-line additive patch did three things: print a plain-text MIB line, skip
     curses when stdout is not a TTY, and call `_exit(0)` at the end of a `--loadbin`
     pass. `--loadbin` always loops, because `repeat=true` at `LTE-Tracker.cpp:220`.
   - It decoded cell 301's MIB in about 6 s and 650 MB per run.
   - It was rejected for a zero-patch route. CellSearch's table already carries every
     MIB field except SFN, and SFN has no coverage-mapping value. That is not worth
     carrying an AGPL source patch, a slow and heavy binary, and a fixture that only
     decodes because the tracker loops it.
   - That plan is preserved in git history at commit fc13404
     (`docs/superpowers/plans/2026-10-03-lte-mib-decode-spike.md`).
5. **SFN and PLMN remain unavailable.** CellSearch computes the SFN but doesn't print
   it. PLMN needs SIB1, which nothing in the vendored codebase decodes.

This supersedes the "Fixture source", "Vendored source patch" and "Architecture"
sections above. There is no source patch, no `generate_mib_fixture.py`, no
`offline_mib_scanner.py` and no new fixture. `capture/cellular/offline_scanner.py`
parses the MIB fields from CellSearch's summary table, verified against the existing
`cell301_1815.3mhz.bin`.
```

- [ ] **Step 2: Update `capture/cellular/README.md`**

Replace the intro paragraph:

```markdown
Passive LTE cell broadcast decode: currently PSS/SSS cell search only (Cell ID, PSS ID,
RX power, residual frequency offset). Vendors LTE-Cell-Scanner
(https://github.com/JiaoXianjun/LTE-Cell-Scanner), built unmodified via CMake.
```

with:

```markdown
Passive LTE cell broadcast decode via `CellSearch`: PSS/SSS cell search (Cell ID, PSS ID,
RX power, residual frequency offset) plus the fields of its CRC-checked MIB decode
(antenna ports, CP type, n_RB / bandwidth, PHICH duration and resource; no SFN, no
PLMN). Vendors LTE-Cell-Scanner (https://github.com/JiaoXianjun/LTE-Cell-Scanner), built
unmodified via CMake.
```

Replace the Status paragraph:

```markdown
**Status**: hardware-free DSP spike complete — `CellSearch` correctly detects a known
cell (Cell ID 301) from a recorded IQ file, see `testdata/`. MIB/SIB1-3 decode
(`LTE-Tracker`, a separate tool) and real xA9 hardware validation are deferred to later
plans; see `docs/superpowers/specs/2026-08-11-cellular-dsp-spike-design.md`.
```

with:

```markdown
**Status**: hardware-free DSP spike complete — `CellSearch` correctly detects a known
cell (Cell ID 301) and its MIB fields (2 antenna ports, normal CP, 100 RB / 20 MHz,
PHICH normal/one) from a recorded IQ file, see `testdata/`. SIB decode (and so PLMN)
does not exist in the vendored code, and real xA9 hardware validation is deferred; see
`docs/superpowers/specs/2026-08-11-cellular-dsp-spike-design.md` and the amendment in
`docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md`.
```

Leave the Building and License sections unchanged. They remain accurate, because the
submodule is still built unmodified.

- [ ] **Step 3: Update the root `README.md` Layout line**

Replace:

```markdown
- `capture/cellular/` — vendored LTE-Cell-Scanner submodule (PSS/SSS cell search only, unmodified) + normalizer; gain-control/live-radio work deferred
```

with:

```markdown
- `capture/cellular/` — vendored LTE-Cell-Scanner submodule (PSS/SSS cell search + MIB fields, unmodified) + normalizer; gain-control/live-radio work deferred
```

- [ ] **Step 4: Verify the submodule is untouched**

```bash
git -C capture/cellular/vendor/lte-cell-scanner status --short
git -C capture/cellular/vendor/lte-cell-scanner rev-parse HEAD
```

Expected: only `?? build/`, and `e7f71cbd4fa9f5ee5b97a58b2fb236ac13e4b8b1`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md capture/cellular/README.md README.md
git commit -m "docs: amend MIB spike spec for the CellSearch summary-table route"
```

---

## What this plan deliberately does not cover

- **`UnifiedRecord`/`normalizer.py` wiring.** This is out of scope per the spec. When it
  happens, `n_rb_dl` and `bandwidth_mhz` are the natural first fields to wire, since
  channel bandwidth is the MIB field with direct coverage-mapping value. The others,
  ports, CP and PHICH config, are mostly diagnostic.
- **SFN.** CellSearch computes it but never prints it. Surfacing it would take a
  vendored-source patch, which this plan exists to avoid, and SFN has no
  coverage-mapping value. The LTE-Tracker route that does print SFN is preserved at
  `fc13404`.
- **SIB1-3 and PLMN.** There is no SIB decode anywhere in the vendored codebase.
- **Real-stdout coverage of the 1 MHz frequency-match radius.** `--loadbin` scans a
  single frequency, so no real stdout has a block and row at different printed
  frequencies. Review follow-ups after this plan pinned the radius, the row
  assignment and the value mappings with synthetic rows built from the real column
  layout instead.
- **Canonical records from the deduplicated summary table.** The parser returns one
  dict per raw "Detected a … cell!" block, and CellSearch prints a block for every
  detection before `dedup()`, so one cell can yield several dicts (a pre-existing
  behavior). Since review, only the block `dedup()` kept gets the row's MIB fields;
  the others get `None`. Emitting one record per summary-table row instead would
  remove the duplicate per-cell records. Decide it when the normalizer is wired.
- **LTE-Tracker and TDD tracking.** They are no longer used.
- **Follow-ups (pre-existing issues, flagged here, not fixed):**
  - **`generate_fixture.py` sample format.**
    `capture/cellular/testdata/generate_fixture.py` reads HackRF's signed int8 samples
    as offset-binary uint8, applying `- 128`. That flips each sample's sign bit, which
    effectively hard-limits the signal. The raw capture's histogram clusters at bytes
    0–2 and 253–255, the signature of signed int8. The committed
    `cell301_1815.3mhz.bin` and its −9.45 dB RX-power ground truth inherit this.
    Fixing it means regenerating the fixture and re-deriving the ground truth.
  - **IT++ rebuilt on every run.** `capture/cellular/build.sh` tests for IT++ with
    `ldconfig -p | grep -q libitpp`. On Fedora `/usr/local/lib` is not in the ldconfig
    cache, so that check never matches and every run re-clones and rebuilds IT++
    (about 40 s plus network). Detecting `/usr/local/lib/libitpp.so` directly, or
    adding an `ld.so.conf.d` entry, would fix it.
- **Real xA9 hardware validation.** There is no physical SDR in this environment.
