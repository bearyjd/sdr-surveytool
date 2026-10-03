import subprocess

import pytest

from capture.cellular.offline_scanner import parse_cellsearch_output, run_cellsearch

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


def test_run_cellsearch_raises_when_process_fails_with_no_cells(monkeypatch):
    """A non-zero exit combined with zero parsed cells indicates a real
    failure (missing shared library, malformed IQ file, crash) rather than a
    legitimate 'no peaks found' run, and should not be silently swallowed."""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else kwargs.get("args"),
            returncode=1,
            stdout="",
            stderr="error while loading shared libraries: libitpp.so.9",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="libitpp.so.9"):
        run_cellsearch(
            "/fake/CellSearch",
            "/fake/iq.bin",
            freq_start_hz=1815300000,
        )
