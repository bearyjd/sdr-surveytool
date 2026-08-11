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
