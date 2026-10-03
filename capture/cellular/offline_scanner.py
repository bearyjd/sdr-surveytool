from __future__ import annotations

import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

_CELL_BLOCK = re.compile(
    r"Detected a (?P<duplex>FDD|TDD) cell! At freqeuncy (?P<freq_mhz>[\d.]+)MHz.*?"
    r"cell ID:\s*(?P<cell_id>\d+).*?"
    r"PSS ID:\s*(?P<pss_id>\d+).*?"
    r"RX power level:\s*(?P<rx_power_db>-?[\d.]+)\s*dB.*?"
    r"residual frequency offset:\s*(?P<freq_offset_hz>-?[\d.]+)\s*Hz",
    re.DOTALL,
)

_SUMMARY_HEADER = "Detected the following cells:"

# One row of the "Detected the following cells:" summary table CellSearch
# prints last (columns: DPX CID A fc freq-offset RXPWR C nRB P PR
# CrystalCorrectionFactor). Only cells whose MIB passed CRC reach it, so A
# (antenna ports), C (CP type), nRB, P (PHICH duration) and PR (PHICH
# resource) are decoded MIB fields. The U/UNK placeholders for unknown
# values deliberately don't match.
_SUMMARY_ROW = re.compile(
    r"^(?P<duplex>FDD|TDD)\s+(?P<cell_id>\d+)\s+(?P<n_ports>[124])\s+"
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
                "duplex": match.group("duplex"),
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


def _matching_row(cell: dict, rows: list[dict]) -> dict | None:
    """The summary row for the same cell: same duplex mode and cell ID, and
    the nearest frequency within _SAME_CELL_MHZ. None if there is no such
    row, e.g. when the output was cut off before or inside the table."""
    candidates = [
        row
        for row in rows
        if row["duplex"] == cell["duplex"]
        and row["cell_id"] == cell["cell_id"]
        and abs(row["freq_mhz"] - cell["freq_mhz"]) < _SAME_CELL_MHZ
    ]
    return min(
        candidates,
        key=lambda row: abs(row["freq_mhz"] - cell["freq_mhz"]),
        default=None,
    )


def parse_cellsearch_output(stdout: str) -> list[dict]:
    """Parses CellSearch's stdout for 'Detected a <FDD|TDD> cell!' blocks,
    extracting the per-cell fields it prints inline, then merges in the
    MIB fields (antenna ports, CP type, n_RB and derived bandwidth, PHICH
    duration and resource) from its final summary table. CellSearch only
    reports a cell after a CRC-checked MIB decode, but it never prints the
    SFN, and it reports no RSRP/RSRQ/SINR or SIB fields (so no PLMN). A
    cell with no matching table row gets None MIB fields; if the table was
    printed at all, that is also logged as a warning. Note "freqeuncy"
    reproduces a real typo in CellSearch's own output text, not a mistake
    here."""
    rows = _parse_summary_rows(stdout)
    table_printed = _SUMMARY_HEADER in stdout
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
        row = _matching_row(cell, rows)
        if row is None:
            if table_printed:
                logger.warning(
                    "CellSearch summary table has no row for detected cell %r "
                    "(%r, %r MHz); its MIB fields are left as None",
                    cell["cell_id"],
                    cell["duplex"],
                    cell["freq_mhz"],
                )
            mib_fields = dict.fromkeys(_MIB_KEYS)
        else:
            mib_fields = {key: row[key] for key in _MIB_KEYS}
        cells.append({**cell, **mib_fields})
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
