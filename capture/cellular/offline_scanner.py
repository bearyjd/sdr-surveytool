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
    cells = parse_cellsearch_output(result.stdout)
    if not cells and result.returncode != 0:
        raise RuntimeError(
            f"CellSearch exited with code {result.returncode} and no cells were "
            f"parsed from its output; stderr:\n{result.stderr}"
        )
    return cells
