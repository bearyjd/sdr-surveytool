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
