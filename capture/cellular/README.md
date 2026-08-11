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

`testdata/cell301_1815.3mhz.bin` is a decimated derivative of
`vendor/lte-cell-scanner/regression_test_signal_file/f1815.3_s19.2_bw20_0.08s_hackrf-1.bin`,
LTE-Cell-Scanner's own bundled regression capture, produced by
`testdata/generate_fixture.py`. As a derivative of AGPL-3.0-licensed data, this
fixture file inherits AGPL-3.0 licensing — it is not covered by the MIT license
at the repository root (`LICENSE`).
