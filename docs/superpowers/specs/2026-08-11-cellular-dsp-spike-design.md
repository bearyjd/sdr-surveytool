# Cellular DSP Spike Design

Scope: Phase 1 of the multi-modality survey tool's build sequencing (see
`docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md` §10, step 1) —
the hardware-free half of the cellular DSP spike. Validates that LTE-Cell-Scanner's
PSS/SSS cell search logic works, entirely offline, against a recorded LTE IQ file.

**Amendment (post-research):** this design was revised after actually cloning, building,
and running LTE-Cell-Scanner in a sandbox. Two scope narrowings resulted, both
deliberate, documented here rather than silently applied:

1. **MIB/SIB1-3 decode is deferred**, not part of this plan. LTE-Cell-Scanner splits
   cell search (`CellSearch`, a clean single-shot batch tool) from SIB decode
   (`LTE-Tracker`, architected as a continuous producer/tracker/searcher/display-thread
   live-tracking tool). Whether `LTE-Tracker` can cleanly do offline single-file batch
   SIB decode is unconfirmed and would be a separate research spike. This plan closes
   the PSS/SSS cell-search milestone only (Cell ID, PSS ID, RX power, residual frequency
   offset — no RSRP/RSRQ/SINR, no MIB/SIB fields).
2. **No synthetic IQ generation.** No free/open-source tool to synthesize a known-Cell-ID
   LTE downlink waveform was found (srsRAN ships none; MATLAB LTE Toolbox is the only
   confirmed option and is paid). Ground truth instead comes from LTE-Cell-Scanner's own
   bundled recorded capture (`regression_test_signal_file/f1815.3_s19.2_bw20_0.08s_hackrf-1.bin`),
   after required preprocessing — see "Test data" below. This still satisfies the
   original architecture doc's "recorded/synthetic LTE IQ files" language (§3: "can be
   validated in advance against recorded/synthetic LTE IQ files").

**Out of scope for this spec** (see "Deferred" below): anything touching real bladeRF
xA9 hardware. This environment has no physical SDR, so the hardware-integration half of
the Build Sequencing step-1 spike (§10 step 3: "validate ... locks onto a real signal on
real xA9 hardware") is a separate, later plan.

## Legal scope boundary

Unchanged from the design doc and `capture/cellular/README.md`: broadcast-channel decode
only (PSS/SSS/PBCH/PDSCH SIB). No paging-channel decoding, no RRC connection setup,
nothing that identifies or tracks individual subscribers. LTE-Cell-Scanner has no
RRC/paging code at all (verified: `CellSearch` only does PSS/SSS cell search; SIB decode,
also within the legal boundary, lives in the separate `LTE-Tracker` binary, deferred).

## Architecture

```
capture/cellular/
  vendor/lte-cell-scanner/   # git submodule, upstream JiaoXianjun/LTE-Cell-Scanner
  offline_scanner.py         # subprocess wrapper: runs vendored CellSearch, parses stdout
  normalizer.py               # parsed dict -> UnifiedRecord (mirrors capture/wifi)
  testdata/
    cell301_1815.3mhz.bin     # preprocessed fixture (decimated + headered), real capture
    cell301_1815.3mhz.expected.json  # verified ground truth for the above
tests/capture/cellular/
  test_normalizer.py          # unit tests, fixture dict -> UnifiedRecord
  test_offline_scanner.py     # unit tests, stdout-parsing regex against captured text
  test_offline_decode.py      # integration: run CellSearch against testdata, assert decode
```

Process boundary: the vendored `CellSearch` binary is built unmodified (CMake config
changes only, no source patches) and invoked as a subprocess against a pre-recorded IQ
file — it never touches a live radio. `capture/cellular/offline_scanner.py` runs it,
captures stdout, and regex-parses the known text format (verified real output, see
"Build & test strategy") into a plain dict. `capture/cellular/normalizer.py` then turns
that dict into `schema.records.UnifiedRecord`, using the same `Identifier`/`Modality`/
`Signal` shape as WiFi/Bluetooth. This mirrors the existing `capture/wifi` pattern
(`kismet_client.py` talks to Kismet's REST API and returns dicts; `normalizer.py` is pure
and unit-tested) — `offline_scanner.py` plays the `kismet_client.py` role, subprocess
instead of HTTP.

No `capture/cellular/service.py` (live-capture wrapper) or gain-control layer is written
in this phase — there is no live radio to wrap yet. No MIB/SIB1-3 fields are populated
(`Signal.rsrp`/`rsrq`/`snr` stay `None`) — `CellSearch` doesn't report them; only
`Signal.rssi` (from `RX power level`) and `Identifier.cell_id`/`plmn` (PLMN not reported
by `CellSearch` either — cell_id only) are populated this phase.

## Vendoring

LTE-Cell-Scanner is added as a git submodule under `capture/cellular/vendor/`. Built via
its own CMake (`cmake -DUSE_OPENCL=0 -DUSE_BLADERF=0 -DUSE_HACKRF=0`), producing
`CellSearch` unmodified — no source patches, no JSON-output flag added. **Note:**
`rtl-sdr-devel` is a required build-time dependency even for this hardware-free build —
verified the CMakeLists.txt's `USE_RTLSDR` flag is dead code; the RTLSDR `FIND_PACKAGE`
branch runs unconditionally whenever BladeRF and HackRF are both off. This is a build-time
library only; no RTL-SDR hardware is touched at runtime. `libitpp` is not packaged on
Fedora and must be built from source (SourceForge git mirror) and installed to
`/usr/local` with `LD_LIBRARY_PATH=/usr/local/lib` set at both build and run time. License
is AGPL-3.0 — subprocess isolation (not linking) avoids compile-time obligations; a
NOTICE documenting the dependency and its network-use disclosure clause is added to
`capture/cellular/README.md` for whoever later wires this into a network-facing service.

## Test data

Ground truth comes from LTE-Cell-Scanner's own bundled recorded capture,
`regression_test_signal_file/f1815.3_s19.2_bw20_0.08s_hackrf-1.bin` (19.2 Msps, HackRF,
1815.3 MHz), after required preprocessing verified in a build spike:

1. **Decimate 10x** (19.2 Msps → 1.92 Msps = `CellSearch`'s hardcoded internal rate,
   `FS_LTE/16`) via a band-limited FIR + downsample, producing exactly `CAPLENGTH`
   (153,600) samples.
2. **Prepend the 128-byte `--loadbin` header** `write_header_to_bin()` expects (8 magic
   `double`/`uint64` pairs; exact byte layout documented in the implementation plan).
3. Commit the resulting file as `capture/cellular/testdata/cell301_1815.3mhz.bin`
   (~300KB, small enough for git) plus the preprocessing script that produced it (for
   reproducibility, not re-run at test time).

Verified real decode output from this fixture (this is the pass/fail gate for
`test_offline_decode.py`): **Cell ID 301, PSS ID 1, RX power level −9.44976 dB, residual
frequency offset 14302.6 Hz, FDD, 100 RB (20 MHz), 2 antenna ports.**

No synthetic IQ generation — no free tool for it exists (see Amendment above). No
public-recording stretch goal either — the vendored repo's own fixture already serves
that role, so a second external file adds no value here.

## Build & test strategy

- Build steps (dependency install, IT++-from-source, CMake invocation, fixture
  preprocessing) are captured precisely in the implementation plan, verified by an actual
  from-scratch build in a Fedora sandbox.
- If no C++ toolchain / IT++ / RTL-SDR devel headers are available in a given
  environment, the `CellSearch`-dependent tests are marked skip (not fail) — consistent
  with how `KismetClient` and bleak's `service.run` are already excluded from unit-test
  scope as thin I/O wrappers per the WiFi/BT plan.
- `normalizer.py` and `offline_scanner.py`'s stdout-parsing regex are pure Python, tested
  purely with fixture text/dicts — always run, no toolchain dependency.
- `test_offline_decode.py` is the integration test that actually exercises the compiled
  `CellSearch` binary against the real fixture; this is the "does the DSP spike work"
  proof — asserting Cell ID 301 / PSS ID 1 / RX power ≈ −9.45 dB is decoded correctly.
- Runtime envelope for that integration test: ~5s wall, ~420MB peak RSS (verified) — set
  a generous subprocess timeout (e.g. 30s) rather than assuming instant completion.

## Deferred (separate, later plan)

- MIB/SIB1-3 decode via `LTE-Tracker` — offline single-shot batch-decode feasibility is
  unconfirmed for that tool (built for continuous live tracking); needs its own spike.
- libbladeRF2 native AD9361 gain-control layer (replacing LTE-Cell-Scanner's original
  LMS6002D-era gain API).
- Live-radio capture service (`capture/cellular/service.py`) analogous to
  `capture/wifi/service.py` / `capture/bluetooth/service.py`.
- Real xA9 hardware validation (Build Sequencing §10 step 3) — gated on physical
  hardware access, out of reach in this environment.
- Wiring cellular records into the live ingest pipeline (`ingest.service`) and dashboard
  (`viz.app`) — natural once a live capture service exists, not before.
