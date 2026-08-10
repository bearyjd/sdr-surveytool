# Cellular DSP Spike Design

Scope: Phase 1 of the multi-modality survey tool's build sequencing (see
`docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md` §10, step 1) —
the hardware-free half of the cellular DSP spike. Validates that LTE-Cell-Scanner's
PSS/SSS cell search and MIB/SIB1-3 decode logic works, entirely offline, against
recorded/synthetic LTE IQ files.

**Out of scope for this spec** (see "Deferred" below): anything touching real bladeRF
xA9 hardware. This environment has no physical SDR, so the hardware-integration half of
the Build Sequencing step-1 spike (§10 step 3: "validate ... locks onto a real signal on
real xA9 hardware") is a separate, later plan.

## Legal scope boundary

Unchanged from the design doc and `capture/cellular/README.md`: broadcast-channel decode
only (PSS/SSS/PBCH/PDSCH SIB). No paging-channel decoding, no RRC connection setup,
nothing that identifies or tracks individual subscribers. LTE-Cell-Scanner has no
RRC/paging code to begin with, so this is enforced by only vendoring/building the
search+decode modules, not by stripping code that exists.

## Architecture

```
capture/cellular/
  vendor/lte-cell-scanner/   # git submodule, upstream fork
  cmake/                     # build wrapper — builds only search+decode CLI target
  lte_scan_offline            # compiled CLI: reads an IQ file, emits decode JSON
  normalizer.py               # decode JSON -> UnifiedRecord (mirrors capture/wifi)
  testdata/
    synthetic_known_cell.iq   # srsRAN-generated, ground truth baked into test
    synthetic_known_cell.json # expected decode output for the above
    public_recording.iq       # optional, best-effort, if a usable licensed file is found
tests/capture/cellular/
  test_normalizer.py          # unit tests, fixture JSON -> UnifiedRecord
  test_offline_decode.py      # integration: run CLI against testdata, assert decode
```

Process boundary: the C++ core is compiled into a standalone CLI,
`lte_scan_offline --iq-file <path> --out-json <path>`, that never touches a live radio —
it only reads pre-recorded IQ files. This mirrors the existing `capture/wifi` pattern
(`kismet_client.py` talks to Kismet's REST API; `normalizer.py` is pure and unit-tested).
`capture/cellular/normalizer.py` parses the CLI's JSON output into
`schema.records.UnifiedRecord`, using the same `Identifier`/`Modality`/`Signal` shape as
WiFi/Bluetooth.

No `capture/cellular/service.py` (live-capture wrapper) or gain-control layer is written
in this phase — there is no live radio to wrap yet.

## Vendoring

LTE-Cell-Scanner is added as a git submodule under `capture/cellular/vendor/`. The build
wrapper (CMake) compiles only the PSS/SSS search and MIB/SIB1-3 decode source files plus
whatever support code they require, producing the `lte_scan_offline` CLI. The upstream
project's own bladeRF LMS6002D gain-control I/O is not built or ported in this phase.

## Test data

1. **Synthetic (required, primary correctness check)** — an LTE downlink IQ file
   generated with srsRAN's signal-generation tooling, stamped with a known Cell ID and
   known MIB/SIB1-3 content. `test_offline_decode.py` asserts the CLI's decoded output
   exactly matches this known ground truth. This is the pass/fail gate for "Phase 1
   (DSP spike) closed."
2. **Public recording (best-effort, not a blocker)** — if an openly licensed real-world
   LTE IQ capture can be found with clear provenance, it gets a looser test (decodes *a*
   valid Cell ID/MIB, not an exact match, since ground truth is only as good as the
   source's claims). If nothing suitable turns up quickly, this is skipped without
   blocking the milestone.

Test IQ files are kept small (short capture windows) to stay reasonable for git.

## Build & test strategy

- CMake wrapper builds the CLI once. If no C++ toolchain is available in a given
  environment, the C++-dependent tests are marked skip (not fail) — consistent with how
  `KismetClient` and bleak's `service.run` are already excluded from unit-test scope as
  thin I/O wrappers per the WiFi/BT plan.
- `normalizer.py` is pure Python, tested purely with fixture JSON — always runs, no
  toolchain dependency.
- `test_offline_decode.py` is the integration test that actually exercises the compiled
  CLI against the synthetic fixture; this is the one meaningful new "does the DSP spike
  work" proof.

## Deferred (separate, later plan)

- libbladeRF2 native AD9361 gain-control layer (replacing LTE-Cell-Scanner's original
  LMS6002D-era gain API).
- Live-radio capture service (`capture/cellular/service.py`) analogous to
  `capture/wifi/service.py` / `capture/bluetooth/service.py`.
- Real xA9 hardware validation (Build Sequencing §10 step 3) — gated on physical
  hardware access, out of reach in this environment.
- Wiring cellular records into the live ingest pipeline (`ingest.service`) and dashboard
  (`viz.app`) — natural once a live capture service exists, not before.
