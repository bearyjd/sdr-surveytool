# Unknown-Signal Capture Design

Scope: Build Sequencing step 4 (`docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md`
§10) — a custom GNU Radio flowgraph on `gr-soapy` doing threshold energy detection,
1-2s triggered IQ snippet capture with cooldown, and metadata write into the unified
record schema. Feeds `Part 4` (agentic signal characterization, §9), which is out of
scope here — this is capture only.

## Environment check

GNU Radio and SoapySDR aren't installed in this environment yet, but are packaged and
available: `dnf search` confirms `gnuradio` 3.10.12.0 (Fedora 43) and `SoapySDR`/
`python3-SoapySDR` are installable. No `gr-soapy`-named package exists separately —
`gnuradio.soapy` has been part of core GNU Radio since 3.9/3.10, consistent with the
3.10.12 version available here. No physical bladeRF or any other SDR hardware exists
in this environment, so — as with the cellular DSP spikes — anything requiring a real
radio is out of reach; this spike targets what's genuinely testable without one.

## Architecture

```
capture/unknown/
  flowgraph.py         # GNU Radio top-level flowgraph, swappable source
  energy_trigger.py     # pure trigger/cooldown logic (power threshold, per-freq cooldown)
  snippet_writer.py      # writes captured IQ as SigMF (.sigmf-data + .sigmf-meta)
  normalizer.py           # snippet capture event -> UnifiedRecord
  service.py               # entrypoint: builds flowgraph against gr-soapy, wires RecordEmitter
tests/capture/unknown/
  test_energy_trigger.py    # pure logic: threshold crossing, cooldown windowing
  test_flowgraph_integration.py  # synthetic IQ source -> asserts snippet + UnifiedRecord
```

**Source is swappable**: in production, `flowgraph.py` builds a `gr-soapy` source block
tuned to a single fixed center frequency, sampling the widest instantaneous bandwidth
the hardware supports (bladeRF xA9/AD9361: up to ~56MHz) — no retuning/frequency-sweep
scheduling in this spike (deferred; see "Out of scope" below). In tests, the same
downstream trigger/capture chain is fed by a GNU Radio file/vector source playing
synthetic IQ (quiet noise, an injected above-threshold burst, quiet again) instead of
`gr-soapy` — this is the only way to actually exercise the real signal-processing chain
without hardware, matching the "decouple detection logic from the SDR source" approach
used for the cellular spikes' file-vs-live-radio split, one layer up at the GNU Radio
block level.

**Trigger metric**: moving-average power (magnitude-squared) over the incoming IQ
stream, compared against a configurable dB threshold above a noise floor. Chosen over
FFT-based per-bin peak detection for this first spike — simpler, well-understood GNU
Radio primitives, easier to drive with synthetic test bursts. Trades off frequency
localization within the wide capture (deferred to Part 4's downstream characterization,
which has the full snippet to analyze).

**Cooldown**: after a snippet capture at the current tuned center frequency, new
triggers at that frequency are ignored for a fixed, configurable window (e.g. 30s
default) before re-arming. Implemented as a dict keyed by center frequency even though
this spike only ever uses one frequency — forward-compatible with future frequency-
sweeping without a rewrite.

**Snippet format**: SigMF (`.sigmf-data` raw complex64 samples + `.sigmf-meta` JSON with
`sample_rate`/`center_freq`/capture datetime/etc.) via the `sigmf-python` library.
Consistent with the repo's pre-existing `.gitignore` `*.sigmf-data` rule (added in the
initial repo scaffold, anticipating this), standard in the SDR/RFML community, and
compatible with the public RFML datasets Part 4 will train against later.

**Ingest wiring**: unlike the cellular DSP spikes (deliberately pre-ingest, proof-only),
this module wires through `capture.common.emitter.RecordEmitter` like `capture/wifi`
and `capture/bluetooth` already do — it's meant to be a real deployable capture
service, not a hardware-free validation exercise. `service.py`'s error handling matches
the WiFi/BT fault-tolerance pattern: transient SoapySDR/hardware errors are caught and
logged per-cycle, not allowed to kill the whole capture process.

## Testing

- `energy_trigger.py`'s threshold/cooldown logic is pure (feed it numpy power-value
  arrays directly, no GNU Radio scheduler involved) — unit-tested directly, matching
  how `capture/wifi/normalizer.py` and the cellular normalizers stayed pure functions.
- `tests/capture/unknown/test_flowgraph_integration.py` drives the full flowgraph with
  a synthetic IQ source (quiet → burst → quiet) and asserts: a SigMF snippet gets
  written, a correctly-populated `UnifiedRecord` gets emitted (modality UNKNOWN,
  populated `center_freq`/`bandwidth_estimate`/`peak_power`/`iq_snippet_path`/
  `snippet_duration_ms`/`sample_rate`), and that injecting a second burst inside the
  cooldown window does NOT produce a second trigger.
- Exact GNU Radio block APIs, SoapySDR device-arg syntax, and precise trigger/cooldown
  parameter defaults are not yet verified against a real build in this environment (GNU
  Radio isn't installed) — the implementation plan needs a real build-and-run step
  (install GNU Radio/SoapySDR, build a minimal flowgraph, verify it actually runs
  against synthetic IQ) before exact code can be written without placeholders, same
  pattern used for both cellular spikes' planning phases.

## Out of scope for this plan

- **Frequency-sweeping / scanning across a range.** This spike uses one fixed center
  frequency at the widest available instantaneous bandwidth. Scan scheduling
  (dwell time per frequency, ordering, coverage gaps) is real added complexity,
  deferred until the single-frequency version works.
- **FFT-based per-bin peak detection / frequency localization within a wide capture.**
  Deferred; Part 4's downstream characterization can localize from the full snippet.
- **Part 4 (agentic signal characterization)** — this plan produces the input records,
  not the classifier.
- **Real bladeRF xA9 hardware validation** — no physical SDR in this environment, same
  constraint as both cellular spikes.
- **FPGA offload** — explicitly deferred project-wide per the design doc's Build
  Sequencing, only revisited if profiling shows a real bottleneck.
