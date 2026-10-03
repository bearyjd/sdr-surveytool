# sdr-surveytool

Multi-modality coverage-mapping/survey tool: passive cellular, WiFi, Bluetooth, and
wideband unknown-signal surveying, normalized into a single unified record schema.

See [docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md](docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md)
for the full architecture design.

## Layout

- `capture/cellular/` — vendored LTE-Cell-Scanner submodule (PSS/SSS cell search only, unmodified) + normalizer; gain-control/live-radio work deferred
- `capture/wifi/` — Kismet integration + normalizer
- `capture/bluetooth/` — bleak-based BLE scanner + normalizer
- `capture/unknown/` — custom GNU Radio flowgraph (gr-soapy) + trigger/IQ capture + normalizer
- `gps/` — u-blox M8N reader, shared GPS fix service
- `dsp/` — shared numpy-only signal math (power, occupied bandwidth), no I/O; reused by `agent/`
- `schema/` — unified record schema (versioned) + validators
- `ingest/` — normalizer/geotagger — the only writer to storage
- `agent/` — Part 4 agentic signal-characterization (tools, feature extraction, routing)
- `storage/` — Postgres/PostGIS models + migrations, object-storage client
- `viz/` — local field-verification dashboard (map + heatmap)
- `fpga/` — future-phase bladeRF HDL work, deferred until profiling justifies it

## Deployment notes

- `ingest` and `capture/unknown` must run as **one dedicated uid**. The snippet
  staging and store directories must be owned by it with mode `0700`, and both must
  be on **one filesystem and mount**. Snippets are hard-linked from staging into the
  store. Both services check this at startup.
- Capture services (WiFi, Bluetooth, unknown) never block on ingest. The shared
  emitter connects lazily, so they can start first. Each connect or send times out
  after 5 s and is retried once, so a stalled ingest costs the record being sent
  after about 2 x 5 s. That record is dropped and logged, and capture carries on.
- Unknown-signal capture flags such as `--min-free-bytes` (disk floor, default
  2 GiB) and `--max-clock-drift-s` (re-anchor threshold, default 2 s) are documented
  in [capture/unknown/README.md](capture/unknown/README.md).

## Hardware

bladeRF 2.0 micro xA9, Jetson Orin Nano (field/vehicle-mounted host), u-blox M8N GPS,
NVIDIA DGX Spark (stationary training workstation for RFML/TorchSig model development).
