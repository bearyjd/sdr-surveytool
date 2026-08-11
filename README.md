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
- `schema/` — unified record schema (versioned) + validators
- `ingest/` — normalizer/geotagger — the only writer to storage
- `agent/` — Part 4 agentic signal-characterization (tools, feature extraction, routing)
- `storage/` — Postgres/PostGIS models + migrations, object-storage client
- `viz/` — local field-verification dashboard (map + heatmap)
- `fpga/` — future-phase bladeRF HDL work, deferred until profiling justifies it

## Hardware

bladeRF 2.0 micro xA9, Jetson Orin Nano (field/vehicle-mounted host), u-blox M8N GPS,
NVIDIA DGX Spark (stationary training workstation for RFML/TorchSig model development).
