# Multi-Modality Coverage-Mapping Survey Tool — Design

## 1. Purpose & Scope

A B2B SaaS coverage-mapping/survey appliance that passively surveys cellular, WiFi, and
Bluetooth signals, plus wideband unknown-signal detection, sold as a subscription
analytics service to carriers (coverage gaps) and potentially facility/venue customers
(WiFi/BT coverage). Deployed as a vehicle-mounted field appliance for drive-by surveys.

### Legal/scope boundaries (hard constraints on every module)

- **Cellular**: decode only publicly broadcast, unencrypted system information
  (LTE MIB/SIB or NR equivalent) — the same broadcast channels every phone uses to find
  and camp on a network. No paging-channel decoding, no RRC connection setup, nothing
  that identifies or tracks individual subscribers.
- **WiFi/Bluetooth**: passive beacon/advertisement scanning only (802.11 beacon/probe
  frames, BLE advertising packets). No association, no deauth, no packet injection, no
  payload/traffic capture, no decryption/cracking.
- **Unknown-signal agent (Part 4)**: output is classification tag + confidence +
  reasoning only, written back to the record's own metadata. It is a dead end in the
  pipeline — it must never trigger, chain into, or hand data to the cellular/WiFi/BT
  decode modules, regardless of what it classifies a signal as.
- **No module branches on a classification**: nothing may read `metadata.tag` or
  `metadata.classification_status` to decide whether to start, retune or chain a
  capture or decode. `tests/test_no_tag_branching.py` enforces this over `capture/`
  and `ingest/` with an explicit allowlist (today: the unknown-signal normalizer's
  single write of `unclassified`).

## 2. Hardware

| Component | Choice | Rationale |
|---|---|---|
| SDR | bladeRF 2.0 micro xA9 | 2×2 MIMO (diversity RX / future direction-finding), Cyclone V GX FPGA, no 2×2 option exists in Ettus's lineup near this price point (their affordable Bus Series is 1×1-only; 2×2/RFNoC-capable Ettus gear starts at $10k+ E320/N310) |
| Compute host | NVIDIA Jetson Orin Nano | Vehicle-mounted (power is not a hard constraint); CUDA GPU runs Part 4's local modulation-classification model and offloads some wideband-scan DSP via cuFFT, directly serving the stated goal of local signal characterization |
| GPS | u-blox M8N | As specified |
| WiFi/BT capture | Standard USB adapters | Kismet (WiFi) + bleak (BT) |
| FPGA offload | Deferred | Bladerf's Cyclone V is user-programmable via Nuand's open FPGA image + Quartus Prime (no RFNoC-style block framework — raw HDL). Explicitly **not** part of the MVP critical path; triggered only by measured DSP bottlenecks from host-CPU profiling (e.g., cellular correlation search or wideband energy-detection scan can't keep up in real time) |
| Training/dev workstation | NVIDIA DGX Spark (stationary) | Second compute tier, not a field-appliance replacement: trains RFML/modulation-classification models (TorchSig) and runs bulk signal-library analysis / GPU-accelerated DSP experimentation off-vehicle. CUDA ecosystem parity with the Jetson avoids cross-platform driver/software translation between training and deployment. Trained models are exported and deployed down to the Jetson for lightweight real-time field inference — the DGX Spark itself never goes in the vehicle. |

## 3. Part 1 Research Findings (summary)

### Cellular
- **gr-lte**: dead (GNU Radio 3.7 dependency, last push 2020). Ruled out.
- **srsRAN_Project** (5G): archived 2026-06-01, succeeded by OCUDU. **srsRAN_4G** still
  maintained but is a full UE stack (RRC/paging code exists, must be explicitly gated to
  stay in legal scope); bladeRF support historically present but xA9-specific status
  unverified; UHD/USRP is its best-tested backend, not bladeRF.
- **LTE-Cell-Scanner**: recommended base. Structurally incapable of exceeding the legal
  boundary (no RRC/paging code exists at all — it's scan-only). Extracts Cell ID,
  MIB, SIB1-3, RSRP/RSRQ/SINR. Risk: its bladeRF backend was written for the original
  bladeRF's LMS6002D gain API; compiles against libbladeRF2 but was only
  hardware-verified on an original bladeRF x40 — **xA9 (AD9361) compatibility is
  unverified, not confirmed.**
- **Plan**: port LTE-Cell-Scanner's PHY search + SIB-decode core onto a gain-control
  layer written directly against libbladeRF2's native AD9361 API. First spike: validate
  cell search locks on real xA9 hardware before further investment. The DSP/decode
  logic itself can be validated in advance against recorded/synthetic LTE IQ files with
  zero hardware, decoupling that risk from the hardware-integration risk.
- **OpenCellID**: CC BY-SA 4.0, commercial use OK with ShareAlike caveats — reference
  only. **WiGLE**: commercial API licensing currently suspended — do not depend on it.

### WiFi / Bluetooth / unknown-signal
- **WiFi**: Kismet — actively maintained, passive by default (deauth disabled),
  headless/systemd/REST support for unattended field use.
- **Bluetooth**: bleak, standalone — not Kismet's BT (its BLE scan mode sends SCAN_REQ,
  technically active), not bluepy (abandoned since 2018). bleak is actively maintained
  and purpose-built for passive advertisement scanning.
- **Unknown-signal**: custom GNU Radio flowgraph over **gr-soapy** (gr-osmosdr is
  deprecated) rather than gr-inspector, whose GNU Radio 3.10+ compatibility is
  unresolved and which is GUI-centric rather than built for an automated
  trigger→snippet→cooldown loop.
- **WiGLE**: reference-only, commercial licensing suspended.

**Unverified items flagged by research, to validate early**: LTE-Cell-Scanner on real
xA9 hardware; Kismet's exact Pi/Jetson resource footprint; Kismet's full BT/WiFi JSON
field list (docs defer to a live field-explorer).

## 4. Unified Record Schema

```jsonc
{
  "timestamp": "ISO8601 UTC",
  "lat": "float",
  "lon": "float",
  "altitude": "float, meters",
  "gps_fix_quality": "int (fix type/quality indicator from u-blox M8N)",
  "survey_id": "string",
  "operator_id": "string",
  "modality": "cellular | wifi | bluetooth | unknown",
  "identifier": {
    // cellular:
    "cell_id": "string?",
    "plmn": "string? (MCC-MNC)",
    "band": "string?",
    // wifi:
    "bssid": "string?",
    "ssid": "string?",
    "channel": "int?",
    // bluetooth:
    "bt_mac": "string?",
    "device_name": "string?",
    // unknown:
    "center_freq": "float? Hz",
    "bandwidth_estimate": "float? Hz"
  },
  "signal": {
    "rssi": "float",
    "rsrp": "float?",
    "rsrq": "float?",
    "snr": "float?",
    "peak_power": "float?"
  },
  "metadata": {
    "encryption_type_if_broadcast_visible": "string?",
    "quality_flags": "object (gps fix quality, sample density, etc.)",
    "sample_count_in_grid_cell": "int",
    "iq_snippet_path": "string?",
    "snippet_duration_ms": "int?",
    "sample_rate": "float?",
    "classification_status": "unclassified | manually_tagged | auto_classified | needs_review",
    "tag": "string?",
    "confidence": "float?",
    "reasoning": "string?"
  }
}
```

Only the `ingest` service writes records in this shape to storage; capture plugins emit
it, they never touch storage.

## 5. Repo Structure (monorepo)

```
sdr-surveytool/
  capture/
    cellular/     # LTE-Cell-Scanner fork + xA9 gain-API port + normalizer
    wifi/         # Kismet integration + normalizer
    bluetooth/    # bleak-based scanner + normalizer
    unknown/      # custom GNU Radio flowgraph (gr-soapy) + trigger/IQ capture + normalizer
  gps/            # u-blox M8N reader, shared fix service
  schema/         # unified record schema (versioned) + validators
  ingest/         # normalizer/geotagger — the only writer to storage
  agent/          # Part 4 classification agent (tools, feature extraction, routing)
  storage/        # Postgres/PostGIS models+migrations, object-storage client
  viz/            # local field-verification dashboard (map + heatmap)
  fpga/           # future-phase bladeRF HDL work — stubbed until profiling justifies it
  docs/
```

Monorepo: module boundaries (not repo boundaries) deliver the "independent and
swappable" plugin goal for a solo/small-team build.

## 6. Pipeline Architecture

Each capture source is a genuinely different technology (Kismet: C++ service + REST
API; bleak: async Python library; unknown-signal: GNU Radio flowgraph; cellular: ported
C++ tool) — they cannot share an in-process plugin interface. The only shared contract
is: **emit unified-schema JSON records onto a local queue.**

```
[cellular proc] ─┐
[wifi/Kismet]  ───┼─→ local queue (unified-schema JSON) ─→ [ingest] ─→ Postgres/PostGIS
[bluetooth/bleak]─┤                                            │
[unknown/GNU Radio]┘                                          └─→ IQ snippets → local disk
                                                                 │  (S3-compatible later)
[GPS/u-blox service] ── nearest-fix lookup ────────────────────┘
```

- `ingest` is the only writer to storage: validates against the schema, attaches the
  nearest GPS fix + quality flag, computes `sample_count_in_grid_cell`.
- Queue: a lightweight, dependency-light local broker (exact choice — e.g. a simple
  NDJSON-over-Unix-socket versus Redis/NATS — deferred to implementation planning;
  not a schema-affecting decision).

## 7. Storage

- **Postgres/PostGIS**: structured unified records, geospatial queries (coverage-gap
  polygons, per-grid-cell aggregation).
- **Local flat-file storage** for IQ snippets (`metadata.iq_snippet_path`), swappable
  for S3-compatible object storage later without a schema change.

## 8. Visualization

A separate, lightweight local web app (live map + signal heatmap) reading from
Postgres, for field verification only — explicitly not the eventual carrier-facing
product dashboard.

## 9. Part 4 — Agentic Signal Characterization

**Trigger**: records with `modality: "unknown"` and `classification_status:
"unclassified"`.

**Agent pipeline**:
1. Feature extraction — bandwidth, center frequency, symbol-rate estimate, modulation
   classification (spectral-shape based), burst vs. continuous, hop pattern. The
   classification model is **trained offline on the DGX Spark using TorchSig**
   (RFML/modulation-classification training, bulk signal-library analysis), then
   exported and run for inference **on the Jetson's GPU in the field** — training and
   deployment share the CUDA ecosystem, avoiding cross-platform driver/software
   translation. This serves the stated goal of on-device signal characterization
   without a per-snippet cloud round-trip.
2. Library matching — compare against SigMF-tagged corpora / RadioML-style references
   to propose candidate signal identities.
3. Context correlation — cross-reference center frequency against public
   band-plan/allocation tables (FCC ULS, ITU) for the survey region.
4. Confidence-based routing — high-confidence → `auto_classified`; low-confidence →
   flagged for human review with reasoning attached.

**Model**: Sonnet — bounded tool use with structured output, not open-ended reasoning.

**Hard-boundary enforcement (structural, not just convention)**: the agent process
authenticates to Postgres as a login role in `surveytool_agent`, which has **no privilege
on `survey_records` at all** (column GRANTs cannot express "these keys of one JSON
column"). It reads only the `security_barrier` view `agent_pending_unknown` (pending
unknown rows: snippet path and flags, sample rate, frequency, peak power and duration; no
location or survey/operator IDs) and writes only through
`classify_unknown(...)`, a `SECURITY DEFINER` function owned by a NOLOGIN owner role. The
function sets exactly `classification_status`, `tag`, `confidence` and `reasoning`, and
only on a row that is still a pending unknown row in the same atomic `UPDATE`, so a human
tag always wins a race. The agent refuses to start unless its role passes an allowlist:
no membership beyond the agent role, no dangerous role attribute, no table or column
privilege on `survey_records`, no CREATE or TEMPORARY anywhere, and no executable
SECURITY DEFINER function but `classify_unknown`. Operationally, the agent's container is
the outer boundary: it runs as the ingest uid with the snippet store mounted read-only,
no ingest socket, and egress only to the database and the Anthropic API. AST tests over
its code (an import allowlist, banned calls) are tripwires for regressions, not the wall.
See
`docs/superpowers/specs/2026-10-03-signal-classification-agent-design.md` and
`storage/sql/agent_boundary.sql`.

## 10. Build Sequencing

1. **Cellular DSP spike (zero hardware cost)**: validate LTE-Cell-Scanner's PSS/SSS
   search + MIB/SIB decode against recorded/synthetic IQ files.
2. **WiFi/BT pipeline first (low risk, mature tooling)**: Kismet + bleak → ingest →
   Postgres/PostGIS → local dashboard, proving the schema/storage/viz pipeline
   end-to-end.
3. **Cellular hardware spike (parallel, time-boxed)**: order bladeRF xA9, validate
   LTE-Cell-Scanner's gain-control port locks onto a real signal on real xA9 hardware.
   Gate further cellular engineering investment on this result.
4. **Unknown-signal capture**: custom GNU Radio flowgraph on gr-soapy — threshold
   energy detection, 1–2s IQ snippet, cooldown, metadata write.
5. **Part 4 agent**: build once unknown-signal records exist to classify; DB-role
   boundary enforcement ships with the agent's first version, not retrofitted later.
   Modulation-classification model is trained on the DGX Spark (TorchSig, using
   captured unknown-signal snippets plus public RFML datasets), then exported and
   deployed to the Jetson for field inference.
6. **FPGA offload**: deferred, revisited only if profiling from steps 1–4 shows a real
   host-CPU DSP bottleneck.

## 11. Open Items Deferred to Implementation Planning (not schema/architecture-affecting)

- Exact local queue/broker technology choice.
- Kismet field-name mapping (concrete JSON field list) — verify against a running
  instance.
- Confirm bladeRF xA9 current pricing (research used unverified older figures).
