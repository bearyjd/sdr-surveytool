# capture/unknown

Wideband unknown-signal detection: a GNU Radio flowgraph on `gr-soapy` (bladeRF xA9)
doing moving-average energy detection at one fixed center frequency, 1 s triggered IQ
snippet capture (0.1 s pre-trigger + 0.9 s post-trigger by default) with a per-frequency
cooldown, written as SigMF and emitted as unified records (`modality: "unknown"`,
`classification_status: "unclassified"`). These records are the input to the Part 4
characterization agent in `agent/`.

**Status**: implemented and tested against synthetic IQ through the real GNU Radio
scheduler. Never run against real SDR hardware (none available). Frequency sweeping,
per-bin FFT detection, Part 4 classification and FPGA offload are out of scope; see
`docs/superpowers/specs/2026-08-11-unknown-signal-capture-design.md`.

## Layout

- `sample_clock.py`: sample index <-> UTC time. All timing is sample-derived.
- `energy_trigger.py`: pure threshold + per-frequency cooldown logic.
- `snippet_assembler.py`: pure streaming pre/post-trigger buffer.
- `snippet_writer.py`: SigMF pair writer (staging directory).
- `normalizer.py`: snippet capture event -> `UnifiedRecord`.
- `flowgraph.py`: GNU Radio graph (the only module importing `gnuradio` at import time).
- `service.py`: `sdr-capture-unknown` entry point, gr-soapy source, fault tolerance.

The dBFS power stats and the occupied-bandwidth estimate live in the shared top-level
`dsp/spectral.py` (numpy only), so the Part 4 agent can reuse them without importing
`capture/`.

## Units and timing

- Powers are **dBFS**: 10*log10 of mean |x|^2, where a full-scale complex sample
  (|x| = 1.0, Soapy CF32 scaling) is 0 dBFS. They are uncalibrated, not dBm.
  `signal.rssi` holds the mean in-burst power, `signal.peak_power` the peak of the
  1 ms moving-average power, and `signal.snr` is rssi minus `--noise-floor-dbfs`.
  Records carry `quality_flags: {"power_units": "dBFS"}`.
- `timestamp` is the trigger sample's time: the wall clock read once at stream start,
  plus sample_index / sample_rate. Cooldowns compare those same sample-derived times.
  The rate is the one read back from the SDR (drivers round unsupported rates); a
  mismatch with `--sample-rate` is logged. Dropped samples (SDR overflow) make
  sample time lag wall time, and an NTP/GPS step moves wall time.
  - **Drift rebuild:** while samples are arriving and the two differ by more than
    `--max-clock-drift-s` (default 2 s, minimum 0.5 s), the radio session is
    rebuilt, which re-anchors sample time. A stream with no samples at all is the
    stall watchdog's case instead (5 s).
  - **Backoff:** sessions that keep drifting within 60 s of opening escalate the
    restart backoff rather than rebuilding in a tight loop.
  - **Backward steps:** cooldowns persist across sessions as absolute UTC. After the
    wall clock steps backward, any stored trigger time later than the new session's
    anchor is clamped to the anchor (and logged), so the step can't extend a
    cooldown.
- `identifier.bandwidth_estimate` is the 99%-power occupied bandwidth of the burst
  (resolution sample_rate / 1024). It is within +-5% at >= 15 dB SNR and up to +24% at
  the 10 dB trigger margin. Bursts shorter than 1024 samples are overestimated.

## Snippet handoff (capture never writes to storage)

Capture writes `<stamp>_<freq>Hz_<id>.sigmf-data` + `.sigmf-meta` into `--staging-dir`
and emits the record with `iq_snippet_path` set to the staged absolute `.sigmf-data` path.
Each file is written under a hidden temporary name, fsynced, and renamed into place,
data first and meta last; the staging directory is fsynced after both renames. So a
`.sigmf-meta` never appears without its complete data, even across a crash. Every
measurement runs before anything is written. If the record cannot be emitted, the
staged pair is deleted. Capture can start before ingest: the emitter connects
lazily, and while ingest is down, emits fail per snippet and their staged pairs are
deleted.

Ingest's `LocalSnippetStore` treats the path as untrusted:

- **What it accepts:** only a `.sigmf-data` file directly inside ingest's
  `--snippet-staging-dir` (absolute path, symlinks not followed), named exactly as the
  capture writer names it (`storage.snippet_store.STAGED_DATA_NAME`).
- **File checks:** both files must be regular files with a single hard link. The data
  file's size must be plausible cf32: non-empty, a multiple of 8 bytes, and at most
  `--max-snippet-bytes` (default 4,915,200,000, the largest snippet capture can be
  configured to write). Ingest never parses SigMF.
- **How it adopts:** it hard-links both into `--snippet-store-dir` without replacing
  anything there. It checks that each linked inode is the one it checked and has
  exactly two links, and only then removes the staged names.
- **Rejections:** a rejected snippet, including any I/O error during adoption,
  doesn't lose the detection. The record is still persisted, with
  `iq_snippet_path: null` and `quality_flags.snippet_rejected` set to the reason.
- **Failed persist:** if persisting the record fails after its snippet was adopted
  (e.g. the database is down), ingest removes the adopted pair from the store again
  instead of orphaning it.

**Deployment requirements** (checked at startup, which fails with an actionable
message):

- **One dedicated uid.** Capture and ingest must run as the same dedicated user. The
  staging and store directories must be owned by that uid with no group/other access
  (created `0700`; snippet files are `0600`).
- **One filesystem and mount.** Staging and store must be on the same filesystem and
  mount, because hard links cannot cross either. Ingest proves this at startup with
  a real hard link.
- **Matching paths.** Both services resolve their directories to absolute paths and
  log them, since relative defaults resolve against each process's working
  directory. Run both from the same directory, or pass the same absolute staging
  path to both.

## System dependencies (Fedora 43, verified)

GNU Radio and SoapySDR are system packages, not pip-installable:

```bash
sudo dnf install gnuradio SoapySDR python3-SoapySDR
```

This gives GNU Radio 3.10.12 with `gnuradio.soapy` built in (no separate gr-soapy
package). The bindings live in the system Python's site-packages, so a virtualenv needs
`--system-site-packages` to see them. The Python dependencies (`numpy`, `sigmf`) come
from `pyproject.toml`.

**bladeRF driver (unverified here)**: Fedora 43's repositories contain no bladeRF or
SoapyBladeRF package (`dnf list 'bladeRF*' 'soapy-blade*'` finds nothing). Build
libbladeRF (https://github.com/Nuand/bladeRF) and the SoapySDR module SoapyBladeRF
(https://github.com/pothosware/SoapyBladeRF) from source per their READMEs, then confirm
with `SoapySDRUtil --find="driver=bladerf"`. Without the module or a device attached,
the service logs `SoapySDR::Device::make() no match` and retries with backoff.

## Running

```bash
sdr-ingest --gps-fix-quality 0            # stages from data/snippet-staging by default
sdr-capture-unknown --survey-id s1 --operator-id op1 \
    --center-freq 915e6 --noise-floor-dbfs -60
```

`--noise-floor-dbfs` is required: measure it at your gain and frequency first. All
numeric settings must be finite; `--sample-rate` is capped at 61.44e6 (the AD9361
maximum), and the averaging, pre- and post-trigger windows at 5 s each.
`--sample-rate` defaults to 20e6. The bladeRF reaches ~56e6, but the Python chain
sustained only ~58 MS/s on a fast x86 desktop and has not been profiled on the Jetson.
A 1 s snippet is 160 MB on disk at 20 MS/s (448 MB at 56 MS/s). In memory it is
240 MB (672 MB), because a float32 power array rides along with the cf32 samples.

Disk and memory stay bounded:

- **Low disk:** a snippet that would leave less than `--min-free-bytes` (default
  2 GiB) free on the staging disk is not written. Staging, the store and the
  database share that disk. A continuous emitter at the default 30 s cooldown
  writes ~19 GB/h at 20 MS/s.
- **Dropped IQ keeps its detection:** when the IQ isn't written, the detection and
  its measurements are still emitted, with `iq_snippet_path: null` and
  `quality_flags.snippet_dropped` set to why:
  - `low_disk` or `staging_unavailable`;
  - `processing_error`: the write failed;
  - `queue_full`: see the next item.
- **Slow consumer:** at most two completed snippets wait between the radio thread
  and the writer. Further ones lose their IQ rather than blocking the radio, but a
  cheap summary still becomes a `queue_full` record: trigger time, peak and mean
  power, and no bandwidth estimate. Building the summary costs ~13 ms on the radio
  thread at 20 MS/s.
- **Source buffering:** the SDR source gets 100 ms of output buffer, so a briefly
  GIL-starved Python tap doesn't overflow it.

The service restarts the radio session when the stream stalls for 5 s. SIGTERM shuts
it down the same way Ctrl-C does.

## Licenses

`sigmf` (sigmf-python) is LGPL-3.0-or-later, used as an unmodified library dependency.
GNU Radio is GPL-3.0-or-later; this module imports it at runtime as a system package.
