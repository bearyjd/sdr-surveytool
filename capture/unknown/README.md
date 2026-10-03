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
- **Corrupt samples:** NaN/inf IQ samples (DMA or driver corruption) are left out of
  every measurement and counted in `quality_flags.non_finite_samples`; the raw IQ is
  still written. A snippet with no finite samples is dropped and logged rather than
  emitted as a record ingest could not validate.
- **Frequency:** `identifier.center_freq` (and the SigMF frequency, and the cooldown
  key) is the frequency read back from the SDR after tuning. A mismatch with
  `--center-freq` is logged.
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
  (resolution sample_rate / 1024).
  - **Noise floor:** measured per FFT bin from the quiet frames of the snippet's
    pre-trigger samples, meaning the frames whose mean power is below the trigger
    threshold (`dsp.spectral.quiet_reference`, `noise_floor_psd`).
    - **Why quiet frames only:** pre-trigger samples are not always signal-free. An
      always-on emitter above the threshold (e.g. an LTE downlink) retriggers as soon
      as its cooldown expires, so it fills them at full power, and as a reference it
      would cancel itself.
    - **What it buys:** the floor follows the SDR's anti-alias roll-off and colored
      noise, and it subtracts an emitter that was already on below the threshold.
      The estimate therefore describes the burst that triggered capture.
  - **Accuracy:** across 20 seeds, within -3%..+3% for 10-70 kHz brick-wall and
    70-85% shaped bursts at 11-20 dB SNR. This holds on white noise and on noise
    through 60%/80%-passband roll-off filters. Bursts only a few 1024-sample frames
    long are overestimated.
  - **Unreliable estimates:** these carry
    `quality_flags.bandwidth_estimate_unreliable: true`:
    - a burst filling more than 90% of the band, or wrapping its edges;
    - a snippet with fewer than 8 quiet pre-trigger frames, which falls back to a
      flat median floor. Typical causes are a trigger right after stream start, or
      an always-on emitter that retriggered. Its bandwidth is still estimated
      against the median, which is fine on flat noise but blind to roll-off.

## Snippet handoff (capture never writes to storage)

Capture writes `<stamp>_<freq>Hz_<id>.sigmf-data` + `.sigmf-meta` into `--staging-dir`
and emits the record with `iq_snippet_path` set to the staged absolute `.sigmf-data` path.
Each file is written under a hidden temporary name, fsynced, and renamed into place,
data first and meta last; the staging directory is fsynced after both renames. So a
`.sigmf-meta` never appears without its complete data, even across a crash. The
meta marks the trigger with standard SigMF annotations: `pre_trigger` (the noise
reference) over `[0, trigger)` and `burst` from the trigger sample on. Every
measurement runs before anything is written. If the record cannot be emitted, the
staged pair is deleted. Capture can start before ingest: the emitter connects
lazily, and while ingest is down, emits fail per snippet and their staged pairs are
deleted. Connects and sends time out after 5 s, so a wedged ingest that stops reading
fails emits the same way instead of blocking capture.

Ingest's `LocalSnippetStore` treats the path as untrusted:

- **What it accepts:** only a `.sigmf-data` file directly inside ingest's
  `--snippet-staging-dir` (absolute path, symlinks not followed), named exactly as the
  capture writer names it (`storage.snippet_store.STAGED_DATA_NAME`).
- **File checks:** both files must be regular files with a single hard link. The data
  file's size must be plausible cf32: non-empty, a multiple of 8 bytes, and at most
  `--max-snippet-bytes` (default 4,915,200,000, the largest snippet capture can be
  configured to write). The meta must be 1 byte to 1 MiB. Ingest never parses SigMF.
- **How it adopts:** it hard-links both into `--snippet-store-dir` without replacing
  anything there. It checks that each linked inode is the one it checked and has
  exactly two links, and fsyncs the store directory before returning, so a database
  row never references a name a power cut could lose. Only then does it remove the
  staged names (and fsync staging). Transient errors (EINTR, EAGAIN, EMFILE, ENFILE)
  are retried up to 3 times before counting as a rejection.
- **Rejections:** a rejected snippet, including any I/O error during adoption,
  doesn't lose the detection. The record is still persisted, with
  `iq_snippet_path: null` and `quality_flags.snippet_rejected` set to the reason.
- **Failed persist:** if persisting the record fails after its snippet was adopted,
  ingest removes the adopted pair, and reverts the grid-density count, only when the
  row is known absent. That means the failure came before COMMIT was issued and a
  fresh query confirms no row references the snippet. Any failure during or after
  COMMIT is in doubt (on Postgres the row can commit a moment after the connection
  drops), and so is a failed check: in those cases the files and count are kept. An
  orphan is recoverable; a dangling reference is not.

**Deployment requirements** (checked at startup, which fails with an actionable
message):

- **One dedicated uid.** Capture and ingest must run as the same dedicated user. The
  staging and store directories must be owned by that uid with no group/other access
  (created `0700`; snippet files are `0600`).
- **One filesystem and mount.** Staging and store must be on the same filesystem and
  mount, because hard links cannot cross either. Ingest proves this at startup with
  a real hard link.
- **Opt-in on the ingest side.** Ingest only adopts snippets when given both
  `--snippet-staging-dir` and `--snippet-store-dir`. Without them it starts exactly
  as for WiFi/BT-only deployments: unknown-signal records are still persisted, but
  without IQ (`iq_snippet_path: null`, `quality_flags.snippet_rejected:
  "no_snippet_store"`).
- **Absolute, matching paths.** Staging and store directories must be absolute
  paths; relative ones are rejected at startup, since capture and ingest would
  resolve them against different working directories. Capture's `--staging-dir`
  defaults to `/var/lib/sdr-surveytool/snippet-staging` and must equal ingest's
  `--snippet-staging-dir`. Create the parent once for the service user, e.g.
  `sudo install -d -o sdr -g sdr -m 700 /var/lib/sdr-surveytool`.

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
sdr-ingest --gps-fix-quality 0 \
    --snippet-staging-dir /var/lib/sdr-surveytool/snippet-staging \
    --snippet-store-dir /var/lib/sdr-surveytool/snippets   # opt in to storing IQ
sdr-capture-unknown --survey-id s1 --operator-id op1 \
    --center-freq 915e6 --noise-floor-dbfs -60
```

Ingest's database URL comes from `SURVEYTOOL_DATABASE_URL` or a systemd credential, never
from a password on its command line (see the root README).

`--noise-floor-dbfs` is required: measure it at your gain and frequency first. All
numeric settings must be finite; `--sample-rate` is capped at 61.44e6 (the AD9361
maximum), and the averaging, pre- and post-trigger windows at 5 s each.
`--cooldown-s` must be at least 1 s and at least one snippet (pre + post). A
cooldown starts at the trigger and survives session rebuilds, even when the capture
never completed.

Startup also refuses a configuration whose snippet buffers could exceed
`--max-snippet-memory-bytes` (default 2 GiB, sized for an 8 GB Jetson). The
estimate is 12 B/sample for 2 queued, 1 processing and 1 assembling snippet, plus
2 × 12 B × pre for the trigger history and 20 B/sample of measurement transients.
The defaults need 1.41 GB; 56 MS/s with 1 s snippets needs 3.94 GB.
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

## Known gaps (follow-ups)

- **Always-on emitters:** an emitter that never drops below the threshold is
  re-captured once per cooldown. It has no quiet reference, so its bandwidth is
  always flagged unreliable (median floor).
- **Ambiguous delivery:** when an emit fails after ingest actually received the
  record, capture deletes the staged IQ that ingest is about to adopt. Fixing this
  needs an acknowledgement or outbox protocol between capture and ingest, for all
  modalities.
- **Orphan cleanup:** no startup janitor removes staging or store orphans left by
  crashes.
- **Queue-full overflow:** beyond 64 queue-full summaries in one session, detections
  are lost. That case is bounded and counted (`queue_full_summary_lost`).
- **Same-uid tampering:** a process running as the service uid can tamper with
  staged files. That is the documented trust boundary.

## Licenses

`sigmf` (sigmf-python) is LGPL-3.0-or-later, used as an unmodified library dependency.
GNU Radio is GPL-3.0-or-later; this module imports it at runtime as a system package.
