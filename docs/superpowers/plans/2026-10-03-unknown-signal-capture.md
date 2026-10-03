# Unknown-Signal Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `capture/unknown`, a deployable capture service. A GNU Radio flowgraph on
`gr-soapy` detects energy at one fixed center frequency and captures a 1 s triggered IQ
snippet as SigMF, with a per-frequency cooldown. Each snippet becomes a
`modality: "unknown"` `UnifiedRecord` that flows through `RecordEmitter` into ingest. Ingest,
the only writer to storage, then adopts the snippet files into a local snippet store.

**Architecture:** GNU Radio does the per-sample DSP: |x|² followed by a trailing moving
average. A thin Python sink block (`SnippetTap`) forwards each aligned (iq, power) chunk
and its absolute sample offset to `SnippetAssembler`. The assembler is pure numpy and owns
the pre-trigger ring buffer and post-trigger collection, and asks a pure `find_trigger`
function about threshold and cooldown. All timing is sample-derived: the wall clock is read
once per stream start, and every timestamp and cooldown after that is
`anchor + sample_index / sample_rate`. Completed snippets go to the service thread, which
writes them as SigMF into a capture-owned *staging* directory and emits records that point
at the staged file. `IngestService` then moves the pair into `storage.snippet_store.LocalSnippetStore`
and persists a copy of the record that carries the final path. Pure numpy signal math (dBFS
power, occupied bandwidth) lives in a new top-level `dsp/` package, so the Part 4 agent can
reuse it without ever importing `capture/`.

**Tech Stack:** Python ≥ 3.10. GNU Radio 3.10.12 (`gnuradio.blocks`, `gnuradio.soapy`) and
SoapySDR 0.8.1 are system dnf packages. `numpy` and `sigmf` (sigmf-python ≥ 1.13,
LGPL-3.0-or-later) are pip dependencies. The project also uses its existing pydantic schema,
SQLAlchemy storage, Unix-socket queue, and pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-unknown-signal-capture-design.md` (architecture:
`docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md` §4 schema, §6
pipeline, §7 storage).

## Global Constraints

- Python >= 3.10 (`pyproject.toml`). Verified on Python 3.14.7, GNU Radio 3.10.12.0,
  SoapySDR 0.8.1, numpy 2.4.6, sigmf 1.13.0 (Fedora 43).
- **Never run `pip install -e`** in a worktree. The user-site editable install of
  `sdr-surveytool` points at the main checkout, and re-pointing it breaks every other
  checkout. Install new third-party dependencies with `pip install --user` (Task 5 does this
  for `sigmf`). Always run tests from the repository root with `python -m pytest`, so
  `conftest.py` puts the checkout first on `sys.path`.
- Only `ingest` writes to storage (design doc §4, §6). Capture writes SigMF files only into
  its staging directory (`--staging-dir`, default `data/snippet-staging`). Ingest's
  `LocalSnippetStore` moves them into `--snippet-store-dir` (default `data/snippets`). The
  `data/` directory is already gitignored.
- All capture timing is sample-derived (`SampleClock`). Trigger timestamps, SigMF capture
  datetimes and cooldowns never read the wall clock after stream start. No test sleeps or
  depends on real time. The only monotonic clock (the stall watchdog in `_drain`) is injected
  in its tests.
- Only `capture/unknown/flowgraph.py` imports `gnuradio` at module import time.
  `service.py` imports it lazily inside `_open_session`/`_build_soapy_source`, which tests
  never call. Every other module and its tests run without GNU Radio. GNU Radio tests start
  with `pytest.importorskip("gnuradio")`.
- **`dsp/` is shared math only.** It is a new top-level package holding numpy-only helpers,
  with no I/O, no GNU Radio, and no imports from `capture/`. Part 4's `agent/` will reuse it
  for feature extraction and is structurally barred from importing `capture/`. A
  subprocess test (Task 4) enforces this. Capture-specific trigger and cooldown logic stays
  in `capture/unknown/energy_trigger.py`.
- Powers are **dBFS**: 10·log10 of mean |x|², where a full-scale complex sample (|x| = 1.0,
  Soapy CF32) is 0 dBFS. They are uncalibrated, not dBm.
  - `signal.rssi` is the mean in-burst power, because the schema requires `rssi`.
  - `signal.peak_power` is the peak of the 1 ms moving-average power.
  - `signal.snr` is rssi minus the configured noise floor.
  - Records carry `quality_flags={"power_units": "dBFS"}`.
- Unknown records set `classification_status=UNCLASSIFIED` explicitly (Part 4 selects on it,
  and the schema default is `None`). Like every other capture module, they use placeholder
  `lat=0.0, lon=0.0, gps_fix_quality=None`.
- Immutability: records are only ever changed via `model_copy`, and cooldown maps are rebuilt
  by `record_trigger`. `SnippetAssembler` is the one deliberately mutable object, a
  streaming buffer, and its docstring explains why.
- Out of scope (spec): frequency sweeping, FFT per-bin detection/localization, Part 4
  classification, real-hardware validation, FPGA offload.
- **Deviations from the spec and lead decisions, each surfaced deliberately:**
  1. `--sample-rate` defaults to **20e6**, not the spec's "widest (~56 MHz)". The full chain
     peaked at 58–63 MS/s on an x86 desktop (Intel Core Ultra 9 185H). The Jetson Orin Nano
     is unprofiled and certainly slower. 56e6 remains settable.
  2. Fedora 43 has **no bladeRF or SoapyBladeRF package** (`dnf list 'bladeRF*'
     'soapy-blade*'` finds nothing). The README documents a from-source build, which is
     unverified here.
  3. `numpy` and `sigmf` become **core** dependencies rather than an optional extra. This
     follows the existing pyproject structure, where modality runtime deps such as `bleak`
     live in core.
  4. Task 1 changes `conftest.py`, a test-infrastructure fix nobody asked for. Without it, in
     any git worktree, `tests/ingest/*` silently imports `ingest.service` from the **main
     checkout** (see Verified facts, Test infrastructure), so Task 8's ingest tests could
     never pass there.
  5. The noise floor is an operator-supplied, **required** CLI value (`--noise-floor-dbfs`).
     There is no adaptive floor estimation, and the trigger threshold is that floor plus
     `--threshold-db` (default 10).

> **Amendment (post-review, 2026-10-03):** the code review and the security review
> of the executed branch led to the follow-up commits after `cb64986`. Where the task
> code blocks below differ, **the code on the branch is authoritative**. Changes:
>
> - **Bounded resources:**
>   - `--min-free-bytes` floor (default 2 GiB) before writing a snippet;
>   - snippet queue capped at 2 with `put_nowait` drop-and-count;
>   - `cooldown_seconds > 0` and `post_trigger_samples >= 1` enforced, since a zero
>     window hung `process()`;
>   - staged pair deleted when an emit fails;
>   - `occupied_bandwidth_hz` batched in float32, so peak memory no longer scales
>     with the snippet (6 MiB for a 61 MiB input, down from 244 MiB).
> - **Snippet handoff:**
>   - atomic SigMF writes (temp names, data renamed first, meta last, files `0600`);
>   - snippet dirs created `0700`, resolved to absolute paths and logged, and
>     required at startup to be owned by the service uid
>     (`storage.snippet_store.ensure_private_dir`);
>   - `adopt()` rewritten to use `abspath`, lstat for single-link regular files,
>     `os.link(follow_symlinks=False)` with dev/ino verification, link-both-then-unlink
>     with rollback, and a same-filesystem check at construction;
>   - rejections raise `SnippetRejected(reason)`, so ingest persists the record with
>     `iq_snippet_path=None` and `quality_flags.snippet_rejected`;
>   - untrusted names are formatted with `!r`.
> - **Service:**
>   - `_open_session` tested against the real scheduler;
>   - `start()` moved inside try/finally;
>   - `_drain` delivers queued snippets before raising, and also ends the session
>     when sample time drifts more than `--max-clock-drift-s` (default 2 s) from
>     wall time;
>   - SDR source given 100 ms of `min_output_buffer` (GNU Radio accepted up to
>     5.6M items);
>   - SIGTERM shuts down like Ctrl-C;
>   - processed snippets released before waiting for the next;
>   - pyright narrowing and types added, and CLI parsing split.
> - **Task 11 Step 3:** `cooldown_seconds=0.0` is now rejected by `CaptureSettings`.
>   Use `1.0` to see `assert [0.5, 2.0, 4.0] == [0.5, 4.0]`.
> - **Second review round:**
>   - drift is judged only while samples advance (a stall stays a stall);
>   - `--max-clock-drift-s` must be at least 0.5 s;
>   - the sample rate is read back from the SDR (`get_sample_rate(0)`) and carried
>     on `CapturedSnippet.sample_rate`;
>   - quick drift rebuilds (within 60 s) escalate the backoff;
>   - low disk or an unavailable staging dir drops only the IQ: the record is
>     emitted with `quality_flags.snippet_dropped`;
>   - `adopt()` accepts only `STAGED_DATA_NAME`, maps any `OSError` to
>     `SnippetRejected("os_error")`, requires `st_nlink == 2` after linking, and
>     tolerates a vanished source after both links;
>   - the store constructor proves staging-to-store hard links with a real probe
>     instead of comparing `st_dev`.
> - **Adversarial review round:**
>   - ingest discards an adopted pair when its record fails to persist
>     (`SnippetStore.discard`);
>   - SigMF files are fsynced before rename, and the staging dir after;
>   - adopt checks the cf32 size (`bad_size`, `--max-snippet-bytes`);
>   - `RecordEmitter` connects lazily, so capture can start before ingest;
>   - cooldowns from the future are clamped to a new session's anchor;
>   - snippets are measured before anything is written, and a write failure emits
>     a `processing_error` record;
>   - settings must be finite, with `sample_rate <= 61.44e6` and windows `<= 5 s`;
>   - queue-full drops keep a summary emitted as a `queue_full` record.
> - **Final counts:** the full suite gives `230 passed, 1 skipped`, and the
>   `-W error` subset (`tests/capture/unknown tests/dsp
>   tests/storage/test_snippet_store.py`) gives `160 passed`.

## Verified facts (build-and-run spike, 2026-10-03)

Each fact below comes from running code in this environment. No facts are taken from
documentation alone.

**GNU Radio 3.10.12 block APIs (pybind signatures, introspected):**
- `blocks.complex_to_mag_squared(vlen: int = 1)`
- `blocks.moving_average_ff(length: int, scale: float, max_iter: int = 4096, vlen: int = 1)`
  uses a **trailing** window: output *i* = scale · Σ in[*i*−length+1 … *i*], with zero history
  before the stream starts. With length 4, an impulse at index 10 appears at outputs 10–13.
  Both blocks are 1:1, so the tap's two inputs stay sample-aligned.
- `blocks.vector_source_c(data, repeat=False, vlen=1, tags=[])` accepts a numpy complex64
  array directly.
- `blocks.file_source(itemsize, filename, repeat=False, offset=0, len=0)`, with
  `gr.sizeof_gr_complex == 8`.
- Consider a Python `gr.sync_block` with `in_sig=[np.complex64, np.float32], out_sig=None`.
  Its `work()` gets equal-length numpy arrays (complex64, float32), and
  `self.nitems_read(0) == self.nitems_read(1)` is the absolute stream index of
  `input_items[0][0]`.
- `top_block.run()` returns once a finite source is exhausted. `start()` / `stop()` / `wait()`
  work on an infinite source.
- **An exception escaping a Python block's `work()` hangs the flowgraph forever.** GNU Radio
  logs `thread_body_wrapper :error: ...`, the block's thread dies, and `run()`/`wait()` never
  return. Returning `int(gr.WORK_DONE)` (== −1) ends the graph cleanly, with both finite and
  infinite sources. `stop()` + `wait()` return promptly even on a hung graph.
- **Python block objects must stay referenced while the graph runs.** If the Python object is
  garbage-collected, the process aborts (`terminate ... AttributeError: 'Thread' object has no
  attribute 'stop'`) or segfaults: GNU Radio called methods on whatever object reused the
  memory. C++ blocks created inside a builder function survive its return, because the
  flowgraph holds them.
- Throughput on an Intel Core Ultra 9 185H, 22 threads:

  | Chain | Throughput |
  |---|---|
  | `vector_source` → `head` → `null_sink` | 900 MS/s |
  | + `complex_to_mag_squared` + `moving_average_ff(56000)` | 139 MS/s |
  | + `SnippetTap`/`SnippetAssembler`, no trigger | **58 MS/s** |
  | + `SnippetTap`/`SnippetAssembler`, one 1 s snippet | **63 MS/s** |

  Snippet size (computed, not measured): a 1 s snippet is 8 bytes/sample on disk (cf32),
  which is 160 MB at 20 MS/s or 448 MB at 56 MS/s. In memory, `CapturedSnippet` also carries
  a float32 power array, making it 12 bytes/sample: 240 MB or 672 MB, before the
  bandwidth estimator's transient arrays.
- Runtime: the integration test (5.5 s of signal at 100 kS/s) takes 0.86 s. The 41
  `tests/capture/unknown` tests plus the 9 `tests/dsp` tests take ~1.3 s.

**gr-soapy (bundled in GNU Radio 3.10, no separate package):**
- Constructor: `soapy.source(device: str, type: str, nchan: int, dev_args: str,
  stream_args: str, tune_args: list[str], other_settings: list[str])`.
- Per-channel setters: `set_sample_rate(ch, rate)`, `set_frequency(ch, freq)`,
  `set_gain_mode(ch, automatic: bool)`, `set_gain(ch, gain)`, `set_bandwidth(ch, bw)`.
- bladeRF syntax, from GNU Radio's own `/usr/share/gnuradio/grc/blocks/soapy_bladerf_source.block.yml`:
  `soapy.source('driver=bladerf', "fc32", 1, dev_args, '', [''], [''])`.
- With no device or no driver module, the constructor raises
  `RuntimeError('SoapySDR::Device::make() no match')`. The service's real retry loop was run
  against this and logged/retried with backoff of 1, 2, 4, 8 s.

**sigmf-python 1.13.0** (LGPL-3.0-or-later; requires numpy, jsonschema, defusedxml;
`Requires-Python >=3.10`; classifiers through 3.14):
- Write the data with `np.asarray(iq, "<c8").tofile(data_path)`, then build
  `SigMFFile(data_file=..., global_info={sigmf.DATATYPE_KEY: "cf32_le", sigmf.SAMPLE_RATE_KEY: ...})`,
  call `.add_capture(0, metadata={sigmf.FREQUENCY_KEY: ..., sigmf.DATETIME_KEY: ...})`, and
  finish with `.tofile(meta_path)`.
- `tofile` auto-adds `core:sha512`, `core:version` "1.2.6", `core:num_channels` and
  `core:offset`. It refuses to overwrite (`SigMFFileExistsError`).
- The `SigMFFile.*_KEY` class attributes emit `DeprecationWarning` in 1.13. Use the
  module-level `sigmf.*_KEY` constants.
- `sigmf.utils.SIGMF_DATETIME_ISO8601_FMT == "%Y-%m-%dT%H:%M:%S.%fZ"`.
- `sigmf.sigmffile.fromfile()` accepts the `.sigmf-data` path, the `.sigmf-meta` path, or
  the base path. Pairs match by basename (the meta has no dataset key), so moving both files
  together keeps the pair readable.

**Bandwidth estimator (numpy only).** Measured on synthetic brick-wall band-limited bursts
(fs = 100 kHz, 1024-point FFT):
- *99%-power over burst frames, median-subtracted* (chosen): within ±5% at ≥ 15 dB SNR,
  across 2–85 kHz widths, off-center included, and within ±4% across 40 seeds at 20 dB.
  At the 10 dB trigger margin it reads −2% to +24%. A tone measures 4 bins.
- Bursts shorter than one 1024-sample frame are overestimated 6–8×. That is 18 µs at
  56 MS/s, so it rarely matters in practice.
- *Count of bins > 6 dB above the 10th-percentile floor* (rejected): Hann-window leakage
  inflates it at high SNR (2 kHz at 30 dB reads 8 kHz, and a tone reads 22 bins).

**Test infrastructure:**
- In a git worktree, `tests/ingest/__init__.py` makes pytest's importlib mode register the
  *test* directory as the top-level `ingest` package. `ingest.service` is then resolved by the
  user-site editable finder, from the **main checkout**, even in a full-suite run.
- Verified by probe: `ingest.service.__file__ == /var/home/user/Documents/vibe-code/sdr-surveytool/ingest/service.py`.
- `python -c "import ingest"` does not show this, because it only happens under pytest.
- Pre-importing the real packages in `conftest.py` fixes it. Task 1 makes that change.

## Review Focus

These are inputs the spec implies but does not spell out. The owning task's tests pin each
one.

1. **No SDR, an unplugged SDR, or a wedged stream.** The service must log and retry with
   backoff. It must never die, and never hang silently. Tested: Task 10's `run()` retries,
   its stall watchdog, and tap-error re-raise. Task 9 tests the tap's WORK_DONE handling.
2. **A burst whose samples straddle `work()` chunk boundaries, while GNU Radio reuses its
   input buffers.** The snippet must be identical for every chunk size. Tested: Task 3's
   parametrized test feeds 1–5000-sample chunks through one buffer that is overwritten after
   every call.
3. **An emitter still above threshold when its cooldown expires, and cooldown across a
   flowgraph rebuild** (sample numbering restarts at 0). Expected: one new snippet per
   cooldown window (level-triggered), and no immediate retrigger after a rebuild. Tested:
   Task 2 trigger tests, Task 3's rebuilt-assembler test, and Task 10's `run()` test.
4. **Stream edges.** A trigger within the first 0.1 s after a (re)start yields a shorter
   snippet with an honest `snippet_duration_ms`. A stream that ends mid-capture emits nothing
   half-written. Tested: Task 3 and Task 10.
5. **An untrusted `iq_snippet_path` off the ingest socket.** An absolute path elsewhere, a
   `../` traversal, a symlink out of staging, the wrong suffix, or an overwrite must all be
   rejected without moving anything. A rejected record is not persisted and does not bump
   grid density. Tested: Task 7 and Task 8.

## File Structure

| File | Responsibility |
|---|---|
| `conftest.py` (modify) | Pre-import real packages so tests exercise this checkout |
| `capture/unknown/sample_clock.py` | Sample index ↔ UTC time (`SampleClock`) |
| `capture/unknown/energy_trigger.py` | Pure threshold and per-frequency cooldown (`find_trigger`, `record_trigger`) |
| `capture/unknown/snippet_assembler.py` | Pure streaming pre/post-trigger buffer (`SnippetAssembler`, `CapturedSnippet`) |
| `capture/unknown/snippet_writer.py` | SigMF pair into the staging directory |
| `capture/unknown/normalizer.py` | `SnippetCaptureEvent` → `UnifiedRecord` |
| `capture/unknown/flowgraph.py` | GNU Radio graph and `SnippetTap` (the only module importing gnuradio at import) |
| `capture/unknown/service.py` | `CaptureSettings`, `process_snippet`, sessions, watchdog, gr-soapy source, CLI |
| `capture/unknown/README.md` (modify) | System deps, units, handoff, running |
| `dsp/__init__.py`, `dsp/spectral.py`, `dsp/README.md` | Shared numpy-only math: dBFS power stats, 99% occupied bandwidth (reused by Part 4) |
| `README.md` (modify) | Add `dsp/` to the Layout list |
| `storage/snippet_store.py` | `SnippetStore` protocol and `LocalSnippetStore.adopt()` with path validation |
| `ingest/service.py`, `ingest/main.py` (modify) | Adopt staged snippets before persisting; CLI dirs |
| `pyproject.toml` (modify) | numpy/sigmf deps, `capture.unknown` package, `sdr-capture-unknown` script |
| `tests/...` | One test module per unit, plus `tests/capture/unknown/test_flowgraph_integration.py` |

---

### Task 1: Make worktree tests exercise this checkout's code

**Files:**
- Modify: `conftest.py` (append after the existing `sys.path` insert)
- Test: `tests/ingest/test_import_resolution.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a guarantee that every later task's tests import `capture`, `ingest`, `schema`,
  `storage` and `viz` from this checkout. Task 8 depends on this: without it, a worktree's
  `tests/ingest/test_service.py` runs against the main checkout's `ingest/service.py`.

- [ ] **Step 1: Write the failing guard test**

```python
# tests/ingest/test_import_resolution.py
"""Guards against silently testing another checkout's code.

Under --import-mode=importlib, tests/ingest/__init__.py (likewise
tests/storage/ and tests/capture/) gets registered as the top-level
`ingest` package. Submodules such as `ingest.service` then come from any
other finder able to supply them -- in a git worktree, the user-site
editable install pointing at the main checkout -- so the worktree's tests
would run against main's code. conftest.py pre-imports the real packages
to prevent that.
"""
from pathlib import Path

import pytest

import capture.common.emitter
import ingest.service
import storage.repository

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", [capture.common.emitter, ingest.service, storage.repository])
def test_code_under_test_comes_from_this_checkout(module):
    assert Path(module.__file__).resolve().is_relative_to(REPO_ROOT)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/ingest/test_import_resolution.py -v`
Expected in a git worktree: `1 failed, 2 passed`, and the failure is
`test_code_under_test_comes_from_this_checkout[ingest.service]`, because the module was
imported from `/var/home/user/Documents/vibe-code/sdr-surveytool/ingest/service.py`. In the
main checkout this passes even before the fix, since the editable install points at the
checkout itself.

- [ ] **Step 3: Pre-import the real packages in `conftest.py`**

Append to the end of `conftest.py`:

```python

# Import the real top-level packages before pytest collects tests/. Under
# --import-mode=importlib, tests/ingest/__init__.py (likewise tests/storage/,
# tests/capture/) would otherwise be registered as the top-level package of
# the same name, and its submodules would be resolved by whatever other
# finder can supply them -- in a git worktree, the user-site editable install
# pointing at the main checkout. See tests/ingest/test_import_resolution.py.
import capture  # noqa: E402,F401
import ingest  # noqa: E402,F401
import schema  # noqa: E402,F401
import storage  # noqa: E402,F401
import viz  # noqa: E402,F401
```

- [ ] **Step 4: Run the guard and the full suite**

Run: `python -m pytest tests/ingest/test_import_resolution.py -v`
Expected: PASS (3 tests)

Run: `python -m pytest -q`
Expected: `60 passed, 1 skipped`. That is the 57 baseline tests plus these 3. The skip is the
cellular `CellSearch` binary test, which is unrelated.

- [ ] **Step 5: Commit**

```bash
git add conftest.py tests/ingest/test_import_resolution.py
git commit -m "test: make pytest import this checkout's packages, not the editable install's"
```

---

### Task 2: Package skeleton, sample clock, energy trigger

**Files:**
- Create: `capture/unknown/__init__.py` (empty)
- Create: `capture/unknown/sample_clock.py`
- Create: `capture/unknown/energy_trigger.py`
- Modify: `pyproject.toml` (register `capture.unknown`; make numpy a runtime dependency)
- Test: `tests/capture/unknown/test_energy_trigger.py` (no `__init__.py` in
  `tests/capture/unknown/`, matching `tests/capture/wifi/` and `tests/capture/cellular/`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `SampleClock(anchor: datetime, sample_rate: float)`, a frozen dataclass with
    `.time_at(sample_index: int) -> datetime` and
    `.first_index_at_or_after(when: datetime) -> int`.
  - `TriggerConfig(threshold_dbfs: float, cooldown: timedelta)`, frozen.
  - `TriggerEvent(sample_index: int, time: datetime, center_freq_hz: float)`, frozen.
  - `find_trigger(power: np.ndarray, start_index: int, clock: SampleClock, center_freq_hz: float, config: TriggerConfig, last_trigger_at: Mapping[float, datetime]) -> TriggerEvent | None`
  - `record_trigger(last_trigger_at: Mapping[float, datetime], event: TriggerEvent) -> dict[float, datetime]`

Design decisions:
- **Cooldown state is absolute UTC time per center frequency, not sample index.** GNU Radio's
  `nitems_read` restarts at 0 whenever the service rebuilds the flowgraph after a fault, so
  a sample index would be meaningless across a rebuild.
- **The cooldown starts at the trigger sample** (the snippet's start, not its end). With the
  30 s default and a 1 s snippet, the difference is negligible.
- **The trigger is level-triggered**: an emitter still above threshold when its cooldown
  expires triggers again immediately. A continuous emitter is therefore re-sampled once per
  cooldown window instead of recorded once and forgotten.

- [ ] **Step 1: Write the failing tests**

```python
# tests/capture/unknown/test_energy_trigger.py
from datetime import datetime, timedelta, timezone

import numpy as np

from capture.unknown.energy_trigger import (
    TriggerConfig,
    TriggerEvent,
    find_trigger,
    record_trigger,
)
from capture.unknown.sample_clock import SampleClock

# 100 kHz keeps every sample time an exact whole number of microseconds,
# which is timedelta's resolution, so time <-> index round trips are exact.
FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
CLOCK = SampleClock(anchor=ANCHOR, sample_rate=FS)
FREQ = 915e6
CONFIG = TriggerConfig(threshold_dbfs=-30.0, cooldown=timedelta(seconds=1))
QUIET = 1e-4  # -40 dBFS
LOUD = 1e-2  # -20 dBFS


def _power(n: int, loud: slice | None = None) -> np.ndarray:
    power = np.full(n, QUIET, dtype=np.float32)
    if loud is not None:
        power[loud] = LOUD
    return power


def test_sample_clock_maps_index_to_time_and_back():
    assert CLOCK.time_at(150_000) == ANCHOR + timedelta(seconds=1.5)
    assert CLOCK.first_index_at_or_after(ANCHOR + timedelta(seconds=1.5)) == 150_000


def test_no_trigger_when_everything_is_below_threshold():
    assert find_trigger(_power(1000), 0, CLOCK, FREQ, CONFIG, {}) is None


def test_trigger_reports_absolute_index_and_sample_derived_time():
    event = find_trigger(_power(1000, slice(400, 500)), 5_000, CLOCK, FREQ, CONFIG, {})
    assert event == TriggerEvent(
        sample_index=5_400,
        time=ANCHOR + timedelta(seconds=0.054),
        center_freq_hz=FREQ,
    )


def test_power_exactly_at_threshold_triggers():
    power = _power(10)
    power[3] = 10 ** (-30.0 / 10)
    event = find_trigger(power, 0, CLOCK, FREQ, CONFIG, {})
    assert event is not None and event.sample_index == 3


def test_cooldown_suppresses_trigger_at_same_frequency():
    last = {FREQ: CLOCK.time_at(0)}
    # Loud samples at 0.5 s, well inside the 1 s cooldown.
    assert find_trigger(_power(1000, slice(0, 1000)), 50_000, CLOCK, FREQ, CONFIG, last) is None


def test_cooldown_is_tracked_per_center_frequency():
    last = {2.4e9: CLOCK.time_at(0)}
    event = find_trigger(_power(1000, slice(10, 20)), 50_000, CLOCK, FREQ, CONFIG, last)
    assert event is not None and event.sample_index == 50_010


def test_rearms_exactly_when_cooldown_expires():
    last = {FREQ: CLOCK.time_at(0)}
    # Cooldown expires at index 100_000; the chunk covers 99_990..100_009, all loud.
    event = find_trigger(_power(20, slice(0, 20)), 99_990, CLOCK, FREQ, CONFIG, last)
    assert event is not None and event.sample_index == 100_000


def test_continuous_signal_retriggers_once_cooldown_expires():
    """Level-triggered by design: an emitter that never goes quiet is
    re-sampled once per cooldown window instead of recorded once and lost."""
    last = {FREQ: CLOCK.time_at(0)}
    event = find_trigger(_power(200_000, slice(0, 200_000)), 0, CLOCK, FREQ, CONFIG, last)
    assert event is not None and event.sample_index == 100_000


def test_cooldown_survives_a_stream_restart_with_a_new_anchor():
    """Sample indices restart at 0 whenever the flowgraph is rebuilt, so the
    cooldown is kept as absolute time: a trigger 0.2 s before the restart
    still suppresses one 0.3 s after it (0.5 s < 1 s cooldown)."""
    restart_clock = SampleClock(anchor=ANCHOR + timedelta(seconds=0.2), sample_rate=FS)
    last = {FREQ: ANCHOR}
    assert find_trigger(_power(1000, slice(0, 1000)), 30_000, restart_clock, FREQ, CONFIG, last) is None
    event = find_trigger(_power(1000, slice(0, 1000)), 80_000, restart_clock, FREQ, CONFIG, last)
    assert event is not None and event.sample_index == 80_000


def test_record_trigger_returns_new_map_without_mutating_input():
    original = {2.4e9: ANCHOR}
    event = TriggerEvent(sample_index=7, time=CLOCK.time_at(7), center_freq_hz=FREQ)
    updated = record_trigger(original, event)
    assert updated == {2.4e9: ANCHOR, FREQ: CLOCK.time_at(7)}
    assert original == {2.4e9: ANCHOR}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/capture/unknown/test_energy_trigger.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'capture.unknown.energy_trigger'`

- [ ] **Step 3: Create the package and implement**

Create an empty `capture/unknown/__init__.py`.

```python
# capture/unknown/sample_clock.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SampleClock:
    """Maps absolute stream sample indices to UTC datetimes and back.

    All capture timing (trigger timestamps, cooldown windows, SigMF capture
    datetimes) is derived from sample counts, never from the wall clock. The
    wall clock is read once, when a stream starts, to produce `anchor`, so
    timing is deterministic under test and immune to scheduler jitter.
    """

    anchor: datetime  # UTC time of stream sample index 0
    sample_rate: float

    def time_at(self, sample_index: int) -> datetime:
        return self.anchor + timedelta(seconds=sample_index / self.sample_rate)

    def first_index_at_or_after(self, when: datetime) -> int:
        return math.ceil((when - self.anchor).total_seconds() * self.sample_rate)
```

```python
# capture/unknown/energy_trigger.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from capture.unknown.sample_clock import SampleClock


@dataclass(frozen=True)
class TriggerConfig:
    threshold_dbfs: float  # absolute: noise floor (dBFS) + trigger margin (dB)
    cooldown: timedelta


@dataclass(frozen=True)
class TriggerEvent:
    sample_index: int  # absolute stream index of the first above-threshold sample
    time: datetime  # sample-derived UTC time of that sample
    center_freq_hz: float


def find_trigger(
    power: np.ndarray,
    start_index: int,
    clock: SampleClock,
    center_freq_hz: float,
    config: TriggerConfig,
    last_trigger_at: Mapping[float, datetime],
) -> TriggerEvent | None:
    """Return the first sample of `power` at or above the threshold that is
    outside the cooldown window of the last trigger at `center_freq_hz`.

    `power` is linear moving-average |x|^2 with full scale = 1.0 (0 dBFS);
    `power[0]` is stream sample `start_index`. Cooldown is kept as absolute
    time, not sample index, so it survives a flowgraph rebuild that restarts
    sample numbering. Level-triggered: a signal still above threshold when
    its cooldown expires triggers again at the first re-armed sample.
    """
    above = np.flatnonzero(power >= 10.0 ** (config.threshold_dbfs / 10.0))
    last = last_trigger_at.get(center_freq_hz)
    if last is not None:
        rearm_offset = clock.first_index_at_or_after(last + config.cooldown) - start_index
        above = above[above >= rearm_offset]
    if above.size == 0:
        return None
    sample_index = start_index + int(above[0])
    return TriggerEvent(
        sample_index=sample_index,
        time=clock.time_at(sample_index),
        center_freq_hz=center_freq_hz,
    )


def record_trigger(
    last_trigger_at: Mapping[float, datetime], event: TriggerEvent
) -> dict[float, datetime]:
    """Return a new cooldown map with `event` as the latest trigger at its frequency."""
    return {**last_trigger_at, event.center_freq_hz: event.time}
```

In `pyproject.toml`, numpy moves from the `dev` extra into core `dependencies`, because it is
now a runtime dependency. Add `"capture.unknown"` to `[tool.setuptools].packages`. The edited
sections must read:

```toml
dependencies = [
    "pydantic>=2.6",
    "sqlalchemy>=2.0",
    "requests>=2.31",
    "bleak>=0.22",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "numpy>=1.26",  # capture/unknown DSP; also capture/cellular/testdata/generate_fixture.py
]
```

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
]
```

```toml
packages = [
    "schema",
    "capture",
    "capture.common",
    "capture.wifi",
    "capture.bluetooth",
    "capture.cellular",
    "capture.unknown",
    "ingest",
    "storage",
    "viz",
]
```

Do **not** reinstall the package; see Global Constraints. The tests import from the checkout
through `conftest.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/capture/unknown/test_energy_trigger.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add capture/unknown/__init__.py capture/unknown/sample_clock.py capture/unknown/energy_trigger.py tests/capture/unknown/test_energy_trigger.py pyproject.toml
git commit -m "feat: add unknown-signal energy trigger with sample-time cooldown"
```

---

### Task 3: Snippet assembler (pure streaming buffer)

**Files:**
- Create: `capture/unknown/snippet_assembler.py`
- Test: `tests/capture/unknown/test_snippet_assembler.py`

**Interfaces:**
- Consumes: from Task 2, `SampleClock`, `TriggerConfig`, `TriggerEvent`, `find_trigger` and
  `record_trigger`.
- Produces:
  - `CapturedSnippet`, frozen, with fields `iq: np.ndarray` (complex64),
    `power: np.ndarray` (float32, aligned 1:1 with `iq`), `start_index: int`,
    `start_time: datetime` (UTC time of `iq[0]`) and `trigger: TriggerEvent`.
  - `SnippetAssembler(clock: SampleClock, center_freq_hz: float, config: TriggerConfig, pre_trigger_samples: int, post_trigger_samples: int, last_trigger_at: Mapping[float, datetime])`
    with `.process(iq: np.ndarray, power: np.ndarray, start_index: int) -> list[CapturedSnippet]`
    and `.last_trigger_at -> dict[float, datetime]` (returns a copy).

Snippet buffering decision: the GNU Radio block (Task 9) only forwards chunks. All buffering
lives here, in pure numpy, so it is unit-tested with arbitrary chunk sizes and no scheduler:
- The assembler copies whatever it keeps, because GNU Radio reuses `work()`'s input buffers.
- The snippet array is preallocated, so a 448 MB snippet never needs a second concatenated
  copy.
- No trigger is evaluated while a snippet is still being collected.
- A snippet still collecting when input stops is dropped.

- [ ] **Step 1: Write the failing tests**

`_feed` deliberately pushes every chunk through **one reused buffer that is overwritten
after each call**, the way GNU Radio treats `work()` inputs. Without it, a missing `.copy()`
would pass every test. This was mutation-checked: dropping the history `.copy()` fails 5 of
these tests.

```python
# tests/capture/unknown/test_snippet_assembler.py
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from capture.unknown.energy_trigger import TriggerConfig
from capture.unknown.sample_clock import SampleClock
from capture.unknown.snippet_assembler import CapturedSnippet, SnippetAssembler

FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
FREQ = 915e6
CONFIG = TriggerConfig(threshold_dbfs=-30.0, cooldown=timedelta(seconds=1))
PRE = 100
POST = 400


def _assembler(
    clock: SampleClock | None = None,
    last_trigger_at: dict | None = None,
    pre: int = PRE,
) -> SnippetAssembler:
    return SnippetAssembler(
        clock=clock or SampleClock(anchor=ANCHOR, sample_rate=FS),
        center_freq_hz=FREQ,
        config=CONFIG,
        pre_trigger_samples=pre,
        post_trigger_samples=POST,
        last_trigger_at=last_trigger_at or {},
    )


def _signal(n: int, bursts: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Quiet ramp (unique values, so slices are distinguishable) with loud
    bursts. Power is |iq|^2 directly: the assembler only cares that power[i]
    describes iq[i], not how it was averaged."""
    iq = (0.01 * np.exp(1j * np.arange(n) / 10.0)).astype(np.complex64)
    for start, stop in bursts:
        iq[start:stop] *= 10.0
    return iq, (np.abs(iq) ** 2).astype(np.float32)


def _feed(assembler: SnippetAssembler, iq, power, chunk: int) -> list[CapturedSnippet]:
    """Feed in fixed-size chunks through ONE reused buffer pair, scribbled
    over after every call -- exactly what GNU Radio does to work()'s input
    buffers, so any sample the assembler keeps without copying is corrupted."""
    iq_buf = np.empty(chunk, dtype=np.complex64)
    power_buf = np.empty(chunk, dtype=np.float32)
    snippets = []
    for start in range(0, len(iq), chunk):
        n = min(chunk, len(iq) - start)
        iq_buf[:n] = iq[start : start + n]
        power_buf[:n] = power[start : start + n]
        snippets += assembler.process(iq_buf[:n], power_buf[:n], start)
        iq_buf[:] = np.nan
        power_buf[:] = np.nan
    return snippets


def test_quiet_input_produces_no_snippet():
    iq, power = _signal(5_000, [])
    assert _feed(_assembler(), iq, power, 1024) == []


@pytest.mark.parametrize("chunk", [1, 7, 64, 333, 5_000])
def test_snippet_is_exact_slice_around_trigger_for_any_chunking(chunk):
    """GNU Radio hands work() variable-size chunks; a burst straddling chunk
    boundaries must yield the identical snippet as one big chunk."""
    iq, power = _signal(5_000, [(1_000, 1_200)])
    snippets = _feed(_assembler(), iq, power, chunk)
    assert len(snippets) == 1
    snippet = snippets[0]
    assert snippet.trigger.sample_index == 1_000
    assert snippet.start_index == 1_000 - PRE
    assert snippet.start_time == ANCHOR + timedelta(seconds=900 / FS)
    np.testing.assert_array_equal(snippet.iq, iq[900:1_400])
    np.testing.assert_array_equal(snippet.power, power[900:1_400])
    assert snippet.iq.dtype == np.complex64 and snippet.power.dtype == np.float32


def test_trigger_within_first_pre_samples_truncates_pre_trigger_history():
    iq, power = _signal(5_000, [(30, 200)])
    (snippet,) = _feed(_assembler(), iq, power, 16)
    assert snippet.start_index == 0
    assert len(snippet.iq) == 30 + POST
    np.testing.assert_array_equal(snippet.iq, iq[: 30 + POST])


def test_burst_too_close_to_end_of_stream_yields_nothing():
    """A capture still collecting when input stops is dropped, never emitted
    half-written."""
    iq, power = _signal(5_000, [(4_800, 5_000)])
    assert _feed(_assembler(), iq, power, 512) == []


def test_burst_inside_cooldown_is_ignored_and_later_burst_captured():
    # Cooldown 1 s = 100_000 samples after the trigger at 1_000.
    iq, power = _signal(130_000, [(1_000, 1_200), (50_000, 50_200), (120_000, 120_200)])
    assembler = _assembler()
    snippets = _feed(assembler, iq, power, 4096)
    assert [s.trigger.sample_index for s in snippets] == [1_000, 120_000]
    assert assembler.last_trigger_at == {FREQ: ANCHOR + timedelta(seconds=1.2)}


def test_cooldown_carried_into_a_rebuilt_assembler_suppresses_retrigger():
    first = _assembler()
    iq, power = _signal(2_000, [(1_000, 1_200)])
    assert len(_feed(first, iq, power, 512)) == 1

    # Flowgraph rebuilt 0.1 s later: indices restart at 0 under a new anchor.
    rebuilt = _assembler(
        clock=SampleClock(anchor=ANCHOR + timedelta(seconds=0.1), sample_rate=FS),
        last_trigger_at=first.last_trigger_at,
    )
    iq, power = _signal(5_000, [(1_000, 1_200)])
    assert _feed(rebuilt, iq, power, 512) == []


def test_zero_pre_trigger_samples_starts_snippet_at_trigger():
    iq, power = _signal(5_000, [(1_000, 1_200)])
    (snippet,) = _feed(_assembler(pre=0), iq, power, 100)
    assert snippet.start_index == 1_000
    np.testing.assert_array_equal(snippet.iq, iq[1_000:1_400])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/capture/unknown/test_snippet_assembler.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'capture.unknown.snippet_assembler'`

- [ ] **Step 3: Implement**

```python
# capture/unknown/snippet_assembler.py
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from capture.unknown.energy_trigger import (
    TriggerConfig,
    TriggerEvent,
    find_trigger,
    record_trigger,
)
from capture.unknown.sample_clock import SampleClock


@dataclass(frozen=True)
class CapturedSnippet:
    iq: np.ndarray  # complex64; iq[0] is stream sample start_index
    power: np.ndarray  # float32 moving-average |x|^2, aligned 1:1 with iq
    start_index: int
    start_time: datetime  # sample-derived UTC time of iq[0]
    trigger: TriggerEvent


@dataclass
class _ActiveCapture:
    trigger: TriggerEvent
    start_index: int
    iq: np.ndarray  # preallocated pre + post samples
    power: np.ndarray
    filled: int


class SnippetAssembler:
    """Turns a stream of (iq, power) chunks into complete triggered snippets.

    Pure Python/numpy (no GNU Radio import), so it is unit-tested directly
    with arbitrary chunk sizes. Stateful by necessity: it is a streaming
    buffer, and rebuilding the pre-trigger history immutably on every
    scheduler call would copy up to pre_trigger_samples items each time.

    One snippet at a time: no trigger is evaluated while a snippet is still
    being collected, whatever the cooldown. A snippet still collecting when
    the input stops is dropped, never returned half-written.
    """

    def __init__(
        self,
        clock: SampleClock,
        center_freq_hz: float,
        config: TriggerConfig,
        pre_trigger_samples: int,
        post_trigger_samples: int,
        last_trigger_at: Mapping[float, datetime],
    ) -> None:
        self._clock = clock
        self._center_freq_hz = center_freq_hz
        self._config = config
        self._pre = pre_trigger_samples
        self._post = post_trigger_samples
        self._last_trigger_at = dict(last_trigger_at)
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._history_len = 0
        self._active: _ActiveCapture | None = None

    @property
    def last_trigger_at(self) -> dict[float, datetime]:
        return dict(self._last_trigger_at)

    def process(
        self, iq: np.ndarray, power: np.ndarray, start_index: int
    ) -> list[CapturedSnippet]:
        """Consume one chunk; `iq[0]` is stream sample `start_index`. Returns
        every snippet completed within this chunk (usually none)."""
        snippets = []
        pos = 0
        while pos < len(iq):
            if self._active is not None:
                pos = self._collect(iq, power, pos)
                if self._active.filled == len(self._active.iq):
                    snippets.append(self._finish())
                continue
            trigger = find_trigger(
                power[pos:],
                start_index + pos,
                self._clock,
                self._center_freq_hz,
                self._config,
                self._last_trigger_at,
            )
            end = len(iq) if trigger is None else trigger.sample_index - start_index
            self._remember(iq[pos:end], power[pos:end])
            if trigger is None:
                break
            self._begin(trigger)
            pos = end
        return snippets

    def _remember(self, iq: np.ndarray, power: np.ndarray) -> None:
        """Keep (copies of) the most recent pre_trigger_samples. Copies are
        required: GNU Radio reuses its input buffers after work() returns."""
        if self._pre == 0 or len(iq) == 0:
            return
        self._history.append((iq[-self._pre :].copy(), power[-self._pre :].copy()))
        self._history_len += len(self._history[-1][0])
        while self._history_len - len(self._history[0][0]) >= self._pre:
            dropped, _ = self._history.popleft()
            self._history_len -= len(dropped)

    def _begin(self, trigger: TriggerEvent) -> None:
        if self._history:
            pre_iq = np.concatenate([part for part, _ in self._history])[-self._pre :]
            pre_power = np.concatenate([part for _, part in self._history])[-self._pre :]
        else:
            pre_iq = np.empty(0, dtype=np.complex64)
            pre_power = np.empty(0, dtype=np.float32)
        total = len(pre_iq) + self._post
        active = _ActiveCapture(
            trigger=trigger,
            start_index=trigger.sample_index - len(pre_iq),
            iq=np.empty(total, dtype=np.complex64),
            power=np.empty(total, dtype=np.float32),
            filled=len(pre_iq),
        )
        active.iq[: len(pre_iq)] = pre_iq
        active.power[: len(pre_iq)] = pre_power
        self._active = active
        self._last_trigger_at = record_trigger(self._last_trigger_at, trigger)

    def _collect(self, iq: np.ndarray, power: np.ndarray, pos: int) -> int:
        active = self._active
        take = min(len(active.iq) - active.filled, len(iq) - pos)
        active.iq[active.filled : active.filled + take] = iq[pos : pos + take]
        active.power[active.filled : active.filled + take] = power[pos : pos + take]
        active.filled += take
        self._remember(iq[pos : pos + take], power[pos : pos + take])
        return pos + take

    def _finish(self) -> CapturedSnippet:
        active, self._active = self._active, None
        return CapturedSnippet(
            iq=active.iq,
            power=active.power,
            start_index=active.start_index,
            start_time=self._clock.time_at(active.start_index),
            trigger=active.trigger,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/capture/unknown/test_snippet_assembler.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add capture/unknown/snippet_assembler.py tests/capture/unknown/test_snippet_assembler.py
git commit -m "feat: add pure streaming snippet assembler for unknown-signal capture"
```

---

### Task 4: Shared DSP helpers (`dsp/spectral.py`)

**Files:**
- Create: `dsp/__init__.py` (empty), `dsp/spectral.py`, `dsp/README.md`
- Modify: `pyproject.toml` (register `dsp`), `README.md` (one Layout line)
- Test: `tests/dsp/test_spectral.py`. Do **not** create `tests/dsp/__init__.py`: it would
  shadow the real `dsp` package the same way `tests/ingest/__init__.py` shadows `ingest`
  (Task 1).

**Interfaces:**
- Consumes: nothing (numpy and math only; never `capture/`, never `gnuradio`).
- Produces:
  - `dsp.spectral.dbfs(linear_power: float) -> float`
  - `dsp.spectral.peak_power_dbfs(power: np.ndarray) -> float`
  - `dsp.spectral.mean_burst_power_dbfs(power: np.ndarray, threshold_dbfs: float) -> float`
  - `dsp.spectral.occupied_bandwidth_hz(iq: np.ndarray, sample_rate: float, threshold_dbfs: float) -> float`

These live in a top-level `dsp/` package, not `capture/unknown/`. The Part 4 agent
(designed in parallel) reuses them for feature extraction, and its structural rule forbids
`agent/` from importing anything under `capture/`. The last test runs a fresh interpreter
and pins that `dsp.spectral` pulls in nothing from `capture` or `gnuradio`.
Mutation-checked: adding `import capture.unknown.sample_clock` to `spectral.py` fails it with
`AssertionError: ['capture', 'capture.unknown', 'capture.unknown.sample_clock']`.

`bandwidth_estimate` uses option (a) from the lead's decision 4: a cheap numpy estimate
computed from the snippet, not a placeholder. The method and its measured accuracy are under
Verified facts. scipy is deliberately not used, since it is not a project dependency. The
bandwidth tests use a fixed seed, and a 40-seed sweep stayed within ±4%, so the 15% tolerance
is not seed-lucky.

- [ ] **Step 1: Write the failing tests**

```python
# tests/dsp/test_spectral.py
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dsp.spectral import (
    dbfs,
    mean_burst_power_dbfs,
    occupied_bandwidth_hz,
    peak_power_dbfs,
)

FS = 100_000.0
NOISE_POWER = 1e-4  # -40 dBFS
THRESHOLD_DBFS = -30.0


def _noise(rng: np.random.Generator, n: int, power: float) -> np.ndarray:
    scale = math.sqrt(power / 2)
    return (scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


def _band_limited(rng: np.random.Generator, n: int, power: float, bandwidth_hz: float, offset_hz: float) -> np.ndarray:
    """Brick-wall band-limited complex noise: a stand-in for an unknown
    modulated signal with a known true occupied bandwidth."""
    spectrum = np.fft.fft(_noise(rng, n, 1.0))
    freqs = np.fft.fftfreq(n, 1 / FS)
    spectrum[np.abs(freqs - offset_hz) > bandwidth_hz / 2] = 0
    x = np.fft.ifft(spectrum)
    return (x * math.sqrt(power / np.mean(np.abs(x) ** 2))).astype(np.complex64)


def _snippet_with(burst: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """1 s snippet: 0.1 s quiet, burst, quiet tail -- the shape the assembler
    produces (pre-trigger history, trigger, post-trigger)."""
    before = _noise(rng, 10_000, NOISE_POWER)
    during = burst + _noise(rng, len(burst), NOISE_POWER)
    after = _noise(rng, 100_000 - 10_000 - len(burst), NOISE_POWER)
    return np.concatenate([before, during, after])


def test_dbfs_full_scale_is_zero_and_floors_silence():
    assert dbfs(1.0) == 0.0
    assert dbfs(1e-4) == pytest.approx(-40.0)
    assert math.isfinite(dbfs(0.0))


def test_peak_power_is_max_of_moving_average_power():
    power = np.array([1e-4, 1e-2, 1e-3], dtype=np.float32)
    assert peak_power_dbfs(power) == pytest.approx(-20.0, abs=1e-4)


def test_mean_burst_power_averages_only_above_threshold_samples():
    power = np.array([1e-4] * 8 + [1e-2, 1e-2], dtype=np.float32)
    assert mean_burst_power_dbfs(power, THRESHOLD_DBFS) == pytest.approx(-20.0, abs=1e-4)


@pytest.mark.parametrize("bandwidth_hz, offset_hz", [(5_000, -20_000), (20_000, 10_000), (60_000, 0)])
def test_occupied_bandwidth_of_band_limited_burst(bandwidth_hz, offset_hz):
    rng = np.random.default_rng(7)
    # 20 dB above the noise floor, 0.2 s long inside the 1 s snippet.
    burst = _band_limited(rng, 20_000, NOISE_POWER * 100, bandwidth_hz, offset_hz)
    estimate = occupied_bandwidth_hz(_snippet_with(burst, rng), FS, THRESHOLD_DBFS)
    assert estimate == pytest.approx(bandwidth_hz, rel=0.15)


def test_occupied_bandwidth_of_a_tone_is_a_few_bins():
    rng = np.random.default_rng(7)
    tone = (0.1 * np.exp(2j * np.pi * 12_345 * np.arange(20_000) / FS)).astype(np.complex64)
    estimate = occupied_bandwidth_hz(_snippet_with(tone, rng), FS, THRESHOLD_DBFS)
    assert estimate <= 5 * FS / 1024


def test_occupied_bandwidth_of_silence_is_finite():
    estimate = occupied_bandwidth_hz(np.zeros(4096, dtype=np.complex64), FS, THRESHOLD_DBFS)
    assert math.isfinite(estimate) and estimate > 0


def test_dsp_imports_nothing_from_capture_or_gnuradio():
    """agent/ (Part 4) reuses dsp/ and may never import capture/, so dsp/
    must stay free of capture/ and GNU Radio. Fresh interpreter: this test
    session itself has capture/ imported already."""
    probe = (
        "import sys, dsp.spectral; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('capture', 'gnuradio')); "
        "assert not bad, bad"
    )
    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-c", probe], cwd=repo_root, check=True)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/dsp/test_spectral.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'dsp'`

- [ ] **Step 3: Implement**

Create an empty `dsp/__init__.py`.

```python
# dsp/spectral.py
"""Shared numpy-only DSP helpers: no I/O, no GNU Radio, no imports from
capture/. Used by capture/unknown for snippet metadata and meant for reuse
by the Part 4 agent's feature extraction, which may never import capture/.
"""

from __future__ import annotations

import math

import numpy as np

# All powers are dBFS: 10*log10 of linear |x|^2 power where a full-scale
# complex sample (|x| = 1.0, Soapy's CF32 scaling) is 0 dBFS. Uncalibrated:
# not dBm, since no RF power calibration exists for the bladeRF front end.
_SILENCE_FLOOR = 1e-20  # -200 dBFS; keeps log10 finite on all-zero input

_NFFT = 1024
_OCCUPIED_POWER_FRACTION = 0.99


def dbfs(linear_power: float) -> float:
    return 10.0 * math.log10(max(linear_power, _SILENCE_FLOOR))


def peak_power_dbfs(power: np.ndarray) -> float:
    """Peak of the moving-average power (not raw per-sample |x|^2)."""
    return dbfs(float(np.max(power)))


def mean_burst_power_dbfs(power: np.ndarray, threshold_dbfs: float) -> float:
    """Mean of the moving-average power over the samples at or above the
    trigger threshold, i.e. the burst itself, excluding the quiet pre/post
    padding. Falls back to the peak if nothing crosses (cannot happen for an
    assembler snippet, whose trigger sample always does)."""
    above = power[power >= 10.0 ** (threshold_dbfs / 10.0)]
    return dbfs(float(np.mean(above)) if above.size else float(np.max(power)))


def occupied_bandwidth_hz(iq: np.ndarray, sample_rate: float, threshold_dbfs: float) -> float:
    """99%-power occupied bandwidth of the burst inside a snippet.

    Welch-style PSD over only the FFT frames whose mean power reaches the
    trigger threshold (so quiet pre/post padding doesn't dilute the burst),
    minus a per-bin noise floor (the median bin, valid while the burst
    occupies under half the capture bandwidth), then the narrowest span
    holding 99% of the remaining power. Resolution is sample_rate / 1024.
    Verified on synthetic band-limited bursts: within +-5% at >= 15 dB SNR,
    up to +24% at the 10 dB trigger margin; bursts shorter than one
    1024-sample frame are overestimated several-fold.
    """
    nfft = min(_NFFT, len(iq))
    frames = iq[: (len(iq) // nfft) * nfft].reshape(-1, nfft)
    frame_power = np.mean(np.abs(frames) ** 2, axis=1)
    in_burst = frame_power >= 10.0 ** (threshold_dbfs / 10.0)
    if not in_burst.any():
        in_burst = frame_power == frame_power.max()
    window = np.hanning(nfft)
    psd = np.mean(np.abs(np.fft.fft(frames[in_burst] * window, axis=1)) ** 2, axis=0)
    excess = np.clip(np.fft.fftshift(psd) - np.median(psd), 0.0, None)
    total = excess.sum()
    if total <= 0.0:
        return sample_rate / nfft
    cumulative = np.cumsum(excess) / total
    tail = (1.0 - _OCCUPIED_POWER_FRACTION) / 2.0
    low = int(np.searchsorted(cumulative, tail))
    high = int(np.searchsorted(cumulative, 1.0 - tail))
    return (high - low + 1) * sample_rate / nfft
```

````markdown
# dsp

Shared, numpy-only signal-processing helpers: dBFS power statistics and occupied-bandwidth
estimation (`spectral.py`). No I/O, no GNU Radio, and no imports from `capture/`. Used by
`capture/unknown` to describe snippets, and reusable by the Part 4 agent's feature
extraction, which must never import `capture/`.
````

In `pyproject.toml`, add `"dsp"` to `[tool.setuptools].packages`, so the list reads:

```toml
packages = [
    "schema",
    "capture",
    "capture.common",
    "capture.wifi",
    "capture.bluetooth",
    "capture.cellular",
    "capture.unknown",
    "dsp",
    "ingest",
    "storage",
    "viz",
]
```

In the root `README.md` Layout list, insert this line directly above the `schema/` line:

```markdown
- `dsp/` — shared numpy-only signal math (power, occupied bandwidth), no I/O; reused by `agent/`
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/dsp/test_spectral.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add dsp/__init__.py dsp/spectral.py dsp/README.md tests/dsp/test_spectral.py pyproject.toml README.md
git commit -m "feat: add shared dsp package with dBFS power and occupied-bandwidth helpers"
```

---

### Task 5: SigMF snippet writer

**Files:**
- Create: `capture/unknown/snippet_writer.py`
- Modify: `pyproject.toml` (add `sigmf` to core dependencies)
- Test: `tests/capture/unknown/test_snippet_writer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `write_sigmf_snippet(iq: np.ndarray, staging_dir: Path, sample_rate: float, center_freq_hz: float, capture_start: datetime) -> Path`,
  which returns the **absolute** `.sigmf-data` path. The `.sigmf-meta` sibling shares its
  basename. `iq_snippet_path` in records is always this `.sigmf-data` path.

- [ ] **Step 1: Install the dependency (user site, never editable)**

Run: `pip install --user 'sigmf>=1.13'`
Then: `python -c "import sigmf; print(sigmf.__version__)"`
Expected: `1.13.0` or newer.

In `pyproject.toml`, append to `dependencies`, after the numpy line added in Task 2:

```toml
    "sigmf>=1.13",  # capture/unknown snippet files (LGPL-3.0-or-later)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/capture/unknown/test_snippet_writer.py
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import sigmf
from sigmf import sigmffile

from capture.unknown.snippet_writer import write_sigmf_snippet

START = datetime(2026, 10, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)
IQ = (np.arange(1_000) * (1 - 2j) / 1_000).astype(np.complex64)


def test_writes_readable_sigmf_pair_and_returns_absolute_data_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = write_sigmf_snippet(IQ, Path("staging"), 2e6, 915e6, START)

    assert data_path.is_absolute()
    assert data_path.parent == (tmp_path / "staging").resolve()
    assert data_path.suffix == ".sigmf-data"
    assert data_path.with_suffix(".sigmf-meta").is_file()

    recording = sigmffile.fromfile(str(data_path))
    np.testing.assert_array_equal(recording.read_samples(), IQ)
    assert recording.get_global_field(sigmf.DATATYPE_KEY) == "cf32_le"
    assert recording.get_global_field(sigmf.SAMPLE_RATE_KEY) == 2e6
    capture = recording.get_captures()[0]
    assert capture[sigmf.FREQUENCY_KEY] == 915e6
    assert capture[sigmf.DATETIME_KEY] == "2026-10-03T12:00:00.123456Z"


def test_two_snippets_with_identical_start_time_do_not_collide(tmp_path):
    first = write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, START)
    second = write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, START)
    assert first != second
    assert len(list(tmp_path.glob("*.sigmf-data"))) == 2


def test_rejects_naive_capture_start(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        write_sigmf_snippet(IQ, tmp_path, 2e6, 915e6, datetime(2026, 10, 3, 12, 0, 0))
```

- [ ] **Step 3: Run them to verify they fail**

Run: `python -m pytest tests/capture/unknown/test_snippet_writer.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'capture.unknown.snippet_writer'`

- [ ] **Step 4: Implement**

```python
# capture/unknown/snippet_writer.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sigmf
from sigmf import SigMFFile
from sigmf.utils import SIGMF_DATETIME_ISO8601_FMT

_RECORDER = "sdr-surveytool capture.unknown"


def write_sigmf_snippet(
    iq: np.ndarray,
    staging_dir: Path,
    sample_rate: float,
    center_freq_hz: float,
    capture_start: datetime,
) -> Path:
    """Write `iq` as a SigMF pair (`.sigmf-data` raw cf32_le + `.sigmf-meta`
    JSON) directly in `staging_dir` and return the absolute `.sigmf-data`
    path. Staging is capture-owned scratch space: ingest's snippet store
    later moves the pair into storage (only ingest writes to storage).

    `capture_start` is the sample-derived UTC time of iq[0] and becomes the
    SigMF capture's core:datetime. The basename carries a random suffix, so
    concurrent capture processes can never collide.
    """
    if capture_start.tzinfo is None:
        raise ValueError("capture_start must be timezone-aware (UTC)")
    start_utc = capture_start.astimezone(timezone.utc)
    staging_dir = Path(staging_dir).resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{start_utc:%Y%m%dT%H%M%S%fZ}_{center_freq_hz:.0f}Hz_{uuid.uuid4().hex[:8]}"
    data_path = staging_dir / f"{stem}.sigmf-data"
    # '<c8' = little-endian complex64, exactly SigMF's cf32_le.
    np.asarray(iq, dtype="<c8").tofile(data_path)

    meta = SigMFFile(
        data_file=str(data_path),
        global_info={
            sigmf.DATATYPE_KEY: "cf32_le",
            sigmf.SAMPLE_RATE_KEY: float(sample_rate),
            sigmf.RECORDER_KEY: _RECORDER,
            sigmf.DESCRIPTION_KEY: "Energy-triggered unknown-signal snippet",
        },
    )
    meta.add_capture(
        0,
        metadata={
            sigmf.FREQUENCY_KEY: float(center_freq_hz),
            sigmf.DATETIME_KEY: start_utc.strftime(SIGMF_DATETIME_ISO8601_FMT),
        },
    )
    meta.tofile(str(staging_dir / f"{stem}.sigmf-meta"))
    return data_path
```

- [ ] **Step 5: Run the tests to verify they pass, with deprecations as errors**

Run: `python -m pytest tests/capture/unknown/test_snippet_writer.py -v -W error::DeprecationWarning`
Expected: PASS (3 tests). The deprecated `SigMFFile.*_KEY` attributes would fail here, which
is why the module-level `sigmf.*_KEY` constants are used.

- [ ] **Step 6: Commit**

```bash
git add capture/unknown/snippet_writer.py tests/capture/unknown/test_snippet_writer.py pyproject.toml
git commit -m "feat: write unknown-signal snippets as SigMF into a staging directory"
```

---

### Task 6: Normalizer

**Files:**
- Create: `capture/unknown/normalizer.py`
- Test: `tests/capture/unknown/test_normalizer.py`

**Interfaces:**
- Consumes: `schema.records.{ClassificationStatus, Identifier, Metadata, Modality, Signal, UnifiedRecord}`
  (existing).
- Produces:
  - `SnippetCaptureEvent`, frozen, with fields `timestamp: datetime`,
    `center_freq_hz: float`, `sample_rate: float`, `bandwidth_estimate_hz: float`,
    `peak_power_dbfs: float`, `mean_power_dbfs: float`, `noise_floor_dbfs: float`,
    `snippet_path: str` and `snippet_duration_ms: int`.
  - `normalize_snippet_event(event: SnippetCaptureEvent, survey_id: str, operator_id: str) -> UnifiedRecord`

- [ ] **Step 1: Write the failing tests**

```python
# tests/capture/unknown/test_normalizer.py
from datetime import datetime, timezone

import pytest

from capture.unknown.normalizer import SnippetCaptureEvent, normalize_snippet_event
from schema.records import ClassificationStatus, Modality, UnifiedRecord

EVENT = SnippetCaptureEvent(
    timestamp=datetime(2026, 10, 3, 12, 0, 0, 500000, tzinfo=timezone.utc),
    center_freq_hz=915e6,
    sample_rate=2e6,
    bandwidth_estimate_hz=125_000.0,
    peak_power_dbfs=-17.5,
    mean_power_dbfs=-20.0,
    noise_floor_dbfs=-40.0,
    snippet_path="/data/snippet-staging/x.sigmf-data",
    snippet_duration_ms=1000,
)


def test_normalize_snippet_event_maps_fields():
    record = normalize_snippet_event(EVENT, survey_id="s1", operator_id="op1")
    assert record.modality is Modality.UNKNOWN
    assert record.timestamp == EVENT.timestamp
    assert record.survey_id == "s1"
    assert record.operator_id == "op1"
    assert record.identifier.center_freq == 915e6
    assert record.identifier.bandwidth_estimate == 125_000.0
    assert record.signal.rssi == -20.0
    assert record.signal.peak_power == -17.5
    assert record.signal.snr == pytest.approx(20.0)
    assert record.metadata.iq_snippet_path == "/data/snippet-staging/x.sigmf-data"
    assert record.metadata.snippet_duration_ms == 1000
    assert record.metadata.sample_rate == 2e6
    assert record.metadata.quality_flags == {"power_units": "dBFS"}
    # Placeholder position: ingest attaches the real GPS fix.
    assert (record.lat, record.lon, record.gps_fix_quality) == (0.0, 0.0, None)


def test_record_is_queued_for_part4_classification():
    """Part 4 selects records with classification_status == unclassified;
    the schema default is None, so leaving it unset would hide every
    snippet from the classifier."""
    record = normalize_snippet_event(EVENT, survey_id="s1", operator_id="op1")
    assert record.metadata.classification_status is ClassificationStatus.UNCLASSIFIED


def test_record_survives_the_queue_json_round_trip():
    record = normalize_snippet_event(EVENT, survey_id="s1", operator_id="op1")
    assert UnifiedRecord.model_validate_json(record.model_dump_json()) == record
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/capture/unknown/test_normalizer.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'capture.unknown.normalizer'`

- [ ] **Step 3: Implement**

```python
# capture/unknown/normalizer.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from schema.records import (
    ClassificationStatus,
    Identifier,
    Metadata,
    Modality,
    Signal,
    UnifiedRecord,
)


@dataclass(frozen=True)
class SnippetCaptureEvent:
    """Everything known about one staged snippet. Powers are dBFS
    (uncalibrated; full-scale complex sample = 0 dBFS), see
    dsp.spectral."""

    timestamp: datetime  # sample-derived UTC time of the trigger sample
    center_freq_hz: float
    sample_rate: float
    bandwidth_estimate_hz: float
    peak_power_dbfs: float
    mean_power_dbfs: float
    noise_floor_dbfs: float
    snippet_path: str  # absolute path of the staged .sigmf-data file
    snippet_duration_ms: int


def normalize_snippet_event(
    event: SnippetCaptureEvent, survey_id: str, operator_id: str
) -> UnifiedRecord:
    """Converts one snippet capture event into a UnifiedRecord. lat/lon are
    left at 0.0 with gps_fix_quality=None -- ingest.service attaches the real
    fix, matching capture.wifi/bluetooth/cellular.

    Signal.rssi is required by the schema; for this modality it carries the
    mean in-burst power in dBFS (not dBm -- no RF calibration exists), and
    snr is that minus the configured noise floor. quality_flags records the
    unit so a stored row is self-describing next to dBm WiFi/BT rows.
    """
    return UnifiedRecord(
        timestamp=event.timestamp,
        lat=0.0,
        lon=0.0,
        gps_fix_quality=None,
        survey_id=survey_id,
        operator_id=operator_id,
        modality=Modality.UNKNOWN,
        identifier=Identifier(
            center_freq=event.center_freq_hz,
            bandwidth_estimate=event.bandwidth_estimate_hz,
        ),
        signal=Signal(
            rssi=event.mean_power_dbfs,
            snr=event.mean_power_dbfs - event.noise_floor_dbfs,
            peak_power=event.peak_power_dbfs,
        ),
        metadata=Metadata(
            quality_flags={"power_units": "dBFS"},
            iq_snippet_path=event.snippet_path,
            snippet_duration_ms=event.snippet_duration_ms,
            sample_rate=event.sample_rate,
            classification_status=ClassificationStatus.UNCLASSIFIED,
        ),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/capture/unknown/test_normalizer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add capture/unknown/normalizer.py tests/capture/unknown/test_normalizer.py
git commit -m "feat: add unknown-signal normalizer (snippet event -> UnifiedRecord)"
```

---

### Task 7: Local snippet store (ingest's side of the handoff)

**Files:**
- Create: `storage/snippet_store.py`
- Test: `tests/storage/test_snippet_store.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (stdlib only; storage never imports sigmf).
- Produces:
  - `SnippetStore`, a `typing.Protocol` with `adopt(self, staged_data_path: str) -> str`.
    This is the seam for a later S3-compatible store (design doc §7).
  - `LocalSnippetStore(staging_dir: Path | str, root_dir: Path | str)` with
    `.adopt(staged_data_path: str) -> str`, which returns the absolute final `.sigmf-data`
    path. It raises:
    - `ValueError` when the path is not directly inside staging (symlinks are resolved), or
      is not `.sigmf-data`.
    - `FileNotFoundError` when either half of the pair is missing.
    - `FileExistsError` instead of overwriting.

`iq_snippet_path` arrives over the ingest socket, so it is **untrusted** (design doc §6; the
repo's security rules list path traversal and filesystem operations as review triggers).
The validation here is that trust boundary. The data file moves first and the meta last,
because a `.sigmf-meta` in the store marks a complete pair.

- [ ] **Step 1: Write the failing tests**

```python
# tests/storage/test_snippet_store.py
import os

import pytest

from storage.snippet_store import LocalSnippetStore


def _stage(directory, stem: str = "snip") -> tuple:
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / f"{stem}.sigmf-data"
    meta = directory / f"{stem}.sigmf-meta"
    data.write_bytes(b"\x00\x01\x02\x03")
    meta.write_text('{"global": {}}')
    return data, meta


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "staging", tmp_path / "snippets"


def test_adopt_moves_both_files_and_returns_final_data_path(dirs):
    staging, root = dirs
    data, meta = _stage(staging)

    final = LocalSnippetStore(staging, root).adopt(str(data))

    assert final == str((root / "snip.sigmf-data").resolve())
    assert (root / "snip.sigmf-data").read_bytes() == b"\x00\x01\x02\x03"
    assert (root / "snip.sigmf-meta").read_text() == '{"global": {}}'
    assert not data.exists() and not meta.exists()


def test_adopt_rejects_a_path_outside_staging(dirs, tmp_path):
    """iq_snippet_path arrives over the ingest socket, so it is untrusted:
    ingest must never move an arbitrary file into the store."""
    staging, root = dirs
    staging.mkdir()
    victim, _ = _stage(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="staging directory"):
        LocalSnippetStore(staging, root).adopt(str(victim))
    assert victim.exists()


def test_adopt_rejects_dot_dot_traversal_out_of_staging(dirs, tmp_path):
    staging, root = dirs
    staging.mkdir()
    victim, _ = _stage(tmp_path / "elsewhere")
    sneaky = staging / ".." / "elsewhere" / "snip.sigmf-data"
    with pytest.raises(ValueError, match="staging directory"):
        LocalSnippetStore(staging, root).adopt(str(sneaky))
    assert victim.exists()


def test_adopt_rejects_a_symlink_escaping_staging(dirs, tmp_path):
    staging, root = dirs
    victim, _ = _stage(tmp_path / "elsewhere")
    staging.mkdir()
    os.symlink(victim, staging / "link.sigmf-data")
    with pytest.raises(ValueError, match="staging directory"):
        LocalSnippetStore(staging, root).adopt(str(staging / "link.sigmf-data"))
    assert victim.exists()


@pytest.mark.parametrize("suffix", [".sigmf-meta", ".txt", ""])
def test_adopt_rejects_anything_but_a_sigmf_data_file(dirs, suffix):
    staging, root = dirs
    _stage(staging)
    other = staging / f"snip{suffix}"
    other.touch()
    with pytest.raises(ValueError, match=".sigmf-data"):
        LocalSnippetStore(staging, root).adopt(str(other))


def test_adopt_fails_when_meta_sibling_is_missing_and_moves_nothing(dirs):
    staging, root = dirs
    data, meta = _stage(staging)
    meta.unlink()
    with pytest.raises(FileNotFoundError):
        LocalSnippetStore(staging, root).adopt(str(data))
    assert data.exists()


def test_adopt_refuses_to_overwrite_an_existing_snippet(dirs):
    staging, root = dirs
    data, _ = _stage(staging)
    root.mkdir()
    (root / "snip.sigmf-data").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        LocalSnippetStore(staging, root).adopt(str(data))
    assert (root / "snip.sigmf-data").read_bytes() == b"original"
    assert data.exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/storage/test_snippet_store.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'storage.snippet_store'`

- [ ] **Step 3: Implement**

```python
# storage/snippet_store.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

_DATA_SUFFIX = ".sigmf-data"
_META_SUFFIX = ".sigmf-meta"


class SnippetStore(Protocol):
    def adopt(self, staged_data_path: str) -> str:
        """Take ownership of a staged SigMF pair; return its final .sigmf-data path."""
        ...


class LocalSnippetStore:
    """Local flat-file IQ snippet storage (design doc section 7), swappable for
    an S3-compatible store behind the SnippetStore protocol.

    Capture processes write SigMF pairs into a staging directory; ingest --
    the only writer to storage -- adopts them here before persisting the
    record that references them. The staged path comes off the ingest
    socket, so it is treated as untrusted: it must resolve (symlinks
    included) to a .sigmf-data file directly inside the staging directory.
    """

    def __init__(self, staging_dir: Path | str, root_dir: Path | str) -> None:
        self._staging_dir = Path(staging_dir).resolve()
        self._root_dir = Path(root_dir).resolve()

    def adopt(self, staged_data_path: str) -> str:
        data = Path(staged_data_path).resolve()
        if data.parent != self._staging_dir:
            raise ValueError(
                f"Snippet {staged_data_path!r} is not inside the staging "
                f"directory {self._staging_dir}"
            )
        if data.suffix != _DATA_SUFFIX:
            raise ValueError(f"Snippet {staged_data_path!r} is not a {_DATA_SUFFIX} file")
        meta = data.with_suffix(_META_SUFFIX)
        for staged in (data, meta):
            if not staged.is_file():
                raise FileNotFoundError(f"Staged snippet file missing: {staged}")
        final_data = self._root_dir / data.name
        final_meta = self._root_dir / meta.name
        for final in (final_data, final_meta):
            if final.exists():
                raise FileExistsError(f"Snippet already stored: {final}")
        self._root_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(data, final_data)
        # Meta last: a .sigmf-meta in the store marks a complete pair.
        shutil.move(meta, final_meta)
        return str(final_data)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/storage/test_snippet_store.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add storage/snippet_store.py tests/storage/test_snippet_store.py
git commit -m "feat: add local snippet store with staged-path validation"
```

---

### Task 8: Ingest adopts staged snippets before persisting

**Files:**
- Modify: `ingest/service.py` (constructor, `process_one`, new `_adopt_snippet_if_present`)
- Modify: `ingest/main.py` (two CLI arguments, construct the store)
- Test: `tests/ingest/test_service.py` (append), `tests/ingest/test_main.py` (append)

**Interfaces:**
- Consumes: from Task 7, `storage.snippet_store.{SnippetStore, LocalSnippetStore}`.
- Produces:
  - `IngestService(queue_server, session_factory, gps_provider, snippet_store: SnippetStore | None = None)`.
    Existing positional call sites are unchanged.
  - `ingest.main` arguments `--snippet-staging-dir` (default `data/snippet-staging`) and
    `--snippet-store-dir` (default `data/snippets`).

Wiring decision. `ingest/service.py` was read first, and the store wires in cleanly in
`process_one`, so no awkward workaround is needed. Adoption runs **before**
`_attach_grid_density`. The failure behavior:
- **A rejected snippet** (bad path, missing file, or no store configured) raises before any
  state changes, so there is no grid-count rollback. `ingest.main`'s loop already logs it
  with the traceback and continues. The record is not persisted, and its staged files stay
  in staging for manual recovery.
- **A `save_record` failure after adoption** leaves the pair in the store unreferenced: an
  orphan, never a dangling database path. The existing handler still reverts the grid count.

- [ ] **Step 1: Write the failing tests**

In `tests/ingest/test_service.py`, change the imports at the top to:

```python
from datetime import datetime, timezone
from pathlib import Path

import pytest

from capture.common.emitter import RecordEmitter
from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from schema.records import Identifier, Metadata, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord
from storage.repository import save_record as real_save_record
from storage.snippet_store import LocalSnippetStore
```

Append to the end of `tests/ingest/test_service.py`:

```python
def _stage_snippet(staging_dir, stem: str = "snip") -> str:
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / f"{stem}.sigmf-data").write_bytes(b"\x00" * 16)
    (staging_dir / f"{stem}.sigmf-meta").write_text("{}")
    return str((staging_dir / f"{stem}.sigmf-data").resolve())


def _snippet_record(snippet_path: str) -> UnifiedRecord:
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=10.0,
        lon=20.0,
        gps_fix_quality=1,
        survey_id="s",
        operator_id="o",
        modality=Modality.UNKNOWN,
        identifier=Identifier(center_freq=915e6, bandwidth_estimate=20_000.0),
        signal=Signal(rssi=-20.0, peak_power=-17.0),
        metadata=Metadata(iq_snippet_path=snippet_path),
    )


def _snippet_pipeline(tmp_path, snippet_store):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)
    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider, snippet_store=snippet_store)
    return socket_path, server, session_factory, service


def test_process_one_adopts_staged_snippet_and_persists_final_path(tmp_path):
    staging, store_root = tmp_path / "staging", tmp_path / "snippets"
    staged = _stage_snippet(staging)
    socket_path, server, session_factory, service = _snippet_pipeline(
        tmp_path, LocalSnippetStore(staging, store_root)
    )
    try:
        original = _snippet_record(staged)
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(original)

        processed = service.process_one(timeout=2)

        final = str((store_root / "snip.sigmf-data").resolve())
        assert processed.metadata.iq_snippet_path == final
        assert (store_root / "snip.sigmf-meta").is_file()
        assert list(staging.iterdir()) == []
        # The emitted record object is never mutated; ingest works on copies.
        assert original.metadata.iq_snippet_path == staged
        with session_factory() as session:
            rows = session.query(SurveyRecord).all()
            assert len(rows) == 1
            assert rows[0].metadata_["iq_snippet_path"] == final
    finally:
        server.stop()


def test_process_one_rejects_snippet_outside_staging_without_persisting(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    outside = _stage_snippet(tmp_path / "elsewhere")
    socket_path, server, session_factory, service = _snippet_pipeline(
        tmp_path, LocalSnippetStore(staging, tmp_path / "snippets")
    )
    try:
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_snippet_record(outside))
            emitter.emit(_record(lat=10.0, lon=20.0, gps_fix_quality=1))

        with pytest.raises(ValueError, match="staging directory"):
            service.process_one(timeout=2)

        # The rejected record never bumped the density count for its cell.
        assert service.process_one(timeout=2).metadata.sample_count_in_grid_cell == 1
        assert Path(outside).exists()
        with session_factory() as session:
            assert session.query(SurveyRecord).count() == 1
    finally:
        server.stop()


def test_process_one_rejects_snippet_record_when_no_store_configured(tmp_path):
    staged = _stage_snippet(tmp_path / "staging")
    socket_path, server, session_factory, service = _snippet_pipeline(tmp_path, None)
    try:
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_snippet_record(staged))

        with pytest.raises(RuntimeError, match="no snippet store"):
            service.process_one(timeout=2)

        with session_factory() as session:
            assert session.query(SurveyRecord).count() == 0
    finally:
        server.stop()
```

Append to the end of `tests/ingest/test_main.py`:

```python
def test_snippet_directories_default_under_data():
    args = _parse_args(["--gps-fix-quality", "0"])
    assert args.snippet_staging_dir == "data/snippet-staging"
    assert args.snippet_store_dir == "data/snippets"


def test_snippet_directories_passthrough():
    args = _parse_args(
        [
            "--gps-fix-quality", "0",
            "--snippet-staging-dir", "/srv/staging",
            "--snippet-store-dir", "/srv/snippets",
        ]
    )
    assert args.snippet_staging_dir == "/srv/staging"
    assert args.snippet_store_dir == "/srv/snippets"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/ingest -v`
Expected: `5 failed, 26 passed`. The three new service tests fail with
`TypeError: IngestService.__init__() got an unexpected keyword argument 'snippet_store'`, and
the two new main tests fail on the missing arguments. If instead *every* service test
behaves as if your edits don't exist, Task 1's conftest fix is missing.

- [ ] **Step 3: Implement in `ingest/service.py`**

Add the import after `from storage.repository import save_record`:

```python
from storage.snippet_store import SnippetStore
```

Replace the class docstring's first paragraph with:

```python
    """Consumes validated records from a QueueServer, adopts any staged IQ
    snippet into the snippet store, attaches the nearest GPS fix when the
    record didn't already carry a usable one, tracks per-grid-cell sample
    density, and persists via storage.repository.save_record.
```

Replace the constructor signature and the first three assignments with:

```python
    def __init__(
        self,
        queue_server: QueueServer,
        session_factory: sessionmaker | None,
        gps_provider: GpsFixProvider,
        snippet_store: SnippetStore | None = None,
    ) -> None:
        self._queue_server = queue_server
        self._session_factory = session_factory
        self._gps_provider = gps_provider
        self._snippet_store = snippet_store
```

In `process_one`, replace

```python
        record = self._queue_server.get(timeout=timeout)
        record = self._attach_gps_if_missing(record)
```

with

```python
        record = self._queue_server.get(timeout=timeout)
        # Adopt before the grid-density bump: a rejected snippet raises here
        # with nothing to roll back. If save_record later fails, the adopted
        # pair stays in the store unreferenced (an orphan, never a dangling
        # path in the database).
        record = self._adopt_snippet_if_present(record)
        record = self._attach_gps_if_missing(record)
```

Add this method directly above `_has_usable_fix`:

```python
    def _adopt_snippet_if_present(self, record: UnifiedRecord) -> UnifiedRecord:
        """Move a capture-staged SigMF pair into the snippet store and point
        the record at its final location. Capture never writes to storage
        itself (design doc section 6); this is where snippets cross over."""
        staged = record.metadata.iq_snippet_path
        if staged is None:
            return record
        if self._snippet_store is None:
            raise RuntimeError(
                f"Record carries iq_snippet_path {staged!r} but ingest has no "
                "snippet store configured"
            )
        final = self._snippet_store.adopt(staged)
        updated_metadata = record.metadata.model_copy(update={"iq_snippet_path": final})
        return record.model_copy(update={"metadata": updated_metadata})
```

- [ ] **Step 4: Implement in `ingest/main.py`**

Add the import after `from storage.db import init_db, make_engine, make_session_factory`:

```python
from storage.snippet_store import LocalSnippetStore
```

In `_parse_args`, add these two arguments directly after the `--poll-timeout` argument:

```python
    parser.add_argument(
        "--snippet-staging-dir",
        default="data/snippet-staging",
        help="Directory capture processes stage SigMF snippets in. Must match "
        "the capture side's --staging-dir; snippet paths outside it are rejected.",
    )
    parser.add_argument(
        "--snippet-store-dir",
        default="data/snippets",
        help="Directory ingest moves adopted SigMF snippets into.",
    )
```

In `main`, replace

```python
    service = IngestService(server, session_factory, gps_provider)
```

with

```python
    snippet_store = LocalSnippetStore(args.snippet_staging_dir, args.snippet_store_dir)
    service = IngestService(server, session_factory, gps_provider, snippet_store=snippet_store)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/ingest -v`
Expected: PASS (31 tests: 23 pre-existing, the 3 guard tests from Task 1, and 5 new)

- [ ] **Step 6: Commit**

```bash
git add ingest/service.py ingest/main.py tests/ingest/test_service.py tests/ingest/test_main.py
git commit -m "feat: ingest adopts staged IQ snippets into the snippet store before persisting"
```

---

### Task 9: GNU Radio flowgraph

**Files:**
- Create: `capture/unknown/flowgraph.py`
- Test: `tests/capture/unknown/test_flowgraph.py`

**Interfaces:**
- Consumes: from Task 3, `SnippetAssembler` and `CapturedSnippet`. Also GNU Radio
  (`sudo dnf install gnuradio SoapySDR python3-SoapySDR`).
- Produces:
  - `SnippetTap(assembler, on_snippet: Callable[[CapturedSnippet], None])`, a
    `gr.sync_block` with attributes `samples_seen: int` and `error: Exception | None`.
  - `CaptureFlowgraph(top_block: gr.top_block, tap: SnippetTap)`, frozen.
  - `build_flowgraph(source: gr.basic_block, averaging_samples: int, assembler: SnippetAssembler, on_snippet: Callable[[CapturedSnippet], None]) -> CaptureFlowgraph`

Three verified GNU Radio behaviors shape this file:
1. An exception escaping `work()` hangs the graph forever, so the tap catches everything and
   returns `int(gr.WORK_DONE)`.
2. A garbage-collected Python block crashes the process, so `CaptureFlowgraph` keeps the tap
   referenced next to its top block.
3. The tests must never hang the suite, so `_run_bounded` uses `start()`, a bounded `join`,
   and `stop()`. Mutation-checked: without the tap's `try/except`, the tap-failure test fails
   after 10 s with `flowgraph did not finish within 10.0s` instead of hanging.

- [ ] **Step 1: Write the failing tests**

```python
# tests/capture/unknown/test_flowgraph.py
"""Real GNU Radio scheduler tests. GNU Radio is a system (dnf) package, not
pip-installable, so these skip cleanly where it is absent."""
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("gnuradio")

from gnuradio import blocks  # noqa: E402

from capture.unknown.energy_trigger import TriggerConfig  # noqa: E402
from capture.unknown.flowgraph import CaptureFlowgraph, build_flowgraph  # noqa: E402
from capture.unknown.sample_clock import SampleClock  # noqa: E402
from capture.unknown.snippet_assembler import SnippetAssembler  # noqa: E402

FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
AVERAGING = 100  # 1 ms
NOISE_POWER = 1e-4  # -40 dBFS
BURST_POWER = 1e-2  # -20 dBFS
CONFIG = TriggerConfig(threshold_dbfs=-30.0, cooldown=timedelta(seconds=3))
PRE, POST = 10_000, 90_000  # 1.0 s snippets


def _signal(n: int, burst_starts: list[int], burst_len: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(3)
    iq = np.sqrt(NOISE_POWER / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    for start in burst_starts:
        iq[start : start + burst_len] += np.sqrt(BURST_POWER) * np.exp(
            2j * np.pi * 0.1 * np.arange(burst_len)
        )
    return iq.astype(np.complex64)


def _assembler() -> SnippetAssembler:
    return SnippetAssembler(
        clock=SampleClock(anchor=ANCHOR, sample_rate=FS),
        center_freq_hz=915e6,
        config=CONFIG,
        pre_trigger_samples=PRE,
        post_trigger_samples=POST,
        last_trigger_at={},
    )


def _run_bounded(flowgraph: CaptureFlowgraph, timeout: float = 30.0) -> None:
    """start() + bounded wait(): a regression that hangs the scheduler fails
    this test instead of hanging the whole suite."""
    waiter = threading.Thread(target=flowgraph.top_block.wait, daemon=True)
    flowgraph.top_block.start()
    waiter.start()
    waiter.join(timeout)
    if waiter.is_alive():
        flowgraph.top_block.stop()
        waiter.join(5)
        pytest.fail(f"flowgraph did not finish within {timeout}s")


def test_flowgraph_captures_burst_aligned_with_source_samples():
    iq = _signal(200_000, [50_000])
    snippets = []
    flowgraph = build_flowgraph(blocks.vector_source_c(iq, False), AVERAGING, _assembler(), snippets.append)
    _run_bounded(flowgraph)

    assert len(snippets) == 1
    snippet = snippets[0]
    # The trailing moving average crosses -30 dBFS ~9% of a window into the burst.
    assert 50_000 <= snippet.trigger.sample_index < 50_000 + AVERAGING
    assert snippet.start_index == snippet.trigger.sample_index - PRE
    np.testing.assert_array_equal(snippet.iq, iq[snippet.start_index : snippet.start_index + PRE + POST])
    # power[i] is the mean |x|^2 of the AVERAGING samples ending at i.
    i = snippet.trigger.sample_index
    expected = np.mean(np.abs(iq[i - AVERAGING + 1 : i + 1]) ** 2)
    assert snippet.power[PRE] == pytest.approx(expected, rel=1e-4)
    assert flowgraph.tap.samples_seen == len(iq)
    assert flowgraph.tap.error is None


def test_flowgraph_suppresses_burst_inside_cooldown():
    # Bursts at 0.5 s, 2.0 s (inside the 3 s cooldown) and 4.0 s (after it).
    iq = _signal(550_000, [50_000, 200_000, 400_000])
    snippets = []
    flowgraph = build_flowgraph(blocks.vector_source_c(iq, False), AVERAGING, _assembler(), snippets.append)
    _run_bounded(flowgraph)

    starts = [s.trigger.sample_index for s in snippets]
    assert len(starts) == 2
    assert 50_000 <= starts[0] < 50_100
    assert 400_000 <= starts[1] < 400_100


def test_tap_failure_ends_the_flowgraph_instead_of_hanging_it():
    """An exception escaping a Python block's work() kills its scheduler
    thread and leaves wait() blocked forever (verified). The tap must turn
    it into WORK_DONE and expose it as .error."""

    class ExplodingAssembler:
        def process(self, iq, power, start_index):
            raise ValueError("assembler bug")

    flowgraph = build_flowgraph(
        blocks.vector_source_c(_signal(100_000, []), False), AVERAGING, ExplodingAssembler(), lambda s: None
    )
    _run_bounded(flowgraph, timeout=10.0)
    assert isinstance(flowgraph.tap.error, ValueError)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/capture/unknown/test_flowgraph.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'capture.unknown.flowgraph'`.
On a machine without GNU Radio, the module is SKIPPED instead
(`could not import 'gnuradio'`).

- [ ] **Step 3: Implement**

```python
# capture/unknown/flowgraph.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from gnuradio import blocks, gr

from capture.unknown.snippet_assembler import CapturedSnippet, SnippetAssembler

# moving_average_ff recomputes its running sum from scratch every max_iter
# items to bound float32 accumulation drift; 4096 is GNU Radio's default.
_MOVING_AVERAGE_MAX_ITER = 4096


class SnippetTap(gr.sync_block):
    """Terminal block: forwards every aligned (iq, power) chunk, with its
    absolute stream offset, to a pure-Python SnippetAssembler, and hands
    completed snippets to `on_snippet`.

    on_snippet runs on the GNU Radio scheduler thread: it must be quick and
    non-blocking (the service passes an unbounded queue's put). Disk I/O
    here would stall the stream and overflow the SDR.
    """

    def __init__(
        self,
        assembler: SnippetAssembler,
        on_snippet: Callable[[CapturedSnippet], None],
    ) -> None:
        gr.sync_block.__init__(
            self,
            name="unknown_snippet_tap",
            in_sig=[np.complex64, np.float32],
            out_sig=None,
        )
        self._assembler = assembler
        self._on_snippet = on_snippet
        self.samples_seen = 0  # progress counter for the service's stall watchdog
        self.error: Exception | None = None

    def work(self, input_items, output_items) -> int:
        iq, power = input_items
        start = self.nitems_read(0)
        try:
            for snippet in self._assembler.process(iq, power, start):
                self._on_snippet(snippet)
        except Exception as exc:
            # An exception escaping work() kills this block's scheduler thread
            # and leaves the flowgraph hung forever (verified on GNU Radio
            # 3.10.12). WORK_DONE ends the flowgraph cleanly instead; the
            # owner re-raises self.error.
            self.error = exc
            return int(gr.WORK_DONE)
        self.samples_seen = start + len(iq)
        return len(iq)


@dataclass(frozen=True)
class CaptureFlowgraph:
    """Holds the top block AND the Python tap. GNU Radio keeps only a raw
    pointer to Python blocks: if the tap object is garbage collected while
    the graph runs, the process aborts or segfaults (verified), so whoever
    runs the graph must keep this object alive until wait() returns."""

    top_block: gr.top_block
    tap: SnippetTap


def build_flowgraph(
    source: gr.basic_block,
    averaging_samples: int,
    assembler: SnippetAssembler,
    on_snippet: Callable[[CapturedSnippet], None],
) -> CaptureFlowgraph:
    """source (complex64) -> |x|^2 -> trailing moving average -> tap input 1,
    with the raw source also on tap input 0. Both paths are 1:1 sync blocks,
    so tap input i carries sample i and the mean |x|^2 of samples
    i-averaging_samples+1 .. i. The source is any complex64 GNU Radio block:
    gr-soapy in production, a vector/file source in tests."""
    top_block = gr.top_block("unknown_signal_capture")
    magnitude_squared = blocks.complex_to_mag_squared(1)
    moving_average = blocks.moving_average_ff(
        averaging_samples, 1.0 / averaging_samples, _MOVING_AVERAGE_MAX_ITER, 1
    )
    tap = SnippetTap(assembler, on_snippet)
    top_block.connect(source, magnitude_squared, moving_average, (tap, 1))
    top_block.connect(source, (tap, 0))
    return CaptureFlowgraph(top_block=top_block, tap=tap)
```

The first flowgraph in a fresh environment may print
`vmcircbuf_prefs::get :info: ... vmcircbuf_default_factory failed to open`. GNU Radio is
creating its prefs file, and the message is harmless.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/capture/unknown/test_flowgraph.py -v`
Expected: PASS (3 tests) in under a second.

- [ ] **Step 5: Commit**

```bash
git add capture/unknown/flowgraph.py tests/capture/unknown/test_flowgraph.py
git commit -m "feat: add GNU Radio energy-detection flowgraph with snippet tap"
```

---

### Task 10: Capture service, CLI, README

**Files:**
- Create: `capture/unknown/service.py`
- Modify: `capture/unknown/README.md` (replace the contents)
- Modify: `pyproject.toml` (add the `sdr-capture-unknown` script)
- Test: `tests/capture/unknown/test_service.py`

**Interfaces:**
- Consumes:
  - Tasks 2–6: `SampleClock`, `TriggerConfig`, `record_trigger`, `SnippetAssembler`,
    `CapturedSnippet`, `dsp.spectral.{peak_power_dbfs, mean_burst_power_dbfs, occupied_bandwidth_hz}`,
    `write_sigmf_snippet`, `SnippetCaptureEvent` and `normalize_snippet_event`.
  - Task 9: `build_flowgraph`, imported lazily.
  - Existing: `capture.common.emitter.RecordEmitter`.
- Produces:
  - `CaptureSettings`, a frozen dataclass that validates itself. It has a `threshold_dbfs`
    property and `samples(seconds) -> int`.
  - `process_snippet(snippet: CapturedSnippet, settings: CaptureSettings, survey_id: str, operator_id: str) -> UnifiedRecord`,
    which Task 11 calls directly.
  - `run(settings, socket_path, survey_id, operator_id) -> None`.
  - `main(argv=None)`.
  - Private, and monkeypatched by tests: `_open_session(settings, last_trigger_at)` (a
    context manager yielding an iterator of snippets), `_drain(...)`, `_stage_and_emit(...)`
    and `_parse_args(argv) -> tuple[CaptureSettings, argparse.Namespace]`.

Fault tolerance follows the WiFi/BT pattern of per-cycle catch, log and doubling backoff up
to 60 s, adapted to a long-running radio session:
- **Failing to open the radio** (`SoapySDR::Device::make() no match`) backs off 1 → 2 → 4 … 60 s.
- **A session that opened and later died** (stall watchdog or tap error) restarts after 1 s.
- **One snippet failing to stage or emit** is logged and dropped. The radio keeps running.
- **Cooldown state survives rebuilds** as absolute trigger times owned by `run()`.
- **GNU Radio failures are polled.** They are invisible to Python's call stack, so `_drain`
  checks `tap.error` and `tap.samples_seen` progress. That covers an unplugged device, a
  wedged driver, or a source returning WORK_DONE, whatever gr-soapy turns out to do on real
  hardware (unverifiable here).

- [ ] **Step 1: Write the failing tests**

```python
# tests/capture/unknown/test_service.py
import logging
import queue
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from capture.unknown import service
from capture.unknown.energy_trigger import TriggerEvent
from capture.unknown.service import CaptureSettings, process_snippet
from capture.unknown.snippet_assembler import CapturedSnippet
from schema.records import ClassificationStatus, Modality

FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)


def _settings(staging_dir: Path, **overrides) -> CaptureSettings:
    return CaptureSettings(
        **{
            "center_freq_hz": 915e6,
            "sample_rate": FS,
            "noise_floor_dbfs": -40.0,
            "staging_dir": staging_dir,
            **overrides,
        }
    )


def _snippet(trigger_index: int = 60_000, pre: int = 10_000, post: int = 90_000) -> CapturedSnippet:
    n = pre + post
    iq = np.full(n, 0.01, dtype=np.complex64)
    iq[pre : pre + 20_000] = 0.1  # -20 dBFS burst right at the trigger
    power = (np.abs(iq) ** 2).astype(np.float32)
    start = trigger_index - pre
    return CapturedSnippet(
        iq=iq,
        power=power,
        start_index=start,
        start_time=ANCHOR + timedelta(seconds=start / FS),
        trigger=TriggerEvent(
            sample_index=trigger_index,
            time=ANCHOR + timedelta(seconds=trigger_index / FS),
            center_freq_hz=915e6,
        ),
    )


def test_settings_reject_a_threshold_at_or_above_full_scale(tmp_path):
    with pytest.raises(ValueError, match="full scale"):
        _settings(tmp_path, noise_floor_dbfs=-5.0, threshold_db=10.0)


def test_settings_reject_windows_shorter_than_one_sample(tmp_path):
    with pytest.raises(ValueError, match="at least one sample"):
        _settings(tmp_path, averaging_seconds=1e-7)


def test_process_snippet_stages_sigmf_and_builds_record(tmp_path):
    record = process_snippet(_snippet(), _settings(tmp_path / "staging"), "s1", "op1")

    data_path = Path(record.metadata.iq_snippet_path)
    assert data_path.parent == (tmp_path / "staging").resolve()
    assert data_path.is_file() and data_path.with_suffix(".sigmf-meta").is_file()
    assert record.modality is Modality.UNKNOWN
    # Sample-derived, never wall clock: anchor + 60_000 / 100 kHz.
    assert record.timestamp == ANCHOR + timedelta(seconds=0.6)
    assert record.identifier.center_freq == 915e6
    assert record.identifier.bandwidth_estimate > 0
    assert record.signal.peak_power == pytest.approx(-20.0, abs=0.01)
    assert record.signal.rssi == pytest.approx(-20.0, abs=0.01)
    assert record.signal.snr == pytest.approx(20.0, abs=0.01)
    assert record.metadata.snippet_duration_ms == 1000
    assert record.metadata.sample_rate == FS
    assert record.metadata.classification_status is ClassificationStatus.UNCLASSIFIED


def test_snippet_duration_reflects_samples_actually_captured(tmp_path):
    """A trigger in the first 0.1 s truncates the pre-trigger history."""
    record = process_snippet(_snippet(trigger_index=3_000, pre=3_000), _settings(tmp_path), "s1", "op1")
    assert record.metadata.snippet_duration_ms == 930


class _FakeTap:
    def __init__(self) -> None:
        self.samples_seen = 0
        self.error: Exception | None = None


def test_drain_yields_snippets_then_raises_when_stream_stalls():
    snippets: queue.Queue = queue.Queue()
    snippets.put("snippet-1")
    tap = _FakeTap()
    times = iter([0.0, 1.0, 3.0, 9.0])
    drained = service._drain(snippets, tap, stall_seconds=5.0, poll_seconds=0.0, monotonic=lambda: next(times))

    assert next(drained) == "snippet-1"
    tap.samples_seen = 4096  # progress at t=1.0
    with pytest.raises(RuntimeError, match="stalled"):
        next(drained)  # t=3.0 still within 5 s of progress; t=9.0 is not


def test_drain_reraises_a_tap_failure():
    tap = _FakeTap()
    tap.error = ValueError("assembler bug")
    drained = service._drain(queue.Queue(), tap, stall_seconds=5.0, poll_seconds=0.0, monotonic=lambda: 0.0)
    with pytest.raises(RuntimeError, match="tap failed") as excinfo:
        next(drained)
    assert isinstance(excinfo.value.__cause__, ValueError)


class _FakeEmitter:
    def __init__(self, socket_path: str) -> None:
        self.records = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        pass

    def emit(self, record) -> None:
        self.records.append(record)


def test_run_survives_radio_failures_and_keeps_cooldown_across_rebuilds(tmp_path, monkeypatch):
    """Radio missing twice (backoff 1 s, 2 s), then a session that captures
    one snippet before its stream stalls (backoff resets to 1 s because the
    radio did open), then the 4th session must receive that snippet's
    trigger time so the rebuilt flowgraph doesn't immediately retrigger."""
    snippet = _snippet()
    sessions = []

    @contextmanager
    def fake_open_session(settings, last_trigger_at):
        sessions.append(dict(last_trigger_at))
        if len(sessions) <= 2:
            raise RuntimeError("SoapySDR::Device::make() no match")
        if len(sessions) == 4:
            raise KeyboardInterrupt  # ends the test; not an Exception, so not retried

        def stalled():
            yield snippet
            raise RuntimeError("SDR stream stalled")

        yield stalled()

    sleeps = []
    emitters = []
    monkeypatch.setattr(service, "_open_session", fake_open_session)
    monkeypatch.setattr(service.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        service, "RecordEmitter", lambda path: emitters.append(_FakeEmitter(path)) or emitters[-1]
    )

    with pytest.raises(KeyboardInterrupt):
        service.run(_settings(tmp_path), "/unused.sock", "s1", "op1")

    assert sleeps == [1.0, 2.0, 1.0]
    assert sessions[3] == {915e6: snippet.trigger.time}
    assert len(emitters[0].records) == 1


def test_a_failed_emit_drops_only_that_snippet(tmp_path, caplog):
    class BrokenEmitter:
        def emit(self, record) -> None:
            raise OSError("ingest socket gone")

    with caplog.at_level(logging.ERROR):
        service._stage_and_emit(_snippet(), _settings(tmp_path), BrokenEmitter(), "s1", "op1")
    assert "dropping it" in caplog.text


def test_cli_requires_noise_floor_and_builds_settings():
    with pytest.raises(SystemExit):
        service._parse_args(["--survey-id", "s", "--operator-id", "o", "--center-freq", "915e6"])

    settings, args = service._parse_args(
        [
            "--survey-id", "s",
            "--operator-id", "o",
            "--center-freq", "915e6",
            "--noise-floor-dbfs", "-60",
        ]
    )
    assert settings.center_freq_hz == 915e6
    assert settings.sample_rate == 20e6
    assert settings.threshold_dbfs == -50.0
    assert settings.staging_dir == Path("data/snippet-staging")
    assert args.socket_path == "/tmp/sdr-ingest.sock"


def test_cli_rejects_invalid_settings_with_usage_error():
    with pytest.raises(SystemExit):
        service._parse_args(
            [
                "--survey-id", "s",
                "--operator-id", "o",
                "--center-freq", "915e6",
                "--noise-floor-dbfs", "-5",
            ]
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/capture/unknown/test_service.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'service' from 'capture.unknown'`

- [ ] **Step 3: Implement**

```python
# capture/unknown/service.py
from __future__ import annotations

import argparse
import logging
import queue
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from capture.common.emitter import RecordEmitter
from capture.unknown.energy_trigger import TriggerConfig, record_trigger
from capture.unknown.normalizer import SnippetCaptureEvent, normalize_snippet_event
from capture.unknown.sample_clock import SampleClock
from capture.unknown.snippet_assembler import CapturedSnippet, SnippetAssembler
from capture.unknown.snippet_writer import write_sigmf_snippet
from dsp.spectral import mean_burst_power_dbfs, occupied_bandwidth_hz, peak_power_dbfs
from schema.records import UnifiedRecord

logger = logging.getLogger(__name__)

# On consecutive failures to open the radio the sleep doubles from 1 s up to
# this ceiling, matching capture/wifi and capture/bluetooth.
_INITIAL_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0
# No new samples for this long means the SDR stream is dead (unplugged,
# wedged driver, or its source returned WORK_DONE): rebuild the session.
_STALL_SECONDS = 5.0
_POLL_SECONDS = 0.5


@dataclass(frozen=True)
class CaptureSettings:
    center_freq_hz: float
    sample_rate: float
    noise_floor_dbfs: float  # operator-measured; no hardware-validated default exists
    staging_dir: Path
    threshold_db: float = 10.0  # trigger margin above the noise floor
    averaging_seconds: float = 0.001
    pre_trigger_seconds: float = 0.1
    post_trigger_seconds: float = 0.9  # snippet = pre + post = 1.0 s
    cooldown_seconds: float = 30.0
    gain_db: float = 30.0
    device: str = "driver=bladerf"
    device_args: str = ""

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.center_freq_hz <= 0:
            raise ValueError("sample_rate and center_freq_hz must be positive")
        if self.threshold_dbfs >= 0.0:
            raise ValueError(
                f"Trigger threshold {self.threshold_dbfs} dBFS is at or above full "
                "scale (0 dBFS) and could never fire; lower the noise floor or margin"
            )
        if self.pre_trigger_seconds < 0 or self.cooldown_seconds < 0:
            raise ValueError("pre_trigger_seconds and cooldown_seconds must be >= 0")
        if self.samples(self.averaging_seconds) < 1 or self.samples(self.post_trigger_seconds) < 1:
            raise ValueError(
                "averaging_seconds and post_trigger_seconds must each span at least one sample"
            )

    @property
    def threshold_dbfs(self) -> float:
        return self.noise_floor_dbfs + self.threshold_db

    def samples(self, seconds: float) -> int:
        return round(seconds * self.sample_rate)


def process_snippet(
    snippet: CapturedSnippet,
    settings: CaptureSettings,
    survey_id: str,
    operator_id: str,
) -> UnifiedRecord:
    """Stage `snippet` as SigMF and build its UnifiedRecord. All times are
    sample-derived (see SampleClock); the duration is what was actually
    captured, which is shorter than pre + post when the trigger came within
    pre_trigger_seconds of the stream starting."""
    data_path = write_sigmf_snippet(
        snippet.iq,
        settings.staging_dir,
        settings.sample_rate,
        settings.center_freq_hz,
        snippet.start_time,
    )
    event = SnippetCaptureEvent(
        timestamp=snippet.trigger.time,
        center_freq_hz=settings.center_freq_hz,
        sample_rate=settings.sample_rate,
        bandwidth_estimate_hz=occupied_bandwidth_hz(
            snippet.iq, settings.sample_rate, settings.threshold_dbfs
        ),
        peak_power_dbfs=peak_power_dbfs(snippet.power),
        mean_power_dbfs=mean_burst_power_dbfs(snippet.power, settings.threshold_dbfs),
        noise_floor_dbfs=settings.noise_floor_dbfs,
        snippet_path=str(data_path),
        snippet_duration_ms=round(len(snippet.iq) * 1000 / settings.sample_rate),
    )
    return normalize_snippet_event(event, survey_id, operator_id)


def run(
    settings: CaptureSettings, socket_path: str, survey_id: str, operator_id: str
) -> None:
    """Capture forever: one radio session at a time, rebuilt on failure.

    A failing session (no SDR attached, driver error, stalled stream, tap
    failure) is logged and retried rather than killing the capture process:
    field surveys must survive transient faults unattended. Cooldown state
    is kept here, across sessions, as absolute trigger times.
    """
    backoff = _INITIAL_BACKOFF_SECONDS
    last_trigger_at: Mapping[float, datetime] = {}
    with RecordEmitter(socket_path) as emitter:
        while True:
            try:
                with _open_session(settings, last_trigger_at) as snippets:
                    backoff = _INITIAL_BACKOFF_SECONDS  # the radio opened
                    for snippet in snippets:
                        last_trigger_at = record_trigger(last_trigger_at, snippet.trigger)
                        _stage_and_emit(snippet, settings, emitter, survey_id, operator_id)
            except Exception:
                logger.exception(
                    "Unknown-signal capture session failed; retrying in %.1fs", backoff
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)


def _stage_and_emit(
    snippet: CapturedSnippet,
    settings: CaptureSettings,
    emitter: RecordEmitter,
    survey_id: str,
    operator_id: str,
) -> None:
    """A full disk or a dead ingest socket loses this one snippet, not the
    radio session. (If staging succeeded but emit failed, the staged pair is
    left behind in the staging directory.)"""
    try:
        emitter.emit(process_snippet(snippet, settings, survey_id, operator_id))
    except Exception:
        logger.exception(
            "Failed to stage/emit snippet triggered at sample %d; dropping it",
            snippet.trigger.sample_index,
        )


@contextmanager
def _open_session(
    settings: CaptureSettings, last_trigger_at: Mapping[float, datetime]
) -> Iterator[Iterator[CapturedSnippet]]:
    """Open the SDR, start the flowgraph, and yield an iterator of completed
    snippets; always stops the flowgraph on exit. Needs GNU Radio and a real
    SDR, so tests replace it (see tests/capture/unknown/test_service.py)."""
    # Deferred: GNU Radio is a system (dnf) package, not a pip dependency,
    # and the rest of this module must import without it.
    from capture.unknown.flowgraph import build_flowgraph

    # Open the device first: opening and tuning a bladeRF can take seconds, and
    # the anchor must be read as close as possible to sample 0, i.e. right
    # before start(), or every timestamp would be early by the open time.
    source = _build_soapy_source(settings)
    clock = SampleClock(anchor=datetime.now(timezone.utc), sample_rate=settings.sample_rate)
    assembler = SnippetAssembler(
        clock=clock,
        center_freq_hz=settings.center_freq_hz,
        config=TriggerConfig(
            threshold_dbfs=settings.threshold_dbfs,
            cooldown=timedelta(seconds=settings.cooldown_seconds),
        ),
        pre_trigger_samples=settings.samples(settings.pre_trigger_seconds),
        post_trigger_samples=settings.samples(settings.post_trigger_seconds),
        last_trigger_at=last_trigger_at,
    )
    # Unbounded: the cooldown caps snippets at one per cooldown window, and a
    # bounded put() would block the scheduler thread and overflow the SDR.
    snippets: queue.Queue[CapturedSnippet] = queue.Queue()
    flowgraph = build_flowgraph(
        source,
        settings.samples(settings.averaging_seconds),
        assembler,
        snippets.put,
    )
    flowgraph.top_block.start()
    try:
        yield _drain(snippets, flowgraph.tap)
    finally:
        flowgraph.top_block.stop()
        flowgraph.top_block.wait()


def _drain(
    snippets: queue.Queue,
    tap,
    stall_seconds: float = _STALL_SECONDS,
    poll_seconds: float = _POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
) -> Iterator[CapturedSnippet]:
    """Yield snippets as the flowgraph completes them; raise once it dies.

    `tap` is the flowgraph's SnippetTap (anything with samples_seen and
    error). A GNU Radio flowgraph fails silently from Python's point of
    view, so health is polled: a tap error is re-raised, and samples_seen
    frozen for stall_seconds means the SDR stopped streaming. The monotonic
    clock here is health-checking only; no record timing depends on it.
    """
    last_seen = tap.samples_seen
    last_progress = monotonic()
    while True:
        try:
            snippet = snippets.get(timeout=poll_seconds)
        except queue.Empty:
            pass
        else:
            yield snippet
        if tap.error is not None:
            raise RuntimeError("Snippet tap failed; flowgraph stopped") from tap.error
        now = monotonic()
        if tap.samples_seen != last_seen:
            last_seen, last_progress = tap.samples_seen, now
        elif now - last_progress > stall_seconds:
            raise RuntimeError(f"SDR stream stalled: no samples for {stall_seconds:.0f}s")


def _build_soapy_source(settings: CaptureSettings):
    """gr-soapy source for the configured device. Never called by tests (no
    SDR hardware); API verified by introspection against GNU Radio 3.10.12 and
    its bundled soapy_bladerf_source.block.yml template. Without a device or
    driver module, soapy.source raises RuntimeError('SoapySDR::Device::make()
    no match'), which run() logs and retries with backoff."""
    from gnuradio import soapy

    source = soapy.source(settings.device, "fc32", 1, settings.device_args, "", [""], [""])
    source.set_sample_rate(0, settings.sample_rate)
    source.set_frequency(0, settings.center_freq_hz)
    # Manual gain: a dBFS trigger threshold is only meaningful at fixed gain.
    source.set_gain_mode(0, False)
    source.set_gain(0, settings.gain_db)
    return source


def _parse_args(argv: list[str] | None = None) -> tuple[CaptureSettings, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Unknown-signal energy-triggered IQ capture -> ingest queue"
    )
    parser.add_argument("--socket-path", default="/tmp/sdr-ingest.sock")
    parser.add_argument("--survey-id", required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--center-freq", type=float, required=True, help="Hz")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=20e6,
        help="Samples/s (= instantaneous bandwidth). The bladeRF xA9 reaches "
        "~56e6, but the Python capture chain sustained only ~58e6 on a fast "
        "x86 desktop; profile the Jetson before raising this.",
    )
    parser.add_argument(
        "--noise-floor-dbfs",
        type=float,
        required=True,
        help="Measured noise floor in dBFS at this gain and frequency. Required: "
        "there is no hardware-validated default, and a wrong guess either "
        "triggers on noise every cooldown or never triggers at all.",
    )
    parser.add_argument("--threshold-db", type=float, default=10.0)
    parser.add_argument("--averaging-ms", type=float, default=1.0)
    parser.add_argument("--pre-trigger-s", type=float, default=0.1)
    parser.add_argument("--post-trigger-s", type=float, default=0.9)
    parser.add_argument("--cooldown-s", type=float, default=30.0)
    parser.add_argument("--gain-db", type=float, default=30.0)
    parser.add_argument("--device", default="driver=bladerf")
    parser.add_argument("--device-args", default="")
    parser.add_argument(
        "--staging-dir",
        default="data/snippet-staging",
        help="Must match ingest's --snippet-staging-dir.",
    )
    args = parser.parse_args(argv)
    try:
        settings = CaptureSettings(
            center_freq_hz=args.center_freq,
            sample_rate=args.sample_rate,
            noise_floor_dbfs=args.noise_floor_dbfs,
            staging_dir=Path(args.staging_dir),
            threshold_db=args.threshold_db,
            averaging_seconds=args.averaging_ms / 1000.0,
            pre_trigger_seconds=args.pre_trigger_s,
            post_trigger_seconds=args.post_trigger_s,
            cooldown_seconds=args.cooldown_s,
            gain_db=args.gain_db,
            device=args.device,
            device_args=args.device_args,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return settings, args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (`sdr-capture-unknown`)."""
    settings, args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run(
        settings=settings,
        socket_path=args.socket_path,
        survey_id=args.survey_id,
        operator_id=args.operator_id,
    )
```

In `pyproject.toml`, add to `[project.scripts]` after `sdr-capture-bluetooth`:

```toml
sdr-capture-unknown = "capture.unknown.service:main"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/capture/unknown/test_service.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Verify the service module imports without GNU Radio, and the CLI parses**

Run:
```bash
python -c "import sys; sys.modules['gnuradio'] = None; import capture.unknown.service; print('ok')"
python -c "from capture.unknown.service import main; main(['--help'])"
```
Expected: `ok`, then the usage text listing `--center-freq`, `--noise-floor-dbfs`,
`--sample-rate`, `--staging-dir` and the other options.

- [ ] **Step 6: Replace `capture/unknown/README.md`**

````markdown
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
  If the SDR overflows (drops samples), sample time falls behind wall time by the
  dropped duration.
- `identifier.bandwidth_estimate` is the 99%-power occupied bandwidth of the burst
  (resolution sample_rate / 1024). It is within +-5% at >= 15 dB SNR and up to +24% at
  the 10 dB trigger margin. Bursts shorter than 1024 samples are overestimated.

## Snippet handoff (capture never writes to storage)

Capture writes `<stamp>_<freq>Hz_<id>.sigmf-data` + `.sigmf-meta` into `--staging-dir`
and emits the record with `iq_snippet_path` set to the staged absolute `.sigmf-data` path.
Ingest's `LocalSnippetStore` validates the path (it must be a `.sigmf-data` file directly
inside ingest's `--snippet-staging-dir`, symlinks resolved), moves the pair into
`--snippet-store-dir`, and persists the record with the final path. Run both services
from the same working directory, or pass the same absolute staging path to both. Keep
staging and store on one filesystem so the move is an atomic rename.

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

`--noise-floor-dbfs` is required: measure it at your gain and frequency first.
`--sample-rate` defaults to 20e6. The bladeRF reaches ~56e6, but the Python chain
sustained only ~58 MS/s on a fast x86 desktop and has not been profiled on the Jetson.
A 1 s snippet is 160 MB on disk at 20 MS/s (448 MB at 56 MS/s). In memory it is
240 MB (672 MB), because a float32 power array rides along with the cf32 samples.

## Licenses

`sigmf` (sigmf-python) is LGPL-3.0-or-later, used as an unmodified library dependency.
GNU Radio is GPL-3.0-or-later; this module imports it at runtime as a system package.
````

- [ ] **Step 7: Commit**

```bash
git add capture/unknown/service.py tests/capture/unknown/test_service.py capture/unknown/README.md pyproject.toml
git commit -m "feat: add unknown-signal capture service with fault-tolerant radio sessions"
```

---

### Task 11: End-to-end integration test (the closure proof)

**Files:**
- Test: `tests/capture/unknown/test_flowgraph_integration.py`

**Interfaces:**
- Consumes everything: `build_flowgraph` (Task 9), `SnippetAssembler` (Task 3),
  `process_snippet`/`CaptureSettings` (Task 10), `LocalSnippetStore` (Task 7),
  `IngestService(..., snippet_store=)` (Task 8), and the existing `RecordEmitter`,
  `QueueServer` and SQLite storage.
- Produces: nothing further consumes this. It is the spec's
  `test_flowgraph_integration.py`. Its burst at 2.0 s falls inside the cooldown, so the test
  asserts that this burst does NOT produce a record.

All components already exist, so this test passes on its first run. Step 3 proves it can
fail.

- [ ] **Step 1: Write the test**

```python
# tests/capture/unknown/test_flowgraph_integration.py
"""End-to-end closure proof for unknown-signal capture, no SDR hardware:

synthetic IQ file -> GNU Radio file_source -> |x|^2 -> moving average ->
SnippetTap/SnippetAssembler -> SigMF staging -> UnifiedRecord ->
RecordEmitter -> QueueServer -> IngestService (adopts the snippet into the
LocalSnippetStore) -> SQLite.

Bursts at 0.5 s, 2.0 s and 4.0 s with a 3 s cooldown: the 2.0 s burst falls
inside the first trigger's cooldown and must NOT produce a record. Timing
is sample-derived, so the whole 5.5 s of signal runs in well under a second.
"""
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gnuradio")

from gnuradio import blocks, gr  # noqa: E402
from sigmf import sigmffile  # noqa: E402

from capture.common.emitter import RecordEmitter  # noqa: E402
from capture.unknown.energy_trigger import TriggerConfig  # noqa: E402
from capture.unknown.flowgraph import CaptureFlowgraph, build_flowgraph  # noqa: E402
from capture.unknown.sample_clock import SampleClock  # noqa: E402
from capture.unknown.service import CaptureSettings, process_snippet  # noqa: E402
from capture.unknown.snippet_assembler import SnippetAssembler  # noqa: E402
from ingest.gps_fix import GpsFix, StaticGpsFixProvider  # noqa: E402
from ingest.queue_server import QueueServer  # noqa: E402
from ingest.service import IngestService  # noqa: E402
from storage.db import init_db, make_engine, make_session_factory  # noqa: E402
from storage.models import SurveyRecord  # noqa: E402
from storage.snippet_store import LocalSnippetStore  # noqa: E402

FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
NOISE_POWER = 1e-4  # -40 dBFS
BURST_POWER = 1e-2  # -20 dBFS: 20 dB SNR
BURST_BANDWIDTH_HZ = 20_000.0
BURST_SECONDS = 0.2
BURST_STARTS_S = [0.5, 2.0, 4.0]
TOTAL_SECONDS = 5.5
FIX = GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)


def _synthetic_iq() -> np.ndarray:
    rng = np.random.default_rng(11)
    n = round(TOTAL_SECONDS * FS)
    scale = math.sqrt(NOISE_POWER / 2)
    iq = scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    burst_len = round(BURST_SECONDS * FS)
    freqs = np.fft.fftfreq(burst_len, 1 / FS)
    for start_s in BURST_STARTS_S:
        spectrum = np.fft.fft(rng.standard_normal(burst_len) + 1j * rng.standard_normal(burst_len))
        spectrum[np.abs(freqs - 10_000.0) > BURST_BANDWIDTH_HZ / 2] = 0
        burst = np.fft.ifft(spectrum)
        burst *= math.sqrt(BURST_POWER / np.mean(np.abs(burst) ** 2))
        start = round(start_s * FS)
        iq[start : start + burst_len] += burst
    return iq.astype(np.complex64)


def _run_bounded(flowgraph: CaptureFlowgraph, timeout: float = 30.0) -> None:
    waiter = threading.Thread(target=flowgraph.top_block.wait, daemon=True)
    flowgraph.top_block.start()
    waiter.start()
    waiter.join(timeout)
    if waiter.is_alive():
        flowgraph.top_block.stop()
        waiter.join(5)
        pytest.fail(f"flowgraph did not finish within {timeout}s")


def test_synthetic_bursts_become_stored_unknown_records(tmp_path):
    staging, store_root = tmp_path / "staging", tmp_path / "snippets"
    settings = CaptureSettings(
        center_freq_hz=915e6,
        sample_rate=FS,
        noise_floor_dbfs=-40.0,
        staging_dir=staging,
        threshold_db=10.0,
        averaging_seconds=0.001,
        pre_trigger_seconds=0.1,
        post_trigger_seconds=0.9,
        cooldown_seconds=3.0,
    )
    iq_file = tmp_path / "synthetic.cf32"
    _synthetic_iq().tofile(iq_file)

    # 1. Real GNU Radio scheduler over the synthetic file.
    assembler = SnippetAssembler(
        clock=SampleClock(anchor=ANCHOR, sample_rate=FS),
        center_freq_hz=settings.center_freq_hz,
        config=TriggerConfig(
            threshold_dbfs=settings.threshold_dbfs,
            cooldown=timedelta(seconds=settings.cooldown_seconds),
        ),
        pre_trigger_samples=settings.samples(settings.pre_trigger_seconds),
        post_trigger_samples=settings.samples(settings.post_trigger_seconds),
        last_trigger_at={},
    )
    snippets = []
    source = blocks.file_source(gr.sizeof_gr_complex, str(iq_file), False)
    flowgraph = build_flowgraph(source, settings.samples(settings.averaging_seconds), assembler, snippets.append)
    _run_bounded(flowgraph)
    assert flowgraph.tap.error is None
    assert [round(s.trigger.sample_index / FS, 1) for s in snippets] == [0.5, 4.0]

    # 2. Capture -> queue -> ingest -> storage.
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)
    service = IngestService(
        server,
        session_factory,
        StaticGpsFixProvider(FIX),
        snippet_store=LocalSnippetStore(staging, store_root),
    )
    try:
        with RecordEmitter(socket_path) as emitter:
            for snippet in snippets:
                emitter.emit(process_snippet(snippet, settings, "survey-1", "op-1"))
        processed = [service.process_one(timeout=2) for _ in snippets]
    finally:
        server.stop()

    # Capture never left anything behind in staging; ingest took ownership.
    assert list(staging.iterdir()) == []
    with session_factory() as session:
        rows = session.query(SurveyRecord).order_by(SurveyRecord.id).all()
    engine.dispose()  # close the pooled in-memory connection; no ResourceWarning at GC
    assert len(rows) == 2
    for row, record, start_s in zip(rows, processed, [0.5, 4.0]):
        assert row.modality == "unknown"
        # Sample-derived: within one 1 ms averaging window of the burst onset.
        offset = (record.timestamp - ANCHOR).total_seconds() - start_s
        assert 0.0 <= offset < 0.001
        assert row.identifier["center_freq"] == 915e6
        assert row.identifier["bandwidth_estimate"] == pytest.approx(BURST_BANDWIDTH_HZ, rel=0.15)
        assert row.signal["rssi"] == pytest.approx(-20.0, abs=1.0)
        assert row.signal["snr"] == pytest.approx(20.0, abs=1.0)
        assert -20.0 < row.signal["peak_power"] < -15.0
        assert row.metadata_["snippet_duration_ms"] == 1000
        assert row.metadata_["sample_rate"] == FS
        assert row.metadata_["classification_status"] == "unclassified"
        assert row.lat == FIX.lat and row.gps_fix_quality == FIX.fix_quality

        stored = Path(row.metadata_["iq_snippet_path"])
        assert stored.parent == store_root.resolve()
        recording = sigmffile.fromfile(str(stored))
        samples = recording.read_samples()
        assert len(samples) == round(1.0 * FS)
        assert recording.get_captures()[0]["core:frequency"] == 915e6
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/capture/unknown/test_flowgraph_integration.py -v`
Expected: PASS (1 test) in about 1 s. Verified values: trigger timestamps 0.50004 s and
4.00006 s after the anchor, bandwidth 19,922 Hz against a true 20,000 Hz, rssi −19.97 dBFS,
snr 20.03 dB, peak −16.7 dBFS.

- [ ] **Step 3: Prove the cooldown assertion has teeth, then revert**

Temporarily change `cooldown_seconds=3.0` to `cooldown_seconds=0.0` in the test and re-run.
Expected: FAIL with `assert [0.5, 2.0, 4.0] == [0.5, 4.0]`. Revert the change and re-run:
PASS.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: `124 passed, 1 skipped`. The skip is the unrelated cellular `CellSearch` binary
test, and the 33 pre-existing warnings are unchanged. The new modules also pass under
`-W error`: `python -m pytest tests/capture/unknown tests/dsp tests/storage/test_snippet_store.py -q -W error`
gives `59 passed` with exit code 0. The integration test's `engine.dispose()` exists for this
check: without it, an unclosed in-memory SQLite connection sometimes surfaces at GC as
`PytestUnraisableExceptionWarning`.

Without GNU Radio installed, `test_flowgraph.py` and `test_flowgraph_integration.py` SKIP,
and everything else passes. Verified by running with `sys.modules['gnuradio'] = None`:
46 passed, 2 skipped in `tests/capture/unknown tests/dsp`.

- [ ] **Step 5: Commit**

```bash
git add tests/capture/unknown/test_flowgraph_integration.py
git commit -m "test: add end-to-end unknown-signal capture -> ingest integration test"
```

---

## What this plan deliberately does not cover

- **Frequency sweeping or scanning.** The plan uses one fixed center frequency. Cooldowns are
  already keyed by frequency, so sweeping can be added later without a rewrite.
- **FFT per-bin detection and localization within the capture.** Part 4 can localize from
  the full snippet.
- **Part 4 classification.** These records are its input. `classification_status` is set to
  `unclassified` for it.
- **Real bladeRF xA9 validation.** No hardware exists here. Unverified on real hardware:
  - the SoapyBladeRF from-source build;
  - `set_gain_mode(0, False)` support;
  - gr-soapy's behavior on unplug or overflow (the stall watchdog is the hedge);
  - real noise-floor values;
  - sustained throughput on the Jetson.
- **Sample-time drift on SDR overflow.** Dropped samples make sample-derived timestamps lag
  wall time. Re-anchoring from hardware time or `rx_time` tags is deferred.
- **Adaptive noise-floor estimation.** The floor is an operator-measured CLI value.
- **Staging garbage collection.** Pairs orphaned in staging (ingest down, emit failed,
  record rejected) or in the store (`save_record` failed after adoption) are left for manual
  cleanup.
- **The S3-compatible snippet store.** `SnippetStore` is the seam; only the local
  implementation ships.
- **FPGA or GPU offload of the energy detector.** Deferred project-wide until Jetson
  profiling (design doc §10 step 6). The throughput numbers under Verified facts are the
  starting point.
