# LTE MIB Decode Spike Design

Scope: a follow-up to the cellular DSP spike (`docs/superpowers/specs/2026-08-11-cellular-dsp-spike-design.md`),
which deferred MIB/SIB1-3 decode because it depends on `LTE-Tracker` — a different
tool than `CellSearch`, architected for continuous live tracking rather than clean
single-shot offline decode. This spec investigates and scopes what's actually
achievable there, hardware-free.

## Investigation findings (this changes the original goal)

A source-level investigation of the vendored `LTE-Tracker` (same submodule as the
CellSearch spike, `capture/cellular/vendor/lte-cell-scanner`) found:

1. **SIB1-3 decode does not exist anywhere in this codebase.** Grepping the full
   `src/`/`include/` tree found zero SIB-related decode logic. Only MIB decode exists
   (`do_mib_decode()` in `src/tracker_thread.cpp`, ~line 555-769). "MIB/SIB1-3 decode"
   as originally scoped is not achievable with this vendored tool — SIB would require
   either a different tool or writing SIB decode from scratch, both out of scope here.
2. **`LTE-Tracker`'s `--loadbin` mode is a debug hack, not a real batch mode.** It
   loops the input file (by default), busy-feeds it into a producer/tracker/searcher/
   display thread pipeline with no defined "done" state, then after the file drains
   once: `sleep(10s); exit(-1)` (`src/LTE-Tracker.cpp:1634-1636`, via `ABORT`). This
   needs a real patch to behave as a clean single-shot tool.
3. **Live output is ncurses-only.** `display_thread.cpp` uses `initscr()`/`printw()`
   for all live tracking results — no quiet/parseable stdout path exists for decoded
   MIB fields. This also needs a patch.
4. **MIB decode needs 160ms of IQ** (`mib_fifo.size()==16`, one MIB symbol per
   subframe, 16 subframes = 160ms) — 2x longer than the 80ms fixture from the
   CellSearch spike. That fixture's *source* recording is also only 80ms, so a longer
   recording is needed from elsewhere, not just a smaller decimation factor.

**Decision from this investigation**: proceed with a narrowed MIB-only spike (no
SIB), accepting that this requires — for the first time in this project's cellular
work — actually patching the vendored AGPL source (not just CMake config flags), and
sourcing IQ test data from a different upstream repo.

## Fixture source

`JiaoXianjun/LTE-Cell-Scanner-big-file` (AGPLv3, same author/project family) has
1-second real captures — `regression_test_signal_file/f2585_s19.2_bw20_1s_hackrf.bin`
(19.2 Msps, 20 MHz BW), same raw-int8-interleaved-IQ format as the CellSearch spike's
fixture, produced by the identical capture toolchain. 1 second is 12.5x the 80ms
CellSearch fixture and comfortably over the 160ms MIB floor. This repo also has
`.per` files showing SIB1/2/3/5 were successfully decoded against this exact capture
by some other (non-vendored) tool upstream — useful as external corroboration that
this is a real, decodable LTE signal, though it's reference-only since SIB decode
isn't something this project reproduces.

**Not vendored as a submodule** (the full repo is much larger than needed, mostly
unused captures). Instead: `capture/cellular/testdata/generate_mib_fixture.py`
downloads just the one needed file from a pinned commit SHA (checksum-verified),
decimates it 10x (same approach as the CellSearch fixture: windowed-sinc FIR +
downsample to 1.92 Msps), trims it to a few hundred ms (comfortable margin over 160ms,
keeps the committed `.bin` small — exact trim length decided during implementation
once real MIB-decode timing is known), and writes the same `--loadbin`-compatible
128-byte header used before.

## Vendored source patch

A small, tracked patch (documented as a diff against upstream, committed alongside
the submodule pin, not a silent fork) to two things in `src/tracker_thread.cpp`
and/or `src/LTE-Tracker.cpp`:

1. After `do_mib_decode()` successfully unpacks a MIB, print a plain stdout line with
   the decoded fields (cell ID, bandwidth in RB, PHICH duration, PHICH resource,
   SFN) — additive, doesn't require removing the curses UI.
2. Change the file-exhaustion path from `sleep(10s); exit(-1)` (a hardcoded, failure-
   coded debug hack) to `exit(0)` once at least one MIB has been successfully
   decoded and printed — a real single-shot completion signal a subprocess wrapper
   can rely on.

This is a bigger decision than anything touched in the CellSearch spike (which used
CMake config flags only, zero source changes) — modifying vendored AGPL source is a
real, ongoing maintenance surface (a patch to carry forward if the submodule pin ever
updates), acknowledged and accepted for this spike.

## Architecture

```
capture/cellular/
  vendor/lte-cell-scanner/            # existing submodule, same as CellSearch spike
    <patch applied to tracker_thread.cpp / LTE-Tracker.cpp, tracked in this repo>
  offline_mib_scanner.py              # subprocess wrapper + stdout parser (mirrors offline_scanner.py)
  testdata/
    generate_mib_fixture.py           # downloads + decimates + trims + headers the big-file capture
    mib_<cellid>.bin                  # committed fixture (exact name/cell ID known after implementation)
    mib_<cellid>.expected.json        # ground truth (known after implementation's real build-and-run)
tests/capture/cellular/
  test_offline_mib_scanner.py         # unit tests, fixture text -> parsed dict
  test_offline_mib_decode.py          # integration: run patched LTE-Tracker against testdata, assert decode
```

`offline_mib_scanner.py` mirrors `offline_scanner.py`'s shape: a regex-based stdout
parser (pure, unit-testable) plus a thin subprocess-invoking function. Runtime and
exact output format are unknown until the implementation's build-and-run step —
unlike the CellSearch spike, this session's investigation was source-reading only
(no build), so the implementation plan needs a real build-and-run spike (same pattern
used for the CellSearch spike: verify facts by actually building and running before
writing exact code into the plan) to establish ground truth and timing before the
plan can be written without placeholders.

## Explicitly out of scope for this plan

- **No `UnifiedRecord`/`normalizer.py` wiring.** This spike proves the offline decode
  chain works and stops there, matching the "DSP spike only" precedent from the
  CellSearch spike. Wiring MIB fields into the schema is future work.
- **No SIB1-3 decode.** Confirmed not implemented in the vendored codebase; would
  need a different tool or from-scratch implementation, out of scope here.
- **PLMN stays permanently unreachable** through this path — it comes from SIB1,
  which doesn't exist in this codebase. Even after this spike succeeds, `PLMN` is
  still `None`; only bandwidth/PHICH config/SFN become newly available.
- **No curses UI removal.** The patch is additive (new plain-text print), not a
  removal of the existing ncurses display — minimizes the diff against upstream.
- **No real xA9 hardware validation** — same constraint as the CellSearch spike,
  no physical SDR available in this environment.

## Amendment (2026-10-03): MIB fields come from CellSearch; LTE-Tracker route rejected

A build-and-run spike (Fedora 43, LTE-Cell-Scanner submodule @ e7f71cbd) changed this
design's conclusions. The implementation follows
`docs/superpowers/plans/2026-10-03-lte-mib-fields-cellsearch.md`.

1. **CellSearch already decodes the MIB.**
   - `src/CellSearch.cpp:1332` calls `decode_mib()` (`src/searcher.cpp:3840`), which
     accepts a decode only when the CRC matches (`searcher.cpp:3940-3952`).
   - Cells whose MIB fails are dropped (`CellSearch.cpp:1334-1338`).
   - Its final summary table (`CellSearch.cpp:1454-1493`) prints antenna ports, CP
     type, n_RB, PHICH duration and PHICH resource for every reported cell. The
     committed 80 ms fixture already shows `FDD 301 2 1815.3M ... N 100 N one`.
   - `decode_mib()` also computes the SFN (`searcher.cpp:3998-3999`), but CellSearch
     never prints it.
2. **MIB decode needs 40 ms of IQ, not 160 ms.** LTE-Tracker's `do_mib_decode()` buffers
   only slot 1, symbols 0–3 of each frame (`tracker_thread.cpp:570`). So
   `mib_fifo.size()==16` (`tracker_thread.cpp:581`) is 4 frames, one 40 ms PBCH TTI.
   CellSearch decodes the MIB from the 80 ms fixture.
3. **`f2585_s19.2_bw20_1s_hackrf.bin` is a TDD cell, and LTE-Tracker is FDD-only.**
   - The file is from the big-file repo @ 0791cb339a8e88fc531494f2e447ff03bb48ff04,
     SHA-256 ea453bba4fe4edb6c5fa458b3067e4eeb4ef77defbf6acfa3a426371b0850fba. It is a
     plain git blob, not LFS.
   - CellSearch decodes it as TDD cell 216 (2 ports, 100 RB, PHICH normal/one).
   - `tracker_thread.cpp` has no duplex or TDD handling at all. In a 60 s run, a
     patched tracker made about 25,000 MIB attempts on that cell and every one failed
     CRC.
   - Weak FDD captures also never decoded in the tracker: the two −27 dB cells in
     `f1860_s19.2_bw20_1s_hackrf_home.bin`. CellSearch decodes both.
4. **The LTE-Tracker patch route was built, verified, and rejected.**
   - A 25-line additive patch did three things: print a plain-text MIB line, skip
     curses when stdout is not a TTY, and call `_exit(0)` at the end of a `--loadbin`
     pass. `--loadbin` always loops, because `repeat=true` at `LTE-Tracker.cpp:220`.
   - It decoded cell 301's MIB in about 6 s and 650 MB per run.
   - It was rejected for a zero-patch route. CellSearch's table already carries every
     MIB field except SFN, and SFN has no coverage-mapping value. That is not worth
     carrying an AGPL source patch, a slow and heavy binary, and a fixture that only
     decodes because the tracker loops it.
   - That plan is preserved in git history at commit fc13404
     (`docs/superpowers/plans/2026-10-03-lte-mib-decode-spike.md`).
5. **SFN and PLMN remain unavailable.** CellSearch computes the SFN but doesn't print
   it. PLMN needs SIB1, which nothing in the vendored codebase decodes.

This supersedes the "Fixture source", "Vendored source patch" and "Architecture"
sections above. There is no source patch, no `generate_mib_fixture.py`, no
`offline_mib_scanner.py` and no new fixture. `capture/cellular/offline_scanner.py`
parses the MIB fields from CellSearch's summary table, verified against the existing
`cell301_1815.3mhz.bin`.
