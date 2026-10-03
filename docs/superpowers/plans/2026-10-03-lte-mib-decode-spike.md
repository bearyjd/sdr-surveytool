# LTE MIB Decode Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove, with no SDR hardware, that the vendored `LTE-Tracker` decodes an LTE
MIB (DL bandwidth in RB, PHICH duration, PHICH resource, SFN) from a recorded IQ file,
called from Python as a single-shot subprocess.

**Architecture:** Add one small, additive patch file to this repo
(`capture/cellular/patches/lte-tracker-mib-stdout.patch`). `build.sh` applies it to a
throwaway `git archive` export of the pinned submodule commit and builds `LTE-Tracker`
there, so the submodule is never modified. The patch adds three behaviors, all of them
only when stdout is not a terminal: skip curses, print one plain line per decoded MIB,
and exit 0 at the end of a file pass. A new `offline_mib_scanner.py` mirrors
`offline_scanner.py`: a pure regex parser plus a thin subprocess wrapper. The closure
proof reuses the existing committed 80 ms fixture `cell301_1815.3mhz.bin`; no new IQ
data is added.

**Tech Stack:** Python 3.10+ (subprocess, regex), vendored C++ (LTE-Cell-Scanner @
`e7f71cbd`, CMake, IT++, Boost, FFTW, ncurses), `git archive` + `git apply`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md`. This plan
departs from the spec where the build-and-run spike proved it wrong. See "Spike findings"
below; Task 3 appends them to the spec as an amendment.

## Spike findings (verified 2026-10-03 by building and running)

Environment: Fedora 43, GCC 15.3.1, Python 3.14.7, submodule pinned at
`e7f71cbd4fa9f5ee5b97a58b2fb236ac13e4b8b1`. The code, patch text, output and timings in
this plan come from that run, not from reading the source.

1. **Toolchain.** The unchanged `capture/cellular/build.sh` builds `CellSearch`, and
   `tests/capture/cellular/test_offline_decode.py` passes. `LTE-Tracker` is already a
   CMake target in `vendor/lte-cell-scanner/src/CMakeLists.txt`. It builds with the
   same hardware-free flags (`-DUSE_OPENCL=0 -DUSE_BLADERF=0 -DUSE_HACKRF=0
   -DCMAKE_BUILD_TYPE=Release`) and `make LTE-Tracker` in about 15 s. The only
   warning is a pre-existing `-Wformat` at `display_thread.cpp:165`.
2. **MIB decode needs 40 ms of IQ, not 160 ms.** `do_mib_decode()` buffers only slot 1,
   symbols 0–3, which is 4 PBCH symbols per 10 ms frame. So `mib_fifo.size()==16` is 4
   frames, one 40 ms PBCH TTI.
3. **The spec's fixture is TDD, and LTE-Tracker's tracker is FDD-only.** The fixture is
   `JiaoXianjun/LTE-Cell-Scanner-big-file` @ `0791cb339a8e88fc531494f2e447ff03bb48ff04`,
   `regression_test_signal_file/f2585_s19.2_bw20_1s_hackrf.bin`: 38,400,000 bytes,
   SHA-256 `ea453bba4fe4edb6c5fa458b3067e4eeb4ef77defbf6acfa3a426371b0850fba`. It is not
   Git LFS: the API blob SHA `c0a86e53f15063ae03fb1e2b1fd4c318b0a5d3b3` equals
   `git hash-object` of the downloaded 38.4 MB file.
   - The raw samples are signed int8 (HackRF native).
   - After 10x decimation, CellSearch finds **TDD** cell 216 (2 ports, 100 RB, PHICH
     normal/one).
   - `tracker_thread.cpp` has no TDD handling. In a 60 s run the tracker made about
     25,000 MIB attempts and every one failed CRC; the curses display showed SSS SNR
     `-Inf`.
4. **Weak FDD cells also fail in the tracker.** Two captures were tried:
   `f1860_s19.2_bw20_1s_hackrf_home.bin` (SHA-256 `71507cfc…13399bf`) at 160 ms, 320 ms
   and 1 s trims, and `f1860_s1.92_g0_1s_strong_rtlsdr.bin` at 320 ms. Both are FDD
   cells 142 and 86 at about −27 dB, or −43 dB on the RTL-SDR capture. CellSearch
   decodes both cells' MIBs, but in 60 s the tracker never passed CRC. The root cause
   was not investigated.
5. **The existing fixture works.** `capture/cellular/testdata/cell301_1815.3mhz.bin`
   (80 ms, 1.92 Msps, one strong FDD cell at −9.45 dB) decodes a MIB in the patched
   tracker every time.
   - `LTE-Tracker` needs the 1.92 Msps rate: the producer thread cuts each OFDM symbol
     as 128 consecutive samples. So this fixture needs no decimation or trim.
   - **No download script and no new fixture are needed**, which replaces the spec's
     `generate_mib_fixture.py` / `mib_<cellid>.bin`.
6. **`--loadbin` always loops the file.** `repeat=true` is the default
   (`LTE-Tracker.cpp:220`), and `-r` can only set it to true.
   - So the spec's target, `sleep(10s); ABORT(-1)`, is unreachable. The patch instead
     exits at the end of a file pass once a MIB has been printed.
   - It uses `_exit(0)`, not `exit(0)`. Plain `exit` runs static destructors while the
     tracker threads are still running. That trips IT++'s
     `Modulator<T>::set(): Number of symbols and bits2symbols does not match` assertion
     and the process dies with SIGABRT (exit 134).
   - `_exit` also discards any half-buffered `cout` line, so callers only ever see
     whole lines. Each line is flushed by `endl` in a single write.
7. **The curses display must be skipped off-TTY.** Unpatched, with stdout piped and
   stdin at EOF, `getch()` never blocks and the display thread redraws in a tight loop.
   One measured run wrote 776,641,701 bytes of escape codes to stdout in 40 s.
   Interactive runs in a pty are unchanged: verified with `script`, they show the curses
   UI, print no MIB lines and do not auto-exit.
8. **Ground truth for `cell301_1815.3mhz.bin`.** Every decoded MIB is cell 301, 100 RB,
   PHICH duration normal, PHICH resource one. The SFN is 12 or 16.
   - The printed SFN is 4 × the MIB's 8-bit SFN field, i.e. the first radio frame of the
     decoded 40 ms TTI.
   - Cross-check: CellSearch's own `decode_mib()` on the same file sets `cell.sfn = 13`
     (its SFN for the capture's first full frame, at sample 7762) with
     `frame_timing_guess = 3`. So its decoded TTI is SFN 16–19, inside the capture. This
     was seen through a scratch debug print, never committed.
   - The capture covers frames 13–19 in full, so TTI 16–19 is complete. TTI 12–15 is
     completed by the looped replay.
   - CellSearch's summary table also reports `N 100 N one` for this cell, which matches
     (see `REAL_STDOUT` in `tests/capture/cellular/test_offline_scanner.py`).
9. **Runtime.** 20 consecutive runs went through the Python wrapper, binary built by
   `build.sh`.
   - Every run exited 0 with empty stderr.
   - SFN sequences: `(12, 16)` ×14, `(16,)` ×4, `(16, 12, 16)` ×2.
   - Wall time: min 6.01 s, median 6.14 s, max 6.24 s.
   - About 650 MB peak RSS.
   - The 4.8 s calibration PSS cross-correlation dominates. A 30 s timeout leaves about
     5x headroom.
10. **CellSearch already decodes the MIB** (`decode_mib()` with a CRC check; cells that
    fail are dropped). It prints ports, CP, nRB and PHICH in its summary table but never
    the SFN. A CellSearch-only route (parse that table, no source patch) was not taken
    here and remains an open option.

## Global Constraints

- Python >= 3.10, matching `pyproject.toml`. No new Python or system dependencies:
  `git apply` uses git, which `build.sh` already installs.
- Never commit inside `capture/cellular/vendor/lte-cell-scanner`, and never change its
  pin (`e7f71cbd4fa9f5ee5b97a58b2fb236ac13e4b8b1`). Vendored-source changes live only
  in `capture/cellular/patches/lte-tracker-mib-stdout.patch`, which is applied to a
  temporary export.
- After a build, `git -C capture/cellular/vendor/lte-cell-scanner status --short`
  shows only `?? build/`. The existing CellSearch build already creates that
  directory; this plan adds nothing.
- The patch is additive: 25 added lines, 0 removed. All new behavior is gated on
  `!isatty(STDOUT_FILENO)`. The curses UI is not removed.
- No `UnifiedRecord`/`normalizer.py` wiring, no SIB decode. PLMN stays unavailable.
- No live radio. Only pre-recorded `.bin` files are read.
- Never run `pip install -e` from a worktree. Run tests from the repo root with
  `python -m pytest`; the root `conftest.py` puts the repo on `sys.path`.
- Package names and paths are Fedora/dnf-specific, as in the predecessor plan.

## Review Focus

- **A cell that never yields a MIB** (TDD cell, weak or interfering cells).
  LTE-Tracker never exits in that case. The caller should get a `RuntimeError` that
  names the timeout and includes the stdout so far, not a hang and not a raw
  `TimeoutExpired` with bytes in it. Pinned by
  `test_run_lte_tracker_raises_runtime_error_on_timeout` (Task 2).
- **Run-to-run variation in which TTIs are printed.** A run prints 1–3 lines, SFN 12
  and/or 16, in any order. The integration test must assert membership, not an exact
  list (Task 3, `test_offline_mib_decode_matches_known_cell`).
- **A binary that can't run** (missing `libitpp`, unreadable file) exits non-zero with
  no MIB lines. The caller should get a `RuntimeError` carrying stderr. Pinned by
  `test_run_lte_tracker_raises_when_process_fails_with_no_mib` (Task 2).
- **MIB values the fixture doesn't contain**: PHICH resource `1/6` and `1/2` (they
  contain a slash), PHICH duration `extended`, and several cells' lines interleaved.
  Each line should parse on its own. Pinned by
  `test_parse_lte_tracker_output_handles_fractional_phich_and_multiple_cells` (Task 2).
- **A malformed line with fields missing** must not be reported as a MIB. Pinned by
  `test_parse_lte_tracker_output_ignores_truncated_line` (Task 2).

---

### Task 1: Patch LTE-Tracker and build it hardware-free

**Files:**
- Create: `capture/cellular/patches/lte-tracker-mib-stdout.patch`
- Modify: `capture/cellular/build.sh:47-49` (the closing `echo` lines)
- Modify: `capture/cellular/README.md` (intro paragraph, Building section, License notice)

**Interfaces:**
- Consumes: the existing submodule at `capture/cellular/vendor/lte-cell-scanner`, pinned
  at `e7f71cbd`, and the existing `build.sh` (dnf deps, IT++ to `/usr/local`,
  CellSearch).
- Produces:
  - **Binary.** `capture/cellular/build/LTE-Tracker`. It is gitignored by the root
    `.gitignore` rule `build/` and needs `LD_LIBRARY_PATH=/usr/local/lib` at runtime.
  - **Invocation.** `LTE-Tracker -f <freq_hz> --loadbin <file.bin>`.
  - **Output.** When stdout is not a TTY, it prints one line per CRC-passing MIB,
    exactly
    `MIB decoded: cell_id=<int> n_rb_dl=<6|15|25|50|75|100> phich_duration=<normal|extended> phich_resource=<1/6|1/2|one|two> sfn=<int>`.
    `sfn` is a multiple of 4 from 0 to 1020.
  - **Exit.** It exits 0 at the end of the file pass after the first such line. If no
    MIB is ever decoded it never exits.
  - Tasks 2 and 3 rely on this exact line format, exit behavior and binary path.

This task is vendoring tooling plus a C++ patch with no Python logic, so it has no TDD
red/green cycle, the same as the predecessor plan's Task 1. It is verified by building
and running the binary.

- [ ] **Step 1: Create `capture/cellular/patches/lte-tracker-mib-stdout.patch`**

The content must be exactly this. It is a git-format diff against the pinned commit,
paths relative to the LTE-Cell-Scanner root, applied with `git apply` (`-p1`). Blank
context lines are a single space. If your editor strips that space, `git apply` still
accepts the patch; Step 2 checks it.

```diff
diff --git a/src/LTE-Tracker.cpp b/src/LTE-Tracker.cpp
index 1c7e3b5..146fabd 100644
--- a/src/LTE-Tracker.cpp
+++ b/src/LTE-Tracker.cpp
@@ -94,6 +94,8 @@ using namespace itpp;
 using namespace std;
 
 uint8 verbosity=1;
+// Defined in tracker_thread.cpp.
+extern volatile bool mib_decoded;
 
 // Global variables that can be set by the command line. Used for debugging.
 double global_1=0;
@@ -1622,6 +1624,13 @@ int main(
           sampbuf_sync.fifo.push_back(samp_imag);
           offset++;
           if (offset==(unsigned)file_data.length()) {
+            // Single-shot --loadbin: stop at the end of a pass over the
+            // file once a MIB has been printed. _exit, not exit: the other
+            // threads are still running and exit()'s static destructors
+            // abort them (IT++ Modulator assertion, SIGABRT).
+            if (mib_decoded) {
+              _exit(0);
+            }
             if (!repeat) {
 //              cout << "1\n";
               break;
diff --git a/src/display_thread.cpp b/src/display_thread.cpp
index 61838df..a32805e 100644
--- a/src/display_thread.cpp
+++ b/src/display_thread.cpp
@@ -405,6 +405,11 @@ void display_thread(
 ) {
   global_thread_data.display_thread_id=syscall(SYS_gettid);
 
+  // No curses UI when stdout is not a terminal (e.g. a subprocess pipe):
+  // with no tty on stdin, getch() never blocks and the redraw loop spins.
+  if (!isatty(STDOUT_FILENO))
+    return;
+
   // Initialize the curses screen
   initscr();
   start_color();
diff --git a/src/tracker_thread.cpp b/src/tracker_thread.cpp
index 2681e2e..1e47cf5 100644
--- a/src/tracker_thread.cpp
+++ b/src/tracker_thread.cpp
@@ -552,6 +552,9 @@ void pbch_extract_rt(
   ASSERT(idx==n_syms);
 }
 
+// Set once a MIB has been printed in non-interactive mode; read by main().
+volatile bool mib_decoded=false;
+
 int8 do_mib_decode(
   tracked_cell_t & tracked_cell,
   const cvec & syms,
@@ -735,6 +738,14 @@ int8 do_mib_decode(
         boost::mutex::scoped_lock lock(tracked_cell.meas_mutex);
         tracked_cell.mib_decode_failures=0;
       }
+      // Plain-text MIB report for non-interactive (subprocess) use. The
+      // printed SFN is the first radio frame of this 40ms PBCH TTI.
+      if (!isatty(STDOUT_FILENO)) {
+        const char * phich_res_str[]={"1/6","1/2","one","two"};
+        const uint16 sfn=4*(128*c_est_ivec(6)+64*c_est_ivec(7)+32*c_est_ivec(8)+16*c_est_ivec(9)+8*c_est_ivec(10)+4*c_est_ivec(11)+2*c_est_ivec(12)+c_est_ivec(13));
+        cout << "MIB decoded: cell_id=" << tracked_cell.n_id_cell << " n_rb_dl=" << (int)n_rb_dl_est << " phich_duration=" << ((phich_duration_est==phich_duration_t::NORMAL)?"normal":"extended") << " phich_resource=" << phich_res_str[phich_res] << " sfn=" << sfn << endl;
+        mib_decoded=true;
+      }
       for (uint8 t=0;t<16;t++) {
         mib_fifo.pop_front();
       }
```

Why each hunk exists (spike findings 6–7):
- **`display_thread.cpp`.** Skips curses off-TTY. Without it, a piped run floods stdout
  (776 MB in 40 s).
- **`tracker_thread.cpp`.** Prints the line in the CRC-success branch of
  `do_mib_decode()`, the only place the tracker unpacks a MIB, and sets the flag.
  - `phich_res` (0–3) and `n_rb_dl_est` are locals already computed just above.
  - `mib_decoded` follows the codebase's existing `extern volatile bool do_exit;`
    cross-thread flag idiom.
- **`LTE-Tracker.cpp`.** Checks the flag each time the `--loadbin` feeder wraps to the
  start of the file. That is the only place the file's end is observable, because
  `repeat` is always true. It is reachable only in `--loadbin` mode, and the flag is
  only ever set off-TTY.

- [ ] **Step 2: Verify the patch applies cleanly to a pristine export**

```bash
TMP="$(mktemp -d)"
git -C capture/cellular/vendor/lte-cell-scanner archive HEAD | tar -x -C "$TMP"
git -C "$TMP" apply --check --stat "$PWD/capture/cellular/patches/lte-tracker-mib-stdout.patch"
rm -rf "$TMP"
```

Expected:
```
 src/LTE-Tracker.cpp    |    9 +++++++++
 src/display_thread.cpp |    5 +++++
 src/tracker_thread.cpp |   11 +++++++++++
 3 files changed, 25 insertions(+)
```

The patch path must be absolute. `git -C` changes directory before resolving it, so a
relative path fails with `can't open patch`.

- [ ] **Step 3: Extend `capture/cellular/build.sh` to build the patched `LTE-Tracker`**

Replace the last three lines of the file:

```bash
echo "== Done. Binary at $VENDOR_DIR/build/src/CellSearch =="
echo "Note: IT++ installs to /usr/local, which is not on the default loader"
echo "path — set LD_LIBRARY_PATH=/usr/local/lib when running CellSearch."
```

with:

```bash
echo "== Building patched LTE-Tracker (MIB decode; same hardware-free flags) =="
# patches/lte-tracker-mib-stdout.patch is applied to a throwaway export of
# the pinned submodule commit, never to the submodule working tree, so the
# submodule stays byte-identical to upstream.
TRACKER_SRC="$(mktemp -d)"
git -C "$VENDOR_DIR" archive HEAD | tar -x -C "$TRACKER_SRC"
git -C "$TRACKER_SRC" apply "$REPO_ROOT/capture/cellular/patches/lte-tracker-mib-stdout.patch"
mkdir "$TRACKER_SRC/build"
(
    cd "$TRACKER_SRC/build"
    cmake -DUSE_OPENCL=0 -DUSE_BLADERF=0 -DUSE_HACKRF=0 -DCMAKE_BUILD_TYPE=Release ..
    make -j"$(nproc)" LTE-Tracker
)
mkdir -p "$REPO_ROOT/capture/cellular/build"
install -m 755 "$TRACKER_SRC/build/src/LTE-Tracker" "$REPO_ROOT/capture/cellular/build/LTE-Tracker"
rm -rf "$TRACKER_SRC"

echo "== Done. Binaries at $VENDOR_DIR/build/src/CellSearch and $REPO_ROOT/capture/cellular/build/LTE-Tracker =="
echo "Note: IT++ installs to /usr/local, which is not on the default loader"
echo "path — set LD_LIBRARY_PATH=/usr/local/lib when running either binary."
```

Notes:
- **`git archive HEAD`** exports exactly the pinned commit's tracked files. It ignores
  the submodule's untracked `build/` and any local edits.
- **Failure handling.** Under the script's existing `set -euo pipefail`, a patch that
  fails to apply aborts the build. Verified: applying it twice fails with
  `patch does not apply` and exit 1.
- **Temp directory.** The export lives in a `mktemp -d` directory outside any git
  repository, so `git -C "$TRACKER_SRC" apply` behaves like plain `patch -p1`. It is
  removed afterwards, the same way the IT++ section already handles `$ITPP_SRC`.
- **Lint.** `bash -n` and `shellcheck` both pass on the result.

- [ ] **Step 4: Run the build and verify the binary exists**

Run: `./capture/cellular/build.sh`

Expected:
- The script completes with exit 0 in about 45 s on 22 cores. That includes an IT++
  rebuild: on Fedora the `ldconfig -p | grep -q libitpp` check never sees
  `/usr/local/lib`, which is pre-existing and out of scope.
- It ends with:
  ```
  == Done. Binaries at <repo>/capture/cellular/vendor/lte-cell-scanner/build/src/CellSearch and <repo>/capture/cellular/build/LTE-Tracker ==
  ```
- `ls -l capture/cellular/build/LTE-Tracker` shows an executable of about 11 MB.

- [ ] **Step 5: Smoke-run it the way a subprocess caller will (stdout not a terminal)**

```bash
OUT="$(mktemp)"
LD_LIBRARY_PATH=/usr/local/lib capture/cellular/build/LTE-Tracker \
    -f 1815300000 --loadbin capture/cellular/testdata/cell301_1815.3mhz.bin > "$OUT"
echo "exit: $?"
cat "$OUT"
rm -f "$OUT"
```

Expected after about 6 s: `exit: 0`, and stdout ends with 1–3 `MIB decoded:` lines.
Each line reads `cell_id=301 n_rb_dl=100 phich_duration=normal phich_resource=one` with
`sfn=12` or `sfn=16`. Verified full output (the PSS XCORR time varies run to run):

```
OpenCL LTE Tracker (Release) beginning. 1.0 to 1.1.0: An enhanced LTE Cell Scanner/tracker. Xianjun Jiao (putaoshu@msn.com)
  PPM: 120
  correction: 1
Calibrating local oscillator.
Use file begin with 1815.3MHz actual 1815.3MHz 1.92e+06MHz
    Search frequency: 1815.3 to 1815.3 MHz
with freq correction: 0 kHz
    Search PSS at fo: -220 to 215 kHz
PSS XCORR  cost 4.78511s
Calibration succeeded!
   Residual frequency offset: 14302.6 Hz
   New correction factor: 1.0000078789572046656
Searcher process has been launched.
MIB decoded: cell_id=301 n_rb_dl=100 phich_duration=normal phich_resource=one sfn=12
MIB decoded: cell_id=301 n_rb_dl=100 phich_duration=normal phich_resource=one sfn=16
```

There are no curses escape codes and nothing on stderr. If the run instead hangs past
30 s, the patch was not applied; check Step 2.

- [ ] **Step 6: Verify the submodule is untouched**

```bash
git -C capture/cellular/vendor/lte-cell-scanner status --short
git -C capture/cellular/vendor/lte-cell-scanner rev-parse HEAD
git status --short
```

Expected:
- The submodule status shows only `?? build/`, the CellSearch build directory that the
  existing script already creates.
- `rev-parse` prints `e7f71cbd4fa9f5ee5b97a58b2fb236ac13e4b8b1`.
- The top-level status shows only ` M capture/cellular/build.sh`,
  `?? capture/cellular/patches/`, and the submodule's pre-existing ` ? ` untracked-
  content marker. `capture/cellular/build/` must not appear, because it is gitignored.

- [ ] **Step 7: Update `capture/cellular/README.md`**

Replace the intro paragraph:

```markdown
Passive LTE cell broadcast decode: currently PSS/SSS cell search only (Cell ID, PSS ID,
RX power, residual frequency offset). Vendors LTE-Cell-Scanner
(https://github.com/JiaoXianjun/LTE-Cell-Scanner), built unmodified via CMake.
```

with:

```markdown
Passive LTE cell broadcast decode: PSS/SSS cell search (Cell ID, PSS ID, RX power,
residual frequency offset) via `CellSearch`, and MIB decode (bandwidth in resource
blocks, PHICH duration, PHICH resource, SFN) via `LTE-Tracker`. Vendors
LTE-Cell-Scanner (https://github.com/JiaoXianjun/LTE-Cell-Scanner) as an unmodified
submodule; `CellSearch` is built from it as-is, `LTE-Tracker` with one small tracked
patch (`patches/lte-tracker-mib-stdout.patch`).
```

In the Building section, after the paragraph ending `...and is built from source by the
script.`, add:

```markdown
`build.sh` produces two binaries: `vendor/lte-cell-scanner/build/src/CellSearch` and
`build/LTE-Tracker`. The latter is compiled from a temporary `git archive` export of the
pinned submodule commit with `patches/lte-tracker-mib-stdout.patch` applied, so the
submodule working tree is never modified. Both need `LD_LIBRARY_PATH=/usr/local/lib`
at runtime (IT++ is installed there).

## LTE-Tracker patch

`patches/lte-tracker-mib-stdout.patch` (25 added lines, nothing removed) changes
behavior only when stdout is not a terminal (i.e. when run as a subprocess):

- the curses display is skipped (otherwise, with no terminal on stdin, it redraws in a
  tight loop — 776 MB of escape codes in 40 s, measured);
- each MIB that passes CRC is printed as one line, e.g.
  `MIB decoded: cell_id=301 n_rb_dl=100 phich_duration=normal phich_resource=one sfn=16`
  (`sfn` is the first radio frame of the decoded 40ms PBCH TTI, always a multiple of 4);
- in `--loadbin` mode the program exits 0 at the end of the first pass over the file
  after a MIB was printed. `--loadbin` always loops the file, so if no MIB ever decodes
  it never exits — callers must use a timeout.

Run interactively in a terminal, LTE-Tracker behaves exactly as upstream.
```

In the License notice section, replace the first paragraph:

```markdown
`vendor/lte-cell-scanner/` is a git submodule of LTE-Cell-Scanner, licensed AGPL-3.0.
It is vendored unmodified and invoked as a separate subprocess (not linked into this
Python package), which avoids compile-time licensing entanglement, but AGPL-3.0's
network-use disclosure clause still applies if this tool's output ever becomes part of
a network-facing service (e.g. the planned carrier-facing analytics product) — anyone
wiring this in must offer that service's users the corresponding source, including any
modifications. No modifications exist yet; flag this before shipping such a service.
```

with:

```markdown
`vendor/lte-cell-scanner/` is a git submodule of LTE-Cell-Scanner, licensed AGPL-3.0.
The submodule itself is unmodified, but the `LTE-Tracker` binary is built from it with
`patches/lte-tracker-mib-stdout.patch` applied — a modification of AGPL-3.0 source, so
the patch is itself AGPL-3.0 (not covered by the MIT license at the repository root).
Both binaries are invoked as separate subprocesses (not linked into this Python
package), which avoids compile-time licensing entanglement, but AGPL-3.0's network-use
disclosure clause still applies if this tool's output ever becomes part of a
network-facing service (e.g. the planned carrier-facing analytics product) — anyone
wiring this in must offer that service's users the corresponding source, including
that patch. Flag this before shipping such a service.
```

Leave the `testdata/cell301_1815.3mhz.bin` licensing paragraph unchanged.

- [ ] **Step 8: Commit**

```bash
git add capture/cellular/patches/lte-tracker-mib-stdout.patch capture/cellular/build.sh capture/cellular/README.md
git commit -m "feat: build patched LTE-Tracker with plain-text MIB output"
```

---

### Task 2: LTE-Tracker stdout parser and subprocess wrapper

**Files:**
- Create: `capture/cellular/offline_mib_scanner.py`
- Test: `tests/capture/cellular/test_offline_mib_scanner.py`

**Interfaces:**
- Consumes: Task 1's stdout line format
  (`MIB decoded: cell_id=… n_rb_dl=… phich_duration=… phich_resource=… sfn=…`) and its
  invocation `LTE-Tracker -f <freq_hz> --loadbin <file>`. Nothing in this task needs the
  binary to be built.
- Produces:
  - `capture.cellular.offline_mib_scanner.parse_lte_tracker_output(stdout: str) -> list[dict]`.
    Each dict has `cell_id: int`, `n_rb_dl: int`, `phich_duration: str`
    (`"normal"|"extended"`), `phich_resource: str` (`"1/6"|"1/2"|"one"|"two"`) and
    `sfn: int`.
  - `capture.cellular.offline_mib_scanner.run_lte_tracker(binary_path: str, iq_file_path: str, freq_hz: int, timeout_seconds: float = 30.0, extra_env: dict[str, str] | None = None) -> list[dict]`.
    It raises `RuntimeError` on a timeout, or on a non-zero exit with no MIB parsed.
  - Task 3 calls `run_lte_tracker` directly.

`parse_lte_tracker_output` is pure and fully TDD-covered against the verbatim stdout
from Task 1 Step 5. `run_lte_tracker` follows `run_cellsearch`, with one addition:
LTE-Tracker never exits when no MIB decodes, so `TimeoutExpired` is the normal "no MIB"
outcome. It becomes a `RuntimeError`. Its two error paths are unit-tested with
monkeypatch, the same way `test_run_cellsearch_raises_when_process_fails_with_no_cells`
tests `run_cellsearch`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/capture/cellular/test_offline_mib_scanner.py
import subprocess

import pytest

from capture.cellular.offline_mib_scanner import (
    parse_lte_tracker_output,
    run_lte_tracker,
)

# Verbatim stdout from an actual patched `LTE-Tracker -f 1815300000 --loadbin
# capture/cellular/testdata/cell301_1815.3mhz.bin` run with stdout captured
# by subprocess (not a terminal, so no curses output).
REAL_STDOUT = """\
OpenCL LTE Tracker (Release) beginning. 1.0 to 1.1.0: An enhanced LTE Cell Scanner/tracker. Xianjun Jiao (putaoshu@msn.com)
  PPM: 120
  correction: 1
Calibrating local oscillator.
Use file begin with 1815.3MHz actual 1815.3MHz 1.92e+06MHz
    Search frequency: 1815.3 to 1815.3 MHz
with freq correction: 0 kHz
    Search PSS at fo: -220 to 215 kHz
PSS XCORR  cost 4.78511s
Calibration succeeded!
   Residual frequency offset: 14302.6 Hz
   New correction factor: 1.0000078789572046656
Searcher process has been launched.
MIB decoded: cell_id=301 n_rb_dl=100 phich_duration=normal phich_resource=one sfn=12
MIB decoded: cell_id=301 n_rb_dl=100 phich_duration=normal phich_resource=one sfn=16
"""

# Everything LTE-Tracker prints before its first MIB decode — what a run
# against a TDD or too-weak cell has produced when it is killed at timeout.
NO_MIB_STDOUT = REAL_STDOUT.split("MIB decoded:")[0]


def test_parse_lte_tracker_output_extracts_each_mib():
    mibs = parse_lte_tracker_output(REAL_STDOUT)
    assert mibs == [
        {
            "cell_id": 301,
            "n_rb_dl": 100,
            "phich_duration": "normal",
            "phich_resource": "one",
            "sfn": 12,
        },
        {
            "cell_id": 301,
            "n_rb_dl": 100,
            "phich_duration": "normal",
            "phich_resource": "one",
            "sfn": 16,
        },
    ]


def test_parse_lte_tracker_output_returns_empty_list_when_no_mib():
    assert parse_lte_tracker_output(NO_MIB_STDOUT) == []


def test_parse_lte_tracker_output_handles_fractional_phich_and_multiple_cells():
    """The patch spells PHICH resource 1/6 and 1/2 with a slash and prints
    one line per tracked cell; a two-cell capture interleaves them."""
    stdout = (
        "MIB decoded: cell_id=142 n_rb_dl=50 phich_duration=extended "
        "phich_resource=1/6 sfn=1020\n"
        "MIB decoded: cell_id=86 n_rb_dl=25 phich_duration=normal "
        "phich_resource=1/2 sfn=0\n"
    )
    mibs = parse_lte_tracker_output(stdout)
    assert [(m["cell_id"], m["phich_resource"]) for m in mibs] == [
        (142, "1/6"),
        (86, "1/2"),
    ]
    assert mibs[0]["phich_duration"] == "extended"
    assert mibs[0]["sfn"] == 1020


def test_parse_lte_tracker_output_ignores_truncated_line():
    """A line missing any of the five fields is not a MIB and must not be
    parsed as one."""
    stdout = "MIB decoded: cell_id=301 n_rb_dl=100 phich_duration=normal\n"
    assert parse_lte_tracker_output(stdout) == []


def test_run_lte_tracker_raises_when_process_fails_with_no_mib(monkeypatch):
    """A non-zero exit with zero parsed MIBs is a real failure (missing
    shared library, unreadable IQ file, crash) and must not be swallowed."""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else kwargs.get("args"),
            returncode=127,
            stdout="",
            stderr="error while loading shared libraries: libitpp.so.8",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="libitpp.so.8"):
        run_lte_tracker("/fake/LTE-Tracker", "/fake/iq.bin", freq_hz=1815300000)


def test_run_lte_tracker_raises_runtime_error_on_timeout(monkeypatch):
    """LTE-Tracker never exits if no MIB decodes (TDD cell, weak signal), so
    a timeout is the 'no MIB' outcome. TimeoutExpired.stdout is bytes even
    with text=True; the message must include it decoded."""

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=kwargs["timeout"], output=NO_MIB_STDOUT.encode()
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="no MIB within 5.0s") as excinfo:
        run_lte_tracker(
            "/fake/LTE-Tracker",
            "/fake/iq.bin",
            freq_hz=1815300000,
            timeout_seconds=5.0,
        )
    assert "Calibration succeeded!" in str(excinfo.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/capture/cellular/test_offline_mib_scanner.py -v`
Expected: collection error,
`ModuleNotFoundError: No module named 'capture.cellular.offline_mib_scanner'`.

- [ ] **Step 3: Write `capture/cellular/offline_mib_scanner.py`**

```python
from __future__ import annotations

import os
import re
import subprocess

_MIB_LINE = re.compile(
    r"^MIB decoded: cell_id=(?P<cell_id>\d+) n_rb_dl=(?P<n_rb_dl>\d+) "
    r"phich_duration=(?P<phich_duration>normal|extended) "
    r"phich_resource=(?P<phich_resource>1/6|1/2|one|two) sfn=(?P<sfn>\d+)$",
    re.MULTILINE,
)


def parse_lte_tracker_output(stdout: str) -> list[dict]:
    """Parses the 'MIB decoded: ...' lines that the patched LTE-Tracker
    (capture/cellular/patches/lte-tracker-mib-stdout.patch) prints when its
    stdout is not a terminal, one line per successfully decoded MIB. sfn is
    the first radio frame of the decoded 40ms PBCH TTI, so it is always a
    multiple of 4. No PLMN: that comes from SIB1, which LTE-Tracker does not
    decode."""
    mibs = []
    for match in _MIB_LINE.finditer(stdout):
        mibs.append(
            {
                "cell_id": int(match.group("cell_id")),
                "n_rb_dl": int(match.group("n_rb_dl")),
                "phich_duration": match.group("phich_duration"),
                "phich_resource": match.group("phich_resource"),
                "sfn": int(match.group("sfn")),
            }
        )
    return mibs


def run_lte_tracker(
    binary_path: str,
    iq_file_path: str,
    freq_hz: int,
    timeout_seconds: float = 30.0,
    extra_env: dict[str, str] | None = None,
) -> list[dict]:
    """Runs the patched LTE-Tracker binary against a pre-recorded --loadbin
    IQ file and returns parsed MIB dicts. Never touches a live radio.
    LTE-Tracker loops the file until a MIB is decoded, then exits 0 at the
    end of that pass; if no MIB ever decodes (a TDD cell — its tracker is
    FDD-only — or too weak a signal) it never exits, so a timeout means "no
    MIB" and is raised as RuntimeError. extra_env is merged over the current
    environment — used to set LD_LIBRARY_PATH when IT++ isn't on the
    default loader path."""
    env = {**os.environ, **(extra_env or {})}
    try:
        result = subprocess.run(
            [binary_path, "-f", str(freq_hz), "--loadbin", iq_file_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired carries bytes even when text=True was requested.
        partial = (exc.stdout or b"").decode(errors="replace")
        raise RuntimeError(
            f"LTE-Tracker decoded no MIB within {timeout_seconds}s; "
            f"stdout so far:\n{partial}"
        ) from exc
    mibs = parse_lte_tracker_output(result.stdout)
    if not mibs and result.returncode != 0:
        raise RuntimeError(
            f"LTE-Tracker exited with code {result.returncode} and no MIB was "
            f"parsed from its output; stderr:\n{result.stderr}"
        )
    return mibs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/capture/cellular/test_offline_mib_scanner.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add capture/cellular/offline_mib_scanner.py tests/capture/cellular/test_offline_mib_scanner.py
git commit -m "feat: add LTE-Tracker MIB subprocess wrapper and stdout parser"
```

---

### Task 3: Ground truth, integration test, and docs closure

**Files:**
- Create: `capture/cellular/testdata/cell301_1815.3mhz.mib.expected.json`
- Test: `tests/capture/cellular/test_offline_mib_decode.py`
- Modify: `capture/cellular/README.md` (Status paragraph)
- Modify: `README.md:11` (Layout line for `capture/cellular/`)
- Modify: `docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md` (append an
  amendment)

**Interfaces:**
- Consumes:
  - `capture.cellular.offline_mib_scanner.run_lte_tracker` (Task 2).
  - The built binary at `capture/cellular/build/LTE-Tracker` (Task 1).
  - The existing fixture `capture/cellular/testdata/cell301_1815.3mhz.bin`.
- Produces: nothing further consumes this. It is the closure proof for the spike.

As with `test_offline_decode.py`, the integration test runs the real compiled binary
and is skipped, not failed, if the binary hasn't been built. The order and number of
decoded TTIs vary from run to run (spike finding 9), so the test asserts that every
decoded MIB matches the ground truth and that every SFN is one the capture overlaps.
It does not assert an exact list.

- [ ] **Step 1: Write `capture/cellular/testdata/cell301_1815.3mhz.mib.expected.json`**

```json
{
  "cell_id": 301,
  "n_rb_dl": 100,
  "phich_duration": "normal",
  "phich_resource": "one",
  "sfn_candidates": [12, 16]
}
```

Where these values come from (spike finding 8):
- **Static fields.** They match CellSearch's `N 100 N one` for cell 301 on the same
  file.
- **`sfn_candidates`.** These are the two 40 ms TTIs that overlap the 80 ms capture,
  which covers SFN 13–19 in full. CellSearch's own MIB decode of the same file lands on
  TTI 16. TTI 12 is completed by LTE-Tracker's looped replay.

- [ ] **Step 2: Write the integration test**

```python
# tests/capture/cellular/test_offline_mib_decode.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from capture.cellular.offline_mib_scanner import run_lte_tracker

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LTE_TRACKER_BINARY = _REPO_ROOT / "capture/cellular/build/LTE-Tracker"
_FIXTURE = _REPO_ROOT / "capture/cellular/testdata/cell301_1815.3mhz.bin"
_EXPECTED = (
    _REPO_ROOT / "capture/cellular/testdata/cell301_1815.3mhz.mib.expected.json"
)

pytestmark = pytest.mark.skipif(
    not _LTE_TRACKER_BINARY.exists(),
    reason=(
        "patched LTE-Tracker binary not built; run capture/cellular/build.sh "
        "(see capture/cellular/README.md)"
    ),
)


def test_offline_mib_decode_matches_known_cell():
    expected = json.loads(_EXPECTED.read_text())

    mibs = run_lte_tracker(
        str(_LTE_TRACKER_BINARY),
        str(_FIXTURE),
        freq_hz=1815300000,
        extra_env={
            "LD_LIBRARY_PATH": "/usr/local/lib:"
            + os.environ.get("LD_LIBRARY_PATH", "")
        },
    )

    assert mibs, "LTE-Tracker exited 0 but printed no MIB line"
    for mib in mibs:
        assert mib["cell_id"] == expected["cell_id"]
        assert mib["n_rb_dl"] == expected["n_rb_dl"]
        assert mib["phich_duration"] == expected["phich_duration"]
        assert mib["phich_resource"] == expected["phich_resource"]
        # Which TTIs get decoded before the exit depends on thread timing;
        # every one must be a TTI that overlaps the 80ms capture.
        assert mib["sfn"] in expected["sfn_candidates"]
```

- [ ] **Step 3: Run it**

Run: `python -m pytest tests/capture/cellular/test_offline_mib_decode.py -v -rs`

Expected:
- **Binary built (Task 1):** PASS (1 test) in about 6 s. Verified over 20 runs at
  6.01–6.24 s wall with about 650 MB peak RSS, well inside the 30 s `timeout_seconds`
  default.
- **Binary not built:** SKIPPED with
  `patched LTE-Tracker binary not built; run capture/cellular/build.sh (see capture/cellular/README.md)`.

- [ ] **Step 4: Run the full suite to confirm no regressions**

Run: `python -m pytest`

Expected with both binaries built: `65 passed`. That is the 58 baseline plus 6 in
`test_offline_mib_scanner.py` and 1 in `test_offline_mib_decode.py`. The 33
pre-existing DeprecationWarnings from fastapi are unrelated. Without the binaries,
the two binary-dependent tests are SKIPPED instead.

- [ ] **Step 5: Update the docs**

In `capture/cellular/README.md`, replace the Status paragraph:

```markdown
**Status**: hardware-free DSP spike complete — `CellSearch` correctly detects a known
cell (Cell ID 301) from a recorded IQ file, see `testdata/`. MIB/SIB1-3 decode
(`LTE-Tracker`, a separate tool) and real xA9 hardware validation are deferred to later
plans; see `docs/superpowers/specs/2026-08-11-cellular-dsp-spike-design.md`.
```

with:

```markdown
**Status**: hardware-free DSP spikes complete — `CellSearch` correctly detects a known
cell (Cell ID 301) from a recorded IQ file, and the patched `LTE-Tracker` decodes that
cell's MIB (100 RB, PHICH normal/one) from the same file, see `testdata/`. Limits:
LTE-Tracker's tracker is FDD-only (a TDD cell never yields a MIB through it) and has no
SIB decode, so PLMN stays unavailable. Real xA9 hardware validation is deferred; see
`docs/superpowers/specs/2026-08-11-cellular-dsp-spike-design.md` and
`docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md`.
```

In the root `README.md`, replace line 11:

```markdown
- `capture/cellular/` — vendored LTE-Cell-Scanner submodule (PSS/SSS cell search only, unmodified) + normalizer; gain-control/live-radio work deferred
```

with:

```markdown
- `capture/cellular/` — vendored LTE-Cell-Scanner submodule (unmodified; `CellSearch` PSS/SSS cell search + patched `LTE-Tracker` MIB decode) + normalizer; gain-control/live-radio work deferred
```

Append to `docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md`:

```markdown

## Amendment (2026-10-03): build-and-run spike findings

The implementation plan's build-and-run spike
(`docs/superpowers/plans/2026-10-03-lte-mib-decode-spike.md`, Fedora 43, submodule @
e7f71cbd) corrected this design in several places:

1. **MIB decode needs 40 ms of IQ, not 160 ms.** `do_mib_decode()` buffers only slot 1,
   symbols 0–3 of each frame, so `mib_fifo.size()==16` is 4 frames — one PBCH TTI.
2. **`f2585_s19.2_bw20_1s_hackrf.bin` is a TDD cell (ID 216), and LTE-Tracker's tracker
   is FDD-only.** About 25,000 tracker MIB attempts in 60 s all failed CRC, while
   CellSearch decodes the same cell. The file (big-file repo @ 0791cb33, SHA-256
   ea453bba…0fba) is a plain git blob, not LFS.
3. **Weak FDD captures also fail in the tracker.** The two 1860 MHz cells (~−27 dB) in
   `f1860_s19.2_bw20_1s_hackrf_home.bin` never passed CRC in 60 s at 160 ms, 320 ms or
   1 s trims, though CellSearch decodes both.
4. **The existing 80 ms `cell301_1815.3mhz.bin` fixture works.** The patched tracker
   decodes its MIB (cell 301, 100 RB, PHICH normal/one, SFN 12 or 16) in about 6 s.
   No download script or new fixture is needed, which replaces
   `generate_mib_fixture.py` / `mib_<cellid>.bin` above.
5. **`--loadbin` always loops** (`repeat=true` by default, and `-r` can only set it to
   true), so the `sleep(10s); exit(-1)` path is unreachable.
   - The patch instead exits at the end of a file pass once a MIB was printed.
   - It uses `_exit(0)`: plain `exit()` aborts on an IT++ assertion in the
     still-running threads.
6. **The curses display must be skipped when stdout is not a terminal.** Otherwise it
   spins, writing about 776 MB in 40 s. The patch gates all new behavior on that
   condition, and interactive use is unchanged.
7. **CellSearch already decodes the MIB.** It runs `decode_mib()` with a CRC check and
   prints ports, CP, nRB and PHICH in its summary table, but not the SFN. A
   CellSearch-only route (no source patch) remains an option this spike did not take.
```

- [ ] **Step 6: Commit**

```bash
git add capture/cellular/testdata/cell301_1815.3mhz.mib.expected.json tests/capture/cellular/test_offline_mib_decode.py capture/cellular/README.md README.md docs/superpowers/specs/2026-08-11-lte-mib-decode-spike-design.md
git commit -m "test: add end-to-end LTE-Tracker MIB decode integration test"
```

---

## What this plan deliberately does not cover

- **`UnifiedRecord`/`normalizer.py` wiring of MIB fields.** This is out of scope per the
  spec; the spike stops at a proven offline decode chain.
- **SIB1-3 and PLMN.** There is no SIB decode anywhere in the vendored codebase.
- **TDD cells in LTE-Tracker.** The tracker thread assumes FDD PSS/SSS positions, and
  uplink subframes feed its CRS tracking loops. Making it TDD-capable is a real DSP
  change, not a patch-sized one.
- **Why the weak two-cell 1860 MHz captures fail in the tracker.** This was not
  investigated (spike finding 4).
- **A longer or new fixture and the spec's `generate_mib_fixture.py` download script.**
  They are unnecessary: the existing fixture works, and both 1 s candidates fail in the
  tracker.
- **The CellSearch-only alternative.** That route parses its summary table for nRB and
  PHICH and would need a one-line patch to print SFN. It was not taken; it remains an
  open option (spike finding 10).
- **Pre-existing issues, flagged and not fixed:**
  - `testdata/generate_fixture.py` reads HackRF's signed int8 samples as offset uint8.
    That is a sign-bit flip that effectively hard-limits the signal, and the
    committed fixture and its −9.45 dB ground truth inherit it.
  - `build.sh`'s `ldconfig -p | grep -q libitpp` never sees `/usr/local/lib` on Fedora,
    so every run rebuilds IT++.
  - LTE-Tracker's `-r` flag is meaningless, because `repeat` already defaults to true.
- **Removing the curses UI.** The patch only skips it when stdout is not a terminal.
- **Real xA9 hardware validation.** There is no physical SDR in this environment.
