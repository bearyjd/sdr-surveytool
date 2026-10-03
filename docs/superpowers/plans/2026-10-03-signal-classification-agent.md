# Signal Classification Agent (Part 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `agent/`, the Part 4 classifier. It triages every `modality: "unknown"`
record that step 4 stores:
- read its SigMF snippet;
- split the spectrum into occupied regions;
- measure the strongest region;
- ground it in a small cited US band table;
- ask Claude for one forced `record_classification` tool call;
- write `auto_classified` or `needs_review`, plus tag, confidence and reasoning, back into
  that record's own metadata, through a database boundary that allows nothing else.

**Architecture:** The pipeline has three layers.
- **`dsp/` math (numpy only).**
  - `dsp/segmentation.py` builds a Welch PSD over the active frames, masks the DC bin,
    and finds contiguous regions.
  - The primary region is the burst that triggered capture. It is measured against
    step 4's per-bin `dsp.spectral.noise_floor_psd` of the snippet's quiet pre-trigger
    frames. Emitters already on are found in that reference against a local median
    floor, as context only.
  - Without a usable reference, it uses a bias-corrected percentile self floor (never
    the median) and flags the analysis as reduced confidence. The usual cause is an
    always-on emitter that fills its own pre-trigger.
  - It treats the band as circular: a signal straddling the ±fs/2 edge is one region.
  - `dsp/features.py` channelizes the primary region by FFT-bin selection and measures
    fine OBW and center, duty cycle, bursts, PAPR, flatness, and a symbol-rate line in |x|².
- **The `agent/` package.** It is split by responsibility: contained snippet reader,
  band table, analysis, prompt, LLM call, pure routing, and service loop.
- **The database boundary.** The agent's role has no privilege on `survey_records`.
  - It reads a `security_barrier` view and writes only through a `SECURITY DEFINER`
    function owned by a NOLOGIN role.
  - The function's single atomic `UPDATE` re-checks "still pending", so a human tag
    always wins.
  - The gateway refuses to start unless its role passes an allowlist self-check.
- **Deployment.** The agent's container is the outer boundary; the AST tests are
  tripwires.

**Tech Stack:** The project already uses Python ≥ 3.10, numpy, sigmf-python ≥ 1.13,
SQLAlchemy 2, pydantic 2 and pytest. This plan adds `psycopg[binary]` ≥ 3.2 and
`anthropic` ≥ 0.107. PostgreSQL 16 / PostGIS 3.4 (`docker.io/postgis/postgis:16-3.4`)
serves the boundary tests.

**Spec:** `docs/superpowers/specs/2026-10-03-signal-classification-agent-design.md`
(revision 2). Architecture: `docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md`
§1, §4, §9. Step 4, the input: `docs/superpowers/plans/2026-10-03-unknown-signal-capture.md`.

## Global Constraints

- **Base.** This branch sits on finished step 4 at `9615218`
  (`docs: document quiet-frame gating of the bandwidth noise reference`), the tip of
  `worktree-unknown-signal-capture`. Check with `git merge-base --is-ancestor 9615218 HEAD`.
  The step-4 APIs this plan relies on:
  - `storage.db.make_engine(url: str)`, `init_db(engine)`, `make_session_factory(engine)`;
  - `storage.repository.save_record(session, record)`;
  - `dsp.spectral.noise_floor_psd(reference_iq, nfft=1024) -> ndarray`: step 4's per-bin
    noise floor from signal-free reference samples, in natural FFT order, scaled as the
    mean of |FFT(frame × Hann)|². It is the **one** reference-floor implementation;
  - `dsp.spectral.quiet_reference(reference_iq, threshold_dbfs, nfft=1024) -> ndarray | None`:
    the reference frames below a threshold, or None when fewer than 8 remain. A continuous
    emitter that retriggers fills its own pre-trigger and must not cancel itself.
    `dsp.spectral.dbfs` converts the threshold.
    Task 2 also moves step 4's reliability rule (> 90% of the band, or touching both
    edges) into a public `bandwidth_is_reliable`, which step 4's `occupied_bandwidth`
    then calls. Step 4's public signatures are unchanged;
  - the snippet's SigMF annotations. Step 4's writer labels `[0, trigger)` as
    `pre_trigger` (the below-threshold noise reference) and the rest as `burst`;
  - in tests only: `capture.unknown.snippet_writer.write_sigmf_snippet(iq, staging_dir, sample_rate, center_freq_hz, capture_start, trigger_offset=None) -> Path`;
  - in tests only: `storage.snippet_store.LocalSnippetStore(staging_dir, root_dir).adopt(staged_data_path: str) -> str`.
    The constructor creates both directories and checks them (`0700`, owned by this euid),
    and proves that a hard link from staging into the store works. `adopt()` accepts only
    names matching `STAGED_DATA_NAME`, and raises `SnippetRejected(ValueError)`. Tests let
    it create its own directories inside `tmp_path`. **The agent never constructs a
    `LocalSnippetStore`**, and cannot import `storage/`: it needs only the store root path;
  - the `dsp/` package, and the `conftest.py` pre-import fix.
- **Test counts.** Only per-file counts for files this plan owns are authoritative. A run
  that also includes step 4's tests (the full suite, `tests/dsp`, step 4's
  `tests/dsp/test_spectral.py`) is expected to exit 0, and its totals are deliberately not
  stated, because step 4 keeps adding tests. The import-guard totals also grow with every
  module added to `dsp/` or `schema/`, so those runs are checked by exit status too.
- **Task order.** Tasks 4, 5, 7, 9 and 10 each edit `pyproject.toml`, so they are strictly
  sequential. Every task depends on the ones before it anyway.
- **Versions.** Python ≥ 3.10. Verified on Python 3.14.7, numpy 2.4.6, sigmf 1.13.0,
  SQLAlchemy 2.0.51, psycopg 3.3.4 (binary), anthropic 0.107.1, pydantic 2.13.4, and
  PostgreSQL 16.4 with PostGIS 3.4.3, on Fedora 43.
- **Never run `pip install -e`.** The user-site editable install points at the main
  checkout. Install new dependencies with
  `pip install --user 'psycopg[binary]>=3.2' 'anthropic>=0.107'`. Always run tests from
  the repository root with `python -m pytest`.
- **PostgreSQL tests use a throwaway container only.** Never point them at the compose
  dev volume or any other container on the host. They need `SURVEYTOOL_TEST_PG_URL`, an
  admin URL, and skip without it. Each run creates its own database and
  uniquely suffixed roles, because roles are cluster-global, and drops them all. To start
  and stop the container from a distrobox (the bare `podman` alias does not exist in
  non-interactive shells), first pick a free port: `ss -ltn | grep -q ':55432 ' && echo busy`.

  ```bash
  PW=$(python -c "import secrets; print(secrets.token_hex(16))")
  distrobox-host-exec podman run -d --rm --name sdr-part4-pg -e POSTGRES_PASSWORD="$PW" \
      -p 127.0.0.1:55432:5432 docker.io/postgis/postgis:16-3.4
  export SURVEYTOOL_TEST_PG_URL="postgresql+psycopg://postgres:$PW@127.0.0.1:55432/postgres"
  # ... wait until it accepts connections ...
  distrobox-host-exec podman stop sdr-part4-pg   # --rm removes it; nothing persists
  ```
- **No test makes a live API call.** The LLM tests drive the real anthropic SDK over an
  `httpx.MockTransport`. One optional live smoke test runs only when both
  `ANTHROPIC_API_KEY` and `SURVEYTOOL_LIVE_LLM_TEST=1` are set. It was **not** run while
  writing this plan, because no key was available. It is the only check of the model id
  `claude-sonnet-5-5`, forced `tool_choice` and `thinking: disabled` against the real API.
- **Deployment, decided.** The agent runs in its own container:
  - as the **same uid as ingest**, because step 4 keeps the store `0700` and its snippets
    `0600`;
  - with the snippet store bind-mounted **read-only at the identical resolved path** that
    ingest uses, because records hold absolute `resolve()`d paths;
  - with **no ingest socket**;
  - with network egress only to the database and `api.anthropic.com`;
  - with a memory limit of about 1 GiB (the analysis peaks at 608 MB at its cap).

  The container is the boundary. The AST guards (Task 11) are tripwires, and
  `agent/README.md` says so.
- **`dsp/` stays numpy-only:** no I/O, no scipy, no `capture/` imports. scipy 1.16.2 is
  installed system-wide here, but it is not a project dependency and nothing needs it.
- **`dsp/` memory discipline**, as in step 4's `occupied_bandwidth`:
  - Work in complex64/float32 batches of `BATCH_SAMPLES = 1 << 17` samples (1 MiB),
    whatever the capture length.
  - Never promote a whole array to complex128.
  - One measured exception: the channelizer transforms each fixed 65536-sample block in
    complex128, which is also 1 MiB. In complex64, that FFT left round-off spurs that
    read as symbol-rate lines on 14 of 100 clean tones.
  - `test_ffts_run_in_bounded_batches_without_promotion` pins this.
- **The noise floor and the trigger threshold.** Segmentation works in one of two modes.
  - **The trigger threshold is not recorded.** Step 4 gates its reference on its trigger
    threshold. Neither the record, the view nor the SigMF meta carries that threshold;
    the meta has only a comment string. The agent therefore uses the fallback rule: a
    pre-trigger frame is quiet 3 dB below the snippet's active-frame mean. That level
    goes to step 4's `quiet_reference` as `threshold_dbfs`, which then requires ≥ 8
    quiet frames. If step 4 recorded its threshold, the agent could use it instead.
    Step 4's trigger margin defaults to 10 dB (`--threshold-db`), so a burst that
    triggered normally clears 3 dB. One that raises the band power by less falls to
    self mode: measured for a 67.5 kHz burst at 10–12 dB in-band SNR in a 1 MHz band.
  - **Reference mode** (≥ 8 quiet frames):
    - The primary is the burst, measured against `noise_floor_psd` of the quiet frames.
      That floor is raised to the reference's own PSD wherever the reference is higher.
      It follows the receiver's roll-off, and an emitter already on leaves no residual
      that could pose as the burst.
    - Emitters already on are found in the reference PSD against `local_floor`, the
      bias-corrected median of a circular 65-bin window. They go to the prompt as
      context with `present_before_trigger: true` and never become the primary.
  - **Self mode.** It applies when there is no annotation, fewer than 8 quiet frames, or
    an always-on emitter that fills its own pre-trigger.
    - The floor is the bias-corrected 10th-percentile bin, cross-checked against the
      2nd percentile. When they disagree by more than 3 dB, the band is crowded (more
      than ~90% occupied) and the 2nd-percentile floor is used.
    - A median floor would erase any signal occupying half the band.
    - The floor is flat, so it is blind to roll-off. Every region is flagged unreliable,
      and the analysis carries the `no_quiet_noise_reference` reason.
    - An always-on emitter is still found as the primary, never "no region", so it never
      counts toward a halt.
    - The self floor lives in `dsp/segmentation.py`, because step 4 has no self-floor
      estimator.
  - **Reduced confidence.** `SnippetAnalysis.reduced_confidence` lists the reasons:
    `no_quiet_noise_reference`, `non_finite_samples`, and `bandwidth_unreliable` (from
    `bandwidth_is_reliable`). Any reason keeps every band match ungrounded. The prompt
    states the floor source and the reasons.
- **NaN and inf samples** never crash the agent. The reader zeroes and counts them
  (`Snippet.non_finite_samples`), which reduces confidence. A snippet with no finite
  sample at all is unreadable.
- **Records without a snippet.** Step 4 persists a detection whose IQ could not be kept as
  a pending unknown row with `iq_snippet_path = NULL`. Ingest sets
  `quality_flags.snippet_rejected` (a `SnippetRejected` reason, or `no_snippet_store`).
  Capture sets `quality_flags.snippet_dropped`: `low_disk`, `staging_unavailable` or
  `queue_full`.
  - The view exposes both flags.
  - The agent closes these rows out on their own branch, never through the containment
    path: `needs_review` at once, NULL tag, confidence 0, a reasoning that names the flag,
    and no LLM call.
  - They never count toward any halt.
- **`metadata.sample_rate` and `identifier.center_freq` are authoritative.** Step 4 stores
  the SDR's read-back rate and the tuned frequency. The file's `core:sample_rate` and
  `core:frequency` must agree with them.
- **Two service invariants**, stated in `agent/service.py` and tested in Task 10:
  1. **`needs_review` is a judgment about one record, never the result of a systemic
     fault.** An outcome that could be systemic is held pending until a later success
     proves the system works.
  2. **The LLM answers at most once per record per process run.** A decision that could
     not be written is kept and written again, never re-asked. A call that raised before
     returning a response is not an answer.
- **The `agent/` import allowlist** (Task 11):
  - `agent/` imports only a fixed stdlib set, numpy, sigmf, anthropic, pydantic, `dsp`,
    `schema` and `agent` itself, plus sqlalchemy and psycopg in `agent/db_gateway.py`
    only.
  - It never imports `capture/`, `ingest/`, `storage/` or `sys`, so its CLI exits with
    `raise SystemExit(...)`.
  - `dsp/` and `schema/` are scanned too. Tests may import anything.
- **Secrets come from the environment only:** `SURVEYTOOL_AGENT_DATABASE_URL` and
  `ANTHROPIC_API_KEY` for the agent, and an operator-chosen password for its login role.
  Nothing goes in the repo, and none of them is a CLI argument.
- **Immutability.** Results are frozen dataclasses or frozen pydantic models.
  `ClassificationAgent` is the one deliberately stateful object (cursor, held records,
  unwritten decisions, token spend), and its docstring says so.
- **Deviations from the spec**, each forced by the spike or by the plan review:
  1. **The self-check is an allowlist.** The spec checks only `current_user` with
     `has_table_privilege`. The spike showed two holes in that check:
     - a NOINHERIT owner grant passed it and then rewrote a row via `SET ROLE`;
     - a column grant is invisible to it.

     The gateway instead rejects all of the following:
     - any membership but the agent role;
     - SUPERUSER, REPLICATION, BYPASSRLS, CREATEROLE or CREATEDB on either role;
     - any table or column privilege on `survey_records`;
     - CREATE on any schema, or CREATE/TEMPORARY on the database;
     - any executable non-extension SECURITY DEFINER function but `classify_unknown`.

     It requires `'USAGE'` of the agent role rather than `'MEMBER'`.
  2. **The reader does not call `sigmffile.fromfile`.** It follows a meta's
     `core:dataset` to any path, falls back to archive and converter readers, and leaks
     the meta handle on bad JSON. The reader parses at most 1 MiB of meta itself, rejects
     `core:dataset`, and builds `SigMFFile(metadata=..., data_file=<contained path>, skip_checksum=True)`.
  3. **The record and its file must agree** on sample rate and center frequency, to a
     relative 1e-9. Otherwise the snippet counts as unreadable. The cap is 2 s *and* an
     absolute 2^25 samples.
  4. **Failure handling never marks during a systemic fault.** The spec puts records in
     `needs_review` after 3 transient failures, on a model-output failure, and on a
     containment violation. Here:
     - **Transient API errors** rewind the cursor and back off indefinitely, capped at
       300 s, with no halt.
     - **Invalid model output** (validation failure, refusal, `max_tokens`) is held until
       a later answer validates; 5 in a row halt.
     - **A snippet outside the store** is held until a later snippet is analysed; a run
       of 5 snippet failures of any kind halts.
     - At startup, the agent requires an absolute root that one of the newest pending
       snippets reads through.
  5. **OBW is re-measured on the channelized primary region.** At 20 MS/s, the spec's
     coarse width could never ground FRS or airband.
  6. **The symbol rate is withheld below 13 dB region SNR.** After a quiet reference,
     region SNR reads up to 0.9 dB low, so lines appear from about 14 dB in-band SNR.
  7. **The DC mask covers DC±1, not just the DC bin.**
  8. **`classify_unknown` requires a non-NULL reasoning**, and the gateway strips NUL
     characters.
  9. **Extra revokes.** The SQL also revokes ALL on the table from PUBLIC. The PUBLIC
     revokes of CREATE and TEMPORARY are database-wide.
  10. **The tool's `tag` may be `null`.** Thinking is disabled explicitly, and no
      `temperature` is set.
  11. **The band table has 26 entries** (spec: about 20), with tightened
      `expected_obw_hz` ranges:
      - downlinks ≥ 1 MHz, 2 m ≥ 2 kHz;
      - none starts at 0, and none spans more than 250×.
  12. **No occupied region leaves the record pending** and counts toward the
      snippet-failure halt. Step 4 triggered on energy, so finding none is our failure.
  13. **Restarts.** Halts name record ids, `--start-after-id` skips them, and each
      record id is logged before its snippet is read.
  14. **Snippet-less pending rows** (above) are new in step 4.
  15. **`auto_classified` grounds the tag itself.** Its band prefix (before the first
      `:`) must be a grounded entry's id, or its last part must be the modulation label.
      The spec only asks for any grounded match.
  16. **The noise floor is not the spec's median.** The primary is measured against step
      4's per-bin pre-trigger reference when ≥ 8 quiet frames exist. Otherwise it uses a
      percentile self floor with the crowded-band cross-check, with reduced confidence.
      The band is circular.
  17. **Write errors.**
      - Arguments refused by the database (SQLSTATE class 22) halt.
      - A statement or lock timeout leaves the record pending.
      - Any other write error fails the batch, and the kept decision is retried.
      - 5 failed batches in a row halt.
  18. **Context comes from two places.** The spec sends up to 3 other regions of the
      burst spectrum. Here, emitters already on before the trigger are subtracted from
      that spectrum by the reference floor and found in the reference instead
      (`present_before_trigger: true`). The 3 strongest of both kinds go along.
  19. **Reduced confidence** (above) is not in the spec. Any reason keeps every band
      match ungrounded.
  20. **NaN and inf samples** are zeroed and counted rather than rejected. The spec is
      silent on them.

## Verified facts (build-and-run spike, 2026-10-03)

Everything below was run in this environment, against a throwaway PostgreSQL on
127.0.0.1:55432; no other container was touched. The DSP tolerances and the profile were
re-measured after the floor change, not carried over.

**PostgreSQL 16.4 and the boundary SQL**
- **Table shape.** `init_db` (`create_all`) creates `survey_records` with:
  - `id integer` (default `nextval('survey_records_id_seq')`);
  - `json` columns `identifier`, `signal` and `metadata` (`json`, not `jsonb`);
  - btree indexes on `modality` and `survey_id`;
  - the connecting role as owner.
- **JSON behavior.**
  - `->>` works on `json`.
  - There is no `json = json` operator (`UndefinedFunction`), so tests compare
    `json.loads` dicts.
  - `(metadata::jsonb || jsonb_build_object(...))::json` reorders keys by length, then
    bytewise.
- **Install and inlining.**
  - Applying the SQL twice works.
  - The SQL-standard-body predicate (`RETURN ...`) is inlined. `EXPLAIN` shows a Bitmap
    Index Scan on `ix_survey_records_modality` with `modality = 'unknown'` as the index
    condition.
- **Direct access, connected as the agent login role:**

  | Attempt | Result |
  |---|---|
  | `SELECT`/`INSERT`/`UPDATE`/`DELETE`/`TRUNCATE` on `survey_records` | 42501 |
  | `CREATE FUNCTION public.x`, `CREATE FUNCTION pg_temp.x`, `CREATE TEMP TABLE`, `CREATE SCHEMA` | 42501 |
  | `UPDATE` on the view | 0A000 |
  | `SELECT lat` from the view | 42703 |
- **Leaky-qual probe.** It is an admin-planted `COST 0.0000001` function that runs
  `RAISE NOTICE`. Through the barrier view it saw only visible rows. Through the same view
  **without** `security_barrier`, it printed a hidden `manually_tagged` row's path. A
  failing-cast probe leaked nothing even without the barrier, so it is not a useful test.
- **`classify_unknown` behavior.**
  - It raises 22023 for bad arguments (NaN confidence included) and P0002 when the row
    is not pending.
  - The PostgreSQL ARE `$` rejects `'abc\n'`.
  - The real race: the human's uncommitted `UPDATE` blocks the agent's call. Once the
    human commits, READ COMMITTED re-checks the `WHERE` clause, P0002 is raised, and the
    human tag stands.
- **Self-check.**
  - The spec's version missed a NOINHERIT owner grant (`SET ROLE` + `UPDATE` then
    rewrote a wifi row) and column grants.
  - In a fresh PG16 database, the clean agent login has no membership but its role and
    no CREATE or TEMPORARY anywhere. It can execute no non-extension SECURITY DEFINER
    function but `classify_unknown`.
  - Every bypass case in Task 5 is refused.
  - `pg_maintain` exists only from PostgreSQL 17, so that case skips on 16.
  - A superuser's message lists its attributes first.
- **Write-error mapping.**
  - NaN confidence raises 22023, which arrives as `psycopg.DataError` and is mapped to
    `SubmitRejected`.
  - A row lock held past a 0.5 s `statement_timeout` raises 57014, mapped to
    `SubmitTimedOut`.
  - NUL characters are stripped before writing.
- **SQLAlchemy and psycopg.** `exec_driver_sql(sql)` with no parameters still hands
  psycopg an empty dict, so psycopg parses `%` as placeholders. The installer therefore
  uses the raw DB-API cursor.
- **Mutation checks.** Each of the following was tried and fails its named test:
  - dropping `security_barrier`;
  - dropping the membership allowlist;
  - dropping the pending predicate from the `UPDATE`;
  - dropping the `TEMPORARY` revoke;
  - dropping the regex `$`.

**DSP (numpy only)**

Twenty seeds per case, unless noted. Capture: fs = 1 MS/s, 2^18 samples (0.262 s),
noise −50 dBFS, signal −30 dBFS.

| Case | Measured (worst of 20) | Test tolerance |
|---|---|---|
| Tone at +123.456 kHz | center ≤ 3.2 Hz; OBW 183 Hz; flatness ≤ 0.0003; PAPR 0.21–0.25 dB; symbol rate None 20/20 | ±20 Hz; < 400 Hz; < 0.01; < 1 dB |
| Band-limited 50 kHz | OBW −0.4%…+0.6%; center ≤ 388 Hz; flatness ≥ 0.987; PAPR 8.2–8.5 dB; symbol rate None 20/20 | ±3%; ±1 kHz; > 0.95; 7.5–9.5 dB |
| RRC BPSK β 0.35 at 4/8/20 samples/symbol, 20 dB in-band SNR | symbol rate error ≤ 4e-5; OBW 1.16–1.22 × Rs; PAPR 3.9–4.2 dB | ±0.1%; 1.17 × Rs ±6%; 3–5 dB |
| RRC symbol rate vs in-band SNR, self floor | 20/20 correct at ≥ 13 dB (4 and 20 sps); withheld (None) at 10 dB | None below 13 dB |
| The same after a 25 ms quiet reference | region SNR reads up to 0.9 dB under the in-band SNR, because the burst floor is raised to the reference's own noisy PSD: 0/20 and 1/20 at 13 dB, 20/20 at ≥ 15 dB (error ≤ 6e-5), None at ≤ 12 dB | none |
| OOK, ten 5 ms bursts | duty +0.0046; exactly 10 bursts; mean burst +2.4% | ±0.01; exact; ±5% |
| Two emitters (carrier + burst) | burst is primary every time: center ≤ 412 Hz, OBW ±0.6%, duty ≤ 0.0009 off; carrier as context at −9.8…−10.1 dB | ±1 kHz; ±3%; ±0.01; ±0.5 dB |
| 5 kHz signal at 2 MS/s | fine OBW +0.1…+2.5%; coarse +95% | ±6% |
| One band-limited signal filling 55–95% of the band, self floor (10 seeds each) | one region every time, OBW −0.7…−1.0%; always flagged unreliable | ±3% |
| The same at 60–95%, after a 25 ms quiet reference (10 seeds each) | one region every time, OBW −0.7…−1.0%; reliable through 90% (the OBW reads just under 90% of the band), unreliable at 95% | ±3% |
| Two wideband emitters (300 and 250 kHz) and a −45 dBFS tone | regions in order: the stronger wideband, the other, the tone | ±3%; ±1 bin |
| Task 8: 125 kHz burst + carrier at 915 MHz, 50 ms quiet reference | center −576…+569 Hz; OBW −0.8…0%; carrier context ≤ 3.3 Hz off at −9.86…−10.09 dB, `present_before_trigger` every seed; no reduced-confidence reason; grounded in `ism_902_928` every seed; the edge-straddling case unreliable and ungrounded every seed | ±2 kHz; ±5%; ±1 kHz; ±0.5 dB |
| Task 12: the same through step 4's own `process_snippet` | center −544…+206 Hz; OBW −0.8…0%; duty ≤ 0.00012 off; carrier context ≤ 1.03 Hz off; reference floor, grounded and context present every seed | ±2 kHz; ±5%; ±0.01; ±2 kHz |
| 60 kHz signal straddling ±fs/2, reference floor | one region, OBW −0.7…+2.5%, unreliable 20/20 | ±5% |
| 60 kHz signal at +450 kHz, reference floor | one region, reliable 20/20, center ≤ 1.7 kHz off | ±2 bins |
| Colored noise: 60% or 80% flat passband, 15 dB cosine roll-off; a burst after a 25 ms reference | 20 kHz bursts: OBW +7.4% (22 bins), center ≤ 455 Hz off. 300 kHz bursts: OBW −0.7%, center ≤ 1.1 kHz off. One reliable region, no false context, 20/20 in each of the 4 cases | ±15% (20 kHz), ±3% (300 kHz); ±2 bins |
| The same 20 kHz burst, self floor | a false region ≥ 560 kHz wide, unreliable 20/20 | > 500 kHz, unreliable |
| A carrier and a 20 kHz emitter already on, plus a 20 kHz burst, on 80%/15 dB colored noise | primary is the burst 20/20; both earlier emitters found in the reference 20/20 (≤ 6 Hz and ≤ 523 Hz off), never in the burst regions | ±1 bin; ±2 bins |
| Always-on emitter filling 90% of the band on 80%/15 dB colored noise | self floor 20/20; the primary every time, OBW −0.9%, unreliable | ±3% |
| False symbol-rate lines, 100 seeds each | 0 band-limited, 0 tones, 0 random OOK, on the self floor and after a quiet reference alike | n/a |
| Noise only | 0 regions, 20/20 | no regions |

**The noise floor.** These findings shaped the floor design:
- **Median floor.** It gives 0 regions at 55, 70 and 90% occupancy, and returns the tone
  as the only region when two wideband emitters are present. Task 2's mutation step
  shows this.
- **10th percentile alone.** It reads 79× the true noise at 90% occupancy, with the
  signal at about 110×, so it finds no region. Hence the 2nd-percentile cross-check.
- **Reference floor** (`noise_floor_psd` over a 25 ms pre-trigger, 10 seeds):
  - 60–95% occupancy: one region every time, OBW −0.7…−1.0%;
  - 15 dB receiver roll-off at the band edges: only the 20 kHz burst is found. The self
    floor instead reads the in-band noise as a false region over 500 kHz wide;
  - a pre-trigger that holds a continuous emitter is not 3 dB quieter, so the self
    floor is used and the emitter is found. Subtracting that reference would have
    erased it;
  - the 9-bin smoothed floor alone only partly subtracts a carrier that was on before
    the trigger, and the residue showed up among the burst regions. Raising the floor to
    the reference's own PSD removes it, and the reference-side search finds it as
    context. The context test fails without either.
- **Context floor.** In the reference, the median of a 65-bin window read no false
  region in 320 colored-noise references (60% and 80% flat, 15 and 25 dB roll-off, 24
  and 97 frames). A lower envelope (20th percentile) lags a steep skirt: 306 false
  regions in 40 references at 80%/25 dB. Task 2's second mutation step shows this.
- **The trigger threshold** reaches neither the record nor the SigMF meta (only a
  comment string), so the quiet gate uses the 3 dB fallback rule (Global Constraints).
- **A loud transient in part of the reference** does not spoil it: its frames fail the
  quiet gate. Without that gate, the continuous-emitter test fails.
- These findings came while step 4 replaced its percentile floor (at `4b130a3`) with
  `noise_floor_psd` (`706bb3b`), and then added quiet-frame gating (`9615218`).

**Bugs fixed during the sweep**, each now pinned by a test:
- the channel filter's skirt was too wide;
- the active-frame fallback used only 16 frames for a continuous emitter;
- a global-median line detector fired on the |x|² continuum;
- splicing bursts together put artifacts into the envelope spectrum;
- a single-frame envelope periodogram produced false lines;
- a complex64 channelizer FFT produced false lines;
- `synthetic.band_limited` did not wrap frequencies.

**Real snippet at 1 s, 20 MS/s** (written by step 4's writer, read by the shipped
`read_snippet` + `analyse_snippet`, on an Intel Core Ultra 9 185H).
- **Read.** It took about 100 ms, with 0 hash calls; `fromfile(skip_checksum=False)`
  takes 214 ms.
- **Analysis.** The snippet was written with a 0.1 s `pre_trigger` annotation, so the
  quiet-gated reference floor was used. The analysis took 0.81 s, for 0.91 s total and a
  386 MB peak RSS. The self-floor path measured 1.00 s and 385 MB.
- **Result.**
  - Primary: 918.002 MHz, OBW 1005.9 kHz, duty 0.200, grounded in `ism_902_928`.
  - Context, both found in the pre-trigger reference: 910 MHz (the carrier) and
    916.2346 MHz (a 12.5 kHz emitter).
- **At the 2^25-sample cap** (self floor): 1.49 s and 608 MB. The Jetson is unprofiled.

**anthropic 0.107.1** (introspected, plus real request serialization over
`httpx.MockTransport`)
- **Request and response types.**
  - `StopReason` = `end_turn | max_tokens | stop_sequence | tool_use | pause_turn | refusal`.
  - `tool_choice={"type": "tool", "name": ...}` forces the tool, and
    `thinking={"type": "disabled"}` is accepted.
  - `Message.model_validate` rejects malformed shapes.
- **Model id.** The `Model` literal stops at the 4.x models, but the type is
  `Literal | str`, so `claude-sonnet-5-5` is accepted unchecked.
- **API key.** `Anthropic()` constructs without a key, so the agent checks
  `ANTHROPIC_API_KEY` itself.
- **Status handling.**
  - 529 raises `OverloadedError`, which is not an `InternalServerError` and is not
    exported.
  - The SDK retries 408, 409, 429, ≥ 500 and connection errors.
  - The agent therefore classifies transient errors by status code with the same rule.

**eCFR** (Title 47, as of 2026-10-01). Every entry's `source` quote was grepped from
section XML fetched on 2026-10-03 from
`https://www.ecfr.gov/api/versioner/v1/full/2026-10-01/title-47.xml?part=P&section=S`
(needs `curl --compressed`).

| Entry | Edges (MHz) | Confirmed by |
|---|---|---|
| fm_broadcast | 88–108 | §73.201 |
| airband_vhf | 117.975–137 | §2.106(b)(200), footnote 5.200 |
| ham_2m | 144–148 | §97.301(a), Region 2 column |
| marine_vhf | 156–162 | §80.5 |
| frs_gmrs_462 / _467 | 462.540–462.735 / 467.540–467.735 | §95.1763 channel centers ± half the 20 kHz of §95.1773; FRS 12.5 kHz in §95.573 |
| uhf_tv | 470–608 | §73.603(a) |
| lte_b71_600_downlink / _uplink | 617–652 / 663–698 | §27.11(k); §27.5(l) |
| lower700_uplink / _downlink | 698–716 / 728–746 | §27.5(c)(1) blocks A–C; §27.50(c) |
| upper700_c_downlink / _uplink | 746–757 / 776–787 | §27.5(b)(3) |
| cellular_850_uplink / _downlink | 824–849 / 869–894 | §22.905 |
| ism_902_928 | 902–928 | §15.247 heading; §18.301 (915 MHz ± 13.0 MHz) |
| adsb_1090 | 1087.7–1092.3 | §2.106(b)(328)(ii), footnote 5.328AA |
| gnss_rnss_l1 | 1559–1610 | §2.106(b)(328)(iii), footnote 5.328B |
| aws_uplink / aws_downlink | 1695–1780 / 2110–2180 | §27.5(h); §27.50(d) |
| pcs_uplink / pcs_downlink | 1850–1910 / 1930–1990 | §24.200; §24.229 |
| ism_2400 | 2400–2483.5 | §15.247 heading |
| cbrs | 3550–3700 | §96.11(a) |
| unii_5150_5250 / unii_5725_5850 | 5150–5250 / 5725–5850 | §15.407(a)(1)(i), (a)(3)(i) |

Three caveats on the table:
- Where the CFR only pairs bands, "uplink" and "downlink" are conventional.
- `expected_obw_hz` is an engineering estimate, except for the FRS/GMRS bandwidth cap.
- Dropped as unverifiable: NOAA Weather Radio.

**Step-4 facts that matter here**
- Snippet access: capture and ingest run as one uid, with a `0700` store and `0600`
  snippets, checked at startup. Hence the same-uid deployment.
- `pip wheel .` fails before this plan starts: setuptools 70.2 rejects `license = "MIT"`.
  With that fixed in a throwaway copy, the wheel contains the band table JSON and the SQL.

## Review Focus

1. **An API outage at any point.** Expected:
   - no record is marked and the agent never halts;
   - the cursor rewinds, the backoff grows to 300 s, and the same record is retried
     until the API answers.

   Task 10: `test_an_outage_never_marks_and_never_halts` and
   `test_transient_errors_rewind_and_retry_the_same_record`.
2. **A snippet store mounted at the wrong path.** Expected: startup halts with a message
   about the identical resolved path. If it happens mid-run, five held records halt the
   agent with none marked. Task 10: the startup-check tests and
   `test_a_wrong_store_root_halts_before_marking_anything`.
3. **A human tag racing the agent**, between fetch and submit or while the agent waits on
   the row lock. Expected: the human's tag stands. Task 5's two real-PostgreSQL race tests,
   and Task 10.
4. **A record without a usable snippet.** Cases:
   - no snippet at all, including a burst of `queue_full` records;
   - a crafted `.sigmf-meta`;
   - a symlink or `../`;
   - a record that disagrees with its file.

   Expected: for no snippet, `needs_review` naming the flag, with no halt. Otherwise the
   record is held or left pending; no file outside the store is ever opened. Tasks 5, 6,
   10 and 12.
5. **A capture that fills the band, straddles its edge, or holds several wideband
   emitters; a receiver whose noise rolls off at the band edges; emitters already on
   before the trigger; an always-on emitter inside its own pre-trigger reference; NaN
   samples.** Expected:
   - the burst is the primary;
   - earlier emitters are context;
   - the always-on emitter is classified with reduced confidence and never halts the
     agent;
   - nothing crashes, and every reliability flag is honest.

   Reduced confidence or an unreliable bandwidth grounds nothing. Tasks 2, 3, 6, 8 and 10
   (`test_an_always_on_emitter_is_classified_and_never_trips_a_halt`).

## File Structure

| File | Responsibility |
|---|---|
| `schema/records.py` (modify) | `ClassificationStatus.NEEDS_REVIEW` |
| `dsp/spectral.py` (modify) | Public `bandwidth_is_reliable`, shared with step 4 |
| `dsp/synthetic.py` | Deterministic test signals with known answers |
| `dsp/segmentation.py` | Active-frame Welch PSD, reference or self floor, DC mask, circular regions, primary region, context found in the reference |
| `dsp/features.py` | FFT-block channelizer and per-region features |
| `storage/sql/agent_boundary.sql` | Roles, revokes, pending predicate, barrier view, `SECURITY DEFINER` function |
| `storage/agent_boundary.py` | Renders and installs the SQL; the `sdr-agent-boundary` admin CLI |
| `agent/db_gateway.py` | The agent's only DB access: allowlist self-check, fetches, submit with error mapping |
| `agent/snippet_reader.py` | Contained, capped, checksum-free SigMF read |
| `agent/data/band_table_us.json`, `agent/band_table.py` | Cited band table; grounded matching |
| `agent/classifier.py` | `ModulationClassifier` seam; v1 `UnavailableClassifier` |
| `agent/analysis.py` | Segmentation → primary features → context → band matches → classifier |
| `agent/prompt.py` | System prompt; numeric-only user message |
| `agent/llm.py` | Forced-tool request, response validation, transient-error rule |
| `agent/routing.py` | Pure routing on the tag's grounded band prefix |
| `agent/service.py` | The loop, hold-and-release failure handling, the startup probe, the `sdr-agent` CLI |
| `pyproject.toml`, `README.md`, `dsp/README.md`, `storage/README.md`, `agent/README.md`, architecture doc (modify) | Dependencies, packages, scripts, docs |
| `tests/...` | One module per unit, plus `tests/test_no_tag_branching.py`, `tests/agent/test_import_boundary.py` and the end-to-end tests |

---

### Task 1: `needs_review` classification status

**Files:**
- Modify: `schema/records.py` (`ClassificationStatus`)
- Test: `tests/schema/test_records.py` (import list + one test)

**Interfaces:**
- Consumes: nothing.
- Produces: `schema.records.ClassificationStatus.NEEDS_REVIEW == "needs_review"`. Tasks 5,
  9 and 10 depend on it. The change is additive: nothing in `viz/`, `ingest/` or
  `storage/` enumerates statuses (checked with grep).

- [ ] **Step 1: Write the failing test**

<!-- edit: tests/schema/test_records.py -->
Replace

```python
from schema.records import Identifier, Modality, Signal, UnifiedRecord
```

with

```python
from schema.records import (
    ClassificationStatus,
    Identifier,
    Metadata,
    Modality,
    Signal,
    UnifiedRecord,
)
```

Append to `tests/schema/test_records.py`:

<!-- append: tests/schema/test_records.py -->
```python


def test_needs_review_status_round_trips_through_json():
    """Part 4 routes low-confidence or ungrounded results to needs_review."""
    record = _make_record(metadata=Metadata(classification_status="needs_review", tag=None, confidence=0.0))
    restored = UnifiedRecord.model_validate_json(record.model_dump_json())
    assert restored.metadata.classification_status is ClassificationStatus.NEEDS_REVIEW
    assert [status.value for status in ClassificationStatus] == [
        "unclassified",
        "manually_tagged",
        "auto_classified",
        "needs_review",
    ]
```

- [ ] **Step 2: Run it to verify it fails**

<!-- check: t1_red -->
Run: `python -m pytest tests/schema/test_records.py -q`
Expected: FAIL, `1 failed, 3 passed`. The new test fails because `needs_review` is not yet a valid status.


- [ ] **Step 3: Add the status**

In `schema/records.py`:

<!-- edit: schema/records.py -->
Replace

```python
    AUTO_CLASSIFIED = "auto_classified"
```

with

```python
    AUTO_CLASSIFIED = "auto_classified"
    # Part 4 agent: a judgment that a human must resolve (to manually_tagged).
    # Terminal for the agent, and never assigned for systemic faults.
    NEEDS_REVIEW = "needs_review"
```

- [ ] **Step 4: Run the tests to verify they pass**

<!-- check: t1_green -->
Run: `python -m pytest tests/schema/test_records.py -q`
Expected: PASS, `4 passed`.

<!-- check: t1_full -->
Run: `python -m pytest -q`
Expected: `exit 0`: the whole suite passes; the one skip is the unrelated cellular CellSearch binary test.


- [ ] **Step 5: Commit**

<!-- run -->
```bash
git add schema/records.py tests/schema/test_records.py
git commit -m "feat: add needs_review classification status"
```

---

### Task 2: The shared reliability rule, synthetic test signals and spectral segmentation

**Files:**
- Modify: `dsp/spectral.py`. Move step 4's bandwidth-reliability rule into a public
  function that both step 4 and segmentation call. Step 4's public signatures are
  unchanged.
- Create: `dsp/synthetic.py`, `dsp/segmentation.py`
- Test: `tests/dsp/test_segmentation.py`, plus one test appended to step 4's
  `tests/dsp/test_spectral.py`. Do not create `tests/dsp/__init__.py`: it would shadow the
  real `dsp` package, as the step-4 plan's Task 4 explains.

**Interfaces:**
- Consumes: step 4's `dsp.spectral.quiet_reference(reference_iq, threshold_dbfs, nfft)`,
  `noise_floor_psd(reference_iq, nfft=1024)` and `dbfs`.
- Produces:
  - `dsp.spectral.bandwidth_is_reliable(hz, sample_rate, wraps_band_edges: bool) -> bool`:
    False above 90% of the band, or when the signal touches both band edges.
  - `dsp.synthetic` (complex64, mean power `power`, full scale 1.0, circular in
    frequency):
    - `noise(rng, n, power)`;
    - `tone(n, sample_rate, offset_hz, power)`;
    - `band_limited(rng, n, sample_rate, bandwidth_hz, offset_hz, power)`;
    - `rrc_taps(beta, samples_per_symbol, span_symbols=12)`;
    - `rrc_bpsk(rng, n, sample_rate, symbol_rate, beta, offset_hz, power)`;
    - `gate(x, sample_rate, bursts: list[tuple[start_s, duration_s]])`;
    - `colored_noise(rng, n, sample_rate, power, flat_fraction, edge_db)`: noise flat over
      the middle `flat_fraction` of the band, with a raised-cosine roll-off to `-edge_db`
      at ±fs/2, as a receiver's anti-alias filter delivers it.
  - `dsp.segmentation`:
    - constants `NFFT = 1024`, `MAX_CONTEXT_REGIONS = 3` and `BATCH_SAMPLES = 1 << 17`;
    - `SpectralRegion(start_offset_hz, end_offset_hz, center_offset_hz, obw_hz, excess_power, snr_db, bandwidth_reliable, noise_per_bin)`,
      frozen:
      - start/end run continuously past ±fs/2 for a wrapping region, and the center is
        folded into [−fs/2, fs/2);
      - `noise_per_bin` is the mean floor over the region's bins, which Task 3 uses;
    - `Segmentation(regions: tuple[SpectralRegion, ...], before_trigger: tuple[SpectralRegion, ...], floor_source: "pre_trigger" | "self", sample_rate, active_frames, total_frames)`,
      frozen, with `.primary -> SpectralRegion | None`:
      - `regions` are measured during the burst, sorted by excess power; `regions[0]` is
        the primary;
      - `before_trigger` are the emitters already on in the quiet reference (empty in
        self mode), for context only;
    - helpers that Task 3 reuses: `frame_powers(iq, frame_len)`, `active_frame_mask(powers)`,
      `welch_psd(iq, nfft, frame_mask)`, `psd_scale(nfft)`, `occupied_span(excess) -> (low, high)`
      and `runs(mask)`;
    - `self_floor(psd, frames, percentile) -> float`,
      `local_floor(psd, frames) -> ndarray` and
      `circular_regions(above: bool ndarray, max_gap) -> list[index arrays]`;
    - `segment_spectrum(iq, sample_rate, reference_iq=None) -> Segmentation`, which
      raises `ValueError` below `NFFT` samples.

Design decisions, each measured (see Verified facts):
- **Primary floor (reference mode).** The primary is the burst that triggered capture.
  - Step 4 does not record its trigger threshold, so a pre-trigger frame counts as
    quiet 3 dB below the active frames' mean power:
    `quiet_reference(reference_iq, dbfs(active mean / 2))`. Reference mode needs at
    least 8 such frames.
  - The burst-time PSD is measured against step 4's `noise_floor_psd` of those frames,
    raised to the reference's own PSD wherever that is higher. An emitter already on
    then leaves no residual that could pose as the burst.
- **Context (reference mode).** Emitters already on are found in the reference PSD
  itself, against `local_floor`: the bias-corrected median of a circular 65-bin window.
  It follows a receiver roll-off without lagging a steep skirt, and the regions it
  finds are context only.
- **Self floor (no usable reference).** The cases are a missing annotation, fewer than
  8 quiet frames, or an always-on emitter that fills its own pre-trigger.
  - The floor is the bias-corrected 10th-percentile bin, cross-checked against the
    2nd. When they disagree by more than 3 dB, the band is crowded and the
    2nd-percentile floor is used.
  - The floor is flat, so it is blind to roll-off: every region it finds is flagged
    unreliable. Task 8 turns that into reduced confidence. The always-on emitter is
    still found as the primary, never "no region".
- **Regions.**
  - Bins 6 dB above the floor, with gaps of ≤ 2 bins bridged.
  - The band is circular: the scan starts mid-way through the longest gap, so a signal
    straddling ±fs/2 is one region, flagged unreliable.
  - The primary is the region with the largest integrated excess power.
- **Active frames.** Frames 3 dB above the 10th-percentile frame power.
  - If none qualify, every frame is used.
  - If 1–15 qualify, the loudest 16 frames are used.
- **DC mask.** DC±1 is bridged from DC±2.
- **Memory.** Everything runs in complex64/float32 batches of `BATCH_SAMPLES` (1 MiB).
  The Hann window is float32.

- [ ] **Step 1: Write the failing tests**

<!-- write: tests/dsp/test_segmentation.py -->
```python
# tests/dsp/test_segmentation.py
import math

import numpy as np
import pytest

from dsp import synthetic
from dsp.segmentation import NFFT, circular_regions, segment_spectrum

FS = 1e6
N = 1 << 18  # 0.26 s
NOISE = 1e-5  # -50 dBFS across the whole band
BIN_HZ = FS / NFFT


def test_noise_alone_has_no_regions():
    rng = np.random.default_rng(1)
    segmentation = segment_spectrum(synthetic.noise(rng, N, NOISE), FS)
    assert segmentation.regions == ()
    assert segmentation.primary is None


def test_all_zero_input_has_no_regions():
    assert segment_spectrum(np.zeros(4 * NFFT, dtype=np.complex64), FS).regions == ()


def test_tone_is_one_narrow_region_at_its_offset():
    rng = np.random.default_rng(2)
    iq = synthetic.tone(N, FS, 123_456.0, 1e-3) + synthetic.noise(rng, N, NOISE)
    segmentation = segment_spectrum(iq, FS)
    (region,) = segmentation.regions
    assert region.center_offset_hz == pytest.approx(123_456.0, abs=BIN_HZ)
    assert region.obw_hz <= 5 * BIN_HZ
    # A continuous emitter has no quiet frames: every frame is averaged.
    assert segmentation.active_frames == segmentation.total_frames


def test_band_limited_region_width_and_center():
    rng = np.random.default_rng(3)
    iq = synthetic.band_limited(rng, N, FS, 50e3, -200e3, 1e-3) + synthetic.noise(rng, N, NOISE)
    (region,) = segment_spectrum(iq, FS).regions
    assert region.center_offset_hz == pytest.approx(-200e3, abs=BIN_HZ)
    assert region.obw_hz == pytest.approx(50e3, rel=0.05)
    assert region.start_offset_hz < -225e3 + BIN_HZ and region.end_offset_hz > -175e3 - BIN_HZ


def test_two_emitters_primary_is_the_stronger_and_the_other_is_context():
    """A continuous carrier at -40 dBFS and a 0.1 s band-limited burst at
    -30 dBFS: the burst holds more excess power, so it is primary."""
    rng = np.random.default_rng(4)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 50e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    iq = synthetic.tone(N, FS, -250e3, 1e-4) + burst + synthetic.noise(rng, N, NOISE)
    segmentation = segment_spectrum(iq, FS)
    primary, other = segmentation.regions
    assert primary.center_offset_hz == pytest.approx(200e3, abs=BIN_HZ)
    assert other.center_offset_hz == pytest.approx(-250e3, abs=BIN_HZ)
    assert 10 * math.log10(other.excess_power / primary.excess_power) == pytest.approx(-10.0, abs=0.5)
    # Only the burst's frames (and the loudest neighbours) are averaged.
    assert segmentation.active_frames < segmentation.total_frames / 2


def test_dc_bin_is_not_a_region():
    """The receiver's LO leakage sits in the DC bin; it is masked."""
    rng = np.random.default_rng(5)
    iq = synthetic.tone(N, FS, 0.0, 1e-3) + synthetic.noise(rng, N, NOISE)
    assert segment_spectrum(iq, FS).regions == ()


def test_wide_emitter_centred_on_dc_is_still_one_region():
    rng = np.random.default_rng(6)
    iq = synthetic.band_limited(rng, N, FS, 100e3, 0.0, 1e-3) + synthetic.noise(rng, N, NOISE)
    (region,) = segment_spectrum(iq, FS).regions
    assert region.center_offset_hz == pytest.approx(0.0, abs=2 * BIN_HZ)
    assert region.obw_hz == pytest.approx(100e3, rel=0.05)


@pytest.mark.parametrize("occupancy", [0.6, 0.7, 0.85])
def test_a_wideband_emitter_survives_the_self_floor(occupancy):
    """No reference: a median floor sits inside any signal filling half the
    band and erases it; the percentile self floor does not (measured over 10
    seeds: OBW -0.7%..-1.0% from 55% to 95%). It is blind to roll-off, so
    every region it finds is flagged unreliable: reduced confidence."""
    rng = np.random.default_rng(7)
    iq = synthetic.band_limited(rng, N, FS, occupancy * FS, 20e3, 1e-3) + synthetic.noise(rng, N, NOISE)
    segmentation = segment_spectrum(iq, FS)
    (region,) = segmentation.regions
    assert region.obw_hz == pytest.approx(occupancy * FS, rel=0.03)
    assert segmentation.floor_source == "self" and not region.bandwidth_reliable


@pytest.mark.parametrize("occupancy", [0.9, 0.95])
def test_a_crowded_band_is_still_one_region_but_flagged_unreliable(occupancy):
    """At ~90% the 10th-percentile floor lands inside the signal; the 2nd
    percentile disagrees by > 3 dB, is used instead, and the region carries
    step 4's >90% unreliability judgment."""
    rng = np.random.default_rng(8)
    iq = synthetic.band_limited(rng, N, FS, occupancy * FS, 20e3, 1e-3) + synthetic.noise(rng, N, NOISE)
    (region,) = segment_spectrum(iq, FS).regions
    assert region.obw_hz == pytest.approx(occupancy * FS, rel=0.03)
    assert not region.bandwidth_reliable


def test_two_wideband_emitters_and_a_weak_tone():
    rng = np.random.default_rng(9)
    iq = (
        synthetic.band_limited(rng, N, FS, 300e3, -250e3, 1e-3)
        + synthetic.band_limited(rng, N, FS, 250e3, 220e3, 5e-4)
        + synthetic.tone(N, FS, 30e3, 3e-5)
        + synthetic.noise(rng, N, NOISE)
    )
    wide, other, tone = segment_spectrum(iq, FS).regions
    assert (wide.center_offset_hz, wide.obw_hz) == (pytest.approx(-250e3, abs=BIN_HZ), pytest.approx(300e3, rel=0.03))
    assert (other.center_offset_hz, other.obw_hz) == (pytest.approx(220e3, abs=BIN_HZ), pytest.approx(250e3, rel=0.03))
    assert tone.center_offset_hz == pytest.approx(30e3, abs=BIN_HZ)


def test_a_signal_straddling_the_band_edge_is_one_unreliable_region():
    """+fs/2 and -fs/2 are the same frequency: the region wraps, as it does
    for dsp.features.channelize, and touching both edges makes its
    bandwidth unreliable (dsp.spectral.bandwidth_is_reliable)."""
    rng = np.random.default_rng(10)
    iq = _burst_after_reference(rng, synthetic.band_limited(rng, N, FS, 60e3, FS / 2, 1e-3))
    (region,) = segment_spectrum(iq, FS, iq[:PRE]).regions
    assert abs(region.center_offset_hz) == pytest.approx(FS / 2, abs=BIN_HZ)
    assert region.obw_hz == pytest.approx(60e3, rel=0.05)
    assert region.end_offset_hz > FS / 2 > region.start_offset_hz
    assert not region.bandwidth_reliable


def test_a_signal_near_but_inside_the_edge_stays_reliable():
    rng = np.random.default_rng(11)
    iq = _burst_after_reference(rng, synthetic.band_limited(rng, N, FS, 60e3, 450e3, 1e-3))
    (region,) = segment_spectrum(iq, FS, iq[:PRE]).regions
    assert region.center_offset_hz == pytest.approx(450e3, abs=2 * BIN_HZ) and region.bandwidth_reliable


@pytest.mark.parametrize(
    "above, expected",
    [
        ("..##...", ["2,3"]),
        ("#....##", ["5,6,0"]),  # wraps: the last bins and bin 0 are one region
        ("#.#....", ["0,1,2"]),  # a 1-bin gap is bridged
        ("##.....#.", ["7,8,0,1"]),  # bridged across the wrap
        ("#######", ["0,1,2,3,4,5,6"]),
        (".......", []),
        ("##..##.....", ["0,1", "4,5"]),
    ],
)
def test_circular_regions(above, expected):
    mask = np.array([c == "#" for c in above])
    found = [",".join(str(int(i)) for i in region) for region in circular_regions(mask, max_gap=1)]
    assert found == expected


PRE = 25_000  # a 25 ms pre-trigger reference: 24 frames


def _burst_after_reference(rng, burst):
    """Noise throughout, the burst only after the pre-trigger reference."""
    burst = burst.copy()
    burst[:PRE] = 0
    return synthetic.noise(rng, N, NOISE) + burst


@pytest.mark.parametrize("occupancy, reliable", [(0.6, True), (0.85, True), (0.95, False)])
def test_a_quiet_pre_trigger_reference_supplies_the_floor(occupancy, reliable):
    """Step 4's per-bin floor (dsp.spectral.noise_floor_psd) from the
    snippet's pre-trigger samples: measured -0.7%..-1.0% OBW from 60% to
    95% occupancy over 10 seeds; > 90% is unreliable, as in step 4."""
    rng = np.random.default_rng(12)
    iq = _burst_after_reference(rng, synthetic.band_limited(rng, N, FS, occupancy * FS, 20e3, 1e-3))
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    (region,) = segmentation.regions
    assert segmentation.floor_source == "pre_trigger"
    assert region.obw_hz == pytest.approx(occupancy * FS, rel=0.03)
    assert region.bandwidth_reliable is reliable


@pytest.mark.parametrize(
    "flat, bandwidth, offset",
    [(0.6, 20e3, 100e3), (0.8, 20e3, 100e3), (0.6, 300e3, -50e3), (0.8, 300e3, -50e3)],
)
def test_on_colored_noise_the_reference_floor_measures_the_burst(flat, bandwidth, offset):
    """Receiver noise rolls off 15 dB toward the band edges (an 80% or 60%
    flat passband). The per-bin reference floor follows it: one burst
    region and no false context. Measured over 20 seeds: 20 kHz bursts read
    +7.4% (22 bins) within 455 Hz, 300 kHz bursts -0.7% within 1.1 kHz."""
    rng = np.random.default_rng(13)
    burst = synthetic.band_limited(rng, N, FS, bandwidth, offset, 1e-4)
    burst[:PRE] = 0
    iq = synthetic.colored_noise(rng, N, FS, NOISE, flat, 15.0) + burst
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    (region,) = segmentation.regions
    assert region.center_offset_hz == pytest.approx(offset, abs=2 * BIN_HZ)
    assert region.obw_hz == pytest.approx(bandwidth, rel=0.15 if bandwidth < 1e5 else 0.03)
    assert region.bandwidth_reliable and segmentation.before_trigger == ()


def test_on_colored_noise_the_self_floor_is_fooled_and_says_so():
    """Without a reference, a flat floor reads the in-band noise of a 15 dB
    roll-off as one false region hundreds of kHz wide; that is why every
    self-floor region is flagged unreliable."""
    rng = np.random.default_rng(13)
    burst = synthetic.band_limited(rng, N, FS, 20e3, 100e3, 1e-4)
    iq = synthetic.colored_noise(rng, N, FS, NOISE, 0.6, 15.0) + burst
    primary = segment_spectrum(iq, FS).primary
    assert primary.obw_hz > 500e3 and not primary.bandwidth_reliable


def test_emitters_already_on_are_context_from_the_reference():
    """A carrier and a 20 kHz emitter already on before the trigger, on
    colored noise: found in the reference against its own local floor
    (context only) and absent from the burst regions, where the reference
    floor subtracts them; the primary is the burst."""
    rng = np.random.default_rng(17)
    burst = synthetic.band_limited(rng, N, FS, 20e3, 100e3, 1e-4)
    burst[:PRE] = 0
    iq = (
        synthetic.colored_noise(rng, N, FS, NOISE, 0.8, 15.0)
        + synthetic.tone(N, FS, -250e3, 3e-5)
        + synthetic.band_limited(rng, N, FS, 20e3, 300e3, 3e-5)
        + burst
    )
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    (primary,) = segmentation.regions
    assert primary.center_offset_hz == pytest.approx(100e3, abs=2 * BIN_HZ)
    centers = sorted(region.center_offset_hz for region in segmentation.before_trigger)
    assert centers == [pytest.approx(-250e3, abs=BIN_HZ), pytest.approx(300e3, abs=2 * BIN_HZ)]


def test_a_steep_roll_off_is_not_read_as_context():
    """Noise 25 dB down at the edges of an 80% passband: the median of a
    65-bin window follows the skirt; a lower envelope (20th percentile)
    lags it and reads the slope as emitters (306 false regions in 40)."""
    rng = np.random.default_rng(19)
    burst = synthetic.band_limited(rng, N, FS, 20e3, 100e3, 1e-4)
    burst[:PRE] = 0
    iq = synthetic.colored_noise(rng, N, FS, NOISE, 0.8, 25.0) + burst
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    assert segmentation.before_trigger == ()
    assert segmentation.primary.center_offset_hz == pytest.approx(100e3, abs=2 * BIN_HZ)


def test_an_always_on_emitter_is_the_primary_from_the_self_floor():
    """An always-on emitter (an LTE downlink) fills its own pre-trigger, so
    no reference frame is quiet: the self floor finds it as the primary,
    never no region, flagged unreliable."""
    rng = np.random.default_rng(18)
    iq = synthetic.colored_noise(rng, N, FS, NOISE, 0.8, 15.0) + synthetic.band_limited(rng, N, FS, 0.9 * FS, 0.0, 1e-3)
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    assert segmentation.floor_source == "self"
    assert segmentation.primary.obw_hz == pytest.approx(0.9 * FS, rel=0.03)
    assert not segmentation.primary.bandwidth_reliable


def test_a_reference_holding_the_signal_falls_back_to_the_self_floor():
    """A continuous emitter retriggering after its cooldown fills its own
    pre-trigger: subtracting that would erase it. No reference frame is 3 dB
    quieter than the active frames (dsp.spectral.quiet_reference), so the
    self floor is used."""
    rng = np.random.default_rng(14)
    iq = synthetic.noise(rng, N, NOISE) + synthetic.band_limited(rng, N, FS, 300e3, -100e3, 1e-3)
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    (region,) = segmentation.regions
    assert segmentation.floor_source == "self"
    assert region.obw_hz == pytest.approx(300e3, rel=0.03)


def test_the_quiet_frames_of_a_reference_with_a_transient_still_serve():
    """A loud transient in part of the pre-trigger does not spoil the
    reference: its frames fail the quiet gate (dsp.spectral.quiet_reference),
    and step 4's group-median floor would shrug them off anyway."""
    rng = np.random.default_rng(16)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 50e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    iq = synthetic.noise(rng, N, NOISE) + burst
    iq[: 5 * NFFT] += synthetic.band_limited(rng, 5 * NFFT, FS, 400e3, -100e3, 3e-3)
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    assert segmentation.floor_source == "pre_trigger"
    assert segmentation.primary.obw_hz == pytest.approx(50e3, rel=0.05)


def test_a_reference_shorter_than_eight_frames_is_not_used():
    rng = np.random.default_rng(15)
    iq = _burst_after_reference(rng, synthetic.band_limited(rng, N, FS, 50e3, 200e3, 1e-3))
    assert segment_spectrum(iq, FS, iq[: 7 * NFFT]).floor_source == "self"


def test_too_short_capture_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        segment_spectrum(np.zeros(NFFT - 1, dtype=np.complex64), FS)
```

In step 4's `tests/dsp/test_spectral.py`, extend the import:

<!-- edit: tests/dsp/test_spectral.py -->
Replace

```python
from dsp.spectral import (

```

with

```python
from dsp.spectral import (
    bandwidth_is_reliable,

```

and append:

<!-- append: tests/dsp/test_spectral.py -->
```python


@pytest.mark.parametrize(
    "hz, wraps, reliable",
    [(0.9 * FS, False, True), (0.91 * FS, False, False), (1_000.0, True, False), (1_000.0, False, True)],
)
def test_bandwidth_is_reliable(hz, wraps, reliable):
    """The one reliability rule, shared with dsp.segmentation."""
    assert bandwidth_is_reliable(hz, FS, wraps) is reliable
```

- [ ] **Step 2: Run them to verify they fail**

<!-- check: t2_red -->
Run: `python -m pytest tests/dsp/test_segmentation.py tests/dsp/test_spectral.py -q`
Expected: collection ERRORs, `2 errors`, including `ImportError: cannot import name 'bandwidth_is_reliable' from 'dsp.spectral'`.


- [ ] **Step 3: Implement**

In `dsp/spectral.py`, share the reliability rule:

<!-- edit: dsp/spectral.py -->
Replace

```python
    reliable = (
        has_reference
        and hz <= _RELIABLE_FRACTION_OF_BAND * sample_rate
        and not wraps_band_edges
    )
    return OccupiedBandwidth(hz=hz, reliable=reliable)
```

with

```python
    reliable = has_reference and bandwidth_is_reliable(hz, sample_rate, wraps_band_edges)
    return OccupiedBandwidth(hz=hz, reliable=reliable)


def bandwidth_is_reliable(hz: float, sample_rate: float, wraps_band_edges: bool) -> bool:
    """Whether a measured occupied bandwidth can be trusted: not when it
    fills more than 90% of the capture band or when the signal touches both
    band edges (it wraps around them, so its edge-to-edge span is not its
    bandwidth). Shared by occupied_bandwidth and dsp.segmentation."""
    return hz <= _RELIABLE_FRACTION_OF_BAND * sample_rate and not wraps_band_edges
```

Then the new modules:

<!-- write: dsp/synthetic.py -->
```python
# dsp/synthetic.py
"""Deterministic synthetic IQ with known answers (numpy only).

Test support shared by tests/dsp and tests/agent: every generator returns
complex64 at a stated mean power (linear |x|^2, full scale = 1.0), so tests
can mix emitters at known relative levels.
"""

from __future__ import annotations

import math

import numpy as np


def _scaled(x: np.ndarray, power: float) -> np.ndarray:
    return (x * math.sqrt(power / np.mean(np.abs(x) ** 2))).astype(np.complex64)


def noise(rng: np.random.Generator, n: int, power: float) -> np.ndarray:
    """Complex white Gaussian noise."""
    scale = math.sqrt(power / 2)
    return (scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


def colored_noise(
    rng: np.random.Generator,
    n: int,
    sample_rate: float,
    power: float,
    flat_fraction: float,
    edge_db: float,
) -> np.ndarray:
    """Noise as a real receiver delivers it: flat over the middle
    `flat_fraction` of the band, then a raised-cosine roll-off down to
    -edge_db at +-sample_rate/2 (the anti-alias filter's skirt)."""
    spectrum = np.fft.fft(noise(rng, n, 1.0))
    edge = np.abs(np.fft.fftfreq(n, 1 / sample_rate)) / (sample_rate / 2)
    roll = np.clip((edge - flat_fraction) / (1.0 - flat_fraction), 0.0, 1.0)
    gain_db = -edge_db * (1.0 - np.cos(np.pi * roll)) / 2.0
    return _scaled(np.fft.ifft(spectrum * 10 ** (gain_db / 20)), power)


def tone(n: int, sample_rate: float, offset_hz: float, power: float) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (math.sqrt(power) * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)


def band_limited(
    rng: np.random.Generator,
    n: int,
    sample_rate: float,
    bandwidth_hz: float,
    offset_hz: float,
    power: float,
) -> np.ndarray:
    """Brick-wall band-limited noise: an unknown modulated signal whose true
    occupied bandwidth is exactly `bandwidth_hz`. Frequency is circular, as
    in a real capture: a signal centred at +-sample_rate/2 straddles the edge."""
    spectrum = np.fft.fft(noise(rng, n, 1.0))
    freqs = np.fft.fftfreq(n, 1 / sample_rate)
    distance = (freqs - offset_hz + sample_rate / 2) % sample_rate - sample_rate / 2
    spectrum[np.abs(distance) > bandwidth_hz / 2] = 0
    return _scaled(np.fft.ifft(spectrum), power)


def rrc_taps(beta: float, samples_per_symbol: int, span_symbols: int = 12) -> np.ndarray:
    """Unit-energy root-raised-cosine pulse, `span_symbols` symbols long."""
    t = np.arange(-span_symbols * samples_per_symbol // 2, span_symbols * samples_per_symbol // 2 + 1)
    t = t / samples_per_symbol
    taps = np.empty(len(t))
    for i, ti in enumerate(t):
        if ti == 0.0:
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            taps[i] = (beta / math.sqrt(2)) * (
                (1 + 2 / np.pi) * math.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * math.cos(np.pi / (4 * beta))
            )
        else:
            taps[i] = (
                math.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * math.cos(np.pi * ti * (1 + beta))
            ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return taps / math.sqrt(np.sum(taps**2))


def rrc_bpsk(
    rng: np.random.Generator,
    n: int,
    sample_rate: float,
    symbol_rate: float,
    beta: float,
    offset_hz: float,
    power: float,
) -> np.ndarray:
    """Random BPSK through an RRC pulse. sample_rate / symbol_rate must be an
    integer >= 4. Unlike rectangular PSK, its envelope is not constant, so
    |x|^2 carries a spectral line at the symbol rate."""
    sps = round(sample_rate / symbol_rate)
    if sps < 4 or not math.isclose(sps * symbol_rate, sample_rate):
        raise ValueError("sample_rate / symbol_rate must be an integer >= 4")
    symbols = rng.choice([-1.0, 1.0], size=n // sps + 1)
    impulses = np.zeros(len(symbols) * sps)
    impulses[::sps] = symbols
    baseband = np.convolve(impulses, rrc_taps(beta, sps), mode="same")[:n]
    return _scaled(baseband * np.exp(2j * np.pi * offset_hz * np.arange(n) / sample_rate), power)


def gate(x: np.ndarray, sample_rate: float, bursts: list[tuple[float, float]]) -> np.ndarray:
    """Zero `x` outside the (start_s, duration_s) bursts: on-off keying."""
    gated = np.zeros_like(x)
    for start_s, duration_s in bursts:
        start = round(start_s * sample_rate)
        stop = start + round(duration_s * sample_rate)
        gated[start:stop] = x[start:stop]
    return gated
```

<!-- write: dsp/segmentation.py -->
```python
# dsp/segmentation.py
"""Find the occupied spectral regions of a capture (numpy only).

A capture can hold several emitters: step 4's trigger is level-based and its
band is up to ~56 MHz wide. Whole-capture statistics then describe nothing in
particular, so the agent first splits the spectrum into contiguous occupied
regions and analyses the strongest one (dsp.features).

The primary region is the burst that triggered capture. When the snippet
carries a usable reference -- the quiet frames (dsp.spectral.quiet_reference)
of its "pre_trigger" samples -- the burst-time spectrum is measured against
step 4's per-bin floor of them (noise_floor_psd), raised to the reference's
own spectrum where that is higher, so an emitter already on before the
trigger leaves no residual to pose as the burst. Step 4 gates on its trigger
threshold, which the agent does not know, so a frame counts as quiet 3 dB
below the snippet's active frames. The emitters already on are found in the
reference itself, against a roll-off-aware local floor (the median of a
sliding 65-bin window), and are context only.

Without at least 8 quiet frames (no annotation, too short, or a continuous
emitter retriggering after its cooldown, which fills its own pre-trigger),
a self floor is estimated from the burst-time PSD -- a bias-corrected
percentile, never the median, which sits inside any signal occupying half
the band or more and erases it -- and every region is flagged unreliable:
that floor is flat, so it is blind to roll-off.

The band is treated as circular, as dsp.features.channelize treats it: a
signal straddling the +-fs/2 edge is one region, not two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from dsp.spectral import bandwidth_is_reliable, dbfs, noise_floor_psd, quiet_reference

NFFT = 1024
# Work in complex64/float32 batches of this many samples (1 MiB of complex64),
# like dsp.spectral: memory stays a few MiB whatever the capture length, and
# nothing is promoted to complex128.
BATCH_SAMPLES = 1 << 17
_ACTIVE_MARGIN = 2.0  # 3 dB above the quiet-frame level marks an active frame
_MIN_ACTIVE_FRAMES = 16  # fewer than this and the loudest frames are used instead
_REGION_THRESHOLD = 4.0  # 6 dB above the per-bin noise floor
_MERGE_GAP_BINS = 2  # gaps this narrow (the masked DC bin, ripple) are bridged
# A pre-trigger frame is quiet 3 dB below the active frames' mean power
# (dsp.spectral.quiet_reference then needs at least 8 of them).
_QUIET_REFERENCE_RATIO = 2.0
# Self floor: the 10th-percentile bin sits inside a signal filling more than
# ~90% of the band; the 2nd-percentile one stays noise-only up to ~98%. When
# they disagree by more than 3 dB the band is crowded and the deep floor is
# used. Either way every self-floor region is flagged unreliable.
_SELF_FLOOR_PERCENTILE = 10.0
_DEEP_FLOOR_PERCENTILE = 2.0
_CROWDED_RATIO = 2.0
# Context floor for the reference: the median of a 65-bin circular window.
# It follows a filter roll-off without lagging a steep skirt (measured: the
# median read no false region in 320 colored-noise references, 60%/80% flat
# passbands rolling off 15 or 25 dB, 24 or 97 frames; a 20th percentile read
# 306 in 40 at 80%/25 dB), while an emitter narrower than ~30 bins stands above it.
_LOCAL_WINDOW_BINS = 65
_LOCAL_PERCENTILE = 50.0
_OCCUPIED_FRACTION = 0.99
MAX_CONTEXT_REGIONS = 3


@dataclass(frozen=True)
class SpectralRegion:
    """One contiguous occupied region. Frequencies are offsets from the tuned
    center; powers are linear (full-scale complex sample = 1.0). start and
    end run continuously, so a region wrapping the band edge has
    end_offset_hz > sample_rate / 2; its center is folded into
    [-sample_rate / 2, sample_rate / 2)."""

    start_offset_hz: float
    end_offset_hz: float
    center_offset_hz: float  # excess-power-weighted centroid
    obw_hz: float  # narrowest span holding 99% of the region's excess power
    excess_power: float  # integrated power above the noise floor
    snr_db: float  # excess power over the noise in the same bins
    # dsp.spectral.bandwidth_is_reliable: False above 90% of the band or when
    # the region touches both band edges.
    bandwidth_reliable: bool
    noise_per_bin: float  # mean floor over the region's bins, in PSD units


@dataclass(frozen=True)
class Segmentation:
    regions: tuple[SpectralRegion, ...]  # during the burst, strongest first: [0] is the primary
    before_trigger: tuple[SpectralRegion, ...]  # emitters already on in the reference: context only
    floor_source: str  # "pre_trigger" (step 4's reference) or "self" (every region unreliable)
    sample_rate: float
    active_frames: int
    total_frames: int

    @property
    def primary(self) -> SpectralRegion | None:
        return self.regions[0] if self.regions else None


def frame_powers(iq: np.ndarray, frame_len: int) -> np.ndarray:
    """Mean |x|^2 (float32) of each whole frame; a trailing partial frame is dropped."""
    frames = iq[: (len(iq) // frame_len) * frame_len].reshape(-1, frame_len)
    step = max(1, BATCH_SAMPLES // frame_len)
    return np.concatenate(
        [np.mean(np.abs(frames[i : i + step]) ** 2, axis=1) for i in range(0, len(frames), step)]
        or [np.zeros(0, dtype=np.float32)]
    )


def active_frame_mask(powers: np.ndarray) -> np.ndarray:
    """Frames at least 3 dB above the quiet (10th-percentile) frame level.

    None qualify: the emitter is continuous, there are no quiet frames to
    exclude, so every frame is used. Only a few qualify (a burst shorter than
    _MIN_ACTIVE_FRAMES frames): the loudest _MIN_ACTIVE_FRAMES are used, so
    the PSD still averages enough frames to keep noise bins under the region
    threshold.
    """
    active = powers >= np.percentile(powers, 10) * _ACTIVE_MARGIN
    count = int(active.sum())
    if count == 0:
        return np.ones(len(powers), dtype=bool)
    if count < _MIN_ACTIVE_FRAMES:
        active = np.zeros(len(powers), dtype=bool)
        active[np.argsort(powers)[-_MIN_ACTIVE_FRAMES:]] = True
    return active


def psd_scale(nfft: int) -> float:
    """welch_psd's scale relative to dsp.spectral's |FFT(frame x Hann)|^2:
    1 / (sum(w^2) * nfft), so bins sum to the mean power."""
    window = np.hanning(nfft).astype(np.float32)
    return 1.0 / (float(np.sum(window.astype(np.float64) ** 2)) * nfft)


def welch_psd(iq: np.ndarray, nfft: int, frame_mask: np.ndarray) -> np.ndarray:
    """Hann-windowed, fftshifted mean periodogram over the selected frames,
    scaled so its bins sum to the mean power of those frames. Batched in
    complex64; only the nfft-bin accumulator is float64."""
    # float32: a float64 window would promote every frame to complex128.
    window = np.hanning(nfft).astype(np.float32)
    scale = psd_scale(nfft)
    frames = iq[: len(frame_mask) * nfft].reshape(-1, nfft)
    selected = np.flatnonzero(frame_mask)
    step = max(1, BATCH_SAMPLES // nfft)
    total = np.zeros(nfft)
    for i in range(0, len(selected), step):
        batch = frames[selected[i : i + step]] * window
        total += np.sum(np.abs(np.fft.fft(batch, axis=1)) ** 2, axis=0)
    return np.fft.fftshift(total * scale / max(len(selected), 1))


def occupied_span(excess: np.ndarray) -> tuple[int, int]:
    """Indices (low, high), inclusive, of the narrowest span holding 99% of
    `excess`, trimming equal tails."""
    cumulative = np.cumsum(excess) / excess.sum()
    tail = (1.0 - _OCCUPIED_FRACTION) / 2.0
    return int(np.searchsorted(cumulative, tail)), int(np.searchsorted(cumulative, 1.0 - tail))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs, stop exclusive, of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _merged_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in runs(mask):
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def circular_regions(above: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """Bin indices of each occupied region of a circular band, in order.

    Bin len-1 is adjacent to bin 0 (the +-fs/2 edge of an fftshifted PSD).
    The scan starts in the middle of the longest gap, so no region and no
    bridgeable gap is ever cut by the start of the array.
    """
    n = len(above)
    if not above.any():
        return []
    gaps = runs(~above)
    if not gaps:
        return [np.arange(n)]
    longest = max(gaps, key=lambda gap: gap[1] - gap[0])
    if gaps[0][0] == 0 and gaps[-1][1] == n and len(gaps) > 1:  # the gap wraps too
        wrapped = gaps[-1][1] - gaps[-1][0] + gaps[0][1]
        if wrapped > longest[1] - longest[0]:
            longest = (gaps[-1][0], gaps[-1][0] + wrapped)
    if longest[1] - longest[0] <= max_gap:  # every gap bridges: one region
        return [np.arange(n)]
    order = ((longest[0] + longest[1]) // 2 + np.arange(n)) % n
    return [order[start:stop] for start, stop in _merged_runs(above[order], max_gap)]


def self_floor(psd: np.ndarray, frames: int, percentile: float) -> float:
    """Mean noise power per bin of a Welch PSD averaged over `frames` frames,
    from its `percentile`-th bin: each noise bin is a mean of `frames`
    exponentials, so that percentile sits at a known fraction of the noise
    mean (Wilson-Hilferty), and dividing by it removes the low bias."""
    return float(np.percentile(psd, percentile)) / _quantile_fraction(frames, percentile)


def local_floor(psd: np.ndarray, frames: int) -> np.ndarray:
    """Per-bin roll-off-aware floor: the bias-corrected median of a circular
    65-bin window around each bin."""
    half = _LOCAL_WINDOW_BINS // 2
    padded = np.concatenate([psd[-half:], psd, psd[:half]])
    envelope = np.percentile(sliding_window_view(padded, _LOCAL_WINDOW_BINS), _LOCAL_PERCENTILE, axis=1)
    return envelope / _quantile_fraction(frames, _LOCAL_PERCENTILE)


def _quantile_fraction(frames: int, percentile: float) -> float:
    a = 1.0 / (9.0 * frames)
    return (1.0 - a + NormalDist().inv_cdf(percentile / 100.0) * math.sqrt(a)) ** 3


def _bridge_dc(psd: np.ndarray) -> np.ndarray:
    """The DC bin carries the receiver's LO leakage, not an emitter; with a
    Hann window it spreads into DC +-1. Bridge those three bins from DC +-2."""
    dc = len(psd) // 2
    bridged = psd.copy()
    bridged[dc - 1 : dc + 2] = 0.5 * (psd[dc - 2] + psd[dc + 2])
    return bridged


def _quiet_reference(reference_iq: np.ndarray | None, active_power: float) -> np.ndarray | None:
    if reference_iq is None:
        return None
    return quiet_reference(reference_iq, dbfs(active_power / _QUIET_REFERENCE_RATIO), NFFT)


def _self_floor(psd: np.ndarray, frames: int) -> np.ndarray:
    floor = self_floor(psd, frames, _SELF_FLOOR_PERCENTILE)
    deep = self_floor(psd, frames, _DEEP_FLOOR_PERCENTILE)
    return np.full(len(psd), deep if floor > _CROWDED_RATIO * deep else floor)


def _regions(
    psd: np.ndarray, floor: np.ndarray, sample_rate: float, reliable: bool
) -> tuple[SpectralRegion, ...]:
    """Occupied regions of `psd` above `floor`, strongest first."""
    floor = np.maximum(floor, 1e-30)  # all-zero input: no regions, no 0/0
    bin_hz = sample_rate / len(psd)
    dc = len(psd) // 2
    excess = np.clip(psd - floor, 0.0, None)
    regions = []
    for bins in circular_regions(psd > floor * _REGION_THRESHOLD, _MERGE_GAP_BINS):
        # Continuous frequencies: a region wrapping the edge runs past +fs/2.
        freqs = (int(bins[0]) + np.arange(len(bins)) - dc) * bin_hz
        region_excess = excess[bins]
        power = float(region_excess.sum())
        low, high = occupied_span(region_excess)
        obw = (high - low + 1) * bin_hz
        center = float(np.sum(freqs * region_excess) / power)
        wraps = bool(np.isin([0, len(psd) - 1], bins).all())
        regions.append(
            SpectralRegion(
                start_offset_hz=float(freqs[0] - bin_hz / 2),
                end_offset_hz=float(freqs[-1] + bin_hz / 2),
                center_offset_hz=(center + sample_rate / 2) % sample_rate - sample_rate / 2,
                obw_hz=obw,
                excess_power=power,
                snr_db=float(10 * np.log10(power / float(floor[bins].sum()))),
                bandwidth_reliable=reliable and bandwidth_is_reliable(obw, sample_rate, wraps),
                noise_per_bin=float(floor[bins].mean()),
            )
        )
    return tuple(sorted(regions, key=lambda region: region.excess_power, reverse=True))


def segment_spectrum(
    iq: np.ndarray, sample_rate: float, reference_iq: np.ndarray | None = None
) -> Segmentation:
    """Split the capture's burst-time spectrum into occupied regions, and
    find the emitters already on in `reference_iq` (the snippet's
    pre-trigger samples, if any)."""
    if len(iq) < NFFT:
        raise ValueError(f"Need at least {NFFT} samples, got {len(iq)}")
    powers = frame_powers(iq, NFFT)
    active = active_frame_mask(powers)
    psd = _bridge_dc(welch_psd(iq, NFFT, active))
    quiet = _quiet_reference(reference_iq, float(np.mean(powers[active])))
    if quiet is None:
        regions = _regions(psd, _self_floor(psd, int(active.sum())), sample_rate, reliable=False)
        before: tuple[SpectralRegion, ...] = ()
    else:
        quiet_frames = len(quiet) // NFFT
        reference_psd = _bridge_dc(welch_psd(quiet, NFFT, np.ones(quiet_frames, dtype=bool)))
        floor = np.maximum(np.fft.fftshift(noise_floor_psd(quiet, NFFT)) * psd_scale(NFFT), reference_psd)
        regions = _regions(psd, floor, sample_rate, reliable=True)
        before = _regions(reference_psd, local_floor(reference_psd, quiet_frames), sample_rate, reliable=True)
    return Segmentation(
        regions=regions,
        before_trigger=before,
        floor_source="self" if quiet is None else "pre_trigger",
        sample_rate=sample_rate,
        active_frames=int(active.sum()),
        total_frames=len(powers),
    )
```

- [ ] **Step 4: Run the tests to verify they pass, step 4's included**

<!-- check: t2_green -->
Run: `python -m pytest tests/dsp/test_segmentation.py -q`
Expected: PASS, `37 passed`.

<!-- check: t2_spectral -->
Run: `python -m pytest tests/dsp/test_spectral.py -q`
Expected: `exit 0`: step 4's spectral tests, plus the four new reliability cases.

<!-- check: t2_capture -->
Run: `python -m pytest tests/capture/unknown -q`
Expected: `exit 0`: step 4's capture tests still pass on the shared floor.


- [ ] **Step 5: Prove both floors matter, then revert**

Temporarily put back the median self floor the review flagged:

<!-- edit: dsp/segmentation.py -->
Replace

```python
    return np.full(len(psd), deep if floor > _CROWDED_RATIO * deep else floor)
```

with

```python
    return np.full(len(psd), float(np.median(psd)))
```
<!-- check: t2_mutant -->
Run: `python -m pytest tests/dsp/test_segmentation.py -q`
Expected: FAIL, `8 failed, 29 passed`: every self-floor wideband, crowded and two-wideband case, and `test_an_always_on_emitter_is_the_primary_from_the_self_floor`.


Revert it:

<!-- edit: dsp/segmentation.py -->
Replace

```python
    return np.full(len(psd), float(np.median(psd)))
```

with

```python
    return np.full(len(psd), deep if floor > _CROWDED_RATIO * deep else floor)
```
<!-- check: t2_reverted -->
Run: `python -m pytest tests/dsp/test_segmentation.py -q`
Expected: PASS, `37 passed`.


Then make the context floor a lower envelope (a 20th percentile) instead of the median:

<!-- edit: dsp/segmentation.py -->
Replace

```python
_LOCAL_PERCENTILE = 50.0
```

with

```python
_LOCAL_PERCENTILE = 20.0
```
<!-- check: t2_local_mutant -->
Run: `python -m pytest tests/dsp/test_segmentation.py -q`
Expected: FAIL, `1 failed, 36 passed`: `test_a_steep_roll_off_is_not_read_as_context`.


Revert it:

<!-- edit: dsp/segmentation.py -->
Replace

```python
_LOCAL_PERCENTILE = 20.0
```

with

```python
_LOCAL_PERCENTILE = 50.0
```
<!-- check: t2_local_reverted -->
Run: `python -m pytest tests/dsp/test_segmentation.py -q`
Expected: PASS, `37 passed`.


- [ ] **Step 6: Commit**

<!-- run -->
```bash
git add dsp/spectral.py dsp/synthetic.py dsp/segmentation.py tests/dsp/test_segmentation.py tests/dsp/test_spectral.py
git commit -m "feat: add circular spectral segmentation on step 4's noise reference"
```

---

### Task 3: Region features

**Files:**
- Create: `dsp/features.py`
- Modify: `dsp/README.md` (rewrite), `README.md` (the `dsp/` layout line)
- Test: `tests/dsp/test_features.py`

**Interfaces:**
- Consumes: from Task 2, `dsp.segmentation`'s `BATCH_SAMPLES`, `NFFT`, `Segmentation`, `SpectralRegion` (its `noise_per_bin`),
  `active_frame_mask`, `frame_powers`, `occupied_span`, `runs` and `welch_psd`; and
  `dsp.synthetic` (tests only).
- Produces:
  - `CHANNEL_BLOCK = 1 << 16`;
  - `RegionFeatures(center_offset_hz, obw_hz, papr_db, duty_cycle, burst_count, mean_burst_s, spectral_flatness, symbol_rate_hz, snr_db, analysis_rate_hz, bandwidth_reliable)`,
    frozen. `mean_burst_s` and `symbol_rate_hz` may be `None`.
  - `channelize(iq, sample_rate, center_offset_hz, passband_hz) -> (complex64 baseband, rate, noise_bandwidth_hz)`.
  - `region_features(iq, segmentation, region) -> RegionFeatures`.

Design decisions, each measured:
- **Channelizer.** It works by FFT-bin selection in 65536-sample blocks with 50% overlap,
  keeping the middle half of each block.
  - **Memory:** bounded by one block.
  - **Phase:** a `(-1)^(shift·block)` rotation keeps the mixer phase continuous.
  - **Output rate:** ≥ 4× the passband, so |x|² cannot alias.
  - **Skirt:** a raised cosine over 25% of the half passband. A wider skirt let a
    neighbor through.
  - **Precision:** each block is transformed in complex128 (1 MiB), and the output is
    complex64. A complex64 FFT of this length left round-off spurs that read as
    symbol-rate lines on 14 of 100 clean tones.
- **Fine OBW.** It is the 99% span of the fine PSD, minus the *coarse* noise density
  shaped by the filter. The fine PSD's own median sits below the in-passband noise.
- **Symbol rate.** It is the most locally prominent line (10 dB over the median of
  ±32 bins) in a ≥ 8-frame Welch average of |x|².
  - The frames come only from inside on-runs, with each frame's mean removed.
  - It is gated on region SNR ≥ 13 dB.
- **Duty cycle.** The fraction of frames (≥ 32 samples, about 100 µs) that are 6 dB above
  the region's noise.

- [ ] **Step 1: Write the failing tests**

<!-- write: tests/dsp/test_features.py -->
```python
# tests/dsp/test_features.py
"""Features on synthetic IQ with known answers. Every tolerance was set
from a 20-seed sweep (worst case in the comment), with headroom."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dsp import synthetic
from dsp.features import channelize, region_features
from dsp.segmentation import BATCH_SAMPLES, segment_spectrum

FS = 1e6
N = 1 << 18  # 0.262144 s
DURATION = N / FS
NOISE = 1e-5  # -50 dBFS across the band


def _features(iq, fs=FS):
    segmentation = segment_spectrum(iq, fs)
    return region_features(iq, segmentation, segmentation.primary)


def _noise_for_snr(snr_db: float, signal_power: float, signal_bw_hz: float, fs: float = FS) -> float:
    """Total noise power giving `snr_db` inside `signal_bw_hz`."""
    return signal_power / 10 ** (snr_db / 10) * fs / signal_bw_hz


def test_tone():
    rng = np.random.default_rng(10)
    f = _features(synthetic.tone(N, FS, 123_456.0, 1e-3) + synthetic.noise(rng, N, NOISE))
    assert f.center_offset_hz == pytest.approx(123_456.0, abs=20)  # sweep: <= 3.2 Hz
    assert f.obw_hz < 400  # 3 fine bins of 61 Hz
    assert f.symbol_rate_hz is None  # constant envelope
    assert f.spectral_flatness < 0.01  # sweep: <= 0.0003
    assert f.papr_db < 1.0  # sweep: 0.21-0.25 dB
    assert (f.duty_cycle, f.burst_count) == (1.0, 1)


def test_band_limited_noise():
    rng = np.random.default_rng(11)
    f = _features(synthetic.band_limited(rng, N, FS, 50e3, -200e3, 1e-3) + synthetic.noise(rng, N, NOISE))
    assert f.obw_hz == pytest.approx(50e3, rel=0.03)  # sweep: -0.4%..+0.6%
    assert f.center_offset_hz == pytest.approx(-200e3, abs=1_000)  # sweep: <= 388 Hz
    assert f.symbol_rate_hz is None
    assert f.spectral_flatness > 0.95  # sweep: >= 0.987
    assert 7.5 < f.papr_db < 9.5  # Gaussian: sweep 8.2-8.5 dB at the 99.9th percentile


@pytest.mark.parametrize(
    "symbol_rate, offset",
    [(250e3, 100e3), (125e3, 150e3), (50e3, -150e3)],  # 4, 8 and 20 samples/symbol
)
def test_rrc_bpsk_symbol_rate(symbol_rate, offset):
    """RRC (beta 0.35) BPSK has a non-constant envelope, so |x|^2 carries a
    line at the symbol rate. 20 dB SNR inside the signal's bandwidth."""
    rng = np.random.default_rng(12)
    noise = _noise_for_snr(20.0, 1e-3, 1.35 * symbol_rate)
    f = _features(synthetic.rrc_bpsk(rng, N, FS, symbol_rate, 0.35, offset, 1e-3) + synthetic.noise(rng, N, noise))
    assert f.symbol_rate_hz == pytest.approx(symbol_rate, rel=1e-3)  # sweep: <= 4e-5
    assert f.obw_hz == pytest.approx(1.17 * symbol_rate, rel=0.06)  # 99% OBW: 1.16-1.22 x Rs
    assert 3.0 < f.papr_db < 5.0  # sweep: 3.9-4.2 dB


def test_symbol_rate_is_withheld_below_13_db_snr():
    """Measured: at ~10 dB in-band SNR the line is lost and spurious ones
    can win (3/20 seeds gave wrong rates), so it is not estimated at all."""
    rng = np.random.default_rng(13)
    noise = _noise_for_snr(10.0, 1e-3, 1.35 * 50e3)
    f = _features(synthetic.rrc_bpsk(rng, N, FS, 50e3, 0.35, -150e3, 1e-3) + synthetic.noise(rng, N, noise))
    assert f.snr_db < 13.0
    assert f.symbol_rate_hz is None


def test_ook_duty_cycle_and_bursts():
    """Ten 5 ms carrier bursts every 20 ms: duty 0.05 s / 0.262 s."""
    rng = np.random.default_rng(14)
    bursts = [(0.01 + 0.02 * k, 0.005) for k in range(10)]
    iq = synthetic.gate(synthetic.tone(N, FS, 100e3, 1e-3), FS, bursts) + synthetic.noise(rng, N, NOISE)
    f = _features(iq)
    assert f.duty_cycle == pytest.approx(0.05 / DURATION, abs=0.01)  # sweep: +0.0046
    assert f.burst_count == 10
    assert f.mean_burst_s == pytest.approx(0.005, rel=0.05)  # sweep: +2.4% (frame quantization)
    assert f.symbol_rate_hz is None  # a gated tone: no line inside the bursts


def test_gated_rrc_symbol_rate_is_measured_inside_bursts():
    rng = np.random.default_rng(15)
    bursts = [(0.01 + 0.02 * k, 0.005) for k in range(10)]
    iq = synthetic.gate(synthetic.rrc_bpsk(rng, N, FS, 50e3, 0.35, -100e3, 1e-3), FS, bursts)
    f = _features(iq + synthetic.noise(rng, N, NOISE))
    assert f.symbol_rate_hz == pytest.approx(50e3, rel=1e-3)
    assert f.burst_count == 10


def test_two_emitters_features_belong_to_the_primary_only():
    """The continuous carrier must not leak into the burst's duty cycle or
    bandwidth: the primary is channelized before measuring."""
    rng = np.random.default_rng(16)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 50e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    iq = synthetic.tone(N, FS, -250e3, 1e-4) + burst + synthetic.noise(rng, N, NOISE)
    f = _features(iq)
    assert f.center_offset_hz == pytest.approx(200e3, abs=1_000)  # sweep: <= 412 Hz
    assert f.obw_hz == pytest.approx(50e3, rel=0.03)
    assert f.duty_cycle == pytest.approx(0.1 / DURATION, abs=0.01)  # sweep: <= 0.0009
    assert (f.burst_count, f.mean_burst_s) == (1, pytest.approx(0.1, rel=0.02))


def test_narrowband_obw_is_remeasured_at_fine_resolution():
    """At 2 MS/s a coarse bin is 1953 Hz, so a 5 kHz signal spans ~3 bins
    (coarse OBW +95%). The decimated fine PSD recovers it."""
    fs, n = 2e6, 1 << 19
    rng = np.random.default_rng(17)
    iq = synthetic.band_limited(rng, n, fs, 5e3, 300e3, 1e-3) + synthetic.noise(rng, n, NOISE)
    segmentation = segment_spectrum(iq, fs)
    assert segmentation.primary.obw_hz > 1.5 * 5e3
    f = region_features(iq, segmentation, segmentation.primary)
    assert f.obw_hz == pytest.approx(5e3, rel=0.06)  # sweep: +0.1%..+2.5%
    assert f.center_offset_hz == pytest.approx(300e3, abs=300)  # sweep: <= 131 Hz


def test_channelize_preserves_power_alignment_and_length():
    rng = np.random.default_rng(18)
    iq = synthetic.band_limited(rng, N, FS, 20e3, 150e3, 1e-3)
    y, rate, noise_bandwidth = channelize(iq, FS, 150e3, 30e3)
    assert rate == FS * len(y) / N
    assert np.mean(np.abs(y) ** 2) == pytest.approx(1e-3, rel=0.02)
    assert 30e3 < noise_bandwidth < 30e3 * 1.25


def test_ffts_run_in_bounded_batches_without_promotion(monkeypatch):
    """Same memory discipline as dsp.spectral: every FFT input fits in one
    1 MiB batch whatever the capture length, and only the channelizer's
    fixed-size blocks are complex128 (float32 spurs there read as symbol-rate
    lines); everything else stays complex64/float32."""
    rng = np.random.default_rng(19)
    bursts = [(0.01, 0.2)]
    iq = synthetic.gate(synthetic.rrc_bpsk(rng, N, FS, 50e3, 0.35, -100e3, 1e-3), FS, bursts)
    iq = iq + synthetic.noise(rng, N, NOISE)
    seen = []
    for name in ("fft", "ifft", "rfft"):
        original = getattr(np.fft, name)

        def spy(a, *args, _original=original, **kwargs):
            array = np.asarray(a)
            seen.append((sys._getframe(1).f_code.co_name, array.dtype, array.nbytes))
            return _original(a, *args, **kwargs)

        monkeypatch.setattr(np.fft, name, spy)
    f = _features(iq)
    assert f.symbol_rate_hz is not None  # every FFT path ran
    assert {caller for caller, _, _ in seen} == {"welch_psd", "channelize", "_symbol_rate"}
    assert max(nbytes for _, _, nbytes in seen) <= BATCH_SAMPLES * 8
    assert {dtype for caller, dtype, _ in seen if caller != "channelize"} <= {
        np.dtype(np.complex64),
        np.dtype(np.float32),
    }


def test_feature_modules_import_nothing_from_capture_or_gnuradio():
    probe = (
        "import sys, dsp.features, dsp.segmentation, dsp.synthetic; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('capture', 'gnuradio', 'scipy')); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[2], check=True)
```

- [ ] **Step 2: Run them to verify they fail**

<!-- check: t3_red -->
Run: `python -m pytest tests/dsp/test_features.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'dsp.features'`.


- [ ] **Step 3: Implement**

<!-- write: dsp/features.py -->
```python
# dsp/features.py
"""Time-domain and fine-spectral features of one occupied region (numpy only).

The region (dsp.segmentation) is first channelized: shifted to baseband,
FFT-mask filtered to its band plus a margin, and decimated. Every feature is
then measured on that narrowband signal alone, so a second emitter elsewhere
in the capture cannot leak into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dsp.segmentation import (
    BATCH_SAMPLES,
    NFFT,
    Segmentation,
    SpectralRegion,
    active_frame_mask,
    frame_powers,
    occupied_span,
    runs,
    welch_psd,
)

CHANNEL_BLOCK = 1 << 16  # channelizer FFT block; 50% overlap, middle half kept
_OVERSAMPLE = 4.0  # decimated rate >= 4x the region width: |x|^2 cannot alias
_MARGIN = 1.5  # passband = region width x 1.5, at least a few coarse bins
_ROLLOFF = 0.25  # filter skirt, as a fraction of the half passband
_MIN_PASSBAND_BINS = 4
_FINE_NFFT = 1024
_ON_THRESHOLD = 4.0  # duty cycle counts frames 6 dB above the region's noise
_MIN_FRAME = 32  # samples per duty-cycle frame (power estimate within ~18%)
_FRAME_SECONDS = 100e-6
_PAPR_PERCENTILE = 99.9
_MIN_FLATNESS_BINS = 16  # a tone's 3-bin span would read as flat
_ENVELOPE_NFFT = 4096
_MIN_ENVELOPE_NFFT = 256  # shorter on-runs carry no usable line
_MIN_ENVELOPE_FRAMES = 8
_LINE_THRESHOLD = 10.0  # symbol-rate line must stand 10 dB above the median
_MIN_LINE_BIN = 8  # skip the envelope's DC skirt
_LINE_WINDOW = 32  # bins either side that define the local continuum
# Measured: below ~12 dB in-region SNR the line is lost and spurious ones win.
_MIN_LINE_SNR_DB = 13.0


@dataclass(frozen=True)
class RegionFeatures:
    center_offset_hz: float  # refined on the decimated signal, from the tuned center
    obw_hz: float  # 99%-power occupied bandwidth at fine resolution
    papr_db: float  # 99.9th-percentile |x|^2 over the mean, in on-frames
    duty_cycle: float  # fraction of frames 6 dB above the region's noise
    burst_count: int
    mean_burst_s: float | None
    spectral_flatness: float  # Wiener entropy of the occupied bins, 0..1
    symbol_rate_hz: float | None  # strongest |x|^2 line, or None
    snr_db: float
    analysis_rate_hz: float  # decimated sample rate the features were measured at
    bandwidth_reliable: bool  # the region's own judgment (dsp.segmentation)


def channelize(
    iq: np.ndarray, sample_rate: float, center_offset_hz: float, passband_hz: float
) -> tuple[np.ndarray, float, float]:
    """Shift `center_offset_hz` to DC, keep `passband_hz` (raised-cosine
    edges), and decimate by FFT-bin selection, block by block with 50%
    overlap (the middle half of each block is kept, so circular-convolution
    wrap-around is discarded). Working memory is one block whatever the
    capture length. Each block is transformed in complex128 (1 MiB, the same
    as one complex64 batch elsewhere): measured, a float32 FFT of this length
    leaves structured round-off spurs at multiples of rate/16 that the
    symbol-rate detector reads as lines (14 of 100 clean tones). The output,
    and everything downstream, is complex64/float32. Returns (complex64 baseband aligned with `iq`, its sample
    rate, the filter's noise-equivalent bandwidth in Hz). Power is preserved:
    a signal inside the passband keeps its mean |x|^2."""
    block = CHANNEL_BLOCK
    kept = 1 << max(4, math.ceil(math.log2(block * min(1.0, _OVERSAMPLE * passband_hz / sample_rate))))
    out_rate = sample_rate * kept / block
    shift = round(center_offset_hz / sample_rate * block)
    bins = np.arange(-kept // 2, kept // 2)
    taper = _passband(bins * sample_rate / block, passband_hz).astype(np.float32)
    hop, quarter = block // 2, kept // 4
    pieces = []
    for index in range(math.ceil(len(iq) / hop)):
        # Block `index` is centred on input samples [index*hop, (index+1)*hop),
        # zero-padded where it runs off either end of the capture.
        start = index * hop - block // 4
        chunk = np.zeros(block, dtype=np.complex128)
        lo, hi = max(start, 0), min(start + block, len(iq))
        chunk[lo - start : hi - start] = iq[lo:hi]
        selected = np.fft.fft(chunk)[(bins + shift) % block] * taper
        # Bin selection restarts the mixer's phase at every block start:
        # rotate it back to one continuous oscillator (hop = block / 2).
        rotation = -1.0 if (shift * index) % 2 else 1.0
        y = np.fft.ifft(np.fft.ifftshift(selected)) * (kept / block) * rotation
        pieces.append(y[quarter : kept - quarter].astype(np.complex64, copy=False))
    baseband = np.concatenate(pieces)[: round(len(iq) * kept / block)]
    noise_bandwidth = float(np.sum(taper.astype(np.float64) ** 2)) * sample_rate / block
    return baseband, out_rate, noise_bandwidth


def _passband(freqs: np.ndarray, passband_hz: float) -> np.ndarray:
    """1 inside +-passband/2, raised-cosine roll-off to 0 over the next
    _ROLLOFF x passband/2. Narrow on purpose: a wide skirt would let a
    neighbouring emitter into the channel."""
    half = passband_hz / 2
    over = np.clip((np.abs(freqs) - half) / (half * _ROLLOFF), 0.0, 1.0)
    return 0.5 * (1 + np.cos(np.pi * over))


def region_features(iq: np.ndarray, segmentation: Segmentation, region: SpectralRegion) -> RegionFeatures:
    """Features of `region`, measured on its channelized baseband signal."""
    coarse_bin = segmentation.sample_rate / NFFT
    passband = max(region.end_offset_hz - region.start_offset_hz, _MIN_PASSBAND_BINS * coarse_bin) * _MARGIN
    y, rate, noise_bandwidth = channelize(iq, segmentation.sample_rate, region.center_offset_hz, passband)
    # Noise power the coarse floor puts through the channel filter.
    noise = region.noise_per_bin * noise_bandwidth / coarse_bin

    frame = max(_MIN_FRAME, round(rate * _FRAME_SECONDS))
    powers = frame_powers(y, frame)
    on = powers > noise * _ON_THRESHOLD
    bursts = [stop - start for start, stop in runs(on)]
    on_samples = np.repeat(on, frame)
    on_power = np.abs(y[: len(on_samples)][on_samples]) ** 2 if on.any() else np.abs(y) ** 2
    # On-runs as sample ranges, one frame trimmed off each end (edge transients).
    on_runs = [((start + 1) * frame, (stop - 1) * frame) for start, stop in runs(on) if stop - start > 2]

    fine_nfft = min(_FINE_NFFT, len(y))
    fine_frames = frame_powers(y, fine_nfft)
    psd = welch_psd(y, fine_nfft, active_frame_mask(fine_frames))
    freqs = (np.arange(fine_nfft) - fine_nfft // 2) * rate / fine_nfft
    # The coarse noise density, shaped by the channel filter, per fine bin.
    # (The fine PSD's own median would sit below the in-passband noise,
    # because the filter's roll-off attenuates the band edges.)
    fine_noise = (
        region.noise_per_bin * (rate / fine_nfft) / coarse_bin
        * _passband(freqs, passband) ** 2
    )
    excess = np.clip(psd - fine_noise, 0.0, None)
    low, high = occupied_span(excess) if excess.sum() > 0 else (0, fine_nfft - 1)
    middle, half = (low + high) // 2, max(high - low + 1, _MIN_FLATNESS_BINS) // 2
    occupied = psd[max(middle - half, 0) : middle + half + 1]

    return RegionFeatures(
        center_offset_hz=region.center_offset_hz
        + float(np.sum(freqs * excess) / excess.sum() if excess.sum() > 0 else 0.0),
        obw_hz=(high - low + 1) * rate / fine_nfft,
        papr_db=float(10 * np.log10(np.percentile(on_power, _PAPR_PERCENTILE) / np.mean(on_power))),
        duty_cycle=float(on.mean()),
        burst_count=len(bursts),
        mean_burst_s=float(np.mean(bursts) * frame / rate) if bursts else None,
        spectral_flatness=float(np.exp(np.mean(np.log(occupied))) / np.mean(occupied)),
        symbol_rate_hz=_symbol_rate(y, rate, on_runs) if region.snr_db >= _MIN_LINE_SNR_DB else None,
        snr_db=region.snr_db,
        analysis_rate_hz=rate,
        bandwidth_reliable=region.bandwidth_reliable,
    )


def _symbol_rate(y: np.ndarray, rate: float, on_runs: list[tuple[int, int]]) -> float | None:
    """Frequency of the most prominent line in the |y|^2 spectrum, or None.

    Envelope frames are taken only from inside on-runs (sample ranges), each
    with its own mean removed: splicing bursts together would put the splice
    pattern, not the modulation, into the spectrum. Prominence is measured
    against the local continuum (median of the _LINE_WINDOW bins either
    side), because any band-limited envelope has a smooth triangular
    continuum that a global median would mistake for a line. A constant
    envelope (tone, FM, rectangular PSK) legitimately has no line.
    """
    longest = max((stop - start for start, stop in on_runs), default=0)
    total = sum(stop - start for start, stop in on_runs)
    # Largest power of two that still fits a run and averages enough frames
    # to keep the continuum's own fluctuations far below the threshold.
    fit = min(longest, total // _MIN_ENVELOPE_FRAMES)
    if fit < _MIN_ENVELOPE_NFFT:
        return None
    nfft = min(_ENVELOPE_NFFT, 1 << int(math.log2(fit)))
    starts = [s for start, stop in on_runs for s in range(start, stop - nfft + 1, nfft)]
    window = np.hanning(nfft).astype(np.float32)
    step = max(1, BATCH_SAMPLES // nfft)
    spectrum = np.zeros(nfft // 2 + 1)
    for i in range(0, len(starts), step):
        frames = np.abs(np.stack([y[s : s + nfft] for s in starts[i : i + step]])) ** 2
        frames -= frames.mean(axis=1, keepdims=True)
        spectrum += np.sum(np.abs(np.fft.rfft(frames * window, axis=1)) ** 2, axis=0)
    spectrum /= len(starts)
    padded = np.pad(spectrum, _LINE_WINDOW, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * _LINE_WINDOW + 1)
    prominence = spectrum / np.maximum(np.median(windows, axis=1), 1e-300)
    # Skip the DC skirt, and the top edge, where the channel filter's
    # roll-off reflects into the envelope.
    prominence[:_MIN_LINE_BIN] = 0.0
    prominence[-_LINE_WINDOW:] = 0.0
    index = int(np.argmax(prominence))
    if prominence[index] < _LINE_THRESHOLD:
        return None
    a, b, c = np.log(spectrum[index - 1 : index + 2])  # parabolic peak interpolation
    return float((index + 0.5 * (a - c) / (a - 2 * b + c)) * rate / nfft)
```

Replace `dsp/README.md` entirely:

<!-- write: dsp/README.md -->
```markdown
# dsp

Shared, numpy-only signal-processing helpers. No I/O, no GNU Radio, no scipy, and no
imports from `capture/`.

- `spectral.py`: dBFS power statistics and occupied bandwidth (used by `capture/unknown`).
- `segmentation.py`: Welch PSD over active frames, split into occupied regions (Part 4).
  The primary, the burst that triggered capture, is measured against
  `spectral.noise_floor_psd` of the snippet's quiet pre-trigger frames
  (`spectral.quiet_reference`). Emitters already on are found in that reference against
  a local median floor, as context. Without a quiet reference, a percentile self floor
  is used and every region is flagged unreliable. The band is circular (a signal
  straddling +-fs/2 is one region), like `features.channelize`.
- `features.py`: channelize one region and measure it: fine OBW and center, duty cycle,
  bursts, PAPR, spectral flatness, symbol rate (Part 4).
- `synthetic.py`: deterministic test signals with known answers (tone, band-limited noise,
  colored receiver noise, RRC BPSK, on-off gating).

Memory discipline: everything works in complex64/float32 batches of about 1 MiB, whatever
the capture length. The one exception is the channelizer's fixed 65536-sample block,
which is transformed in complex128 (also 1 MiB): a float32 FFT of that length leaves
round-off spurs that read as symbol-rate lines.
```

In `README.md`:

<!-- edit: README.md -->
Replace

```markdown
- `dsp/` — shared numpy-only signal math (power, occupied bandwidth), no I/O; reused by `agent/`
```

with

```markdown
- `dsp/` — shared numpy-only signal math (power, bandwidth, spectral segmentation, region features), no I/O; reused by `agent/`
```

- [ ] **Step 4: Run the tests to verify they pass**

<!-- check: t3_green -->
Run: `python -m pytest tests/dsp/test_features.py -q`
Expected: PASS, `13 passed`, in about 1 s.

<!-- check: t3_dsp -->
Run: `python -m pytest tests/dsp -q`
Expected: `exit 0`: every dsp test, step 4's included.


- [ ] **Step 5: Commit**

<!-- run -->
```bash
git add dsp/features.py dsp/README.md README.md tests/dsp/test_features.py
git commit -m "feat: add channelized per-region signal features to dsp"
```

---

### Task 4: The database boundary SQL and its installer

**Files:**
- Create: `storage/sql/agent_boundary.sql`, `storage/agent_boundary.py`
- Modify: `storage/README.md` (rewrite), `pyproject.toml` (psycopg dependency,
  `sdr-agent-boundary` script, `storage` package data)
- Test: `tests/storage/test_agent_boundary.py`, which needs no database. Task 5 tests the
  SQL against real PostgreSQL.

**Interfaces:**
- Consumes: from step 4, `storage.db.make_engine` and `init_db`.
- Produces:
  - `storage.agent_boundary`:
    - `DEFAULT_AGENT_ROLE = "surveytool_agent"`, `DEFAULT_OWNER_ROLE = "surveytool_classifier_owner"`;
    - `render_agent_boundary_sql(agent_role, owner_role) -> str`, which raises `ValueError`
      unless both names match `^[a-z_][a-z0-9_]{0,62}$` and differ;
    - `install_agent_boundary(admin_engine, agent_role=..., owner_role=...)`;
    - `main(argv)`, exposed as `sdr-agent-boundary --database-url ADMIN_URL [--agent-role] [--owner-role]`.
      It creates the table if needed (`init_db`), then installs.
  - Database objects, which Task 5 consumes:
    - `public.agent_is_pending_unknown(text, json) -> boolean`;
    - the view `public.agent_pending_unknown(id integer, iq_snippet_path text, sample_rate double precision, center_freq double precision, peak_power double precision, snippet_duration_ms integer, snippet_rejected text, snippet_dropped text)`.
      It includes pending rows whose `iq_snippet_path` is NULL;
    - `public.classify_unknown(integer, text, text, double precision, text) -> void`, which
      raises SQLSTATE `22023` on bad arguments and `P0002` when the row is not, or no
      longer, a pending unknown row.

Why placeholders: tests need uniquely named roles per run, because roles are
cluster-global. The file therefore carries `{{agent_role}}` and `{{owner_role}}`, which are
substituted only after strict identifier validation. The database name goes through
`format('%I', current_database())` inside a `DO` block. The file runs in one transaction
on the raw DB-API cursor with no parameters, so neither SQLAlchemy's `:name` parsing nor
psycopg's `%` parsing touches it.

- [ ] **Step 1: Install the PostgreSQL driver (user site, never editable)**

```bash
pip install --user 'psycopg[binary]>=3.2'
python -c "import psycopg; print(psycopg.__version__, psycopg.pq.__impl__)"
```

Expected: `3.2` or newer, and `binary`. Verified with 3.3.4.

- [ ] **Step 2: Write the failing tests**

<!-- write: tests/storage/test_agent_boundary.py -->
```python
# tests/storage/test_agent_boundary.py
"""Installer checks that need no PostgreSQL (the boundary itself is
exercised for real in test_agent_boundary_pg.py)."""

import pytest
from sqlalchemy import create_engine

from storage.agent_boundary import (
    DEFAULT_AGENT_ROLE,
    DEFAULT_OWNER_ROLE,
    install_agent_boundary,
    main,
    render_agent_boundary_sql,
)


def test_renders_every_placeholder():
    sql = render_agent_boundary_sql("sdr_agent_x", "sdr_owner_x")
    assert "{{" not in sql and "}}" not in sql
    assert "CREATE ROLE sdr_agent_x NOLOGIN" in sql
    assert "OWNER TO sdr_owner_x" in sql
    assert "surveytool_agent" not in sql


def test_default_role_names():
    assert (DEFAULT_AGENT_ROLE, DEFAULT_OWNER_ROLE) == ("surveytool_agent", "surveytool_classifier_owner")


@pytest.mark.parametrize(
    "agent_role, owner_role",
    [
        ("Agent", "owner"),
        ("agent; DROP TABLE survey_records", "owner"),
        ("agent", "own'er"),
        ("a-b", "owner"),
        ("", "owner"),
        ("9agent", "owner"),
        ("a" * 64, "owner"),
        ("same", "same"),
    ],
)
def test_rejects_unsafe_or_clashing_role_names(agent_role, owner_role):
    with pytest.raises(ValueError):
        render_agent_boundary_sql(agent_role, owner_role)


def test_refuses_a_non_postgres_engine():
    engine = create_engine("sqlite:///:memory:")
    with pytest.raises(ValueError, match="PostgreSQL"):
        install_agent_boundary(engine)
    engine.dispose()


def test_cli_refuses_a_non_postgres_url():
    with pytest.raises(SystemExit, match="PostgreSQL"):
        main(["--database-url", "sqlite:///survey.db"])
```

- [ ] **Step 3: Run them to verify they fail**

<!-- check: t4_red -->
Run: `python -m pytest tests/storage/test_agent_boundary.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'storage.agent_boundary'`.


- [ ] **Step 4: Implement**

<!-- write: storage/sql/agent_boundary.sql -->
```sql
-- storage/sql/agent_boundary.sql
--
-- The Part 4 agent's database boundary. Idempotent: safe to apply again.
-- Applied only by storage.agent_boundary (an admin CLI), as one transaction,
-- never at ingest or agent startup. {{agent_role}} and {{owner_role}} are
-- substituted by storage.agent_boundary.render_agent_boundary_sql after
-- validating both against ^[a-z_][a-z0-9_]{0,62}$.
--
-- The agent role gets NO privilege on public.survey_records. It reads the
-- security_barrier view public.agent_pending_unknown and writes only through
-- public.classify_unknown, a SECURITY DEFINER function. Both are owned by a
-- NOLOGIN owner role that holds exactly SELECT and UPDATE (metadata) on the
-- table, so a superuser is never the definer.

SET LOCAL search_path = pg_catalog, pg_temp;

-- Roles. Operators create the login role themselves:
--   CREATE ROLE <login> LOGIN PASSWORD '<from your secret store>' IN ROLE {{agent_role}};
DO $roles$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '{{agent_role}}') THEN
        CREATE ROLE {{agent_role}} NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '{{owner_role}}') THEN
        CREATE ROLE {{owner_role}} NOLOGIN;
    END IF;
END
$roles$;
ALTER ROLE {{agent_role}} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE {{owner_role}} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

-- Revokes. CREATE on public is revoked explicitly: clusters upgraded from
-- before PostgreSQL 15 keep the old PUBLIC CREATE default. TEMPORARY is
-- revoked so no role can define a leaky pg_temp function to probe the view.
REVOKE ALL ON TABLE public.survey_records FROM PUBLIC;
REVOKE ALL ON TABLE public.survey_records FROM {{agent_role}};
REVOKE ALL ON TABLE public.survey_records FROM {{owner_role}};
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
DO $temp$
BEGIN
    EXECUTE pg_catalog.format('REVOKE TEMPORARY ON DATABASE %I FROM PUBLIC',
                              pg_catalog.current_database());
END
$temp$;

-- The owner role holds exactly what the view and classify_unknown need.
GRANT SELECT, UPDATE (metadata) ON TABLE public.survey_records TO {{owner_role}};

-- The pending predicate, defined once and used by both the view and
-- classify_unknown. A SQL-standard body is parsed now, so its operators are
-- bound to pg_catalog at creation instead of resolved through the caller's
-- search_path at run time; it is still inlined into the view's plan.
CREATE OR REPLACE FUNCTION public.agent_is_pending_unknown(p_modality text, p_metadata json)
    RETURNS boolean
    LANGUAGE sql
    IMMUTABLE
    RETURN p_modality = 'unknown'
       AND coalesce(p_metadata ->> 'classification_status', 'unclassified') = 'unclassified';
ALTER FUNCTION public.agent_is_pending_unknown(text, json) OWNER TO {{owner_role}};
REVOKE ALL ON FUNCTION public.agent_is_pending_unknown(text, json) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.agent_is_pending_unknown(text, json) TO {{agent_role}};

-- The agent's only read path: pending unknown rows, and only the columns the
-- agent needs (no location, no survey or operator ID, no other modality).
-- A pending row may have no snippet (iq_snippet_path NULL) when ingest
-- rejected it or capture dropped it; the two flags say which, so the agent
-- can close such rows out instead of leaving them pending forever.
-- security_barrier makes PostgreSQL apply this WHERE clause before any
-- non-leakproof condition a caller adds, so a caller's function can never see
-- a hidden row.
CREATE OR REPLACE VIEW public.agent_pending_unknown
    WITH (security_barrier = true) AS
SELECT r.id,
       r.metadata ->> 'iq_snippet_path' AS iq_snippet_path,
       (r.metadata ->> 'sample_rate')::double precision AS sample_rate,
       (r.identifier ->> 'center_freq')::double precision AS center_freq,
       (r.signal ->> 'peak_power')::double precision AS peak_power,
       (r.metadata ->> 'snippet_duration_ms')::integer AS snippet_duration_ms,
       r.metadata -> 'quality_flags' ->> 'snippet_rejected' AS snippet_rejected,
       r.metadata -> 'quality_flags' ->> 'snippet_dropped' AS snippet_dropped
  FROM public.survey_records AS r
 WHERE public.agent_is_pending_unknown(r.modality, r.metadata);
ALTER VIEW public.agent_pending_unknown OWNER TO {{owner_role}};
REVOKE ALL ON TABLE public.agent_pending_unknown FROM PUBLIC;
GRANT SELECT ON TABLE public.agent_pending_unknown TO {{agent_role}};

-- The agent's only write path. SECURITY DEFINER runs as {{owner_role}};
-- search_path pins pg_catalog first and pg_temp explicitly last (unlisted,
-- pg_temp is searched first). Every object reference is schema-qualified.
-- SQLSTATE 22023 (invalid_parameter_value): bad arguments.
-- SQLSTATE P0002 (no_data_found): the row is not, or no longer, a pending
-- unknown row -- e.g. a human tagged it while the LLM call was in flight.
CREATE OR REPLACE FUNCTION public.classify_unknown(
    p_record_id integer,
    p_status text,
    p_tag text,
    p_confidence double precision,
    p_reasoning text
)
    RETURNS void
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE
    v_id integer;
BEGIN
    IF p_status IS NULL OR p_status NOT IN ('auto_classified', 'needs_review') THEN
        RAISE EXCEPTION 'classify_unknown: invalid status %', p_status
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    IF p_confidence IS NULL OR NOT (p_confidence >= 0 AND p_confidence <= 1) THEN
        RAISE EXCEPTION 'classify_unknown: confidence % is outside [0, 1]', p_confidence
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    IF p_tag IS NULL THEN
        IF p_status <> 'needs_review' THEN
            RAISE EXCEPTION 'classify_unknown: a NULL tag requires status needs_review'
                USING ERRCODE = 'invalid_parameter_value';
        END IF;
    ELSIF p_tag !~ '^[a-z0-9][a-z0-9_.:-]{0,63}$' THEN
        RAISE EXCEPTION 'classify_unknown: invalid tag %', p_tag
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    IF p_reasoning IS NULL OR length(p_reasoning) > 4000 THEN
        RAISE EXCEPTION 'classify_unknown: reasoning must be non-NULL and at most 4000 characters'
            USING ERRCODE = 'invalid_parameter_value';
    END IF;

    -- Atomic: the pending check and the write are one statement. A concurrent
    -- writer's committed change is re-checked against the WHERE clause under
    -- READ COMMITTED, so a human tag always wins.
    UPDATE public.survey_records AS r
       SET metadata = (r.metadata::jsonb || jsonb_build_object(
               'classification_status', p_status,
               'tag', p_tag,
               'confidence', p_confidence,
               'reasoning', p_reasoning))::json
     WHERE r.id = p_record_id
       AND public.agent_is_pending_unknown(r.modality, r.metadata)
    RETURNING r.id INTO v_id;
    IF v_id IS NULL THEN
        RAISE EXCEPTION 'classify_unknown: record % is not a pending unknown record', p_record_id
            USING ERRCODE = 'no_data_found';
    END IF;
END
$fn$;
ALTER FUNCTION public.classify_unknown(integer, text, text, double precision, text)
    OWNER TO {{owner_role}};
REVOKE ALL ON FUNCTION public.classify_unknown(integer, text, text, double precision, text)
    FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.classify_unknown(integer, text, text, double precision, text)
    TO {{agent_role}};
```

<!-- write: storage/agent_boundary.py -->
```python
# storage/agent_boundary.py
"""Installs the Part 4 agent's database boundary (storage/sql/agent_boundary.sql).

Admin-only. Run it with a superuser URL through the `sdr-agent-boundary`
script. Ingest and the agent never call it: the agent connects as a role
that could not apply it, and checks at startup that it is in force.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.engine import make_url

from storage.db import init_db, make_engine

DEFAULT_AGENT_ROLE = "surveytool_agent"
DEFAULT_OWNER_ROLE = "surveytool_classifier_owner"

_SQL_PATH = Path(__file__).resolve().parent / "sql" / "agent_boundary.sql"
# Lower-case unquoted identifiers only, so a substituted name is safe both as
# an identifier and inside the file's '...' literals.
_ROLE_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")


def render_agent_boundary_sql(
    agent_role: str = DEFAULT_AGENT_ROLE, owner_role: str = DEFAULT_OWNER_ROLE
) -> str:
    """Return agent_boundary.sql with both role placeholders substituted."""
    for name in (agent_role, owner_role):
        if not _ROLE_NAME.fullmatch(name):
            raise ValueError(f"Invalid role name {name!r}: must match {_ROLE_NAME.pattern}")
    if agent_role == owner_role:
        raise ValueError("The agent role and the owner role must differ")
    return (
        _SQL_PATH.read_text()
        .replace("{{agent_role}}", agent_role)
        .replace("{{owner_role}}", owner_role)
    )


def install_agent_boundary(
    admin_engine: Engine,
    agent_role: str = DEFAULT_AGENT_ROLE,
    owner_role: str = DEFAULT_OWNER_ROLE,
) -> None:
    """Apply the boundary in one transaction. public.survey_records must exist."""
    if admin_engine.dialect.name != "postgresql":
        raise ValueError("The agent boundary needs PostgreSQL")
    sql = render_agent_boundary_sql(agent_role, owner_role)
    with admin_engine.begin() as conn:
        # Straight to the DB-API cursor with no parameters: SQLAlchemy's text()
        # would read ':word' as a bind parameter, and the driver only parses
        # '%' placeholders when parameters are passed, so the file's
        # format('%I', ...) reaches PostgreSQL untouched.
        conn.connection.driver_connection.cursor().execute(sql)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the survey_records table if needed, then install the "
        "Part 4 agent database boundary. Needs a superuser URL."
    )
    parser.add_argument("--database-url", required=True, help="Admin (superuser) PostgreSQL URL.")
    parser.add_argument("--agent-role", default=DEFAULT_AGENT_ROLE)
    parser.add_argument("--owner-role", default=DEFAULT_OWNER_ROLE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    url = make_url(args.database_url)
    if url.get_backend_name() != "postgresql":
        raise SystemExit("--database-url must be a PostgreSQL URL")
    engine = make_engine(url.set(drivername="postgresql+psycopg").render_as_string(hide_password=False))
    try:
        init_db(engine)
        install_agent_boundary(engine, args.agent_role, args.owner_role)
    finally:
        engine.dispose()
    print(f"Agent boundary installed: agent role {args.agent_role}, owner role {args.owner_role}")


if __name__ == "__main__":
    main()
```

Replace `storage/README.md` entirely:

<!-- write: storage/README.md -->
```markdown
# storage

Postgres/PostGIS models for the unified record schema (`models.py`, `db.py`,
`repository.py`), the local flat-file snippet store that ingest adopts IQ snippets into
(`snippet_store.py`), and the Part 4 agent's database boundary (`sql/agent_boundary.sql`,
installed by `agent_boundary.py` / `sdr-agent-boundary`).
```

In `pyproject.toml`, add the dependency, the admin script and the package data:

<!-- edit: pyproject.toml -->
Replace

```toml
    "sigmf>=1.13",  # capture/unknown snippet files (LGPL-3.0-or-later)
]
```

with

```toml
    "sigmf>=1.13",  # capture/unknown snippet files (LGPL-3.0-or-later)
    "psycopg[binary]>=3.2",  # agent/ and storage/agent_boundary.py: PostgreSQL only
]
```

<!-- edit: pyproject.toml -->
Replace

```toml
sdr-capture-unknown = "capture.unknown.service:main"
```

with

```toml
sdr-capture-unknown = "capture.unknown.service:main"
sdr-agent-boundary = "storage.agent_boundary:main"
```

<!-- edit: pyproject.toml -->
Replace

```toml
[tool.pytest.ini_options]
```

with

```toml
[tool.setuptools.package-data]
storage = ["sql/*.sql"]

[tool.pytest.ini_options]
```

- [ ] **Step 5: Run the tests to verify they pass**

<!-- check: t4_green -->
Run: `python -m pytest tests/storage/test_agent_boundary.py -q`
Expected: PASS, `12 passed`.


- [ ] **Step 6: Commit**

<!-- run -->
```bash
git add storage/sql/agent_boundary.sql storage/agent_boundary.py storage/README.md pyproject.toml tests/storage/test_agent_boundary.py
git commit -m "feat: add the Part 4 agent database boundary SQL and admin installer"
```

---

### Task 5: The agent's database gateway, tested on real PostgreSQL

**Files:**
- Create: `agent/__init__.py` (empty), `agent/db_gateway.py`
- Modify: `pyproject.toml` (register the `agent` package)
- Test: `tests/agent/test_db_gateway.py` (no database), and
  `tests/storage/test_agent_boundary_pg.py`, which runs on real PostgreSQL and skips
  without `SURVEYTOOL_TEST_PG_URL`. Do not create `tests/agent/__init__.py`.

**Interfaces:**
- Consumes:
  - from Task 1, `ClassificationStatus`;
  - from Task 4, the SQL objects and `install_agent_boundary`;
  - from step 4, `init_db`, `make_session_factory` and `save_record`.
- Produces, in `agent.db_gateway`:
  - `DEFAULT_AGENT_ROLE`;
  - `DEFAULT_STATEMENT_TIMEOUT_S = 30.0`;
  - exceptions, all `RuntimeError`:
    - `BoundaryViolation`;
    - `RecordNotPending` (P0002);
    - `SubmitRejected` (SQLSTATE class 22, which arrives as `psycopg.DataError`);
    - `SubmitTimedOut` (57014 or 55P03);
  - `PendingRecord(id, iq_snippet_path, sample_rate, center_freq, peak_power, snippet_duration_ms, snippet_rejected=None, snippet_dropped=None)`, frozen;
  - `AgentGateway(engine)`, with:
    - `.fetch_pending(after_id: int, limit: int) -> list[PendingRecord]`, lowest id first;
    - `.fetch_newest_with_snippet(after_id: int, limit: int) -> list[PendingRecord]`,
      newest first, rows with a path only. This is the startup probe;
    - `.submit_classification(record_id: int, status: ClassificationStatus, tag: str | None, confidence: float, reasoning: str)`.
      NUL characters are stripped from the reasoning, and errors are mapped to the
      exceptions above;
    - `.close()`;
  - `verify_boundary(engine, agent_role)`, the allowlist self-check (see deviation 1);
  - `connect_gateway(database_url: str, agent_role: str = DEFAULT_AGENT_ROLE, statement_timeout_s: float = DEFAULT_STATEMENT_TIMEOUT_S) -> AgentGateway`.
    It raises `ValueError` for a non-PostgreSQL URL and `BoundaryViolation` when the
    self-check fails, and always uses the `postgresql+psycopg` driver.

The real-PostgreSQL tests refuse, by name, each of the following:
- a NOINHERIT owner grant, and table or column grants;
- CREATEROLE, BYPASSRLS, REPLICATION and CREATEDB, on the login and on the agent role;
- membership of `pg_write_all_data`, `pg_read_all_data`, `pg_execute_server_program`,
  `pg_write_server_files`, `pg_read_server_files`, `pg_signal_backend` and `pg_maintain`
  (that last case skips before PostgreSQL 17);
- TEMPORARY reached through a helper role granted WITH INHERIT FALSE;
- direct TEMPORARY and CREATE on the database;
- CREATE on `public` and on another schema;
- an executable SECURITY DEFINER function.

- [ ] **Step 1: Write the failing tests**

<!-- write: tests/agent/test_db_gateway.py -->
```python
# tests/agent/test_db_gateway.py
"""Gateway checks that need no PostgreSQL (the self-check, view and
function are exercised for real in tests/storage/test_agent_boundary_pg.py)."""

import pytest

from agent.db_gateway import connect_gateway


@pytest.mark.parametrize("url", ["sqlite:///survey.db", "sqlite:///:memory:", "mysql://u:p@localhost/db"])
def test_refuses_non_postgres_urls(url):
    with pytest.raises(ValueError, match="PostgreSQL"):
        connect_gateway(url)
```

<!-- write: tests/storage/test_agent_boundary_pg.py -->
```python
# tests/storage/test_agent_boundary_pg.py
"""The agent's database boundary on real PostgreSQL (never SQLite).

Skipped unless SURVEYTOOL_TEST_PG_URL is set to a superuser URL on a
THROWAWAY cluster (never the compose dev volume). Each run creates its own
database and uniquely suffixed roles -- roles are cluster-global -- and drops
all of them afterwards.
"""

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.pool import NullPool

from agent.db_gateway import (
    BoundaryViolation,
    PendingRecord,
    RecordNotPending,
    SubmitRejected,
    SubmitTimedOut,
    connect_gateway,
)
from schema.records import (
    ClassificationStatus,
    Identifier,
    Metadata,
    Modality,
    Signal,
    UnifiedRecord,
)
from storage.agent_boundary import install_agent_boundary
from storage.db import init_db, make_session_factory
from storage.repository import save_record

ADMIN_URL = os.environ.get("SURVEYTOOL_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    not ADMIN_URL, reason="SURVEYTOOL_TEST_PG_URL (throwaway PostgreSQL admin URL) not set"
)

INSUFFICIENT_PRIVILEGE = "42501"
INVALID_PARAMETER = "22023"


@dataclass(frozen=True)
class Boundary:
    root_url: URL  # superuser, maintenance database
    admin_url: URL  # superuser, this run's database
    database: str
    agent_url: URL  # a login role IN ROLE agent_role, nothing else
    agent_role: str
    owner_role: str
    suffix: str
    password: str


def _engine(url: URL, **kwargs) -> Engine:
    return create_engine(url, poolclass=NullPool, **kwargs)


def _url(url: URL) -> str:
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def boundary():
    suffix = secrets.token_hex(4)
    database = f"sdr_agent_test_{suffix}"
    agent_role, owner_role = f"sdr_agent_{suffix}", f"sdr_owner_{suffix}"
    login = f"sdr_login_{suffix}"
    password = secrets.token_hex(16)
    root_url = make_url(ADMIN_URL).set(drivername="postgresql+psycopg")
    admin_url = root_url.set(database=database)
    root = _engine(root_url, isolation_level="AUTOCOMMIT")
    with root.connect() as conn:
        conn.exec_driver_sql(f"CREATE DATABASE {database}")
    try:
        admin = _engine(admin_url)
        init_db(admin)
        install_agent_boundary(admin, agent_role, owner_role)
        install_agent_boundary(admin, agent_role, owner_role)  # idempotent
        admin.dispose()
        with root.connect() as conn:
            conn.exec_driver_sql(
                f"CREATE ROLE {login} LOGIN PASSWORD '{password}' IN ROLE {agent_role}"
            )
        yield Boundary(
            root_url=root_url,
            admin_url=admin_url,
            database=database,
            agent_url=admin_url.set(username=login, password=password),
            agent_role=agent_role,
            owner_role=owner_role,
            suffix=suffix,
            password=password,
        )
    finally:
        with root.connect() as conn:
            conn.exec_driver_sql(f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
            leftovers = conn.execute(
                text("SELECT rolname FROM pg_catalog.pg_roles WHERE rolname LIKE :pattern"),
                {"pattern": f"sdr\\_%\\_{suffix}"},
            ).scalars().all()
            for role in sorted(leftovers, key=lambda name: name in (agent_role, owner_role)):
                conn.exec_driver_sql(f"DROP ROLE {role}")
            remaining = conn.execute(
                text("SELECT count(*) FROM pg_catalog.pg_roles WHERE rolname LIKE :pattern"),
                {"pattern": f"sdr\\_%\\_{suffix}"},
            ).scalar_one()
        root.dispose()
        assert remaining == 0


def _insert(
    boundary: Boundary,
    modality: Modality,
    status: ClassificationStatus | None,
    path: str | None = "/srv/snippets/a.sigmf-data",
    sample_rate: float = 100_000.0,
    flags: dict | None = None,
) -> int:
    record = UnifiedRecord(
        timestamp=datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc),
        lat=47.6062,
        lon=-122.3321,
        survey_id="survey-secret",
        operator_id="operator-secret",
        modality=modality,
        identifier=Identifier(center_freq=915e6, bandwidth_estimate=20_000.0),
        signal=Signal(rssi=-20.0, snr=20.0, peak_power=-17.5),
        metadata=Metadata(
            quality_flags={"power_units": "dBFS", **(flags or {})},
            iq_snippet_path=path,
            snippet_duration_ms=1000,
            sample_rate=sample_rate,
            classification_status=status,
            tag="human-tag" if status is ClassificationStatus.MANUALLY_TAGGED else None,
        ),
    )
    admin = _engine(boundary.admin_url)
    try:
        with make_session_factory(admin)() as session:
            return save_record(session, record).id
    finally:
        admin.dispose()


def _admin_scalar(boundary: Boundary, sql: str, **params):
    admin = _engine(boundary.admin_url)
    try:
        with admin.begin() as conn:
            return conn.execute(text(sql), params).scalar()
    finally:
        admin.dispose()


def _metadata(boundary: Boundary, record_id: int) -> dict:
    return json.loads(
        _admin_scalar(
            boundary, "SELECT metadata::text FROM survey_records WHERE id = :id", id=record_id
        )
    )


def _watermark(boundary: Boundary) -> int:
    return _admin_scalar(boundary, "SELECT coalesce(max(id), 0) FROM survey_records")


def _sqlstate(excinfo) -> str | None:
    return getattr(excinfo.value.orig, "sqlstate", None)


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * FROM public.survey_records",
        "INSERT INTO public.survey_records (timestamp, lat, lon, survey_id, operator_id, "
        "modality, identifier, signal, metadata) VALUES (now(), 0, 0, 's', 'o', 'unknown', "
        "'{}', '{}', '{}')",
        "UPDATE public.survey_records SET metadata = '{}'",
        "DELETE FROM public.survey_records",
        "TRUNCATE public.survey_records",
    ],
)
def test_agent_role_has_no_direct_access_to_survey_records(boundary, statement):
    agent = _engine(boundary.agent_url)
    try:
        with pytest.raises(DBAPIError) as excinfo, agent.begin() as conn:
            conn.execute(text(statement))
        assert _sqlstate(excinfo) == INSUFFICIENT_PRIVILEGE
    finally:
        agent.dispose()


def test_view_shows_only_pending_unknown_rows(boundary):
    watermark = _watermark(boundary)
    pending = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED)
    no_status = _insert(boundary, Modality.UNKNOWN, None)
    rejected = _insert(boundary, Modality.UNKNOWN, None, path=None, flags={"snippet_rejected": "bad_size"})
    dropped = _insert(boundary, Modality.UNKNOWN, None, path=None, flags={"snippet_dropped": "low_disk"})
    _insert(boundary, Modality.WIFI, None)
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED)
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.AUTO_CLASSIFIED)
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.NEEDS_REVIEW)

    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        rows = gateway.fetch_pending(after_id=watermark, limit=100)
        assert [row.id for row in rows] == [pending, no_status, rejected, dropped]
        assert rows[0] == PendingRecord(
            id=pending,
            iq_snippet_path="/srv/snippets/a.sigmf-data",
            sample_rate=100_000.0,
            center_freq=915e6,
            peak_power=-17.5,
            snippet_duration_ms=1000,
        )
        # Snippet-less rows stay visible, flagged, so the agent can close them out.
        assert (rows[2].iq_snippet_path, rows[2].snippet_rejected, rows[2].snippet_dropped) == (None, "bad_size", None)
        assert (rows[3].iq_snippet_path, rows[3].snippet_rejected, rows[3].snippet_dropped) == (None, None, "low_disk")
        assert [row.id for row in gateway.fetch_pending(after_id=rejected, limit=100)] == [dropped]
        assert [row.id for row in gateway.fetch_pending(after_id=watermark, limit=1)] == [pending]
    finally:
        gateway.close()


def test_view_projects_only_the_agent_columns(boundary):
    agent = _engine(boundary.agent_url)
    try:
        with agent.connect() as conn:
            columns = list(conn.execute(text("SELECT * FROM public.agent_pending_unknown LIMIT 0")).keys())
        assert columns == [
            "id",
            "iq_snippet_path",
            "sample_rate",
            "center_freq",
            "peak_power",
            "snippet_duration_ms",
            "snippet_rejected",
            "snippet_dropped",
        ]
    finally:
        agent.dispose()


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE FUNCTION public.probe(t text) RETURNS boolean LANGUAGE sql RETURN true",
        "CREATE FUNCTION pg_temp.probe(t text) RETURNS boolean LANGUAGE sql RETURN true",
        "CREATE TEMPORARY TABLE probe (x integer)",
        "CREATE SCHEMA probe",
    ],
)
def test_agent_cannot_create_objects_to_probe_the_view(boundary, statement):
    agent = _engine(boundary.agent_url)
    try:
        with pytest.raises(DBAPIError) as excinfo, agent.begin() as conn:
            conn.execute(text(statement))
        assert _sqlstate(excinfo) == INSUFFICIENT_PRIVILEGE
    finally:
        agent.dispose()


def test_leaky_function_sees_only_rows_the_view_shows(boundary):
    """The agent cannot create a function (previous test), so an
    administrator plants the classic leaky probe for it: near-zero cost, so
    the planner would run it before the view's own filter, and it reports
    every value it is passed. The manually tagged unknown row survives the
    view's index condition (modality = 'unknown'), so without
    security_barrier its path is leaked here; with it, it never is."""
    watermark = _watermark(boundary)
    visible = _insert(boundary, Modality.UNKNOWN, None, path="/srv/snippets/visible.sigmf-data")
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED, path="HIDDEN-manual")
    _insert(boundary, Modality.WIFI, None, path="HIDDEN-wifi")
    leak = f"public.leak_{boundary.suffix}"
    admin = _engine(boundary.admin_url)
    agent = _engine(boundary.agent_url)
    try:
        with admin.begin() as conn:
            conn.exec_driver_sql(
                f"CREATE FUNCTION {leak}(t text) RETURNS boolean LANGUAGE plpgsql "
                "COST 0.0000001 AS $$BEGIN RAISE NOTICE USING MESSAGE = 'saw ' || t; "
                "RETURN true; END$$"
            )
            conn.exec_driver_sql(f"GRANT EXECUTE ON FUNCTION {leak}(text) TO {boundary.agent_role}")
        notices: list[str] = []
        with agent.connect() as conn:
            conn.connection.driver_connection.add_notice_handler(
                lambda diagnostic: notices.append(diagnostic.message_primary)
            )
            ids = conn.execute(
                text(
                    f"SELECT id FROM public.agent_pending_unknown "
                    f"WHERE id > :watermark AND {leak}(iq_snippet_path)"
                ),
                {"watermark": watermark},
            ).scalars().all()
        assert ids == [visible]
        assert notices == ["saw /srv/snippets/visible.sigmf-data"]
    finally:
        with admin.begin() as conn:
            conn.exec_driver_sql(f"DROP FUNCTION IF EXISTS {leak}(text)")
        admin.dispose()
        agent.dispose()


def test_classify_unknown_changes_only_the_four_classification_keys(boundary):
    record_id = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED)
    before = _metadata(boundary, record_id)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        gateway.submit_classification(
            record_id, ClassificationStatus.AUTO_CLASSIFIED, "ism_902_928:lora", 0.9, "Because."
        )
    finally:
        gateway.close()
    # Compared as parsed JSON: the jsonb round trip reorders keys.
    assert _metadata(boundary, record_id) == {
        **before,
        "classification_status": "auto_classified",
        "tag": "ism_902_928:lora",
        "confidence": 0.9,
        "reasoning": "Because.",
    }


def test_needs_review_may_carry_a_null_tag(boundary):
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        gateway.submit_classification(
            record_id, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "Model output failed validation."
        )
    finally:
        gateway.close()
    metadata = _metadata(boundary, record_id)
    assert metadata["classification_status"] == "needs_review"
    assert metadata["tag"] is None and metadata["confidence"] == 0.0


@pytest.mark.parametrize(
    "modality, status",
    [
        (Modality.WIFI, None),
        (Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED),
        (Modality.UNKNOWN, ClassificationStatus.AUTO_CLASSIFIED),
    ],
)
def test_classify_unknown_refuses_rows_that_are_not_pending_unknown(boundary, modality, status):
    record_id = _insert(boundary, modality, status)
    before = _metadata(boundary, record_id)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        with pytest.raises(RecordNotPending):
            gateway.submit_classification(
                record_id, ClassificationStatus.AUTO_CLASSIFIED, "x", 0.9, "r"
            )
    finally:
        gateway.close()
    assert _metadata(boundary, record_id) == before


@pytest.mark.parametrize(
    "status, tag, confidence, reasoning",
    [
        ("manually_tagged", "x", 0.5, "r"),
        ("unclassified", "x", 0.5, "r"),
        ("bogus", "x", 0.5, "r"),
        ("auto_classified", "x", 1.5, "r"),
        ("auto_classified", "x", -0.1, "r"),
        ("auto_classified", "x", float("nan"), "r"),
        ("auto_classified", "x", None, "r"),
        ("auto_classified", "Bad", 0.5, "r"),
        ("auto_classified", "abc\n", 0.5, "r"),
        ("auto_classified", "a" * 65, 0.5, "r"),
        ("auto_classified", "-leading-dash", 0.5, "r"),
        ("auto_classified", None, 0.5, "r"),
        ("needs_review", None, 0.5, "x" * 4001),
        ("needs_review", None, 0.5, None),
    ],
)
def test_classify_unknown_rejects_bad_arguments(boundary, status, tag, confidence, reasoning):
    """Called directly, bypassing the gateway's types: the function itself
    is the boundary."""
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    before = _metadata(boundary, record_id)
    agent = _engine(boundary.agent_url)
    try:
        with pytest.raises(DBAPIError) as excinfo, agent.begin() as conn:
            conn.execute(
                text(
                    "SELECT public.classify_unknown(CAST(:id AS integer), CAST(:status AS text), "
                    "CAST(:tag AS text), CAST(:confidence AS double precision), "
                    "CAST(:reasoning AS text))"
                ),
                {"id": record_id, "status": status, "tag": tag, "confidence": confidence, "reasoning": reasoning},
            )
        assert _sqlstate(excinfo) == INVALID_PARAMETER
    finally:
        agent.dispose()
    assert _metadata(boundary, record_id) == before


def test_human_tag_between_fetch_and_submit_wins(boundary):
    watermark = _watermark(boundary)
    record_id = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        assert [row.id for row in gateway.fetch_pending(watermark, 10)] == [record_id]
        _admin_scalar(
            boundary,
            "UPDATE survey_records SET metadata = (metadata::jsonb || "
            "'{\"classification_status\": \"manually_tagged\", \"tag\": \"human\"}')::json "
            "WHERE id = :id RETURNING id",
            id=record_id,
        )
        with pytest.raises(RecordNotPending):
            gateway.submit_classification(
                record_id, ClassificationStatus.AUTO_CLASSIFIED, "agent", 0.95, "r"
            )
    finally:
        gateway.close()
    metadata = _metadata(boundary, record_id)
    assert (metadata["classification_status"], metadata["tag"]) == ("manually_tagged", "human")


def test_human_tag_committed_while_agent_waits_on_the_row_lock_wins(boundary):
    """The real race: the human's UPDATE holds the row lock, the agent's
    classify_unknown blocks on it, then the human commits. Under READ
    COMMITTED PostgreSQL re-checks the agent's WHERE clause against the
    committed row, so the pending predicate fails and nothing is overwritten."""
    record_id = _insert(boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    admin = _engine(boundary.admin_url)
    outcome: list[BaseException | None] = []

    def submit() -> None:
        try:
            gateway.submit_classification(
                record_id, ClassificationStatus.AUTO_CLASSIFIED, "agent", 0.95, "r"
            )
            outcome.append(None)
        except BaseException as exc:  # handed to the main thread
            outcome.append(exc)

    try:
        with admin.connect() as human:
            human.execute(
                text(
                    "UPDATE survey_records SET metadata = (metadata::jsonb || "
                    "'{\"classification_status\": \"manually_tagged\", \"tag\": \"human\"}')::json "
                    "WHERE id = :id"
                ),
                {"id": record_id},
            )
            agent_thread = threading.Thread(target=submit, daemon=True)
            agent_thread.start()
            deadline = time.monotonic() + 10
            while not _admin_scalar(
                boundary,
                "SELECT count(*) FROM pg_catalog.pg_stat_activity "
                "WHERE datname = current_database() "
                "AND cardinality(pg_catalog.pg_blocking_pids(pid)) > 0",
            ):
                assert time.monotonic() < deadline, "agent never blocked on the row lock"
                time.sleep(0.01)
            human.commit()
        agent_thread.join(10)
        assert not agent_thread.is_alive()
    finally:
        gateway.close()
        admin.dispose()
    assert len(outcome) == 1 and isinstance(outcome[0], RecordNotPending)
    metadata = _metadata(boundary, record_id)
    assert (metadata["classification_status"], metadata["tag"]) == ("manually_tagged", "human")


def test_self_check_rejects_a_superuser_url(boundary):
    with pytest.raises(BoundaryViolation, match="not confined.*has SUPERUSER"):
        connect_gateway(_url(boundary.admin_url), boundary.agent_role)


# (setup statements run as the administrator, extra cleanup, expected problem).
# {login} is a fresh login role IN ROLE the agent role; every case must be refused.
BYPASSES = {
    "owner role without inherit": (["GRANT {owner_role} TO {login} WITH INHERIT FALSE"], [], "member of {owner_role}"),
    "column privilege": (["GRANT UPDATE (metadata) ON public.survey_records TO {login}"], [], "privilege on survey_records"),
    "table privilege": (["GRANT SELECT ON public.survey_records TO {login}"], [], "privilege on survey_records"),
    "createrole": (["ALTER ROLE {login} CREATEROLE"], [], "has CREATEROLE"),
    "bypassrls": (["ALTER ROLE {login} BYPASSRLS"], [], "has BYPASSRLS"),
    "replication": (["ALTER ROLE {login} REPLICATION"], [], "has REPLICATION"),
    "createdb": (["ALTER ROLE {login} CREATEDB"], [], "has CREATEDB"),
    "agent role createdb": (["ALTER ROLE {agent_role} CREATEDB"], ["ALTER ROLE {agent_role} NOCREATEDB"], "{agent_role} has CREATEDB"),
    "pg_write_all_data": (["GRANT pg_write_all_data TO {login}"], [], "member of pg_write_all_data"),
    "pg_read_all_data": (["GRANT pg_read_all_data TO {login}"], [], "member of pg_read_all_data"),
    "pg_execute_server_program": (["GRANT pg_execute_server_program TO {login}"], [], "member of pg_execute_server_program"),
    "pg_write_server_files": (["GRANT pg_write_server_files TO {login}"], [], "member of pg_write_server_files"),
    "pg_read_server_files": (["GRANT pg_read_server_files TO {login}"], [], "member of pg_read_server_files"),
    "pg_signal_backend": (["GRANT pg_signal_backend TO {login}"], [], "member of pg_signal_backend"),
    "pg_maintain": (["GRANT pg_maintain TO {login}"], [], "member of pg_maintain"),
    "temporary through a helper without inherit": (
        [
            "CREATE ROLE sdr_helper_{suffix} NOLOGIN",
            "GRANT TEMPORARY ON DATABASE {database} TO sdr_helper_{suffix}",
            "GRANT sdr_helper_{suffix} TO {login} WITH INHERIT FALSE",
        ],
        ["DROP OWNED BY sdr_helper_{suffix}", "DROP ROLE IF EXISTS sdr_helper_{suffix}"],
        "member of sdr_helper_{suffix}",
    ),
    "temporary": (["GRANT TEMPORARY ON DATABASE {database} TO {login}"], [], "CREATE or TEMPORARY on the database"),
    "create on the database": (["GRANT CREATE ON DATABASE {database} TO {login}"], [], "CREATE or TEMPORARY on the database"),
    "create on public": (["GRANT CREATE ON SCHEMA public TO {login}"], [], "can create objects in schema public"),
    "create on another schema": (
        ["CREATE SCHEMA sdr_extra_{suffix}", "GRANT CREATE ON SCHEMA sdr_extra_{suffix} TO {login}"],
        ["DROP SCHEMA IF EXISTS sdr_extra_{suffix} CASCADE"],
        "can create objects in schema sdr_extra_{suffix}",
    ),
    "security definer function": (
        ["CREATE FUNCTION public.sdr_definer_{suffix}() RETURNS integer LANGUAGE sql SECURITY DEFINER RETURN 1"],
        ["DROP FUNCTION IF EXISTS public.sdr_definer_{suffix}()"],
        "can execute SECURITY DEFINER function sdr_definer_{suffix}()",
    ),
}


@pytest.mark.parametrize("case", BYPASSES)
def test_self_check_refuses_a_login_that_is_not_confined(boundary, case):
    setup, cleanup, problem = BYPASSES[case]
    login = f"sdr_bypass_{boundary.suffix}"
    names = {
        "login": login,
        "owner_role": boundary.owner_role,
        "agent_role": boundary.agent_role,
        "database": boundary.database,
        "suffix": boundary.suffix,
    }
    root = _engine(boundary.root_url, isolation_level="AUTOCOMMIT")
    admin = _engine(boundary.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with root.connect() as conn:
            if case == "pg_maintain" and not conn.execute(
                text("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'pg_maintain'")
            ).first():
                pytest.skip("pg_maintain exists from PostgreSQL 17")
            conn.exec_driver_sql(
                f"CREATE ROLE {login} LOGIN PASSWORD '{boundary.password}' IN ROLE {boundary.agent_role}"
            )
        with admin.connect() as conn:
            for statement in setup:
                conn.exec_driver_sql(statement.format(**names))
        with pytest.raises(BoundaryViolation, match="not confined") as excinfo:
            connect_gateway(
                _url(boundary.admin_url.set(username=login, password=boundary.password)),
                boundary.agent_role,
            )
        assert problem.format(**names) in str(excinfo.value)
    finally:
        with admin.connect() as conn:
            for statement in cleanup:
                conn.exec_driver_sql(statement.format(**names))
            if conn.execute(text("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :r"), {"r": login}).first():
                conn.exec_driver_sql(f"DROP OWNED BY {login}")
        with root.connect() as conn:
            conn.exec_driver_sql(f"DROP ROLE IF EXISTS {login}")
        admin.dispose()
        root.dispose()


def test_self_check_rejects_a_login_outside_the_agent_role(boundary):
    login = f"sdr_stranger_{boundary.suffix}"
    root = _engine(boundary.root_url, isolation_level="AUTOCOMMIT")
    try:
        with root.connect() as conn:
            conn.exec_driver_sql(f"CREATE ROLE {login} LOGIN PASSWORD '{boundary.password}'")
        with pytest.raises(BoundaryViolation, match="does not inherit"):
            connect_gateway(
                _url(boundary.admin_url.set(username=login, password=boundary.password)),
                boundary.agent_role,
            )
    finally:
        with root.connect() as conn:
            conn.exec_driver_sql(f"DROP ROLE IF EXISTS {login}")
        root.dispose()


def test_a_write_the_database_refuses_is_submit_rejected(boundary):
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        with pytest.raises(SubmitRejected, match=f"record {record_id}"):
            gateway.submit_classification(record_id, ClassificationStatus.AUTO_CLASSIFIED, "x", float("nan"), "r")
    finally:
        gateway.close()
    assert _metadata(boundary, record_id)["classification_status"] is None


def test_nul_characters_are_stripped_from_the_reasoning(boundary):
    """PostgreSQL text cannot hold NUL; model output can."""
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        gateway.submit_classification(record_id, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "a\x00b")
    finally:
        gateway.close()
    assert _metadata(boundary, record_id)["reasoning"] == "ab"


def test_a_write_blocked_past_the_statement_timeout_is_submit_timed_out(boundary):
    record_id = _insert(boundary, Modality.UNKNOWN, None)
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role, statement_timeout_s=0.5)
    admin = _engine(boundary.admin_url)
    try:
        with admin.connect() as human:
            human.execute(text("SELECT 1 FROM survey_records WHERE id = :id FOR UPDATE"), {"id": record_id})
            with pytest.raises(SubmitTimedOut, match="57014"):
                gateway.submit_classification(record_id, ClassificationStatus.NEEDS_REVIEW, None, 0.0, "r")
            human.rollback()
    finally:
        gateway.close()
        admin.dispose()
    assert _metadata(boundary, record_id)["classification_status"] is None


def test_newest_pending_snippets_for_the_startup_probe(boundary):
    watermark = _watermark(boundary)
    older = _insert(boundary, Modality.UNKNOWN, None, path="/srv/snippets/older.sigmf-data")
    _insert(boundary, Modality.UNKNOWN, None, path=None, flags={"snippet_dropped": "low_disk"})
    newer = _insert(boundary, Modality.UNKNOWN, None, path="/srv/snippets/newer.sigmf-data")
    _insert(boundary, Modality.UNKNOWN, ClassificationStatus.MANUALLY_TAGGED, path="/srv/snippets/tagged.sigmf-data")
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        assert [r.id for r in gateway.fetch_newest_with_snippet(watermark, 5)] == [newer, older]
        assert [r.id for r in gateway.fetch_newest_with_snippet(watermark, 1)] == [newer]
    finally:
        gateway.close()
```

- [ ] **Step 2: Run them to verify they fail**

<!-- check: t5_red -->
Run: `python -m pytest tests/agent/test_db_gateway.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'agent.db_gateway'`.


- [ ] **Step 3: Implement**

Create an empty `agent/__init__.py`:

<!-- write: agent/__init__.py -->
```python
```

<!-- write: agent/db_gateway.py -->
```python
# agent/db_gateway.py
"""The agent's only database access (design: "Structural boundary").

SQL against the boundary objects installed by storage/sql/agent_boundary.sql
and nothing else: the public.agent_pending_unknown view (read) and the
public.classify_unknown function (write). `connect_gateway` refuses to
return a gateway unless the connected role is confined by that boundary.
The only module in agent/ allowed to import sqlalchemy or psycopg.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from schema.records import ClassificationStatus

DEFAULT_AGENT_ROLE = "surveytool_agent"
DEFAULT_STATEMENT_TIMEOUT_S = 30.0

_NOT_PENDING = "P0002"  # classify_unknown's no_data_found
_TIMED_OUT = {"57014", "55P03"}  # query_canceled (statement_timeout), lock_not_available
_CLASSIFY = "public.classify_unknown(integer, text, text, double precision, text)"

# The self-check is an allowlist. The connected role may be a member of
# nothing but itself and the agent role; neither may carry a dangerous
# attribute, any privilege on survey_records, or the right to create
# objects anywhere; and the only SECURITY DEFINER function it may execute
# (outside extensions) is classify_unknown. Each query returns one row per
# problem, as text.
_PROBLEMS = text(
    """
    SELECT 'member of ' || r.rolname
      FROM pg_catalog.pg_roles AS r
     WHERE pg_catalog.pg_has_role(current_user, r.oid, 'MEMBER')
       AND r.rolname NOT IN (current_user, :agent_role)
    UNION ALL
    SELECT r.rolname || ' has '
           || concat_ws(', ',
                        CASE WHEN r.rolsuper THEN 'SUPERUSER' END,
                        CASE WHEN r.rolreplication THEN 'REPLICATION' END,
                        CASE WHEN r.rolbypassrls THEN 'BYPASSRLS' END,
                        CASE WHEN r.rolcreaterole THEN 'CREATEROLE' END,
                        CASE WHEN r.rolcreatedb THEN 'CREATEDB' END)
      FROM pg_catalog.pg_roles AS r
     WHERE r.rolname IN (current_user, :agent_role)
       AND (r.rolsuper OR r.rolreplication OR r.rolbypassrls OR r.rolcreaterole OR r.rolcreatedb)
    UNION ALL
    SELECT r.rolname || ' has a privilege on survey_records'
      FROM pg_catalog.pg_roles AS r
     WHERE r.rolname IN (current_user, :agent_role)
       AND (pg_catalog.has_table_privilege(
                r.oid, pg_catalog.to_regclass('public.survey_records'),
                'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER')
            OR pg_catalog.has_any_column_privilege(
                r.oid, pg_catalog.to_regclass('public.survey_records'),
                'SELECT, INSERT, UPDATE, REFERENCES'))
    UNION ALL
    SELECT r.rolname || ' can create objects in schema ' || n.nspname
      FROM pg_catalog.pg_roles AS r, pg_catalog.pg_namespace AS n
     WHERE r.rolname IN (current_user, :agent_role)
       AND pg_catalog.has_schema_privilege(r.oid, n.oid, 'CREATE')
    UNION ALL
    SELECT r.rolname || ' has CREATE or TEMPORARY on the database'
      FROM pg_catalog.pg_roles AS r
     WHERE r.rolname IN (current_user, :agent_role)
       AND pg_catalog.has_database_privilege(r.oid, pg_catalog.current_database(), 'CREATE, TEMPORARY')
    UNION ALL
    SELECT 'can execute SECURITY DEFINER function ' || p.oid::pg_catalog.regprocedure::text
      FROM pg_catalog.pg_proc AS p
     WHERE p.prosecdef
       AND pg_catalog.has_function_privilege(current_user, p.oid, 'EXECUTE')
       AND p.oid IS DISTINCT FROM pg_catalog.to_regprocedure(:classify)
       AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_depend AS d
                        WHERE d.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
                          AND d.objid = p.oid AND d.deptype = 'e')
    """
)
# The boundary's two entry points must be usable without SET ROLE.
_BOUNDARY_ACCESS = text(
    """
    SELECT pg_catalog.pg_has_role(current_user, r.oid, 'USAGE'),
           coalesce(pg_catalog.has_table_privilege(
               pg_catalog.to_regclass('public.agent_pending_unknown'), 'SELECT'), false),
           coalesce(pg_catalog.has_function_privilege(
               pg_catalog.to_regprocedure(:classify), 'EXECUTE'), false)
      FROM pg_catalog.pg_roles AS r
     WHERE r.rolname = :agent_role
    """
)
_COLUMNS = """id, iq_snippet_path, sample_rate, center_freq, peak_power, snippet_duration_ms,
           snippet_rejected, snippet_dropped"""
_FETCH = text(
    f"""
    SELECT {_COLUMNS}
      FROM public.agent_pending_unknown
     WHERE id > :after_id
     ORDER BY id
     LIMIT :limit
    """
)
_NEWEST_WITH_SNIPPET = text(
    f"""
    SELECT {_COLUMNS}
      FROM public.agent_pending_unknown
     WHERE id > :after_id AND iq_snippet_path IS NOT NULL
     ORDER BY id DESC
     LIMIT :limit
    """
)
_SUBMIT = text(
    """
    SELECT public.classify_unknown(
        CAST(:record_id AS integer),
        CAST(:status AS text),
        CAST(:tag AS text),
        CAST(:confidence AS double precision),
        CAST(:reasoning AS text))
    """
)


class BoundaryViolation(RuntimeError):
    """The connected role is not confined by the agent database boundary."""


class RecordNotPending(RuntimeError):
    """No pending unknown row has this id any more: for example, a human
    tagged it while the LLM call was in flight. The human's tag stands."""


class SubmitRejected(RuntimeError):
    """The database refused the write's arguments (SQLSTATE class 22, such as
    classify_unknown's 22023): a bug in the agent, not a record judgment."""


class SubmitTimedOut(RuntimeError):
    """The write hit statement_timeout or a lock timeout; the row stays pending."""


@dataclass(frozen=True)
class PendingRecord:
    """One row of public.agent_pending_unknown. Every value is DB data and
    therefore untrusted (the snippet path above all). iq_snippet_path is
    None when ingest rejected the snippet or capture dropped it; the
    matching quality flag, if any, is in snippet_rejected/snippet_dropped."""

    id: int
    iq_snippet_path: str | None
    sample_rate: float | None
    center_freq: float | None
    peak_power: float | None
    snippet_duration_ms: int | None
    snippet_rejected: str | None = None
    snippet_dropped: str | None = None


class AgentGateway:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def fetch_pending(self, after_id: int, limit: int) -> list[PendingRecord]:
        """Pending unknown rows with id > after_id, lowest id first."""
        with self._engine.connect() as conn:
            rows = conn.execute(_FETCH, {"after_id": after_id, "limit": limit}).all()
        return [PendingRecord(*row) for row in rows]

    def fetch_newest_with_snippet(self, after_id: int, limit: int) -> list[PendingRecord]:
        """The newest pending rows that carry a snippet path, newest first:
        the startup probe of the snippet-store mount."""
        with self._engine.connect() as conn:
            rows = conn.execute(_NEWEST_WITH_SNIPPET, {"after_id": after_id, "limit": limit}).all()
        return [PendingRecord(*row) for row in rows]

    def submit_classification(
        self,
        record_id: int,
        status: ClassificationStatus,
        tag: str | None,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Write the four classification keys through classify_unknown.

        Raises RecordNotPending if the row stopped being pending,
        SubmitTimedOut on a statement or lock timeout, and SubmitRejected
        when the arguments are refused. NUL characters, which PostgreSQL
        text cannot hold, are removed from the reasoning first.
        """
        params = {
            "record_id": record_id,
            "status": status.value,
            "tag": tag,
            "confidence": confidence,
            "reasoning": reasoning.replace("\x00", ""),
        }
        try:
            with self._engine.begin() as conn:
                conn.execute(_SUBMIT, params)
        except DBAPIError as exc:
            driver_error = exc.orig if isinstance(exc.orig, psycopg.Error) else None
            sqlstate = None if driver_error is None else driver_error.sqlstate
            if sqlstate == _NOT_PENDING:
                raise RecordNotPending(f"Record {record_id} is no longer pending") from exc
            if sqlstate in _TIMED_OUT:
                raise SubmitTimedOut(f"Writing record {record_id} timed out ({sqlstate})") from exc
            if isinstance(driver_error, psycopg.DataError):
                raise SubmitRejected(f"The database rejected the write for record {record_id}: {driver_error}") from exc
            raise

    def close(self) -> None:
        self._engine.dispose()


def verify_boundary(engine: Engine, agent_role: str) -> None:
    """Raise BoundaryViolation unless the connected role is confined."""
    params = {"agent_role": agent_role, "classify": _CLASSIFY}
    with engine.connect() as conn:
        problems = conn.execute(_PROBLEMS, params).scalars().all()
        access = conn.execute(_BOUNDARY_ACCESS, params).first()
    if problems:
        raise BoundaryViolation(
            "Refusing to run: the database role is not confined to the agent boundary: "
            # Attributes and privileges first: they matter more than memberships.
            + "; ".join(sorted(problems, key=lambda problem: (problem.startswith("member of "), problem))[:10])
            + f". Connect as a login role that is only IN ROLE {agent_role}, and re-run "
            "sdr-agent-boundary."
        )
    if access is None or not all(access):
        raise BoundaryViolation(
            f"Refusing to run: the database role does not inherit {agent_role}, or the "
            "boundary (agent_pending_unknown, classify_unknown) is not installed. "
            "Run sdr-agent-boundary as an administrator first."
        )


def connect_gateway(
    database_url: str,
    agent_role: str = DEFAULT_AGENT_ROLE,
    statement_timeout_s: float = DEFAULT_STATEMENT_TIMEOUT_S,
) -> AgentGateway:
    """Connect, verify the boundary, and return the gateway. PostgreSQL only."""
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("The agent needs a PostgreSQL URL; its boundary only exists there")
    engine = create_engine(
        url.set(drivername="postgresql+psycopg"),
        connect_args={"options": f"-c statement_timeout={round(statement_timeout_s * 1000)}"},
    )
    try:
        verify_boundary(engine, agent_role)
    except BaseException:
        engine.dispose()
        raise
    return AgentGateway(engine)
```

In `pyproject.toml`, register the package:

<!-- edit: pyproject.toml -->
Replace

```toml
    "capture.unknown",
    "dsp",
```

with

```toml
    "capture.unknown",
    "agent",
    "dsp",
```

- [ ] **Step 4: Run the tests without a database**

<!-- check: t5_nopg -->
Run: `python -m pytest tests/agent/test_db_gateway.py tests/storage/test_agent_boundary_pg.py -q`
Expected: `3 passed, 60 skipped`. The PostgreSQL file skips without SURVEYTOOL_TEST_PG_URL.


- [ ] **Step 5: Run the boundary tests against a throwaway PostgreSQL**

Start the container and export `SURVEYTOOL_TEST_PG_URL` as shown under Global
Constraints.

<!-- check: t5_pg -->
Run with PostgreSQL: `python -m pytest tests/storage/test_agent_boundary_pg.py -q`
Expected: PASS, `59 passed, 1 skipped` in about 5 s. The skip is the pg_maintain case, which needs PostgreSQL 17.


Afterwards no `sdr_*` database or role is left in the cluster. The module fixture
asserts this on teardown.

- [ ] **Step 6: Prove the barrier has teeth, then revert**

Temporarily turn `security_barrier` off:

<!-- edit: storage/sql/agent_boundary.sql -->
Replace

```sql
    WITH (security_barrier = true) AS
```

with

```sql
    WITH (security_barrier = false) AS
```
<!-- check: t5_mutant -->
Run with PostgreSQL: `python -m pytest tests/storage/test_agent_boundary_pg.py -q -k leaky`
Expected: FAIL, `1 failed`. The probe's notices now include the hidden row's path, `saw HIDDEN-manual`.


Revert it:

<!-- edit: storage/sql/agent_boundary.sql -->
Replace

```sql
    WITH (security_barrier = false) AS
```

with

```sql
    WITH (security_barrier = true) AS
```
<!-- check: t5_reverted -->
Run with PostgreSQL: `python -m pytest tests/storage/test_agent_boundary_pg.py -q -k leaky`
Expected: PASS, `1 passed`.


- [ ] **Step 7: Commit**

<!-- run -->
```bash
git add agent/__init__.py agent/db_gateway.py pyproject.toml tests/agent/test_db_gateway.py tests/storage/test_agent_boundary_pg.py
git commit -m "feat: add the agent database gateway with a connect-time boundary self-check"
```

---

### Task 6: Contained, capped SigMF snippet reader

**Files:**
- Create: `agent/snippet_reader.py`
- Test: `tests/agent/test_snippet_reader.py`, which writes real step-4 snippets with
  `capture.unknown.snippet_writer`.

**Interfaces:**
- Consumes: sigmf and numpy only.
- Produces, in `agent.snippet_reader`:
  - `SnippetOutsideStore(Exception)`: a judgment about the record (→ `needs_review`);
  - `SnippetUnreadable(Exception)`: stays pending, and counts toward the halt;
  - `Snippet(iq: complex64 ndarray, sample_rate, center_freq_hz, truncated, pre_trigger_samples=0, non_finite_samples=0)`,
    frozen:
    - `pre_trigger_samples` is the length of step 4's `pre_trigger` annotation (start 0,
      integer length), clipped to the samples read, or 0;
    - `non_finite_samples` counts the NaN or inf samples, which the reader zeroes so
      the DSP never sees them. Task 8 turns a non-zero count into reduced confidence.
      A snippet with no finite sample at all is `SnippetUnreadable`;
  - `read_snippet(snippet_path: str | None, store_root: Path, record_sample_rate: float | None = None, record_center_hz: float | None = None) -> Snippet`.
    The record's rate and frequency are authoritative; the file's must agree;
  - constants `MAX_SECONDS = 2.0`, `MAX_SAMPLES = 1 << 25` and `MIN_SAMPLES = 1024`.

- [ ] **Step 1: Write the failing tests**

<!-- write: tests/agent/test_snippet_reader.py -->
```python
# tests/agent/test_snippet_reader.py
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import sigmf.hashing

import agent.snippet_reader
from agent.snippet_reader import (
    SnippetOutsideStore,
    SnippetUnreadable,
    read_snippet,
)
from capture.unknown.snippet_writer import write_sigmf_snippet

FS = 100_000.0
FREQ = 915e6
START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path) -> Path:
    root = tmp_path / "snippets"
    root.mkdir()
    return root


def _write(directory: Path, n: int = 4096, fs: float = FS) -> Path:
    """A real step-4 snippet pair (capture.unknown's own writer)."""
    iq = (np.arange(n) % 7).astype(np.complex64)
    return write_sigmf_snippet(iq, directory, fs, FREQ, START)


def _edit_meta(data_path: Path, edit) -> None:
    meta = data_path.with_suffix(".sigmf-meta")
    content = json.loads(meta.read_text())
    edit(content)
    meta.write_text(json.dumps(content))


def test_reads_a_step4_snippet(store):
    data = _write(store)
    snippet = read_snippet(str(data), store)
    assert snippet.sample_rate == FS and snippet.center_freq_hz == FREQ
    np.testing.assert_array_equal(snippet.iq, (np.arange(4096) % 7).astype(np.complex64))
    assert snippet.iq.dtype == np.complex64 and not snippet.truncated


def test_non_finite_samples_are_zeroed_and_counted(store):
    """As step 4 measures: NaN/inf (DMA or driver corruption) never crash
    the analysis; they are zeroed, counted, and reduce confidence later."""
    iq = (np.arange(4096) % 7 + 1).astype(np.complex64)
    iq[[3, 100]] = np.nan
    iq[7] = np.inf
    data = write_sigmf_snippet(iq, store, FS, FREQ, START)
    snippet = read_snippet(str(data), store)
    assert snippet.non_finite_samples == 3
    assert np.isfinite(snippet.iq).all() and snippet.iq[3] == 0 and snippet.iq[7] == 0


def test_a_snippet_with_no_finite_sample_is_unreadable(store):
    data = write_sigmf_snippet(np.full(4096, np.nan, np.complex64), store, FS, FREQ, START)
    with pytest.raises(SnippetUnreadable, match="NaN or inf"):
        read_snippet(str(data), store)


def test_the_record_must_agree_with_its_file(store):
    data = _write(store)
    assert read_snippet(str(data), store, FS, FREQ).sample_rate == FS
    # The record's read-back rate is authoritative when the two agree.
    assert read_snippet(str(data), store, FS * (1 + 1e-12), FREQ).sample_rate == FS * (1 + 1e-12)
    with pytest.raises(SnippetUnreadable, match="sample rate"):
        read_snippet(str(data), store, 2 * FS, FREQ)
    with pytest.raises(SnippetUnreadable, match="center frequency"):
        read_snippet(str(data), store, FS, FREQ + 1e3)


def test_step4s_pre_trigger_annotation_is_the_noise_reference(store):
    iq = (np.arange(4096) % 7).astype(np.complex64)
    annotated = write_sigmf_snippet(iq, store, FS, FREQ, START, trigger_offset=1024)
    assert read_snippet(str(annotated), store).pre_trigger_samples == 1024
    assert read_snippet(str(_write(store)), store).pre_trigger_samples == 0


@pytest.mark.parametrize(
    "annotation",
    [
        {"core:label": "pre_trigger", "core:sample_start": 5, "core:sample_count": 100},
        {"core:label": "pre_trigger", "core:sample_start": 0, "core:sample_count": True},
        {"core:label": "noise", "core:sample_start": 0, "core:sample_count": 100},
    ],
)
def test_a_malformed_pre_trigger_annotation_means_no_reference(store, annotation):
    data = _write(store)
    _edit_meta(data, lambda m: m.update({"annotations": [annotation]}))
    assert read_snippet(str(data), store).pre_trigger_samples == 0


def test_the_reference_is_clipped_to_what_was_read(store, monkeypatch):
    monkeypatch.setattr(agent.snippet_reader, "MAX_SAMPLES", 2048)
    data = write_sigmf_snippet(np.zeros(8192, np.complex64), store, FS, FREQ, START, trigger_offset=4096)
    assert read_snippet(str(data), store).pre_trigger_samples == 2048


def test_never_hashes_the_data_file(store, monkeypatch):
    """sigmf's default checksum pass reads the whole file (214 ms for a
    1 s / 20 MS/s snippet, against 1.6 ms without)."""
    data = _write(store)
    calls = []
    monkeypatch.setattr(sigmf.hashing, "calculate_sha512", lambda *a, **k: calls.append(a) or "x")
    read_snippet(str(data), store)
    assert calls == []


def test_caps_at_two_seconds_of_the_files_sample_rate(store):
    data = _write(store, n=3 * 4096, fs=4096.0)  # 3 s of samples
    snippet = read_snippet(str(data), store)
    assert len(snippet.iq) == 2 * 4096 and snippet.truncated


def test_caps_at_max_samples_whatever_the_sample_rate(store, monkeypatch):
    """The file's sample rate is untrusted: at 1e12 S/s, '2 s' caps
    nothing. The absolute ceiling (2**25 samples) still does; lowered here
    so the test needs no 256 MiB file."""
    assert agent.snippet_reader.MAX_SAMPLES == 1 << 25
    monkeypatch.setattr(agent.snippet_reader, "MAX_SAMPLES", 2048)
    data = _write(store, n=8192)
    _edit_meta(data, lambda m: m["global"].update({"core:sample_rate": 1e12}))
    snippet = read_snippet(str(data), store)
    assert len(snippet.iq) == 2048 and snippet.truncated


@pytest.mark.parametrize(
    "make_path",
    [
        lambda store, outside: None,
        lambda store, outside: "",
        lambda store, outside: "relative/a.sigmf-data",
        lambda store, outside: str(outside),  # a real snippet, elsewhere
        lambda store, outside: str(store / ".." / outside.parent.name / outside.name),  # ../ traversal
        lambda store, outside: str(outside.with_suffix(".sigmf-meta")),  # wrong suffix
    ],
)
def test_rejects_paths_outside_the_store(store, tmp_path, make_path):
    outside_dir = tmp_path / "elsewhere"
    outside = _write(outside_dir)
    with pytest.raises(SnippetOutsideStore):
        read_snippet(make_path(store, outside), store)


def test_rejects_a_symlink_out_of_the_store(store, tmp_path):
    outside = _write(tmp_path / "elsewhere")
    link = store / outside.name
    link.symlink_to(outside)
    store.joinpath(outside.with_suffix(".sigmf-meta").name).symlink_to(outside.with_suffix(".sigmf-meta"))
    with pytest.raises(SnippetOutsideStore):
        read_snippet(str(link), store)


def test_a_meta_naming_another_dataset_is_never_followed(store, tmp_path):
    """sigmf's fromfile() follows core:dataset (relative to the meta, or
    absolute), so a crafted meta inside the store could make it read any
    file. The reader never calls fromfile, and refuses such a meta."""
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"\1" * 8 * 4096)
    data = _write(store)
    _edit_meta(data, lambda m: m["global"].update({"core:dataset": str(secret)}))
    with pytest.raises(SnippetUnreadable, match="non-conforming dataset"):
        read_snippet(str(data), store)


def test_missing_files_are_unreadable_not_outside(store):
    data = _write(store)
    data.unlink()
    with pytest.raises(SnippetUnreadable):
        read_snippet(str(data), store)
    with pytest.raises(SnippetUnreadable):
        read_snippet(str(store / "never-written.sigmf-data"), store)


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["global"].update({"core:datatype": "ci16_le"}),
        lambda m: m["global"].update({"core:num_channels": 2}),
        lambda m: m["global"].update({"core:sample_rate": -1.0}),
        lambda m: m["global"].pop("core:sample_rate"),
        lambda m: m["captures"][0].pop("core:frequency"),
        lambda m: m.pop("global"),
        # sigmf itself cannot sort annotations with a non-integer length:
        lambda m: m.update({"annotations": [{"core:label": "pre_trigger", "core:sample_start": 0, "core:sample_count": "9"}]}),
    ],
)
def test_malformed_metadata_is_unreadable(store, edit):
    data = _write(store)
    _edit_meta(data, edit)
    with pytest.raises(SnippetUnreadable):
        read_snippet(str(data), store)


def test_oversized_or_garbage_meta_is_unreadable(store):
    data = _write(store)
    meta = data.with_suffix(".sigmf-meta")
    meta.write_text("{not json")
    with pytest.raises(SnippetUnreadable):
        read_snippet(str(data), store)
    meta.write_text(" " * ((1 << 20) + 1))
    with pytest.raises(SnippetUnreadable, match="larger than"):
        read_snippet(str(data), store)


def test_too_short_snippet_is_unreadable(store):
    data = _write(store, n=100)
    with pytest.raises(SnippetUnreadable, match="need 1024"):
        read_snippet(str(data), store)
```

- [ ] **Step 2: Run them to verify they fail**

<!-- check: t6_red -->
Run: `python -m pytest tests/agent/test_snippet_reader.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'agent.snippet_reader'`.


- [ ] **Step 3: Implement**

<!-- write: agent/snippet_reader.py -->
```python
# agent/snippet_reader.py
"""Load a record's SigMF snippet: contained, capped, and without a checksum pass.

The snippet path comes from the database, so it is untrusted, and so is
everything in the .sigmf-meta file. The reader therefore:
- requires both files to resolve (symlinks followed) under the snippet-store root;
- reads at most 1 MiB of .sigmf-meta and parses it itself, then hands sigmf
  the contained data path explicitly. sigmf's fromfile() is not used: it
  follows a meta's core:dataset to any path (absolute ones included), falls
  back to archive/collection/non-SigMF converters, and leaks the meta file
  handle when the JSON is invalid;
- skips sigmf's SHA-512 pass, which would read the whole data file;
- reads at most 2 s of samples, and never more than MAX_SAMPLES;
- zeroes NaN/inf samples (DMA or driver corruption) and counts them, as step
  4 does when it measures, so a corrupt sample never crashes the analysis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sigmf
from sigmf import sigmffile

DATA_SUFFIX = ".sigmf-data"
META_SUFFIX = ".sigmf-meta"
MAX_SECONDS = 2.0
# Absolute ceiling whatever the file claims its sample rate is: 256 MiB of
# cf32. Measured peak RSS of the whole analysis at this size: ~610 MB.
MAX_SAMPLES = 1 << 25
MIN_SAMPLES = 1024  # one coarse FFT frame (dsp.segmentation.NFFT)
_MAX_META_BYTES = 1 << 20


class SnippetOutsideStore(Exception):
    """The record's snippet is not a file inside the store: a judgment about
    the record (needs_review), not an I/O fault."""


class SnippetUnreadable(Exception):
    """The snippet is missing, unreadable, or malformed. The record stays
    pending; many in a row point to a systemic fault (wrong root, unmounted
    disk)."""


@dataclass(frozen=True)
class Snippet:
    iq: np.ndarray  # complex64
    sample_rate: float  # core:sample_rate of the file
    center_freq_hz: float  # core:frequency of the file's first capture
    truncated: bool  # the file held more than the cap
    # Length of step 4's "pre_trigger" annotation (signal-free samples before
    # the trigger, the noise reference), clipped to what was read; 0 if none.
    pre_trigger_samples: int = 0
    non_finite_samples: int = 0  # NaN/inf samples, zeroed in `iq`


def _contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise SnippetOutsideStore(f"{path} resolves to {resolved}, outside the snippet store {root}")
    return resolved


def read_snippet(
    snippet_path: str | None,
    store_root: Path,
    record_sample_rate: float | None = None,
    record_center_hz: float | None = None,
) -> Snippet:
    """Read the snippet at `snippet_path` (a .sigmf-data path from the DB).

    The record's metadata.sample_rate and center frequency are authoritative
    (step 4 stores the SDR's read-back rate); the file's core:sample_rate and
    core:frequency must agree with them, and are used only when the record
    has none."""
    if not snippet_path:
        raise SnippetOutsideStore("The record has no snippet path")
    root = store_root.resolve()
    path = Path(snippet_path)
    if not path.is_absolute() or path.suffix != DATA_SUFFIX:
        raise SnippetOutsideStore(f"{snippet_path!r} is not an absolute {DATA_SUFFIX} path")
    data = _contained(path, root)
    meta = _contained(path.with_suffix(META_SUFFIX), root)
    try:
        with meta.open("rb") as handle:
            raw = handle.read(_MAX_META_BYTES + 1)
        if len(raw) > _MAX_META_BYTES:
            raise SnippetUnreadable(f"{meta} is larger than {_MAX_META_BYTES} bytes")
        metadata = json.loads(raw)
        if sigmf.DATASET_KEY in metadata[sigmf.SigMFFile.GLOBAL_KEY]:
            raise SnippetUnreadable(f"{meta} names a non-conforming dataset; step 4 never writes one")
        recording = sigmffile.SigMFFile(metadata=metadata, data_file=str(data), skip_checksum=True)
        return _samples(recording, data, record_sample_rate, record_center_hz)
    except SnippetUnreadable:
        raise
    except Exception as exc:  # the files are untrusted: any parse or read failure is "unreadable"
        raise SnippetUnreadable(f"Cannot read {data}: {exc!r}") from exc


def _samples(
    recording: sigmffile.SigMFFile,
    data: Path,
    record_sample_rate: float | None,
    record_center_hz: float | None,
) -> Snippet:
    datatype = recording.get_global_field(sigmf.DATATYPE_KEY)
    channels = recording.get_global_field(sigmf.NUM_CHANNELS_KEY, 1)
    sample_rate = recording.get_global_field(sigmf.SAMPLE_RATE_KEY)
    captures = recording.get_captures()
    center = captures[0].get(sigmf.FREQUENCY_KEY) if captures else None
    if datatype != "cf32_le" or channels != 1:
        raise SnippetUnreadable(f"{data}: expected one cf32_le channel, got {datatype} x {channels}")
    for name, value in (("sample rate", sample_rate), ("center frequency", center)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SnippetUnreadable(f"{data}: invalid {name} {value!r}")
    for name, claimed, actual in (
        ("sample rate", record_sample_rate, sample_rate),
        ("center frequency", record_center_hz, center),
    ):
        if claimed is not None and not math.isclose(claimed, actual, rel_tol=1e-9):
            raise SnippetUnreadable(f"{data}: the record's {name} {claimed} disagrees with the file's {actual}")
    if record_sample_rate is not None:
        sample_rate = record_sample_rate
    if record_center_hz is not None:
        center = record_center_hz
    count = min(recording.sample_count, round(MAX_SECONDS * sample_rate), MAX_SAMPLES)
    if count < MIN_SAMPLES:
        raise SnippetUnreadable(f"{data}: {recording.sample_count} samples, need {MIN_SAMPLES}")
    iq = np.asarray(recording.read_samples(0, count), dtype=np.complex64)
    finite = np.isfinite(iq)
    non_finite = int(iq.size - np.count_nonzero(finite))
    if non_finite == iq.size:
        raise SnippetUnreadable(f"{data}: every sample is NaN or inf")
    return Snippet(
        iq=np.where(finite, iq, 0).astype(np.complex64) if non_finite else iq,
        sample_rate=float(sample_rate),
        center_freq_hz=float(center),
        truncated=recording.sample_count > count,
        pre_trigger_samples=_pre_trigger_samples(recording, count),
        non_finite_samples=non_finite,
    )


def _pre_trigger_samples(recording: sigmffile.SigMFFile, count: int) -> int:
    """Step 4 annotates [0, trigger) as "pre_trigger". Anything else (no
    annotation, another start, a non-integer length) means no reference."""
    for annotation in recording.get_annotations():
        length = annotation.get(sigmf.SAMPLE_COUNT_KEY)
        if (
            annotation.get(sigmf.LABEL_KEY) == "pre_trigger"
            and annotation.get(sigmf.SAMPLE_START_KEY) == 0
            and type(length) is int
            and length > 0
        ):
            return min(length, count)
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass, warnings as errors**

<!-- check: t6_green -->
Run: `python -m pytest tests/agent/test_snippet_reader.py -q -W error`
Expected: PASS, `30 passed`, with no warnings.


- [ ] **Step 5: Commit**

<!-- run -->
```bash
git add agent/snippet_reader.py tests/agent/test_snippet_reader.py
git commit -m "feat: add a contained, capped, checksum-free SigMF snippet reader for the agent"
```

---

### Task 7: The cited US band table and grounded matching

**Files:**
- Create: `agent/data/band_table_us.json`, `agent/band_table.py`
- Modify: `pyproject.toml` (`agent` package data)
- Test: `tests/agent/test_band_table.py`

**Interfaces:**
- Consumes: pydantic only.
- Produces, in `agent.band_table`:
  - `BandEntry` (pydantic, frozen, `extra="forbid"`), with fields `id`, `start_hz`,
    `end_hz`, `service`, `typical_signals`, `expected_obw_hz: (min, max)`, `citation`
    (must start `47 CFR `) and `source` (the verbatim eCFR text);
  - `BandTable(region, verified, entries)`;
  - `BandMatch(entry, grounded)`, frozen;
  - `load_band_table(path=DEFAULT_BAND_TABLE) -> BandTable`;
  - `match_bands(entries, center_hz, obw_hz) -> list[BandMatch]`.

    It returns every entry the occupied span `[center − obw/2, center + obw/2]`
    overlaps, lowest start first. An entry is grounded when its edges contain the
    center (inclusive) and `obw_hz` lies within its `expected_obw_hz`.

Every edge in the JSON was checked against eCFR text (see Verified facts, eCFR). If you
change an edge, re-fetch the section and replace the `source` quote. An entry without a
verifiable citation is dropped, never guessed.

`expected_obw_hz` must mean something. No range starts at 0, none spans more than 250×,
downlinks start at 1 MHz, and 2 m starts at 2 kHz. The data test enforces the first two.

- [ ] **Step 1: Write the failing tests**

<!-- write: tests/agent/test_band_table.py -->
```python
# tests/agent/test_band_table.py
import pytest
from pydantic import ValidationError

from agent.band_table import BandEntry, load_band_table, match_bands


def _entry(id_: str, start: float, end: float, obw: tuple[float, float] = (10e3, 30e3)) -> BandEntry:
    return BandEntry(
        id=id_,
        start_hz=start,
        end_hz=end,
        service="test service",
        typical_signals=("narrowband FM",),
        expected_obw_hz=obw,
        citation="47 CFR 0.0",
        source="verbatim",
    )


def test_shipped_table_is_valid_and_cited():
    table = load_band_table()
    assert table.region == "US"
    assert 20 <= len(table.entries) <= 30
    for entry in table.entries:
        assert entry.citation.startswith("47 CFR ")
        assert entry.source.strip()
        assert 47e6 <= entry.start_hz < entry.end_hz <= 6e9
        low, high = entry.expected_obw_hz
        # Plausibility must mean something: no range starts at 0 or spans
        # more than ~2.4 decades.
        assert 0 < low < high <= 250 * low, entry.id


def test_shipped_table_ids_include_the_spec_examples():
    ids = {entry.id for entry in load_band_table().entries}
    assert {
        "fm_broadcast",
        "airband_vhf",
        "frs_gmrs_462",
        "ism_902_928",
        "pcs_downlink",
        "aws_downlink",
        "lower700_downlink",
        "upper700_c_downlink",
        "lte_b71_600_downlink",
        "adsb_1090",
        "gnss_rnss_l1",
        "ism_2400",
        "cbrs",
        "unii_5150_5250",
        "unii_5725_5850",
    } <= ids


@pytest.mark.parametrize(
    "start, end, obw",
    [(46e6, 50e6, (1, 2)), (100e6, 100e6, (1, 2)), (100e6, 99e6, (1, 2)), (5.9e9, 6.1e9, (1, 2)), (100e6, 101e6, (2, 1))],
)
def test_entries_outside_47mhz_6ghz_or_inverted_are_rejected(start, end, obw):
    with pytest.raises(ValidationError):
        _entry("bad", start, end, obw)


def test_entry_without_a_cfr_citation_is_rejected():
    with pytest.raises(ValidationError):
        BandEntry.model_validate({**_entry("x", 100e6, 101e6).model_dump(), "citation": "Wikipedia"})


def test_grounded_needs_center_inside_and_plausible_obw():
    bands = (_entry("frs", 462.54e6, 462.735e6, (2e3, 20e3)),)
    (match,) = match_bands(bands, 462.6e6, 12e3)
    assert match.grounded
    (match,) = match_bands(bands, 462.6e6, 200e3)  # center inside, OBW implausible
    assert not match.grounded


def test_band_edges_are_inclusive():
    bands = (_entry("b", 100e6, 101e6),)
    assert match_bands(bands, 100e6, 20e3)[0].grounded
    assert match_bands(bands, 101e6, 20e3)[0].grounded


def test_signal_overlapping_an_edge_matches_ungrounded():
    """Center just outside, occupied span overlapping: listed, not grounded."""
    bands = (_entry("b", 100e6, 101e6),)
    (match,) = match_bands(bands, 101e6 + 5e3, 20e3)
    assert not match.grounded
    assert match_bands(bands, 101e6 + 11e3, 20e3) == []


def test_overlapping_entries_all_returned_lowest_start_first():
    bands = (_entry("wide", 900e6, 930e6, (5e3, 2e6)), _entry("narrow", 914e6, 916e6, (100e3, 300e3)))
    matches = match_bands(bands, 915e6, 125e3)
    assert [(m.entry.id, m.grounded) for m in matches] == [("wide", True), ("narrow", True)]
    matches = match_bands(bands, 915e6, 1e6)
    assert [(m.entry.id, m.grounded) for m in matches] == [("wide", True), ("narrow", False)]
```

- [ ] **Step 2: Run them to verify they fail**

<!-- check: t7_red -->
Run: `python -m pytest tests/agent/test_band_table.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'agent.band_table'`.


- [ ] **Step 3: Implement**

<!-- write: agent/data/band_table_us.json -->
```json
{
  "region": "US",
  "verified": "Band edges checked 2026-10-03 against the eCFR, Title 47 as of 2026-10-01 (https://www.ecfr.gov/api/versioner/v1/full/2026-10-01/title-47.xml?part=P&section=S). expected_obw_hz is an engineering estimate for the listed signals, not a regulatory value, except where a cited rule caps occupied bandwidth. Downlinks start at 1 MHz (an LTE carrier is never narrower); uplinks at 150 kHz (one UE's SC-FDMA allocation can be a single 180 kHz resource block); 2 m starts at 2 kHz, so a bare carrier or CW there is left ungrounded.",
  "entries": [
    {
      "id": "fm_broadcast",
      "start_hz": 88000000,
      "end_hz": 108000000,
      "service": "FM broadcast",
      "typical_signals": [
        "wideband FM (mono/stereo)",
        "HD Radio hybrid sidebands"
      ],
      "expected_obw_hz": [
        100000,
        450000
      ],
      "citation": "47 CFR 73.201",
      "source": "The FM broadcast band consists of that portion of the radio frequency spectrum between 88 MHz and 108 MHz."
    },
    {
      "id": "airband_vhf",
      "start_hz": 117975000,
      "end_hz": 137000000,
      "service": "Aeronautical mobile (R), VHF air-ground voice and data",
      "typical_signals": [
        "AM voice (25 kHz / 8.33 kHz channels)",
        "VDL Mode 2 D8PSK",
        "ACARS MSK"
      ],
      "expected_obw_hz": [
        2000,
        25000
      ],
      "citation": "47 CFR 2.106(b)(200) (footnote 5.200)",
      "source": "5.200 In the band 117.975-137 MHz, the frequency 121.5 MHz is the aeronautical emergency frequency"
    },
    {
      "id": "ham_2m",
      "start_hz": 144000000,
      "end_hz": 148000000,
      "service": "Amateur radio 2 m band (ITU Region 2)",
      "typical_signals": [
        "narrowband FM voice",
        "APRS 1200 baud AFSK on FM",
        "digital voice"
      ],
      "expected_obw_hz": [
        2000,
        20000
      ],
      "citation": "47 CFR 97.301(a)",
      "source": "2 m | 144-146 | 144-148 | 144-148 (ITU Region 1 | Region 2 | Region 3 columns)"
    },
    {
      "id": "marine_vhf",
      "start_hz": 156000000,
      "end_hz": 162000000,
      "service": "Maritime mobile VHF",
      "typical_signals": [
        "FM voice (16K0F3E)",
        "DSC",
        "AIS GMSK (near 162 MHz)"
      ],
      "expected_obw_hz": [
        5000,
        25000
      ],
      "citation": "47 CFR 80.5",
      "source": "in the 156-162 MHz band"
    },
    {
      "id": "frs_gmrs_462",
      "start_hz": 462540000,
      "end_hz": 462735000,
      "service": "FRS / GMRS 462 MHz channels (channel centers 462.5500-462.7250 MHz, widened by half the 20 kHz authorized bandwidth)",
      "typical_signals": [
        "narrowband FM voice",
        "GMRS repeater outputs",
        "FRS/GMRS digital data bursts"
      ],
      "expected_obw_hz": [
        2000,
        20000
      ],
      "citation": "47 CFR 95.1763(a),(b); 95.1773(a),(b); 95.563",
      "source": "The channel center frequencies are: 462.5500, ... and 462.7250 MHz. / The authorized bandwidth is 20 kHz for GMRS transmitters operating on any of the 462 MHz main channels"
    },
    {
      "id": "frs_gmrs_467",
      "start_hz": 467540000,
      "end_hz": 467735000,
      "service": "FRS / GMRS 467 MHz channels (channel centers 467.5500-467.7250 MHz, widened by half the 20 kHz authorized bandwidth)",
      "typical_signals": [
        "narrowband FM voice",
        "GMRS repeater inputs"
      ],
      "expected_obw_hz": [
        2000,
        20000
      ],
      "citation": "47 CFR 95.1763(c),(d); 95.1773(a),(b); 95.563",
      "source": "The channel center frequencies are: 467.5500, ... and 467.7250 MHz. / The authorized bandwidth is 20 kHz for GMRS transmitters operating on any of ... the 467 MHz main channels"
    },
    {
      "id": "uhf_tv",
      "start_hz": 470000000,
      "end_hz": 608000000,
      "service": "UHF television broadcast (channels 14-36)",
      "typical_signals": [
        "ATSC 1.0 8-VSB",
        "ATSC 3.0 OFDM"
      ],
      "expected_obw_hz": [
        4000000,
        6200000
      ],
      "citation": "47 CFR 73.603(a)",
      "source": "14 | 470-476 ... 36 | 602-608"
    },
    {
      "id": "lte_b71_600_downlink",
      "start_hz": 617000000,
      "end_hz": 652000000,
      "service": "600 MHz band downlink (LTE band 71 / NR n71)",
      "typical_signals": [
        "LTE/NR FDD downlink"
      ],
      "expected_obw_hz": [
        1000000,
        20000000
      ],
      "citation": "47 CFR 27.11(k); 27.5(l)",
      "source": "600 MHz downlink band (617-652 MHz)"
    },
    {
      "id": "lte_b71_600_uplink",
      "start_hz": 663000000,
      "end_hz": 698000000,
      "service": "600 MHz band uplink (LTE band 71 / NR n71)",
      "typical_signals": [
        "LTE/NR FDD uplink (SC-FDMA)"
      ],
      "expected_obw_hz": [
        150000,
        20000000
      ],
      "citation": "47 CFR 27.11(k); 27.5(l)",
      "source": "600 MHz uplink band (663-698 MHz)"
    },
    {
      "id": "lower700_uplink",
      "start_hz": 698000000,
      "end_hz": 716000000,
      "service": "Lower 700 MHz blocks A-C uplink (LTE bands 12/17)",
      "typical_signals": [
        "LTE FDD uplink (SC-FDMA)"
      ],
      "expected_obw_hz": [
        150000,
        20000000
      ],
      "citation": "47 CFR 27.5(c)(1); 27.50(c)",
      "source": "Block A: 698-704 MHz and 728-734 MHz; Block B: 704-710 MHz and 734-740 MHz; and Block C: 710-716 MHz and 740-746 MHz / uplink operations in the 698-716 MHz band"
    },
    {
      "id": "lower700_downlink",
      "start_hz": 728000000,
      "end_hz": 746000000,
      "service": "Lower 700 MHz blocks A-C downlink (LTE bands 12/17)",
      "typical_signals": [
        "LTE FDD downlink"
      ],
      "expected_obw_hz": [
        1000000,
        20000000
      ],
      "citation": "47 CFR 27.5(c)(1)",
      "source": "Block A: 698-704 MHz and 728-734 MHz; Block B: 704-710 MHz and 734-740 MHz; and Block C: 710-716 MHz and 740-746 MHz"
    },
    {
      "id": "upper700_c_downlink",
      "start_hz": 746000000,
      "end_hz": 757000000,
      "service": "Upper 700 MHz C block, conventionally downlink (LTE band 13)",
      "typical_signals": [
        "LTE FDD downlink"
      ],
      "expected_obw_hz": [
        1000000,
        20000000
      ],
      "citation": "47 CFR 27.5(b)(3)",
      "source": "Block C in the 746-757 MHz and 776-787 MHz bands"
    },
    {
      "id": "upper700_c_uplink",
      "start_hz": 776000000,
      "end_hz": 787000000,
      "service": "Upper 700 MHz C block, conventionally uplink (LTE band 13)",
      "typical_signals": [
        "LTE FDD uplink (SC-FDMA)"
      ],
      "expected_obw_hz": [
        150000,
        20000000
      ],
      "citation": "47 CFR 27.5(b)(3)",
      "source": "Block C in the 746-757 MHz and 776-787 MHz bands"
    },
    {
      "id": "cellular_850_uplink",
      "start_hz": 824000000,
      "end_hz": 849000000,
      "service": "Cellular 850 MHz, conventionally uplink (LTE band 5)",
      "typical_signals": [
        "LTE FDD uplink (SC-FDMA)",
        "NB-IoT / LTE-M uplink"
      ],
      "expected_obw_hz": [
        150000,
        20000000
      ],
      "citation": "47 CFR 22.905",
      "source": "869-880 MHz paired with 824-835 MHz ... 891.5-894 MHz paired with 846.5-849 MHz"
    },
    {
      "id": "cellular_850_downlink",
      "start_hz": 869000000,
      "end_hz": 894000000,
      "service": "Cellular 850 MHz, conventionally downlink (LTE band 5)",
      "typical_signals": [
        "LTE FDD downlink",
        "NB-IoT / LTE-M"
      ],
      "expected_obw_hz": [
        1000000,
        20000000
      ],
      "citation": "47 CFR 22.905",
      "source": "869-880 MHz paired with 824-835 MHz ... 891.5-894 MHz paired with 846.5-849 MHz"
    },
    {
      "id": "ism_902_928",
      "start_hz": 902000000,
      "end_hz": 928000000,
      "service": "902-928 MHz unlicensed (Part 15) / ISM (Part 18)",
      "typical_signals": [
        "LoRa / LoRaWAN chirp spread spectrum",
        "FHSS and FSK telemetry, smart meters",
        "UHF RFID readers",
        "802.15.4g"
      ],
      "expected_obw_hz": [
        10000,
        2200000
      ],
      "citation": "47 CFR 15.247; 18.301",
      "source": "Operation within the bands 902-928 MHz, 2400-2483.5 MHz, and 5725-5850 MHz"
    },
    {
      "id": "adsb_1090",
      "start_hz": 1087700000,
      "end_hz": 1092300000,
      "service": "1090 MHz aeronautical (Mode S / ADS-B extended squitter)",
      "typical_signals": [
        "Mode S / ADS-B 1 Mbit/s pulse-position modulation",
        "Mode A/C replies"
      ],
      "expected_obw_hz": [
        500000,
        10000000
      ],
      "citation": "47 CFR 2.106(b)(328)(ii) (footnote 5.328AA)",
      "source": "The frequency band 1087.7-1092.3 MHz is also allocated to the aeronautical mobile-satellite (R) service (Earth-to-space) on a primary basis, limited to the space station reception of Automatic Dependent Surveillance-Broadcast (ADS-B) emissions from aircraft transmitters"
    },
    {
      "id": "gnss_rnss_l1",
      "start_hz": 1559000000,
      "end_hz": 1610000000,
      "service": "Radionavigation-satellite (GPS L1, Galileo E1, BeiDou B1, GLONASS L1)",
      "typical_signals": [
        "GNSS spread spectrum (normally below the noise floor of an untuned SDR)"
      ],
      "expected_obw_hz": [
        1000000,
        32000000
      ],
      "citation": "47 CFR 2.106(b)(328)(iii) (footnote 5.328B)",
      "source": "The use of the bands 1164-1300 MHz, 1559-1610 MHz and 5010-5030 MHz by systems and networks in the radionavigation-satellite service"
    },
    {
      "id": "aws_uplink",
      "start_hz": 1695000000,
      "end_hz": 1780000000,
      "service": "AWS-1 and AWS-3 uplink (LTE bands 4/66, unpaired AWS-3 1695-1710 MHz)",
      "typical_signals": [
        "LTE FDD uplink (SC-FDMA)"
      ],
      "expected_obw_hz": [
        150000,
        20000000
      ],
      "citation": "47 CFR 27.5(h); 27.50(d)(4)",
      "source": "1710-1755 MHz, 2110-2155 MHz, 1695-1710 MHz, 1755-1780 MHz, and 2155-2180 MHz bands"
    },
    {
      "id": "pcs_uplink",
      "start_hz": 1850000000,
      "end_hz": 1910000000,
      "service": "Broadband PCS, conventionally uplink (LTE band 2/25)",
      "typical_signals": [
        "LTE FDD uplink (SC-FDMA)"
      ],
      "expected_obw_hz": [
        150000,
        20000000
      ],
      "citation": "47 CFR 24.200; 24.229",
      "source": "authorized in the 1850-1910 and 1930-1990 MHz bands"
    },
    {
      "id": "pcs_downlink",
      "start_hz": 1930000000,
      "end_hz": 1990000000,
      "service": "Broadband PCS, conventionally downlink (LTE band 2/25)",
      "typical_signals": [
        "LTE FDD downlink"
      ],
      "expected_obw_hz": [
        1000000,
        20000000
      ],
      "citation": "47 CFR 24.200; 24.229",
      "source": "authorized in the 1850-1910 and 1930-1990 MHz bands"
    },
    {
      "id": "aws_downlink",
      "start_hz": 2110000000,
      "end_hz": 2180000000,
      "service": "AWS-1 and AWS-3 downlink (LTE bands 4/66)",
      "typical_signals": [
        "LTE FDD downlink"
      ],
      "expected_obw_hz": [
        1000000,
        20000000
      ],
      "citation": "47 CFR 27.5(h); 27.50(d)(1)-(2)",
      "source": "1710-1755 MHz, 2110-2155 MHz, 1695-1710 MHz, 1755-1780 MHz, and 2155-2180 MHz bands"
    },
    {
      "id": "ism_2400",
      "start_hz": 2400000000,
      "end_hz": 2483500000,
      "service": "2.4 GHz unlicensed (Part 15)",
      "typical_signals": [
        "802.11b/g/n/ax Wi-Fi (20/40 MHz)",
        "Bluetooth / BLE (1-2 MHz FHSS)",
        "802.15.4 / Zigbee",
        "microwave ovens"
      ],
      "expected_obw_hz": [
        500000,
        40000000
      ],
      "citation": "47 CFR 15.247",
      "source": "Operation within the bands 902-928 MHz, 2400-2483.5 MHz, and 5725-5850 MHz"
    },
    {
      "id": "cbrs",
      "start_hz": 3550000000,
      "end_hz": 3700000000,
      "service": "Citizens Broadband Radio Service",
      "typical_signals": [
        "LTE/NR TDD (band 48 / n48)"
      ],
      "expected_obw_hz": [
        4000000,
        60000000
      ],
      "citation": "47 CFR 96.11(a)",
      "source": "authorized in the 3550-3700 MHz frequency band"
    },
    {
      "id": "unii_5150_5250",
      "start_hz": 5150000000,
      "end_hz": 5250000000,
      "service": "U-NII-1 (Part 15 unlicensed national information infrastructure)",
      "typical_signals": [
        "802.11a/n/ac/ax Wi-Fi (20/40/80/160 MHz)"
      ],
      "expected_obw_hz": [
        10000000,
        160000000
      ],
      "citation": "47 CFR 15.407(a)(1)(i)",
      "source": "operating in the band 5.15-5.25 GHz"
    },
    {
      "id": "unii_5725_5850",
      "start_hz": 5725000000,
      "end_hz": 5850000000,
      "service": "U-NII-3 / 5.8 GHz unlicensed",
      "typical_signals": [
        "802.11a/n/ac/ax Wi-Fi",
        "5.8 GHz FPV video",
        "point-to-point links"
      ],
      "expected_obw_hz": [
        5000000,
        160000000
      ],
      "citation": "47 CFR 15.407(a)(3)(i); 15.247",
      "source": "For the band 5.725-5.850 GHz"
    }
  ]
}
```

<!-- write: agent/band_table.py -->
```python
# agent/band_table.py
"""The curated US band table (agent/data/band_table_us.json) and grounded matching.

Deliberately small: about twenty well-known allocations an urban drive
survey will hit, each with an eCFR citation and the verbatim text that
confirmed its edges. A match is *grounded* when the signal's center lies
inside the band AND its occupied bandwidth is plausible for that band.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_BAND_TABLE = Path(__file__).resolve().parent / "data" / "band_table_us.json"
MIN_HZ = 47e6
MAX_HZ = 6e9


class BandEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    start_hz: float
    end_hz: float
    service: str = Field(min_length=1)
    typical_signals: tuple[str, ...] = Field(min_length=1)
    expected_obw_hz: tuple[float, float]
    citation: str = Field(pattern=r"^47 CFR \d")
    source: str = Field(min_length=1)  # verbatim eCFR text that confirmed the edges

    @model_validator(mode="after")
    def _sane(self) -> BandEntry:
        if not MIN_HZ <= self.start_hz < self.end_hz <= MAX_HZ:
            raise ValueError(f"{self.id}: edges must satisfy {MIN_HZ:g} <= start < end <= {MAX_HZ:g}")
        low, high = self.expected_obw_hz
        if not 0 <= low < high <= MAX_HZ:
            raise ValueError(f"{self.id}: expected_obw_hz must be 0 <= min < max")
        return self


class BandTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    region: str
    verified: str
    entries: tuple[BandEntry, ...] = Field(min_length=1)


@dataclass(frozen=True)
class BandMatch:
    entry: BandEntry
    grounded: bool


def load_band_table(path: Path = DEFAULT_BAND_TABLE) -> BandTable:
    table = BandTable.model_validate(json.loads(path.read_text()))
    ids = [entry.id for entry in table.entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate band ids")
    return table


def match_bands(entries: tuple[BandEntry, ...], center_hz: float, obw_hz: float) -> list[BandMatch]:
    """Every entry the signal's occupied span [center - obw/2, center + obw/2]
    overlaps, marked grounded or not, lowest start first."""
    low, high = center_hz - obw_hz / 2, center_hz + obw_hz / 2
    return [
        BandMatch(
            entry=entry,
            grounded=entry.start_hz <= center_hz <= entry.end_hz
            and entry.expected_obw_hz[0] <= obw_hz <= entry.expected_obw_hz[1],
        )
        for entry in sorted(entries, key=lambda entry: entry.start_hz)
        if entry.start_hz <= high and entry.end_hz >= low
    ]
```

In `pyproject.toml`, ship the JSON with the package:

<!-- edit: pyproject.toml -->
Replace

```toml
[tool.setuptools.package-data]
storage = ["sql/*.sql"]
```

with

```toml
[tool.setuptools.package-data]
agent = ["data/*.json"]
storage = ["sql/*.sql"]
```

- [ ] **Step 4: Run the tests to verify they pass**

<!-- check: t7_green -->
Run: `python -m pytest tests/agent/test_band_table.py -q`
Expected: PASS, `12 passed`.


- [ ] **Step 5: Commit**

<!-- run -->
```bash
git add agent/data/band_table_us.json agent/band_table.py pyproject.toml tests/agent/test_band_table.py
git commit -m "feat: add the cited US band table with grounded matching"
```

---

### Task 8: Classifier seam, snippet analysis and the prompt

**Files:**
- Create: `agent/classifier.py`, `agent/analysis.py`, `agent/prompt.py`
- Test: `tests/agent/test_analysis_prompt.py`

**Interfaces:**
- Consumes:
  - `dsp.segmentation.segment_spectrum`, `NFFT` and `MAX_CONTEXT_REGIONS` (Task 2);
  - `dsp.features.region_features` and `RegionFeatures` (Task 3);
  - `agent.snippet_reader.Snippet` (Task 6);
  - `agent.band_table.BandEntry`, `BandMatch` and `match_bands` (Task 7).
- Produces:
  - `agent.classifier`:
    - `ModulationPrediction(label, confidence)`, frozen;
    - the `ModulationClassifier` protocol, with `.predict(iq, sample_rate) -> ModulationPrediction | None`;
    - `UnavailableClassifier`, which always returns `None`.
  - `agent.analysis`:
    - `ContextRegion(center_hz, obw_hz, power_relative_to_primary_db, present_before_trigger: bool | None)`.
      The flag is True for an emitter found in the pre-trigger reference, False for one
      that is new during the burst, and None in self mode, where there is no reference to
      tell. Context is the strongest `MAX_CONTEXT_REGIONS` of both kinds;
    - `SnippetAnalysis(tuned_center_hz, sample_rate, noise_reference, reduced_confidence: tuple[str, ...], analysed_seconds, truncated, coarse_resolution_hz, primary: RegionFeatures | None, signal_center_hz, context, band_matches, modulation)`,
      with `.grounded_band_ids -> frozenset[str]` and `.modulation_label -> str | None`:
      - `noise_reference` is the floor source. Segmentation gets the snippet's
        `pre_trigger` samples as its reference;
      - `reduced_confidence` holds any of `"no_quiet_noise_reference"` (self floor),
        `"non_finite_samples"` and `"bandwidth_unreliable"`. Any reason keeps every band
        match ungrounded;
    - `analyse_snippet(snippet, bands, classifier) -> SnippetAnalysis`.
  - `agent.prompt`:
    - `SYSTEM_PROMPT`;
    - `build_user_message(analysis) -> str`. It raises `ValueError` without a primary
      region. The message holds the floor source and `reduced_confidence`, numeric
      features, region summaries (with `present_before_trigger`), band entries and the
      classifier output only.

- [ ] **Step 1: Write the failing tests**

<!-- write: tests/agent/test_analysis_prompt.py -->
```python
# tests/agent/test_analysis_prompt.py
import json

import numpy as np
import pytest

from agent.analysis import analyse_snippet
from agent.band_table import load_band_table
from agent.classifier import ModulationPrediction, UnavailableClassifier
from agent.prompt import build_user_message
from agent.snippet_reader import Snippet
from dsp import synthetic

FS = 1e6
N = 1 << 18
TUNED = 915e6
BANDS = load_band_table().entries


PRE = round(0.05 * FS)  # the pre-trigger: everything before the burst


def _snippet(iq: np.ndarray, pre_trigger: int = PRE, non_finite: int = 0) -> Snippet:
    return Snippet(
        iq=iq, sample_rate=FS, center_freq_hz=TUNED, truncated=False,
        pre_trigger_samples=pre_trigger, non_finite_samples=non_finite,
    )


def _two_emitters() -> np.ndarray:
    """125 kHz burst at 915.2 MHz (primary) + carrier at 914.75 MHz (context)."""
    rng = np.random.default_rng(20)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 125e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    return synthetic.tone(N, FS, -250e3, 1e-4) + burst + synthetic.noise(rng, N, 1e-5)


class _AlwaysPredicts:
    def predict(self, iq, sample_rate):
        return ModulationPrediction(label="lora", confidence=0.7)


def test_analysis_places_the_primary_in_absolute_frequency_and_grounds_it():
    analysis = analyse_snippet(_snippet(_two_emitters()), BANDS, UnavailableClassifier())
    assert analysis.signal_center_hz == pytest.approx(915.2e6, abs=2e3)
    assert analysis.primary.obw_hz == pytest.approx(125e3, rel=0.05)
    (context,) = analysis.context
    assert context.center_hz == pytest.approx(914.75e6, abs=1e3)
    assert context.power_relative_to_primary_db == pytest.approx(-10.0, abs=0.5)
    assert context.present_before_trigger is True  # found in the pre-trigger reference
    assert (analysis.noise_reference, analysis.reduced_confidence) == ("pre_trigger", ())
    assert [(m.entry.id, m.grounded) for m in analysis.band_matches] == [("ism_902_928", True)]
    assert analysis.modulation is None and analysis.grounded_band_ids == {"ism_902_928"}


def test_without_a_quiet_reference_confidence_is_reduced_and_nothing_grounds():
    """The self floor is blind to receiver roll-off (a narrow burst on
    colored noise reads hundreds of kHz wide), so it never grounds."""
    analysis = analyse_snippet(_snippet(_two_emitters(), pre_trigger=0), BANDS, UnavailableClassifier())
    assert analysis.noise_reference == "self"
    assert "no_quiet_noise_reference" in analysis.reduced_confidence
    assert analysis.signal_center_hz == pytest.approx(915.2e6, abs=2e3)
    assert analysis.grounded_band_ids == frozenset()
    assert analysis.context[0].present_before_trigger is None  # no reference to tell


def test_non_finite_samples_reduce_confidence():
    analysis = analyse_snippet(_snippet(_two_emitters(), non_finite=3), BANDS, UnavailableClassifier())
    assert analysis.reduced_confidence == ("non_finite_samples",)
    assert analysis.grounded_band_ids == frozenset()


def test_an_always_on_emitter_is_still_the_primary():
    """An always-on emitter fills its own pre-trigger: no quiet reference,
    so the self floor finds it (with reduced confidence), never nothing."""
    rng = np.random.default_rng(23)
    iq = synthetic.band_limited(rng, N, FS, 0.9 * FS, 0.0, 1e-3) + synthetic.noise(rng, N, 1e-5)
    analysis = analyse_snippet(_snippet(iq), BANDS, UnavailableClassifier())
    assert analysis.primary.obw_hz == pytest.approx(0.9 * FS, rel=0.03)
    assert analysis.noise_reference == "self"
    assert "no_quiet_noise_reference" in analysis.reduced_confidence


def test_out_of_table_signal_is_ungrounded_unless_the_classifier_predicts():
    snippet = Snippet(iq=_two_emitters(), sample_rate=FS, center_freq_hz=433.92e6, truncated=False, pre_trigger_samples=PRE)
    unpredicted = analyse_snippet(snippet, BANDS, UnavailableClassifier())
    assert (unpredicted.grounded_band_ids, unpredicted.modulation_label) == (frozenset(), None)
    analysis = analyse_snippet(snippet, BANDS, _AlwaysPredicts())
    assert analysis.band_matches == () and analysis.modulation_label == "lora"


def test_an_unreliable_bandwidth_grounds_nothing():
    """A 125 kHz signal straddling the +-fs/2 edge of a 914.5 MHz tuning
    sits at 915.0 MHz inside 902-928 MHz, but touching both band edges makes
    its bandwidth unreliable, so the match is listed, not grounded."""
    rng = np.random.default_rng(22)
    burst = synthetic.band_limited(rng, N, FS, 125e3, FS / 2, 1e-3)
    burst[:PRE] = 0
    analysis = analyse_snippet(
        Snippet(iq=burst + synthetic.noise(rng, N, 1e-5), sample_rate=FS, center_freq_hz=914.5e6, truncated=False, pre_trigger_samples=PRE),
        BANDS,
        UnavailableClassifier(),
    )
    assert analysis.noise_reference == "pre_trigger"
    assert not analysis.primary.bandwidth_reliable
    assert analysis.reduced_confidence == ("bandwidth_unreliable",)
    assert [(m.entry.id, m.grounded) for m in analysis.band_matches] == [("ism_902_928", False)]
    assert analysis.grounded_band_ids == frozenset()


def test_empty_spectrum_has_no_primary():
    rng = np.random.default_rng(21)
    analysis = analyse_snippet(_snippet(synthetic.noise(rng, N, 1e-5)), BANDS, UnavailableClassifier())
    assert analysis.primary is None and analysis.band_matches == () and not analysis.grounded_band_ids
    with pytest.raises(ValueError):
        build_user_message(analysis)


def test_prompt_holds_only_numeric_features_and_curated_entries():
    message = build_user_message(analyse_snippet(_snippet(_two_emitters()), BANDS, UnavailableClassifier()))
    payload = json.loads(message[message.index("{") :])
    assert set(payload) == {"capture", "primary_emitter", "other_emitters", "band_table_matches", "modulation_classifier"}
    assert payload["primary_emitter"]["center_mhz"] == pytest.approx(915.2, abs=0.002)
    assert payload["primary_emitter"]["symbol_rate_khz"] is None
    assert payload["band_table_matches"][0]["id"] == "ism_902_928"
    assert payload["band_table_matches"][0]["grounded"] is True
    assert payload["other_emitters"][0]["resolution_khz"] == pytest.approx(FS / 1024 / 1e3, rel=1e-4)
    assert set(payload["capture"]) == {
        "tuned_center_mhz", "sample_rate_msps", "analysed_seconds", "truncated", "noise_reference",
        "reduced_confidence",
    }
    assert (payload["capture"]["noise_reference"], payload["capture"]["reduced_confidence"]) == ("pre_trigger", [])
    assert payload["other_emitters"][0]["present_before_trigger"] is True
    assert set(payload["primary_emitter"]) == {
        "center_mhz", "occupied_bandwidth_khz", "snr_db", "duty_cycle", "burst_count", "mean_burst_ms",
        "papr_db", "spectral_flatness", "symbol_rate_khz", "analysis_rate_msps", "bandwidth_reliable",
    }
    for forbidden in ("survey", "operator", "latitude", "longitude", "author", "description", "path", "sigmf"):
        assert forbidden not in message.lower()
```

- [ ] **Step 2: Run them to verify they fail**

<!-- check: t8_red -->
Run: `python -m pytest tests/agent/test_analysis_prompt.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'agent.analysis'`.


- [ ] **Step 3: Implement**

<!-- write: agent/classifier.py -->
```python
# agent/classifier.py
"""The modulation-classifier seam (design decision 2).

v1 ships only UnavailableClassifier. The TorchSig model trained on the DGX
Spark and exported to the Jetson plugs in here later, behind the same
protocol; until then routing can only ground a result in the band table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ModulationPrediction:
    label: str
    confidence: float


class ModulationClassifier(Protocol):
    def predict(self, iq: np.ndarray, sample_rate: float) -> ModulationPrediction | None: ...


class UnavailableClassifier:
    """No trained model exists yet: never a prediction."""

    def predict(self, iq: np.ndarray, sample_rate: float) -> ModulationPrediction | None:
        return None
```

<!-- write: agent/analysis.py -->
```python
# agent/analysis.py
"""Everything the agent measures about one snippet, before the LLM sees it.

Segmentation picks the primary region: the burst that triggered capture,
measured against the snippet's quiet pre-trigger reference when there is
one. Features are measured on that region alone. Up to three other
emitters go along as context: those already on before the trigger (found in
the reference) and any other burst-time regions. The primary's absolute
center and fine OBW are matched against the curated band table; the
modulation classifier gets the capture. Anything that makes the
measurements less trustworthy is listed in reduced_confidence and keeps
every band match ungrounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent.band_table import BandEntry, BandMatch, match_bands
from agent.classifier import ModulationClassifier, ModulationPrediction
from agent.snippet_reader import Snippet
from dsp.features import RegionFeatures, region_features
from dsp.segmentation import MAX_CONTEXT_REGIONS, NFFT, Segmentation, SpectralRegion, segment_spectrum


@dataclass(frozen=True)
class ContextRegion:
    center_hz: float  # absolute
    obw_hz: float  # coarse: resolution is sample_rate / NFFT
    power_relative_to_primary_db: float
    # True: already on in the pre-trigger reference; False: new during the
    # burst; None: no reference to tell.
    present_before_trigger: bool | None


@dataclass(frozen=True)
class SnippetAnalysis:
    tuned_center_hz: float
    sample_rate: float
    noise_reference: str  # the floor's source: "pre_trigger" or "self" (dsp.segmentation)
    reduced_confidence: tuple[str, ...]  # why the measurements are less trustworthy
    analysed_seconds: float
    truncated: bool
    coarse_resolution_hz: float
    primary: RegionFeatures | None  # None: no occupied region at all
    signal_center_hz: float | None  # absolute center of the primary region
    context: tuple[ContextRegion, ...]
    band_matches: tuple[BandMatch, ...]
    modulation: ModulationPrediction | None

    @property
    def grounded_band_ids(self) -> frozenset[str]:
        """Ids of the band-table entries the primary is grounded in. Routing
        accepts a tag only under one of these prefixes."""
        return frozenset(match.entry.id for match in self.band_matches if match.grounded)

    @property
    def modulation_label(self) -> str | None:
        return None if self.modulation is None else self.modulation.label


def analyse_snippet(
    snippet: Snippet, bands: tuple[BandEntry, ...], classifier: ModulationClassifier
) -> SnippetAnalysis:
    reference = snippet.iq[: snippet.pre_trigger_samples] if snippet.pre_trigger_samples else None
    segmentation = segment_spectrum(snippet.iq, snippet.sample_rate, reference)
    primary_region = segmentation.primary
    primary = None if primary_region is None else region_features(snippet.iq, segmentation, primary_region)
    signal_center = None if primary is None else snippet.center_freq_hz + primary.center_offset_hz
    reasons = _reduced_confidence(segmentation.floor_source, snippet.non_finite_samples, primary)
    context = () if primary_region is None else _context(snippet, segmentation, primary_region)
    # Any reduced-confidence reason keeps every match ungrounded: an
    # unreliable bandwidth makes the OBW plausibility check meaningless.
    matches = () if primary is None else tuple(
        BandMatch(match.entry, match.grounded and not reasons)
        for match in match_bands(bands, signal_center, primary.obw_hz)
    )
    return SnippetAnalysis(
        tuned_center_hz=snippet.center_freq_hz,
        sample_rate=snippet.sample_rate,
        noise_reference=segmentation.floor_source,
        reduced_confidence=reasons,
        analysed_seconds=len(snippet.iq) / snippet.sample_rate,
        truncated=snippet.truncated,
        coarse_resolution_hz=snippet.sample_rate / NFFT,
        primary=primary,
        signal_center_hz=signal_center,
        context=context,
        band_matches=matches,
        modulation=classifier.predict(snippet.iq, snippet.sample_rate),
    )


def _reduced_confidence(floor_source: str, non_finite: int, primary: RegionFeatures | None) -> tuple[str, ...]:
    reasons = []
    if floor_source == "self":
        reasons.append("no_quiet_noise_reference")  # the self floor is blind to roll-off
    if non_finite:
        reasons.append("non_finite_samples")
    if primary is not None and not primary.bandwidth_reliable:
        reasons.append("bandwidth_unreliable")
    return tuple(reasons)


def _context(snippet: Snippet, segmentation: Segmentation, primary: SpectralRegion) -> tuple[ContextRegion, ...]:
    """The strongest other emitters: already on before the trigger, or new
    during the burst (unknown without a reference)."""
    during = None if segmentation.floor_source == "self" else False
    candidates = [(region, during) for region in segmentation.regions[1:]]
    candidates += [(region, True) for region in segmentation.before_trigger]
    candidates.sort(key=lambda candidate: candidate[0].excess_power, reverse=True)
    return tuple(
        ContextRegion(
            center_hz=snippet.center_freq_hz + region.center_offset_hz,
            obw_hz=region.obw_hz,
            power_relative_to_primary_db=10 * math.log10(region.excess_power / primary.excess_power),
            present_before_trigger=before,
        )
        for region, before in candidates[:MAX_CONTEXT_REGIONS]
    )
```

<!-- write: agent/prompt.py -->
```python
# agent/prompt.py
"""What the LLM is told. Numeric features, region summaries, curated
band-table entries and the classifier output only: never SigMF free text
(author/description are an injection surface), never location, survey or
operator IDs."""

from __future__ import annotations

import json

from agent.analysis import SnippetAnalysis

SYSTEM_PROMPT = """\
You classify unknown RF emissions captured by a passive drive-survey receiver in the US.
You receive measured features of one capture and the curated band-table entries its
frequency falls in. Call record_classification exactly once.

- tag: the most likely identity, lower case, "<band-table id>:<signal>", e.g.
  "ism_902_928:lora" or "pcs_downlink:lte". Only a tag prefixed with the id of a grounded
  entry can be accepted without human review. Use null if you cannot propose one.
- confidence: the probability your tag is right. Above 0.85 only when the features agree
  with a grounded band-table entry. Spectral features alone rarely justify high confidence.
  bandwidth_reliable false means the occupied bandwidth could not be measured (the signal
  fills the capture or wraps its edge); reduced_confidence lists why the measurements are
  less trustworthy, and no tag is accepted without review while it is non-empty.
  Other emitters with present_before_trigger true were already on before the capture
  triggered; the primary emitter is the one that triggered it.
- reasoning: the evidence, and any alternative identities you considered.

Your output is a label stored for human review. It never triggers any other action."""

_SIGNIFICANT = 6


def _round(value: float | None) -> float | None:
    return None if value is None else float(f"{value:.{_SIGNIFICANT}g}")


def build_user_message(analysis: SnippetAnalysis) -> str:
    primary = analysis.primary
    if primary is None:
        raise ValueError("No primary region: there is nothing to classify")
    payload = {
        "capture": {
            "tuned_center_mhz": _round(analysis.tuned_center_hz / 1e6),
            "sample_rate_msps": _round(analysis.sample_rate / 1e6),
            "analysed_seconds": _round(analysis.analysed_seconds),
            "truncated": analysis.truncated,
            "noise_reference": analysis.noise_reference,
            "reduced_confidence": list(analysis.reduced_confidence),
        },
        "primary_emitter": {
            "center_mhz": _round(analysis.signal_center_hz / 1e6),
            "occupied_bandwidth_khz": _round(primary.obw_hz / 1e3),
            "snr_db": _round(primary.snr_db),
            "duty_cycle": _round(primary.duty_cycle),
            "burst_count": primary.burst_count,
            "mean_burst_ms": _round(None if primary.mean_burst_s is None else primary.mean_burst_s * 1e3),
            "papr_db": _round(primary.papr_db),
            "spectral_flatness": _round(primary.spectral_flatness),
            "symbol_rate_khz": _round(None if primary.symbol_rate_hz is None else primary.symbol_rate_hz / 1e3),
            "analysis_rate_msps": _round(primary.analysis_rate_hz / 1e6),
            "bandwidth_reliable": primary.bandwidth_reliable,
        },
        "other_emitters": [
            {
                "center_mhz": _round(region.center_hz / 1e6),
                "occupied_bandwidth_khz": _round(region.obw_hz / 1e3),
                "resolution_khz": _round(analysis.coarse_resolution_hz / 1e3),
                "power_relative_to_primary_db": _round(region.power_relative_to_primary_db),
                "present_before_trigger": region.present_before_trigger,
            }
            for region in analysis.context
        ],
        "band_table_matches": [
            {
                "id": match.entry.id,
                "service": match.entry.service,
                "range_mhz": [_round(match.entry.start_hz / 1e6), _round(match.entry.end_hz / 1e6)],
                "expected_occupied_bandwidth_khz": [
                    _round(match.entry.expected_obw_hz[0] / 1e3),
                    _round(match.entry.expected_obw_hz[1] / 1e3),
                ],
                "typical_signals": list(match.entry.typical_signals),
                "citation": match.entry.citation,
                "grounded": match.grounded,
            }
            for match in analysis.band_matches
        ],
        "modulation_classifier": None
        if analysis.modulation is None
        else {"label": analysis.modulation.label, "confidence": _round(analysis.modulation.confidence)},
    }
    return (
        "Features of one unknown-signal capture. Frequencies are absolute; powers are "
        "uncalibrated (dBFS-relative).\n" + json.dumps(payload, indent=1)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

<!-- check: t8_green -->
Run: `python -m pytest tests/agent/test_analysis_prompt.py -q`
Expected: PASS, `8 passed`.


- [ ] **Step 5: Commit**

<!-- run -->
```bash
git add agent/classifier.py agent/analysis.py agent/prompt.py tests/agent/test_analysis_prompt.py
git commit -m "feat: add snippet analysis, the classifier seam and the agent prompt"
```

---

### Task 9: The forced-tool LLM call and routing

**Files:**
- Create: `agent/llm.py`, `agent/routing.py`
- Modify: `pyproject.toml` (anthropic dependency)
- Test: `tests/agent/test_llm.py`, `tests/agent/test_routing.py`, and
  `tests/agent/test_llm_live.py` (optional, skipped by default)

**Interfaces:**
- Consumes: anthropic, pydantic and `ClassificationStatus`. The live test also uses
  Tasks 6–8.
- Produces:
  - `agent.llm`:
    - `DEFAULT_MODEL = "claude-sonnet-5-5"`, `DEFAULT_MAX_TOKENS = 1024`,
      `TOOL_NAME = "record_classification"`, `TAG_PATTERN`, `MAX_REASONING_CHARS = 2000`;
    - `RECORD_CLASSIFICATION_TOOL`;
    - `Classification(tag: str | None, confidence: float, reasoning: str)`, frozen,
      strict, `extra="forbid"`;
    - `LlmResult(classification, failure, tokens_used)`;
    - the `MessagesClient` protocol;
    - `build_request(system, user_message, model, max_tokens) -> dict`;
    - `parse_response(message) -> LlmResult`;
    - `request_classification(client, system, user_message, model, max_tokens) -> LlmResult`
      (API exceptions propagate);
    - `is_transient(exc) -> bool`.
  - `agent.routing`:
    - `AUTO_CLASSIFY_CONFIDENCE = 0.85`, `MAX_REASONING_CHARS = 4000`;
    - `Decision(status, tag, confidence, reasoning)`, frozen;
    - `route(classification, grounded_band_ids: frozenset[str], modulation_label: str | None) -> Decision`.
      It returns `auto_classified` only when the tag's band prefix (before the first `:`)
      is a grounded id, or its last part equals the modulation label;
    - `needs_review(reason) -> Decision`, with a NULL tag, confidence 0 and the reason
      truncated to 4000 characters.

Failure shapes treated as validation failures: `stop_reason` of `max_tokens` or
`refusal`, anything but exactly one `record_classification` block, and any pydantic error.
The error text excludes the model's input (`include_input=False`) so the reasoning
stays bounded.

- [ ] **Step 1: Install the SDK (user site, never editable)**

```bash
pip install --user 'anthropic>=0.107'
python -c "import anthropic; print(anthropic.__version__)"
```

Expected: `0.107.1` or newer.

- [ ] **Step 2: Write the failing tests**

<!-- write: tests/agent/test_llm.py -->
```python
# tests/agent/test_llm.py
"""The request goes through the real anthropic SDK (0.107) over an httpx
MockTransport, so the wire format and the response parsing are the SDK's
own: no network, no API key."""

import json

import anthropic
import httpx
import pytest
from anthropic.types import Message

from agent.llm import (
    RECORD_CLASSIFICATION_TOOL,
    TOOL_NAME,
    Classification,
    is_transient,
    parse_response,
    request_classification,
)

MODEL = "claude-sonnet-5-5"


def _message(stop_reason: str, content: list[dict]) -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 800, "output_tokens": 90},
        "content": content,
    }


def _tool_use(tool_input: dict, name: str = TOOL_NAME) -> dict:
    return {"type": "tool_use", "id": "toolu_01", "name": name, "input": tool_input}


GOOD = {"tag": "ism_902_928:lora", "confidence": 0.9, "reasoning": "Chirp-like, 125 kHz."}


def _client(handler) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_request_is_one_forced_tool_call_with_thinking_off():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_message("tool_use", [_tool_use(GOOD)]))

    result = request_classification(_client(handler), "system text", "user text", MODEL, 1024)
    (body,) = sent
    assert body == {
        "model": MODEL,
        "max_tokens": 1024,
        "system": "system text",
        "messages": [{"role": "user", "content": "user text"}],
        "tools": [RECORD_CLASSIFICATION_TOOL],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "thinking": {"type": "disabled"},
    }
    assert result.classification == Classification(**GOOD)
    assert result.failure is None and result.tokens_used == 890


def test_tool_schema_matches_the_db_contract():
    schema = RECORD_CLASSIFICATION_TOOL["input_schema"]
    assert schema["required"] == ["tag", "confidence", "reasoning"]
    assert schema["properties"]["tag"]["pattern"] == "^[a-z0-9][a-z0-9_.:-]{0,63}$"
    assert schema["properties"]["reasoning"]["maxLength"] == 2000
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "message, failure",
    [
        (_message("max_tokens", [_tool_use({"tag": "x"})]), "stop_reason was max_tokens"),
        (_message("refusal", []), "stop_reason was refusal"),
        (_message("end_turn", [{"type": "text", "text": "It is LoRa."}]), "got 0"),
        (_message("tool_use", [_tool_use(GOOD, name="something_else")]), "got 0"),
        (_message("tool_use", [_tool_use(GOOD), _tool_use(GOOD)]), "got 2"),
        (_message("tool_use", [_tool_use({**GOOD, "tag": "Not A Tag"})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "tag": "abc\n"})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "confidence": 1.5})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "confidence": "0.9"})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "reasoning": "x" * 2001})]), "validation"),
        (_message("tool_use", [_tool_use({"tag": "x", "confidence": 0.5})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "extra": 1})]), "validation"),
    ],
)
def test_failure_shapes_are_validation_failures(message, failure):
    result = parse_response(Message.model_validate(message))
    assert result.classification is None
    assert failure in result.failure
    assert result.tokens_used == 890
    assert len(result.failure) < 1000  # never echoes the (untrusted, long) input


def test_null_tag_is_a_valid_classification():
    result = parse_response(Message.model_validate(_message("tool_use", [_tool_use({**GOOD, "tag": None})])))
    assert result.classification.tag is None


@pytest.mark.parametrize(
    "status, transient",
    [(408, True), (409, True), (429, True), (500, True), (503, True), (529, True),
     (400, False), (401, False), (403, False), (404, False), (413, False), (422, False)],
)
def test_transient_statuses_match_the_sdks_own_retry_rule(status, transient):
    def handler(request):
        return httpx.Response(status, json={"type": "error", "error": {"type": "x", "message": "m"}})

    with pytest.raises(anthropic.APIStatusError) as excinfo:
        request_classification(_client(handler), "s", "u", MODEL, 16)
    assert is_transient(excinfo.value) is transient


def test_connection_errors_and_timeouts_are_transient():
    def refuse(request):
        raise httpx.ConnectError("refused")

    def stall(request):
        raise httpx.ReadTimeout("slow")

    for handler in (refuse, stall):
        with pytest.raises(anthropic.APIConnectionError) as excinfo:
            request_classification(_client(handler), "s", "u", MODEL, 16)
        assert is_transient(excinfo.value)
    assert not is_transient(ValueError("a bug"))
```

<!-- write: tests/agent/test_routing.py -->
```python
# tests/agent/test_routing.py
import pytest

from agent.llm import Classification
from agent.routing import AUTO_CLASSIFY_CONFIDENCE, needs_review, route
from schema.records import ClassificationStatus

GROUNDED = frozenset({"ism_902_928"})


def _c(tag="ism_902_928:lora", confidence=0.9, reasoning="Because.") -> Classification:
    return Classification(tag=tag, confidence=confidence, reasoning=reasoning)


def test_threshold_is_085():
    assert AUTO_CLASSIFY_CONFIDENCE == 0.85


def test_confident_tag_under_a_grounded_band_is_auto_classified_verbatim():
    decision = route(_c(confidence=0.85), GROUNDED, None)
    assert decision.status is ClassificationStatus.AUTO_CLASSIFIED
    assert (decision.tag, decision.confidence, decision.reasoning) == ("ism_902_928:lora", 0.85, "Because.")


def test_a_modulation_prediction_grounds_a_tag_ending_in_its_label():
    decision = route(_c(tag="unknown_band:lora", confidence=0.9), frozenset(), "lora")
    assert decision.status is ClassificationStatus.AUTO_CLASSIFIED


@pytest.mark.parametrize(
    "classification, grounded, modulation, why",
    [
        (_c(confidence=0.8499), GROUNDED, None, "below 0.85"),
        (_c(confidence=1.0), frozenset(), None, "no grounded band-table match"),
        (_c(tag=None, confidence=1.0), GROUNDED, None, "proposed no tag"),
        # A confident tag naming a band the signal is not grounded in.
        (_c(tag="pcs_downlink:lte", confidence=0.99), GROUNDED, None, "'pcs_downlink' is not a grounded"),
        (_c(tag="lora", confidence=0.99), GROUNDED, None, "'lora' is not a grounded"),
        (_c(tag="unknown_band:fsk", confidence=0.99), frozenset(), "lora", "'unknown_band' is not a grounded"),
    ],
)
def test_everything_else_needs_review_keeping_the_models_answer(classification, grounded, modulation, why):
    decision = route(classification, grounded, modulation)
    assert decision.status is ClassificationStatus.NEEDS_REVIEW
    assert (decision.tag, decision.confidence) == (classification.tag, classification.confidence)
    assert decision.reasoning.startswith("Because.") and why in decision.reasoning


def test_routed_reasoning_fits_the_db_limit():
    decision = route(_c(confidence=0.1, reasoning="x" * 2000), frozenset(), None)
    assert len(decision.reasoning) <= 4000


def test_needs_review_has_null_tag_zero_confidence_and_bounded_reason():
    decision = needs_review("y" * 5000)
    assert decision.status is ClassificationStatus.NEEDS_REVIEW
    assert (decision.tag, decision.confidence) == (None, 0.0)
    assert len(decision.reasoning) == 4000
```

<!-- write: tests/agent/test_llm_live.py -->
```python
# tests/agent/test_llm_live.py
"""Optional live smoke test: the only check of the model id, the forced
tool_choice and thinking=disabled against the real API. Skipped unless
ANTHROPIC_API_KEY is set AND SURVEYTOOL_LIVE_LLM_TEST=1 (it costs tokens)."""

import os

import anthropic
import numpy as np
import pytest

from agent.analysis import analyse_snippet
from agent.band_table import load_band_table
from agent.classifier import UnavailableClassifier
from agent.llm import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, request_classification
from agent.prompt import SYSTEM_PROMPT, build_user_message
from agent.snippet_reader import Snippet
from dsp import synthetic

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("SURVEYTOOL_LIVE_LLM_TEST") == "1"),
    reason="live LLM test: set ANTHROPIC_API_KEY and SURVEYTOOL_LIVE_LLM_TEST=1",
)


def test_live_forced_tool_call_returns_a_valid_classification():
    rng = np.random.default_rng(1)
    n, fs = 1 << 18, 1e6
    iq = synthetic.noise(rng, n, 1e-5) + synthetic.gate(
        synthetic.band_limited(rng, n, fs, 125e3, 200e3, 1e-3), fs, [(0.05, 0.1)]
    )
    analysis = analyse_snippet(
        Snippet(iq=iq, sample_rate=fs, center_freq_hz=915e6, truncated=False, pre_trigger_samples=50_000),
        load_band_table().entries,
        UnavailableClassifier(),
    )
    client = anthropic.Anthropic(max_retries=2, timeout=60.0)
    result = request_classification(
        client, SYSTEM_PROMPT, build_user_message(analysis), DEFAULT_MODEL, DEFAULT_MAX_TOKENS
    )
    assert result.classification is not None, result.failure
    assert result.tokens_used > 0
```

- [ ] **Step 3: Run them to verify they fail**

<!-- check: t9_red -->
Run: `python -m pytest tests/agent/test_llm.py tests/agent/test_routing.py -q`
Expected: collection ERRORs, `2 errors`, starting with `ModuleNotFoundError: No module named 'agent.llm'`.


- [ ] **Step 4: Implement**

<!-- write: agent/llm.py -->
```python
# agent/llm.py
"""One forced `record_classification` tool call per record, and nothing else.

The model gets exactly one tool, the forced output tool, so it can never
take an action. Thinking is explicitly disabled: it cannot be combined with
a forced tool_choice. No prompt caching: the constant system prompt is
below the minimum cacheable size, so caching would buy nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
from anthropic.types import Message
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 1024
TOOL_NAME = "record_classification"
TAG_PATTERN = r"^[a-z0-9][a-z0-9_.:-]{0,63}$"  # also enforced by classify_unknown
MAX_REASONING_CHARS = 2000

RECORD_CLASSIFICATION_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Record your classification of the unknown RF emission described in the "
        "user message. This is your only output; call it exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tag": {
                "type": ["string", "null"],
                "pattern": TAG_PATTERN,
                "description": (
                    "Lower-case identity label, e.g. 'ism_902_928:lora' or 'fm_broadcast:wbfm'. "
                    "Prefix with the band table id when an entry fits. null if you cannot "
                    "propose any identity."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Probability that the tag is correct.",
            },
            "reasoning": {
                "type": "string",
                "maxLength": MAX_REASONING_CHARS,
                "description": "The evidence for the tag, and any alternative identities.",
            },
        },
        "required": ["tag", "confidence", "reasoning"],
        "additionalProperties": False,
    },
}


class Classification(BaseModel):
    """The validated tool input. Anything else the model sends is a failure."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tag: str | None = Field(pattern=TAG_PATTERN)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=MAX_REASONING_CHARS)


@dataclass(frozen=True)
class LlmResult:
    classification: Classification | None  # None: the response failed validation
    failure: str | None  # why it failed, for the needs_review reasoning
    tokens_used: int


class _Messages(Protocol):
    def create(self, **kwargs: Any) -> Message: ...


class MessagesClient(Protocol):
    """The slice of anthropic.Anthropic the agent uses (tests inject fakes)."""

    @property
    def messages(self) -> _Messages: ...


def build_request(system: str, user_message: str, model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_message}],
        "tools": [RECORD_CLASSIFICATION_TOOL],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "thinking": {"type": "disabled"},
    }


def parse_response(message: Message) -> LlmResult:
    tokens = message.usage.input_tokens + message.usage.output_tokens
    if message.stop_reason in ("max_tokens", "refusal"):
        return LlmResult(None, f"stop_reason was {message.stop_reason}", tokens)
    calls = [block for block in message.content if block.type == "tool_use" and block.name == TOOL_NAME]
    if len(calls) != 1:
        return LlmResult(None, f"expected one {TOOL_NAME} call, got {len(calls)}", tokens)
    try:
        classification = Classification.model_validate(calls[0].input)
    except ValidationError as exc:
        return LlmResult(None, f"tool input failed validation: {exc.errors(include_url=False, include_input=False)}", tokens)
    return LlmResult(classification, None, tokens)


def request_classification(
    client: MessagesClient, system: str, user_message: str, model: str, max_tokens: int
) -> LlmResult:
    """One API call. API exceptions propagate: see is_transient."""
    return parse_response(client.messages.create(**build_request(system, user_message, model, max_tokens)))


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: a network failure or timeout, or a status the SDK
    itself retries (408, 409, 429, >= 500). Matched on the status code
    because 529 raises OverloadedError, which is not an InternalServerError
    and is not exported by anthropic 0.107."""
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    return isinstance(exc, anthropic.APIStatusError) and (
        exc.status_code in (408, 409, 429) or exc.status_code >= 500
    )
```

<!-- write: agent/routing.py -->
```python
# agent/routing.py
"""Confidence-based routing (pure). auto_classified needs a tag, confidence
>= 0.85, AND grounding of that very tag: its band prefix (the part before
the first ':') is the id of a grounded band-table entry, or its last part is
the modulation classifier's label. Everything else is needs_review, which is
terminal for the agent."""

from __future__ import annotations

from dataclasses import dataclass

from agent.llm import Classification
from schema.records import ClassificationStatus

AUTO_CLASSIFY_CONFIDENCE = 0.85
MAX_REASONING_CHARS = 4000  # classify_unknown rejects anything longer


@dataclass(frozen=True)
class Decision:
    status: ClassificationStatus
    tag: str | None
    confidence: float
    reasoning: str


def route(
    classification: Classification, grounded_band_ids: frozenset[str], modulation_label: str | None
) -> Decision:
    tag = classification.tag
    if tag is None:
        why = "the model proposed no tag"
    elif classification.confidence < AUTO_CLASSIFY_CONFIDENCE:
        why = f"confidence {classification.confidence:.2f} is below {AUTO_CLASSIFY_CONFIDENCE:.2f}"
    elif not grounded_band_ids and modulation_label is None:
        why = "no grounded band-table match and no modulation prediction"
    elif tag.split(":")[0] not in grounded_band_ids and tag.split(":")[-1] != modulation_label:
        why = (
            f"the tag's band prefix {tag.split(':')[0]!r} is not a grounded band-table entry "
            f"({', '.join(sorted(grounded_band_ids)) or 'none'})"
        )
    else:
        return Decision(
            ClassificationStatus.AUTO_CLASSIFIED,
            classification.tag,
            classification.confidence,
            classification.reasoning,
        )
    return Decision(
        ClassificationStatus.NEEDS_REVIEW,
        classification.tag,
        classification.confidence,
        f"{classification.reasoning}\n\nRouted to needs_review: {why}.",
    )


def needs_review(reason: str) -> Decision:
    """A judgment about the record that produced no usable classification:
    NULL tag, zero confidence, the reason as the reasoning."""
    return Decision(ClassificationStatus.NEEDS_REVIEW, None, 0.0, reason[:MAX_REASONING_CHARS])
```

In `pyproject.toml`:

<!-- edit: pyproject.toml -->
Replace

```toml
    "sigmf>=1.13",  # capture/unknown snippet files (LGPL-3.0-or-later)
    "psycopg[binary]>=3.2",
```

with

```toml
    "sigmf>=1.13",  # capture/unknown snippet files (LGPL-3.0-or-later)
    "anthropic>=0.107",  # agent/ (Part 4) LLM call
    "psycopg[binary]>=3.2",
```

- [ ] **Step 5: Run the tests to verify they pass**

<!-- check: t9_green -->
Run: `python -m pytest tests/agent/test_llm.py tests/agent/test_routing.py tests/agent/test_llm_live.py -q`
Expected: PASS, `39 passed, 1 skipped`. The skip is the live test.


The live test is skipped. To run it, which costs a few thousand tokens:
`ANTHROPIC_API_KEY=... SURVEYTOOL_LIVE_LLM_TEST=1 python -m pytest tests/agent/test_llm_live.py -v`.
It was not run while writing this plan.

- [ ] **Step 6: Commit**

<!-- run -->
```bash
git add agent/llm.py agent/routing.py pyproject.toml tests/agent/test_llm.py tests/agent/test_routing.py tests/agent/test_llm_live.py
git commit -m "feat: add the forced record_classification LLM call and confidence routing"
```

---

### Task 10: The agent service, its failure handling and the `sdr-agent` CLI

**Files:**
- Create: `agent/service.py`
- Modify: `pyproject.toml` (the `sdr-agent` script)
- Test: `tests/agent/test_service.py`. The gateway and LLM are in-memory fakes. Sleep and
  the clock are injected, so no real time passes.

**Interfaces:**
- Consumes: everything from Tasks 5–9. Exact names:
  - `read_snippet`, `SnippetOutsideStore`, `SnippetUnreadable`;
  - `analyse_snippet`, `build_user_message`, `SYSTEM_PROMPT`;
  - `request_classification`, `is_transient`;
  - `route`, `needs_review`;
  - `AgentGateway`, `PendingRecord`, `RecordNotPending`, `SubmitRejected`,
    `SubmitTimedOut`, `connect_gateway`;
  - `load_band_table`.
- Produces, in `agent.service`:
  - `SystemicFault(RuntimeError)`;
  - `AgentSettings(snippet_root, model, max_tokens, batch_size=20, max_consecutive_snippet_failures=5, max_consecutive_bad_outputs=5, daily_token_budget=2_000_000, backoff_seconds=2.0, max_backoff_seconds=300.0)`;
  - `ClassificationAgent(gateway, client, settings, bands, classifier=None, sleep=time.sleep, now=utc_now, start_after_id=0)`,
    with:
    - `.check_snippet_root()`, the startup probe;
    - `.run_batch() -> int`;
    - `.take_retry_delay() -> float | None`;
  - `run(agent, poll_seconds, stopping, sleep, max_failed_batches=5)`;
  - `main(argv)`, exposed as `sdr-agent --snippet-store-dir DIR [--agent-role] [--model] [--max-tokens] [--daily-token-budget] [--poll-seconds] [--band-table] [--start-after-id]`.

Failure handling, as tested:

| Situation | Outcome |
|---|---|
| Startup: relative root, or none of the 5 newest pending snippets reads | `SystemicFault` ("mount the store read-only at the identical resolved path") |
| No snippet (`iq_snippet_path` NULL) | `needs_review` at once: NULL tag, the flag named, no LLM call, never counted toward a halt |
| Snippet outside the store | Held pending. Goes to `needs_review` once a later snippet is analysed |
| Always-on emitter (no quiet pre-trigger frame) | Classified from the self floor with `reduced_confidence`; it is the primary, never "no region", and never counts toward a halt |
| Some samples NaN or inf | Zeroed and counted; classified with `reduced_confidence` |
| Snippet unreadable (every sample non-finite included), no occupied region, or feature extraction failed | Stays pending |
| 5 snippet failures of any kind in a row | `SystemicFault` naming the ids and `--start-after-id`, with nothing marked |
| Transient API error | Cursor rewound to the record, backoff 2, 4, … s (capped at 300 s), retried indefinitely; never marked, never halts |
| Any other API error | `SystemicFault` naming the record |
| Invalid model output (validation failure, refusal, `max_tokens`) | Held pending. Goes to `needs_review` once a later answer validates; 5 in a row → `SystemicFault` |
| Write refused (`SubmitRejected`) | `SystemicFault` |
| Write timed out (`SubmitTimedOut`) | Logged; stays pending; the cursor moves on |
| Any other write error | The batch fails. The kept decision is written when the record is fetched again, with no second LLM call. 5 failed batches → `SystemicFault` |
| `RecordNotPending` | Logged and skipped |
| Daily budget spent | Sleeps until UTC midnight |

- [ ] **Step 1: Write the failing tests**

<!-- write: tests/agent/test_service.py -->
```python
# tests/agent/test_service.py
"""The agent loop against an in-memory gateway and a scripted LLM client.
No database, no network, no real time (sleep and the clock are injected)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import httpx
import numpy as np
import pytest
from anthropic.types import Message

from agent.band_table import load_band_table
from agent.db_gateway import PendingRecord, RecordNotPending, SubmitRejected, SubmitTimedOut
from agent.service import AgentSettings, ClassificationAgent, SystemicFault, main, run
from capture.unknown.snippet_writer import write_sigmf_snippet
from dsp import synthetic
from schema.records import ClassificationStatus

FS = 1e6
TUNED = 915e6
START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
GOOD = {"tag": "ism_902_928:lora", "confidence": 0.9, "reasoning": "125 kHz bursts in 902-928 MHz."}
AUTO, REVIEW = ClassificationStatus.AUTO_CLASSIFIED, ClassificationStatus.NEEDS_REVIEW


class FakeGateway:
    def __init__(self, records, not_pending=(), fail_once=(), timed_out=(), rejected=()):
        self.records = records
        self.not_pending, self.timed_out, self.rejected = set(not_pending), set(timed_out), set(rejected)
        self.fail_once = set(fail_once)
        self.fetches: list[tuple[int, int]] = []
        self.submitted: list[tuple] = []

    def fetch_pending(self, after_id, limit):
        self.fetches.append((after_id, limit))
        return [r for r in self.records if r.id > after_id][:limit]

    def fetch_newest_with_snippet(self, after_id, limit):
        withs = [r for r in self.records if r.id > after_id and r.iq_snippet_path is not None]
        return sorted(withs, key=lambda r: r.id, reverse=True)[:limit]

    def submit_classification(self, record_id, status, tag, confidence, reasoning):
        if record_id in self.fail_once:
            self.fail_once.discard(record_id)
            raise ConnectionError("database went away")
        for ids, error in ((self.not_pending, RecordNotPending), (self.timed_out, SubmitTimedOut), (self.rejected, SubmitRejected)):
            if record_id in ids:
                raise error(f"record {record_id}")
        self.submitted.append((record_id, status, tag, confidence, reasoning))


class ScriptedClient:
    """Each create() pops the next scripted outcome: a tool-input dict, an
    exception instance, or a full Message dict."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.messages = self

    def create(self, **kwargs) -> Message:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if "content" not in outcome:
            outcome = _message([{"type": "tool_use", "id": "t", "name": "record_classification", "input": outcome}])
        return Message.model_validate(outcome)


def _message(content: list, stop_reason: str = "tool_use", tokens: int = 500) -> dict:
    return {
        "id": "msg", "type": "message", "role": "assistant", "model": "m", "stop_reason": stop_reason,
        "stop_sequence": None, "usage": {"input_tokens": tokens, "output_tokens": 0}, "content": content,
    }


REFUSAL = _message([], stop_reason="refusal")


def _status_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError("error", response=httpx.Response(status, request=request), body=None)


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


@pytest.fixture
def store(tmp_path) -> Path:
    return tmp_path / "snippets"


def _snippet_record(store: Path, record_id: int, quiet: bool = False, always_on: bool = False) -> PendingRecord:
    """A real step-4 SigMF pair with its 50 ms pre_trigger annotation: a
    125 kHz burst at 915.2 MHz from the trigger on, noise only, or an
    always-on emitter filling 90% of the band (and its own pre-trigger)."""
    rng = np.random.default_rng(record_id)
    n = 1 << 18
    iq = synthetic.noise(rng, n, 1e-5)
    if always_on:
        iq = iq + synthetic.band_limited(rng, n, FS, 0.9 * FS, 0.0, 1e-3)
    elif not quiet:
        iq = iq + synthetic.gate(synthetic.band_limited(rng, n, FS, 125e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    path = write_sigmf_snippet(iq, store, FS, TUNED, START, trigger_offset=round(0.05 * FS))
    return PendingRecord(record_id, str(path), FS, TUNED, -20.0, 262)


def _gone(store: Path, record_id: int) -> PendingRecord:
    return PendingRecord(record_id, str(store / f"gone{record_id}.sigmf-data"), FS, TUNED, None, None)


def _agent(gateway, client, store, sleeps=None, now=None, **settings) -> ClassificationAgent:
    return ClassificationAgent(
        gateway,
        client,
        AgentSettings(snippet_root=store, **settings),
        load_band_table().entries,
        sleep=(sleeps.append if sleeps is not None else lambda seconds: None),
        now=now or (lambda: START),
    )


def _drain(agent, batches: int) -> list:
    """Run `batches` batches; return the retry delays the agent asked for."""
    delays = []
    for _ in range(batches):
        agent.run_batch()
        delays.append(agent.take_retry_delay())
    return delays


def test_grounded_confident_result_is_auto_classified(store):
    gateway = FakeGateway([_snippet_record(store, 1)])
    assert _agent(gateway, ScriptedClient([GOOD]), store).run_batch() == 1
    assert gateway.submitted == [(1, AUTO, "ism_902_928:lora", 0.9, GOOD["reasoning"])]
    assert gateway.fetches == [(0, 20)]


def test_low_confidence_keeps_the_models_tag_for_review(store):
    gateway = FakeGateway([_snippet_record(store, 1)])
    _agent(gateway, ScriptedClient([{**GOOD, "confidence": 0.6}]), store).run_batch()
    ((_, status, tag, confidence, reasoning),) = gateway.submitted
    assert (status, tag, confidence) == (REVIEW, "ism_902_928:lora", 0.6)
    assert "below 0.85" in reasoning


def test_a_tag_outside_the_grounded_band_needs_review(store):
    gateway = FakeGateway([_snippet_record(store, 1)])
    _agent(gateway, ScriptedClient([{**GOOD, "tag": "pcs_downlink:lte", "confidence": 0.99}]), store).run_batch()
    ((_, status, tag, _, reasoning),) = gateway.submitted
    assert (status, tag) == (REVIEW, "pcs_downlink:lte")
    assert "'pcs_downlink' is not a grounded band-table entry (ism_902_928)" in reasoning


@pytest.mark.parametrize(
    "rejected, dropped, named",
    [
        ("bad_size", None, "quality_flags.snippet_rejected = 'bad_size'"),
        ("no_snippet_store", None, "quality_flags.snippet_rejected = 'no_snippet_store'"),
        (None, "low_disk", "quality_flags.snippet_dropped = 'low_disk'"),
        (None, "staging_unavailable", "quality_flags.snippet_dropped = 'staging_unavailable'"),
        (None, "queue_full", "quality_flags.snippet_dropped = 'queue_full'"),
        (None, None, "no snippet_rejected or snippet_dropped flag"),
    ],
)
def test_a_record_without_a_snippet_is_closed_out_for_review(store, rejected, dropped, named):
    """Ingest rejected the snippet or capture dropped it: the detection was
    kept without IQ. Nothing to classify, so needs_review at once, with a
    NULL tag and the flag named, and no LLM call."""
    gateway = FakeGateway([PendingRecord(1, None, FS, TUNED, -20.0, None, rejected, dropped)])
    client = ScriptedClient([])
    _agent(gateway, client, store).run_batch()
    ((_, status, tag, confidence, reasoning),) = gateway.submitted
    assert (status, tag, confidence) == (REVIEW, None, 0.0)
    assert named in reasoning and client.requests == []


def test_a_burst_of_queue_full_records_never_trips_a_halt(store):
    records = [PendingRecord(i, None, FS, TUNED, None, None, None, "queue_full") for i in range(1, 31)]
    gateway = FakeGateway(records)
    agent = _agent(gateway, ScriptedClient([]), store, batch_size=50)
    assert agent.run_batch() == 30
    assert [s[0] for s in gateway.submitted] == list(range(1, 31))


def test_invalid_answers_are_held_until_one_validates(store):
    gateway = FakeGateway([_snippet_record(store, i) for i in (1, 2, 3)])
    _agent(gateway, ScriptedClient([{**GOOD, "tag": "NOT VALID"}, REFUSAL, GOOD]), store).run_batch()
    assert [(s[0], s[1], s[2]) for s in gateway.submitted] == [(1, REVIEW, None), (2, REVIEW, None), (3, AUTO, "ism_902_928:lora")]
    assert gateway.submitted[0][4].startswith("Model output failed validation")
    assert "stop_reason was refusal" in gateway.submitted[1][4]


def test_a_held_invalid_answer_waits_for_proof_the_model_works(store):
    gateway = FakeGateway([_snippet_record(store, 1), _snippet_record(store, 2)])
    _agent(gateway, ScriptedClient([GOOD, REFUSAL]), store).run_batch()
    assert [s[0] for s in gateway.submitted] == [1]


@pytest.mark.parametrize(
    "bad",
    [REFUSAL, _message([], stop_reason="max_tokens"), {**GOOD, "confidence": "high"}],
)
def test_invalid_answers_in_a_row_halt_with_nothing_marked(store, bad):
    """Schema drift, a refusing model, or too few max_tokens fail every
    record alike: halt rather than mark the backlog."""
    gateway = FakeGateway([_snippet_record(store, i) for i in range(1, 7)])
    with pytest.raises(SystemicFault, match=r"5 model answers in a row .*records \[1, 2, 3, 4, 5\]"):
        _agent(gateway, ScriptedClient([bad] * 6), store).run_batch()
    assert gateway.submitted == []


def test_a_snippet_outside_the_store_is_held_until_a_snippet_reads(store, tmp_path):
    outside = _snippet_record(tmp_path / "elsewhere", 1)
    gateway = FakeGateway([outside, _snippet_record(store, 2)])
    client = ScriptedClient([GOOD])
    _agent(gateway, client, store).run_batch()
    assert [(s[0], s[1], s[2]) for s in gateway.submitted] == [(1, REVIEW, None), (2, AUTO, "ism_902_928:lora")]
    assert "outside the snippet store" in gateway.submitted[0][4]
    assert len(client.requests) == 1  # never asked about record 1


def test_a_wrong_store_root_halts_before_marking_anything(store, tmp_path):
    """Every path resolves elsewhere (e.g. the store mounted at another
    path): five in a row halt, and not one record is marked."""
    gateway = FakeGateway([_snippet_record(tmp_path / "elsewhere", i) for i in range(1, 7)])
    with pytest.raises(SystemicFault, match="5 snippets in a row could not be analysed") as excinfo:
        _agent(gateway, ScriptedClient([]), store).run_batch()
    assert "--snippet-store-dir" in str(excinfo.value) and gateway.submitted == []


def test_unreadable_snippets_stay_pending_and_five_in_a_row_halt(store):
    store.mkdir()
    gateway = FakeGateway([_gone(store, i) for i in range(1, 6)])
    with pytest.raises(SystemicFault, match="5 snippets in a row could not be analysed") as excinfo:
        _agent(gateway, ScriptedClient([]), store).run_batch()
    assert "records [1, 2, 3, 4, 5]" in str(excinfo.value)
    assert "--start-after-id 5" in str(excinfo.value)
    assert gateway.submitted == []


def test_a_snippet_with_no_occupied_region_stays_pending_and_counts(store):
    """Step 4 triggered on energy, so an empty spectrum means the analysis
    failed: never marked, and five in a row halt."""
    gateway = FakeGateway([_snippet_record(store, i, quiet=True) for i in range(1, 6)])
    client = ScriptedClient([])
    with pytest.raises(SystemicFault, match="no occupied region"):
        _agent(gateway, client, store).run_batch()
    assert gateway.submitted == [] and client.requests == []


def test_an_always_on_emitter_is_classified_and_never_trips_a_halt(store):
    """An always-on emitter (an LTE downlink) retriggers after every
    cooldown and fills its own pre-trigger, so there is no quiet reference.
    It is still found as the primary region every time -- never 'no
    region' -- with reduced confidence, so it goes to review, not a halt."""
    gateway = FakeGateway([_snippet_record(store, i, always_on=True) for i in range(1, 8)])
    client = ScriptedClient([{**GOOD, "tag": "ism_902_928:lte", "confidence": 0.95}] * 7)
    assert _agent(gateway, client, store).run_batch() == 7
    assert len(client.requests) == 7
    assert [(s[0], s[1]) for s in gateway.submitted] == [(i, REVIEW) for i in range(1, 8)]
    assert all("no grounded band-table match" in s[4] for s in gateway.submitted)


def test_an_analysed_snippet_resets_the_streak(store):
    store.mkdir()
    records = [_gone(store, 1), _gone(store, 2), _gone(store, 3), _gone(store, 4), _snippet_record(store, 5), _gone(store, 6), _gone(store, 7)]
    gateway = FakeGateway(records)
    assert _agent(gateway, ScriptedClient([GOOD]), store).run_batch() == 7
    assert [s[0] for s in gateway.submitted] == [5]


def test_transient_errors_rewind_and_retry_the_same_record(store):
    gateway = FakeGateway([_snippet_record(store, 1), _snippet_record(store, 2)])
    client = ScriptedClient([_status_error(529), _connection_error(), GOOD, GOOD])
    agent = _agent(gateway, client, store)
    assert _drain(agent, 3) == [2.0, 4.0, None]
    assert [after for after, _ in gateway.fetches] == [0, 0, 0]
    assert [s[0] for s in gateway.submitted] == [1, 2]


def test_an_outage_never_marks_and_never_halts(store):
    gateway = FakeGateway([_snippet_record(store, 1), _snippet_record(store, 2)])
    agent = _agent(gateway, ScriptedClient([_status_error(503)] * 12), store)
    assert _drain(agent, 12) == [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 300.0, 300.0, 300.0, 300.0]
    assert gateway.submitted == []


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_non_retryable_api_errors_halt_immediately(store, status):
    gateway = FakeGateway([_snippet_record(store, 1)])
    client = ScriptedClient([_status_error(status)])
    with pytest.raises(SystemicFault, match="Non-retryable Anthropic API error on record 1"):
        _agent(gateway, client, store).run_batch()
    assert gateway.submitted == [] and len(client.requests) == 1


def test_a_human_tag_that_wins_the_race_is_skipped(store):
    gateway = FakeGateway([_snippet_record(store, 1), _snippet_record(store, 2)], not_pending={1})
    _agent(gateway, ScriptedClient([GOOD, GOOD]), store).run_batch()
    assert [s[0] for s in gateway.submitted] == [2]


def test_the_llm_is_never_asked_twice_for_a_record(store):
    """The write fails (the database went away): the batch fails, the record
    is fetched again, and the kept decision is written without a second call."""
    gateway = FakeGateway([_snippet_record(store, 1)], fail_once={1})
    client = ScriptedClient([GOOD])
    agent = _agent(gateway, client, store)
    with pytest.raises(ConnectionError):
        agent.run_batch()
    agent.run_batch()
    assert len(client.requests) == 1
    assert gateway.submitted == [(1, AUTO, "ism_902_928:lora", 0.9, GOOD["reasoning"])]


def test_a_write_rejected_by_the_database_halts(store):
    gateway = FakeGateway([_snippet_record(store, 1)], rejected={1})
    with pytest.raises(SystemicFault, match="record 1"):
        _agent(gateway, ScriptedClient([GOOD]), store).run_batch()


def test_a_timed_out_write_leaves_the_record_pending_and_moves_on(store):
    gateway = FakeGateway([_snippet_record(store, 1), _snippet_record(store, 2)], timed_out={1})
    client = ScriptedClient([GOOD, GOOD])
    agent = _agent(gateway, client, store)
    assert agent.run_batch() == 2
    assert [s[0] for s in gateway.submitted] == [2] and len(client.requests) == 2
    assert agent.run_batch() == 0


def test_cursor_only_moves_forward(store):
    gateway = FakeGateway([_snippet_record(store, 3), _snippet_record(store, 7)])
    agent = _agent(gateway, ScriptedClient([GOOD, GOOD]), store, batch_size=1)
    assert [agent.run_batch(), agent.run_batch(), agent.run_batch()] == [1, 1, 0]
    assert gateway.fetches == [(0, 1), (3, 1), (7, 1)]


def test_spent_budget_pauses_until_utc_midnight(store):
    gateway = FakeGateway([_snippet_record(store, 1), _snippet_record(store, 2)])
    sleeps: list[float] = []
    agent = _agent(gateway, ScriptedClient([GOOD, GOOD]), store, sleeps=sleeps, daily_token_budget=400)
    agent.run_batch()  # record 1 spends 500 tokens; record 2 must wait
    assert sleeps == [timedelta(hours=12).total_seconds()]
    assert [s[0] for s in gateway.submitted] == [1, 2]


def test_budget_resets_when_the_day_rolls_over(store):
    gateway = FakeGateway([_snippet_record(store, 1), _snippet_record(store, 2)])
    sleeps: list[float] = []
    clock = [START]
    agent = _agent(gateway, ScriptedClient([GOOD, GOOD]), store, sleeps=sleeps, now=lambda: clock[0], daily_token_budget=400, batch_size=1)
    agent.run_batch()
    clock[0] = START + timedelta(days=1)
    agent.run_batch()
    assert sleeps == []


class _Batches:
    """A stand-in agent whose batches follow a script of outcomes."""

    def __init__(self, outcomes):
        self.outcomes, self.calls, self.delays = list(outcomes), 0, []

    def run_batch(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        fetched, self.delay = outcome
        return fetched

    def take_retry_delay(self):
        delay, self.delay = getattr(self, "delay", None), None
        return delay


def test_run_survives_a_failing_batch_and_stops_on_request():
    agent = _Batches([ConnectionError("database went away"), (0, None)])
    sleeps: list[float] = []
    run(agent, 30.0, lambda: agent.calls >= 2, sleeps.append)
    assert agent.calls == 2 and sleeps == [30.0]


def test_run_halts_after_five_failed_batches_in_a_row():
    agent = _Batches([ConnectionError("x")] * 5)
    with pytest.raises(SystemicFault, match="5 batches in a row failed"):
        run(agent, 1.0, lambda: False, lambda s: None)


def test_run_sleeps_the_agents_retry_delay():
    agent = _Batches([(1, 8.0), (0, None)])
    sleeps: list[float] = []
    run(agent, 30.0, lambda: agent.calls >= 2, sleeps.append)
    assert sleeps == [8.0]


def test_run_lets_a_systemic_fault_through():
    with pytest.raises(SystemicFault):
        run(_Batches([SystemicFault("outage")]), 1.0, lambda: False, lambda s: None)


def test_startup_check_requires_an_absolute_root(store):
    agent = _agent(FakeGateway([]), ScriptedClient([]), Path("data/snippets"))
    with pytest.raises(SystemicFault, match="must be absolute"):
        agent.check_snippet_root()


def test_startup_check_passes_when_a_recent_snippet_reads(store, tmp_path):
    """The newest is broken, the next one reads: the mount is right."""
    gateway = FakeGateway([_snippet_record(store, 1), _gone(store, 2), PendingRecord(3, None, FS, TUNED, None, None)])
    _agent(gateway, ScriptedClient([]), store).check_snippet_root()
    _agent(FakeGateway([]), ScriptedClient([]), store).check_snippet_root()  # nothing to probe yet


def test_startup_check_halts_when_the_store_is_mounted_elsewhere(store, tmp_path):
    gateway = FakeGateway([_snippet_record(tmp_path / "elsewhere", i) for i in (1, 2)])
    with pytest.raises(SystemicFault, match="identical resolved path"):
        _agent(gateway, ScriptedClient([]), store.resolve()).check_snippet_root()
    assert gateway.submitted == []


def test_cli_takes_secrets_from_the_environment_and_fails_fast(monkeypatch, tmp_path):
    args = ["--snippet-store-dir", str(tmp_path)]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SURVEYTOOL_AGENT_DATABASE_URL", "postgresql://agent@localhost/surveytool")
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        main(args)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("SURVEYTOOL_AGENT_DATABASE_URL")
    with pytest.raises(SystemExit, match="SURVEYTOOL_AGENT_DATABASE_URL"):
        main(args)
    monkeypatch.setenv("SURVEYTOOL_AGENT_DATABASE_URL", "sqlite:///survey.db")
    with pytest.raises(SystemExit, match="PostgreSQL"):
        main(args)
```

- [ ] **Step 2: Run them to verify they fail**

<!-- check: t10_red -->
Run: `python -m pytest tests/agent/test_service.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'agent.service'`.


- [ ] **Step 3: Implement**

<!-- write: agent/service.py -->
```python
# agent/service.py
"""The Part 4 classification agent: fetch, analyse, ask, route, write back.

Two invariants:
- needs_review is a judgment about one record, never the consequence of a
  systemic fault. An outcome that could be systemic is held, still pending,
  until a later success shows the system works; too many in a row halt the
  agent with them still pending;
- the LLM answers at most once per record per process run. A decision that
  could not be written is kept and written again, never re-asked. (A call
  that raised before returning a response is not an answer.)

Outcomes per record:
- no snippet (ingest rejected it or capture dropped it): needs_review at once,
  naming the quality flag; no LLM call; never counts toward a halt;
- snippet outside the store: held; marked needs_review once a later snippet
  is analysed (the store root is then known to be right);
- snippet unreadable, no occupied region, or feature extraction failed: left
  pending (step 4 triggered on energy, so finding none is our failure);
- max_consecutive_snippet_failures of the above in a row: halt;
- transient API error: the cursor is rewound to the record and the loop backs
  off exponentially (capped) and retries it indefinitely; never marked;
- any other API error (auth, bad request, unknown model): halt;
- invalid model output (validation failure, refusal, max_tokens): held;
  marked needs_review once a later answer validates;
  max_consecutive_bad_outputs in a row: halt;
- write rejected by the database (bad arguments): halt; write timed out: left
  pending; any other write error fails the batch, and the kept decision is
  written when the record is fetched again;
- daily token budget spent: pause until the UTC day rolls over.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

from agent.analysis import SnippetAnalysis, analyse_snippet
from agent.band_table import DEFAULT_BAND_TABLE, BandEntry, load_band_table
from agent.classifier import ModulationClassifier, UnavailableClassifier
from agent.db_gateway import (
    DEFAULT_AGENT_ROLE,
    AgentGateway,
    BoundaryViolation,
    PendingRecord,
    RecordNotPending,
    SubmitRejected,
    SubmitTimedOut,
    connect_gateway,
)
from agent.llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    LlmResult,
    MessagesClient,
    is_transient,
    request_classification,
)
from agent.prompt import SYSTEM_PROMPT, build_user_message
from agent.routing import Decision, needs_review, route
from agent.snippet_reader import SnippetOutsideStore, SnippetUnreadable, read_snippet

logger = logging.getLogger(__name__)

_SDK_MAX_RETRIES = 2  # inside each of our calls, before our own backoff
_SDK_TIMEOUT_SECONDS = 60.0
_STARTUP_PROBES = 5


class SystemicFault(RuntimeError):
    """Not about any one record: halt rather than mark the backlog."""


class _RetryLater(Exception):
    def __init__(self, delay: float) -> None:
        super().__init__(delay)
        self.delay = delay


@dataclass(frozen=True)
class AgentSettings:
    snippet_root: Path
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    batch_size: int = 20
    max_consecutive_snippet_failures: int = 5
    max_consecutive_bad_outputs: int = 5
    daily_token_budget: int = 2_000_000
    backoff_seconds: float = 2.0
    max_backoff_seconds: float = 300.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _no_snippet_reason(record: PendingRecord) -> str:
    """The detection was persisted without its IQ, so there is nothing to
    classify. The flag value is DB data: shown truncated and quoted."""
    if record.snippet_rejected is not None:
        return (
            "No IQ snippet to classify: ingest rejected it "
            f"(quality_flags.snippet_rejected = {record.snippet_rejected[:64]!r})."
        )
    if record.snippet_dropped is not None:
        return (
            "No IQ snippet to classify: capture dropped it "
            f"(quality_flags.snippet_dropped = {record.snippet_dropped[:64]!r})."
        )
    return "No IQ snippet to classify, and no snippet_rejected or snippet_dropped flag says why."


class ClassificationAgent:
    """Stateful by necessity: it carries the fetch cursor, the held records,
    the decisions not yet written and the day's token spend from one record
    to the next."""

    def __init__(
        self,
        gateway: AgentGateway,
        client: MessagesClient,
        settings: AgentSettings,
        bands: tuple[BandEntry, ...],
        classifier: ModulationClassifier | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = _utc_now,
        start_after_id: int = 0,
    ) -> None:
        self._gateway = gateway
        self._client = client
        self._settings = settings
        self._bands = bands
        self._classifier = classifier or UnavailableClassifier()
        self._sleep = sleep
        self._now = now
        self._last_id = start_after_id
        # (record id, reason, verdict): verdict is the needs_review to write
        # once the root is proven right (outside-store), else None (stays pending).
        self._snippet_failures: list[tuple[int, str, Decision | None]] = []
        self._bad_outputs: list[tuple[int, Decision]] = []
        self._unwritten: dict[int, Decision] = {}
        self._transient_streak = 0
        self._retry_in: float | None = None
        self._budget_day = now().date()
        self._tokens_today = 0

    def check_snippet_root(self) -> None:
        """Before touching any record: the root must be absolute, and one of
        the newest pending snippets must read through it. Step 4 stores
        resolve()d absolute paths, so a store mounted anywhere else would
        make every snippet 'outside the store'."""
        root = self._settings.snippet_root
        if not root.is_absolute():
            raise SystemicFault(f"--snippet-store-dir must be absolute, got {str(root)!r}")
        newest = self._gateway.fetch_newest_with_snippet(self._last_id, _STARTUP_PROBES)
        problems = []
        for record in newest:
            try:
                read_snippet(record.iq_snippet_path, root, record.sample_rate, record.center_freq)
                return
            except (SnippetOutsideStore, SnippetUnreadable) as exc:
                problems.append(f"record {record.id}: {exc}")
        if problems:
            raise SystemicFault(
                f"None of the newest pending snippets reads under {root}: "
                + "; ".join(problems)
                + ". Mount the snippet store read-only at the identical resolved path ingest uses."
            )
        logger.info("No pending snippet to probe %s with yet", root)

    def take_retry_delay(self) -> float | None:
        """Seconds to wait before retrying a record whose LLM call hit a
        transient error, or None."""
        delay, self._retry_in = self._retry_in, None
        return delay

    def run_batch(self) -> int:
        """Process the next pending records (ids above the cursor); returns
        how many were fetched. Raises SystemicFault to halt."""
        records = self._gateway.fetch_pending(self._last_id, self._settings.batch_size)
        for record in records:
            try:
                self._process(record)
            except _RetryLater as retry:
                self._retry_in = retry.delay  # the cursor stays before this record
                break
            self._last_id = record.id
        return len(records)

    def _process(self, record: PendingRecord) -> None:
        logger.info("Record %d: snippet %s", record.id, record.iq_snippet_path)
        if record.iq_snippet_path is None:
            self._submit(record.id, needs_review(_no_snippet_reason(record)))
            return
        decision = self._unwritten.get(record.id) or self._decide(record)
        if decision is not None:
            self._submit(record.id, decision)

    def _decide(self, record: PendingRecord) -> Decision | None:
        """The decision to write now, or None when the record stays pending
        or is held."""
        analysis = self._analyse(record)
        if analysis is None:
            return None
        result = self._ask(record.id, build_user_message(analysis))
        if result.classification is None:
            self._hold_bad_output(record.id, needs_review(f"Model output failed validation: {result.failure}"))
            return None
        decision = route(result.classification, analysis.grounded_band_ids, analysis.modulation_label)
        self._unwritten[record.id] = decision  # before anything else can fail
        self._release_bad_outputs()
        return decision

    def _analyse(self, record: PendingRecord) -> SnippetAnalysis | None:
        try:
            snippet = read_snippet(
                record.iq_snippet_path, self._settings.snippet_root, record.sample_rate, record.center_freq
            )
        except SnippetOutsideStore as exc:
            return self._snippet_failed(record.id, str(exc), needs_review(f"Snippet rejected: {exc}"))
        except SnippetUnreadable as exc:
            return self._snippet_failed(record.id, str(exc), None)
        try:
            analysis = analyse_snippet(snippet, self._bands, self._classifier)
        except Exception as exc:
            logger.exception("Feature extraction failed for record %d", record.id)
            return self._snippet_failed(record.id, f"feature extraction failed: {exc!r}", None)
        if analysis.primary is None:
            return self._snippet_failed(record.id, "no occupied region stood above the noise floor", None)
        failures, self._snippet_failures = self._snippet_failures, []
        for record_id, _, verdict in failures:
            if verdict is not None:
                self._submit(record_id, verdict)
        return analysis

    def _snippet_failed(self, record_id: int, reason: str, verdict: Decision | None) -> None:
        self._snippet_failures.append((record_id, reason, verdict))
        logger.warning("Record %d held pending: %s", record_id, reason)
        if len(self._snippet_failures) >= self._settings.max_consecutive_snippet_failures:
            ids = [failed for failed, _, _ in self._snippet_failures]
            raise SystemicFault(
                f"{len(ids)} snippets in a row could not be analysed (records {ids}; last: "
                f"{reason}). Check --snippet-store-dir and the store's read-only mount, or "
                f"restart with --start-after-id {record_id} to skip them."
            )
        return None

    def _hold_bad_output(self, record_id: int, verdict: Decision) -> None:
        self._bad_outputs.append((record_id, verdict))
        logger.warning("Record %d held pending: %s", record_id, verdict.reasoning)
        if len(self._bad_outputs) >= self._settings.max_consecutive_bad_outputs:
            ids = [held for held, _ in self._bad_outputs]
            raise SystemicFault(
                f"{len(ids)} model answers in a row were refused or failed validation (records "
                f"{ids}; last: {verdict.reasoning}). Check the model id, the tool schema and "
                f"the prompt; restart with --start-after-id {record_id} to skip them."
            )

    def _release_bad_outputs(self) -> None:
        """An answer just validated, so the model works: each held invalid
        answer was about its own record, and goes to review."""
        while self._bad_outputs:
            record_id, verdict = self._bad_outputs[0]
            self._submit(record_id, verdict)
            self._bad_outputs.pop(0)

    def _ask(self, record_id: int, user_message: str) -> LlmResult:
        self._wait_for_budget()
        try:
            result = request_classification(
                self._client, SYSTEM_PROMPT, user_message, self._settings.model, self._settings.max_tokens
            )
        except Exception as exc:
            if not is_transient(exc):
                raise SystemicFault(
                    f"Non-retryable Anthropic API error on record {record_id}: {exc!r}. "
                    f"If the record itself causes it, restart with --start-after-id {record_id}."
                ) from exc
            self._transient_streak += 1
            delay = min(
                self._settings.backoff_seconds * 2 ** (self._transient_streak - 1),
                self._settings.max_backoff_seconds,
            )
            logger.warning("Transient LLM failure on record %d (%r); retrying it in %.0f s", record_id, exc, delay)
            raise _RetryLater(delay) from exc
        self._transient_streak = 0
        self._tokens_today += result.tokens_used
        return result

    def _wait_for_budget(self) -> None:
        now = self._now()
        if now.date() != self._budget_day:
            self._budget_day, self._tokens_today = now.date(), 0
        if self._tokens_today < self._settings.daily_token_budget:
            return
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
        logger.warning(
            "Daily token budget (%d) spent; pausing until %s", self._settings.daily_token_budget, midnight
        )
        self._sleep((midnight - now).total_seconds())
        self._budget_day, self._tokens_today = midnight.date(), 0

    def _submit(self, record_id: int, decision: Decision) -> None:
        try:
            self._gateway.submit_classification(
                record_id, decision.status, decision.tag, decision.confidence, decision.reasoning
            )
        except RecordNotPending:
            logger.info("Record %d was no longer pending (a human got there first); skipped", record_id)
        except SubmitTimedOut as exc:
            logger.warning("%s; record %d stays pending", exc, record_id)
        except SubmitRejected as exc:
            raise SystemicFault(str(exc)) from exc
        else:
            logger.info(
                "Record %d -> %s (%s, %.2f)", record_id, decision.status.value, decision.tag, decision.confidence
            )
        self._unwritten.pop(record_id, None)


def run(
    agent: ClassificationAgent,
    poll_seconds: float,
    stopping: Callable[[], bool],
    sleep: Callable[[float], None],
    max_failed_batches: int = 5,
) -> None:
    """Loop until `stopping()`. A batch that fails outside any record (the
    database going away) is retried after `poll_seconds`, and
    max_failed_batches in a row halt; SystemicFault propagates."""
    failed = 0
    while not stopping():
        try:
            fetched = agent.run_batch()
            failed = 0
        except SystemicFault:
            raise
        except Exception as exc:
            failed += 1
            logger.exception("Batch failed (%d in a row)", failed)
            if failed >= max_failed_batches:
                raise SystemicFault(f"{failed} batches in a row failed (last: {exc!r})") from exc
            fetched = 0
        delay = agent.take_retry_delay()
        if stopping():
            break
        if delay is not None:
            sleep(delay)
        elif fetched == 0:
            sleep(poll_seconds)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 4 unknown-signal classification agent")
    parser.add_argument(
        "--snippet-store-dir",
        required=True,
        help="Absolute path of ingest's snippet store, mounted read-only at the identical "
        "resolved path ingest uses (stored snippet paths are absolute).",
    )
    parser.add_argument("--agent-role", default=DEFAULT_AGENT_ROLE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--daily-token-budget", type=int, default=AgentSettings.daily_token_budget)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--band-table", default=str(DEFAULT_BAND_TABLE))
    parser.add_argument(
        "--start-after-id",
        type=int,
        default=0,
        help="Only consider records with a higher id. A halt names the records involved; "
        "use this to skip them once their cause is understood.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Secrets come from the environment only, never the command line:
    SURVEYTOOL_AGENT_DATABASE_URL (the agent login role) and ANTHROPIC_API_KEY."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    database_url = os.environ.get("SURVEYTOOL_AGENT_DATABASE_URL")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    if not database_url:
        raise SystemExit("SURVEYTOOL_AGENT_DATABASE_URL is not set")
    try:
        gateway = connect_gateway(database_url, args.agent_role)
    except (BoundaryViolation, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    settings = AgentSettings(
        snippet_root=Path(args.snippet_store_dir),
        model=args.model,
        max_tokens=args.max_tokens,
        daily_token_budget=args.daily_token_budget,
    )
    agent = ClassificationAgent(
        gateway,
        anthropic.Anthropic(api_key=api_key, max_retries=_SDK_MAX_RETRIES, timeout=_SDK_TIMEOUT_SECONDS),
        settings,
        load_band_table(Path(args.band_table)).entries,
        start_after_id=args.start_after_id,
    )
    stop_requested: list[int] = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_requested.append(signum))
    try:
        agent.check_snippet_root()
        run(agent, args.poll_seconds, lambda: bool(stop_requested), time.sleep)
    except SystemicFault as exc:
        raise SystemExit(f"Agent halted: {exc}") from exc
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
```

In `pyproject.toml`:

<!-- edit: pyproject.toml -->
Replace

```toml
sdr-capture-unknown = "capture.unknown.service:main"
sdr-agent-boundary
```

with

```toml
sdr-capture-unknown = "capture.unknown.service:main"
sdr-agent = "agent.service:main"
sdr-agent-boundary
```

- [ ] **Step 4: Run the tests to verify they pass**

<!-- check: t10_green -->
Run: `python -m pytest tests/agent/test_service.py -q`
Expected: PASS, `42 passed` in about 5 s.


- [ ] **Step 5: Prove that the LLM is never asked twice, then revert**

Temporarily drop the kept decision, so a record whose write failed is asked again:

<!-- edit: agent/service.py -->
Replace

```python
        decision = self._unwritten.get(record.id) or self._decide(record)
```

with

```python
        decision = self._decide(record)
```
<!-- check: t10_mutant -->
Run: `python -m pytest tests/agent/test_service.py -q`
Expected: FAIL, `1 failed, 41 passed`: `test_the_llm_is_never_asked_twice_for_a_record`.


Revert it:

<!-- edit: agent/service.py -->
Replace

```python
        decision = self._decide(record)
```

with

```python
        decision = self._unwritten.get(record.id) or self._decide(record)
```
<!-- check: t10_reverted -->
Run: `python -m pytest tests/agent/test_service.py -q`
Expected: PASS, `42 passed`.


- [ ] **Step 6: Commit**

<!-- run -->
```bash
git add agent/service.py pyproject.toml tests/agent/test_service.py
git commit -m "feat: add the classification agent service loop and sdr-agent CLI"
```

---

### Task 11: Structural guards and the boundary documentation

**Files:**
- Test: `tests/agent/test_import_boundary.py` and `tests/test_no_tag_branching.py`, both
  AST-based.
- Modify: `docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md` (§1, §4,
  §9), `agent/README.md` (rewrite), `README.md` (the `agent/` layout line)

**Interfaces:**
- Consumes: the source tree as text. It imports nothing from `agent/`, `capture/` or
  `ingest/`.
- Produces two guards. Both are **tripwires**; the container is the boundary, and the
  docs say so.
  - **Import allowlist over `agent/`, `dsp/` and `schema/`.**
    - Allowed: `__future__`, `argparse`, `collections`, `dataclasses`, `datetime`,
      `enum`, `functools`, `json`, `logging`, `math`, `os` (only `environ` and `getenv`,
      never aliased), `pathlib`, `re`, `signal`, `statistics`, `time`, `typing`,
      `anthropic`, `numpy`, `pydantic`, `sigmf`, `agent`, `dsp`, `schema`. `sqlalchemy`
      and `psycopg` are allowed in `agent/db_gateway.py` only.
    - Banned names: `__import__`, `exec`, `eval`, `compile`, `getattr`, `open` and
      `__builtins__`. Banned: `logging.config` and `logging.handlers`.
    - Each bypass the plan review named is a test case: `import os as o; o.system`,
      `getattr(os, 'system')`, `SysLogHandler(address=...)`.
  - **The architecture rule for `capture/` and `ingest/`.** References to `tag`,
    `classification_status` or `ClassificationStatus` are allowed only on an exact
    allowlist: as names, attributes, keywords, arguments, imports, and as whole words
    inside string constants, which catches raw SQL such as `metadata->>'tag'`.
    - Today the allowlist holds only the step-4 normalizer's single `UNCLASSIFIED`
      write: three AST nodes in `capture/unknown/normalizer.py`.
    - A scan of the tree's string constants found no other hits; "staging" does not
      match the word `tag`.

Both tests pass on their first run, because the code already obeys them. Step 2 proves
they can fail.

- [ ] **Step 1: Write the guards**

<!-- write: tests/agent/test_import_boundary.py -->
```python
# tests/agent/test_import_boundary.py
"""A tripwire, not the boundary: the container is the boundary (no ingest
socket, the snippet store read-only, egress only to the database and
api.anthropic.com; see agent/README.md). This AST scan catches the obvious
ways the agent's code could grow a path out of it: shelling out, opening
sockets, loading code dynamically, writing files, logging to the network,
or importing capture/ingest/storage. It scans agent/ and the two packages
the agent imports, dsp/ and schema/."""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ALLOWED_MODULES = {
    "__future__", "argparse", "collections", "dataclasses", "datetime", "enum", "functools",
    "json", "logging", "math", "os", "pathlib", "re", "signal", "statistics", "time", "typing",
    "anthropic", "numpy", "pydantic", "sigmf",
    "agent", "dsp", "schema",
}
DATABASE_MODULES = {"sqlalchemy", "psycopg"}  # agent/db_gateway.py only
DATABASE_GATEWAY = "agent/db_gateway.py"
BANNED_NAMES = {"__import__", "exec", "eval", "compile", "getattr", "open", "__builtins__"}
BANNED_LOGGING = {"config", "handlers"}  # file and network log handlers
OS_ALLOWED = {"environ", "getenv"}


def violations(source: str, path: str = "agent/example.py") -> list[str]:
    allowed = ALLOWED_MODULES | (DATABASE_MODULES if path == DATABASE_GATEWAY else set())
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    found.append(f"import {alias.name}")
                elif alias.name == "os" and alias.asname is not None:
                    found.append(f"import os as {alias.asname}")
                elif alias.name.split(".")[:2] in (["logging", "config"], ["logging", "handlers"]):
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top = node.module.split(".")[0]
            if top not in allowed:
                found.append(f"from {node.module} import ...")
            elif top == "os":
                found += [f"from os import {a.name}" for a in node.names if a.name not in OS_ALLOWED]
            elif node.module in ("logging.config", "logging.handlers") or (
                node.module == "logging" and any(a.name in BANNED_LOGGING for a in node.names)
            ):
                found.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            found.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr not in OS_ALLOWED:
                found.append(f"os.{node.attr}")
            elif node.value.id == "logging" and node.attr in BANNED_LOGGING:
                found.append(f"logging.{node.attr}")
    return found


SCANNED = sorted(path for package in ("agent", "dsp", "schema") for path in (REPO / package).rglob("*.py"))


def test_the_packages_are_scanned():
    names = {str(path.relative_to(REPO)) for path in SCANNED}
    assert {"agent/service.py", "agent/db_gateway.py", "dsp/segmentation.py", "schema/records.py"} <= names


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: str(p.relative_to(REPO)))
def test_module_stays_inside_the_boundary(path):
    assert violations(path.read_text(), str(path.relative_to(REPO))) == []


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess",
        "from subprocess import run",
        "import socket",
        "import ctypes",
        "import importlib",
        "from importlib import import_module",
        "import multiprocessing",
        "import sys",
        "import requests",
        "import shutil",
        "import capture.unknown.service",
        "from ingest.queue_server import QueueServer",
        "from storage.db import make_engine",
        "__import__('subprocess')",
        "exec('x = 1')",
        "eval('1 + 1')",
        "compile('x', 'f', 'exec')",
        "open('/etc/passwd')",
        "__builtins__['exec']('x')",
        # The bypasses the plan review named:
        "import os as o\no.system('ls')",
        "import os\ngetattr(os, 'system')('ls')",
        "from logging.handlers import SysLogHandler\nSysLogHandler(address=('collector', 514))",
        "import logging.handlers\nlogging.handlers.SysLogHandler(address=('collector', 514))",
        "import logging\nlogging.config.dictConfig({})",
        "from logging import handlers",
        "import os\nos.system('ls')",
        "import os\nos.popen('ls')",
        "import os\nos.execv('/bin/sh', [])",
        "import os\nos.remove('/x')",
        "from os import system",
        # Database drivers outside the gateway:
        "import sqlalchemy",
        "from psycopg import connect",
    ],
)
def test_checker_catches(source):
    assert violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "from __future__ import annotations",
        "from collections.abc import Callable",
        "import os\nkey = os.environ.get('ANTHROPIC_API_KEY')",
        "from os import environ",
        "import logging\nlogging.getLogger(__name__).info('x')",
        "import numpy as np",
        "from pathlib import Path\nPath('x').open('rb')",
        "from agent.llm import TOOL_NAME",
        "from dsp.features import region_features",
        "from schema.records import ClassificationStatus",
    ],
)
def test_checker_allows(source):
    assert violations(source) == []


def test_database_drivers_are_allowed_only_in_the_gateway():
    source = "import psycopg\nfrom sqlalchemy import text"
    assert violations(source, DATABASE_GATEWAY) == []
    assert violations(source, "agent/service.py") != []
```

<!-- write: tests/test_no_tag_branching.py -->
```python
# tests/test_no_tag_branching.py
"""Architecture rule (design doc section 1): no module may read
metadata.tag or classification_status to decide whether to start, retune
or chain a capture or decode. capture/ and ingest/ may not reference them
at all -- as names, attributes, keywords, arguments, imports, or words in
string constants (raw SQL such as metadata->>'tag') -- except for an
explicit allowlist: today, the step-4 normalizer's single write of
UNCLASSIFIED. A tripwire, like tests/agent/test_import_boundary.py."""

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WATCHED = {"tag", "classification_status", "ClassificationStatus"}
WATCHED_WORD = re.compile(r"\b(tag|classification_status|ClassificationStatus)\b")
ALLOWED = Counter(
    {
        ("capture/unknown/normalizer.py", "import", "ClassificationStatus"): 1,
        ("capture/unknown/normalizer.py", "name", "ClassificationStatus"): 1,
        ("capture/unknown/normalizer.py", "keyword", "classification_status"): 1,
    }
)


def references(source: str) -> Counter:
    found: Counter = Counter()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in WATCHED:
            found["name", node.id] += 1
        elif isinstance(node, ast.Attribute) and node.attr in WATCHED:
            found["attribute", node.attr] += 1
        elif isinstance(node, ast.keyword) and node.arg in WATCHED:
            found["keyword", node.arg] += 1
        elif isinstance(node, ast.arg) and node.arg in WATCHED:
            found["argument", node.arg] += 1
        elif isinstance(node, ast.alias) and node.name in WATCHED:
            found["import", node.name] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for word in WATCHED_WORD.findall(node.value):
                found["string", word] += 1
    return found


def test_capture_and_ingest_never_branch_on_classification():
    found: Counter = Counter()
    for package in ("capture", "ingest"):
        for path in sorted((REPO / package).rglob("*.py")):
            for (kind, name), count in references(path.read_text()).items():
                found[str(path.relative_to(REPO)), kind, name] += count
    assert found == ALLOWED


@pytest.mark.parametrize(
    "source",
    [
        "if record.metadata.tag == 'lte': start_decoder()",
        "if row['classification_status'] == 'auto_classified': retune()",
        "status = metadata.get('tag')",
        "from schema.records import ClassificationStatus",
        "def chain(tag): ...",
        "Metadata(tag='x')",
        # Raw SQL reads the key without any Python name:
        "rows = conn.execute(text(\"SELECT id FROM survey_records WHERE metadata->>'tag' = 'lte'\"))",
        "query = 'SELECT metadata->>\\'classification_status\\' FROM survey_records'",
    ],
)
def test_checker_catches(source):
    assert references(source)


def test_staging_and_other_words_containing_tag_are_not_flagged():
    assert not references("STAGING = 'data/snippet-staging'  # stage, staged, tags")
```
<!-- check: t11_first -->
Run: `python -m pytest tests/agent/test_import_boundary.py tests/test_no_tag_branching.py -q`
Expected: PASS, `exit 0`.


- [ ] **Step 2: Prove the import guard has teeth, then revert**

Temporarily add `import subprocess` to `agent/service.py`:

<!-- edit: agent/service.py -->
Replace

```python
import signal
import time
```

with

```python
import signal
import subprocess
import time
```
<!-- check: t11_mutant -->
Run: `python -m pytest tests/agent/test_import_boundary.py -q`
Expected: FAIL, `1 failed`, with `assert ['import subprocess'] == []`.


Revert it:

<!-- edit: agent/service.py -->
Replace

```python
import signal
import subprocess
import time
```

with

```python
import signal
import time
```
<!-- check: t11_reverted -->
Run: `python -m pytest tests/agent/test_import_boundary.py -q`
Expected: PASS, `exit 0`.


Mutation-checked the same way while writing this plan: adding
`if record.metadata.tag == "lte": pass` to `ingest/service.py` fails
`test_capture_and_ingest_never_branch_on_classification` with
`{('ingest/service.py', 'attribute', 'tag'): 1}`.

- [ ] **Step 3: Update the architecture doc and the READMEs**

Add the architecture rule to §1 of `docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md`:

<!-- edit: docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md -->
Replace

```markdown
  pipeline — it must never trigger, chain into, or hand data to the cellular/WiFi/BT
  decode modules, regardless of what it classifies a signal as.

## 2. Hardware
```

with

```markdown
  pipeline — it must never trigger, chain into, or hand data to the cellular/WiFi/BT
  decode modules, regardless of what it classifies a signal as.
- **No module branches on a classification**: nothing may read `metadata.tag` or
  `metadata.classification_status` to decide whether to start, retune or chain a
  capture or decode. `tests/test_no_tag_branching.py` enforces this over `capture/`
  and `ingest/` with an explicit allowlist (today: the unknown-signal normalizer's
  single write of `unclassified`).

## 2. Hardware
```

In the §4 schema:

<!-- edit: docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md -->
Replace

```markdown
    "classification_status": "unclassified | manually_tagged | auto_classified",
```

with

```markdown
    "classification_status": "unclassified | manually_tagged | auto_classified | needs_review",
```

Replace the §9 enforcement paragraph:

<!-- edit: docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md -->
Replace

```markdown
**Hard-boundary enforcement (structural, not just convention)**: the agent process
authenticates to Postgres as a dedicated role with **UPDATE privilege limited to
`metadata.classification_status`, `metadata.tag`, `metadata.confidence`,
`metadata.reasoning` only** on `unknown`-modality records. It has no privilege to write
anywhere else and no code path to invoke capture/decode modules — the boundary holds
even under careless future extension, not just by developer discipline.
```

with

```markdown
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
```

Replace `agent/README.md` entirely:

<!-- write: agent/README.md -->
````markdown
# agent

Part 4 signal classification. Triages `modality: "unknown"` records: it reads the
record's SigMF snippet, splits the spectrum into occupied regions, measures the strongest
one (bandwidth, center, duty cycle, bursts, PAPR, flatness, symbol rate), matches it
against a small curated US band table with eCFR citations, and asks Claude for one
forced `record_classification` tool call. Routing writes `auto_classified` only when the
confidence is >= 0.85 and the tag's band prefix (`ism_902_928` in `ism_902_928:lora`) is
a band-table entry the signal is grounded in; everything else goes to `needs_review`.
Any `reduced_confidence` reason (no quiet pre-trigger noise reference, NaN or inf
samples, an unreliable bandwidth) keeps every band match ungrounded.
The tag, confidence and reasoning go back into the record's own metadata, and nothing
else does. A record whose snippet ingest rejected or capture dropped (`iq_snippet_path`
NULL, `quality_flags.snippet_rejected` / `snippet_dropped` set) is closed out as
`needs_review` without an LLM call.

`needs_review` is a judgment about one record, never the result of a systemic fault:
outcomes that could be systemic (a snippet outside the store, an invalid model answer)
are held pending until a later success proves the system works, transient API errors
are retried indefinitely with backoff, and a run of failures halts the agent with the
records still pending. The LLM answers at most once per record per run.

The modulation classifier is a seam (`classifier.py`): v1 ships `UnavailableClassifier`;
the TorchSig model trained offline on the DGX Spark plugs in later for Jetson inference.

## The boundary

The agent is a dead end in the pipeline: it never triggers, chains into, or hands data
to the cellular/WiFi/BT decode modules.

- **The container is the boundary.** The agent runs in its own container as the same uid
  as ingest (step 4 keeps the snippet store `0700` and every snippet `0600`, owned by that
  uid). The store is bind-mounted **read-only at the identical resolved path ingest uses**:
  records hold absolute `resolve()`d paths, and the agent refuses to start if none of the
  newest pending snippets reads under `--snippet-store-dir`. The ingest socket is **not**
  mounted. Network egress is limited to the database and `api.anthropic.com`. Give it a
  memory limit of about 1 GiB: the analysis peaks at ~610 MB at its 2^25-sample cap.
- **The database.** The agent's login role (`IN ROLE surveytool_agent`, and in nothing
  else) has no privilege on `survey_records`; it reads the `agent_pending_unknown` view and
  writes only through the `classify_unknown` function (`storage/sql/agent_boundary.sql`,
  installed by `sdr-agent-boundary`). At startup the agent checks an allowlist: no
  membership but that role, no dangerous role attribute, no privilege on the table, no
  CREATE or TEMPORARY anywhere, no other executable SECURITY DEFINER function.
- **The LLM.** The model has exactly one tool, the forced output tool, and sees only
  numeric features and band-table entries: no SigMF free text, location, survey or
  operator IDs.
- **Tripwires, not walls.** `tests/agent/test_import_boundary.py` (an import allowlist
  and banned calls over `agent/`, `dsp/` and `schema/`) and `tests/test_no_tag_branching.py`
  (no reference to tags or classification status in `capture/` or `ingest/`) catch obvious
  regressions in review. They do not stop a determined bypass; the container does.

## Running

```bash
# Once, as a database administrator (superuser URL):
sdr-agent-boundary --database-url postgresql://surveytool:...@localhost:5432/surveytool
# Create the agent's login role yourself, password from your secret store:
#   CREATE ROLE surveytool_agent_login LOGIN PASSWORD '...' IN ROLE surveytool_agent;

export SURVEYTOOL_AGENT_DATABASE_URL=postgresql://surveytool_agent_login:...@localhost:5432/surveytool
export ANTHROPIC_API_KEY=...
sdr-agent --snippet-store-dir /absolute/path/of/ingest/data/snippets
```

Each record's id is logged before its snippet is read. If the agent halts, or a snippet
crashes it, the message or the last log line names the records; restart with
`--start-after-id <id>` to skip them once the cause is understood.
````

In `README.md`:

<!-- edit: README.md -->
Replace

```markdown
- `agent/` — Part 4 agentic signal-characterization (tools, feature extraction, routing)
```

with

```markdown
- `agent/` — Part 4 classification agent (features, curated band table, one forced LLM tool call, routing); writes back only through its DB boundary
```
<!-- check: t11_green -->
Run: `python -m pytest tests/agent tests/test_no_tag_branching.py -q`
Expected: PASS, `exit 0`.


- [ ] **Step 4: Commit**

<!-- run -->
```bash
git add tests/agent/test_import_boundary.py tests/test_no_tag_branching.py docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md agent/README.md README.md
git commit -m "test: enforce the agent import allowlist and the no-tag-branching rule; document the boundary"
```

---

### Task 12: End-to-end closure proofs

**Files:**
- Test: `tests/agent/test_end_to_end.py`, which needs no database.
- Modify: `tests/storage/test_agent_boundary_pg.py`. Add imports and append the
  real-PostgreSQL closure test.

**Interfaces:**
- Consumes everything:
  - step 4's `process_snippet`, `CaptureSettings`, `CapturedSnippet`, `TriggerEvent`
    (in-memory test), `write_sigmf_snippet` (PostgreSQL test) and `LocalSnippetStore.adopt`;
  - `ClassificationAgent` and `AgentSettings` (Task 10);
  - `load_band_table` (Task 7);
  - `connect_gateway` (Task 5);
  - `dsp.synthetic` (Task 2).
- Produces: nothing further consumes these. They are the spec's end-to-end test, plus the
  same path through the real boundary.

The scenario:
- A 125 kHz burst 200 kHz above a 915 MHz tuning lands at 915.2 MHz, inside
  `ism_902_928`, with a plausible OBW. That is a grounded match, so a 0.92-confidence
  answer is `auto_classified`.
- A carrier 250 kHz below the tuning, already on before the trigger, is found in the
  reference: context with `present_before_trigger: true`. `reduced_confidence` is empty.
- A second pending record has no snippet, because capture dropped it for `low_disk`. It is
  closed out as `needs_review`, with no LLM call: in memory, and through the real view
  and function.
- Both tests first run the startup probe (`check_snippet_root`) against the real store
  root.
- The in-memory test runs step 4's own `process_snippet`, then builds the
  `PendingRecord` from that record's own sample rate and frequency. The record and its
  SigMF file must therefore agree, as the reader requires. The `pre_trigger` annotation
  step 4 writes supplies the noise reference, and the prompt says so
  (`noise_reference: "pre_trigger"`). Its tolerances hold over 20 seeds (Verified facts).

Both tests pass on their first run, because all the parts already exist.

- [ ] **Step 1: Write the end-to-end test (no database)**

<!-- write: tests/agent/test_end_to_end.py -->
```python
# tests/agent/test_end_to_end.py
"""Closure proof, no hardware, no network, no database:

synthetic IQ -> step 4's own process_snippet (measurement, SigMF writer
with its pre_trigger annotation, UnifiedRecord) -> ingest's
LocalSnippetStore.adopt -> a PendingRecord built from that record's own
sample rate and frequency (so the record and its file must agree) ->
ClassificationAgent (reader, segmentation, features, band table, prompt)
-> a fake LLM -> exact submit_classification arguments.

A 125 kHz burst 200 kHz above a 915 MHz tuning lands at 915.2 MHz inside
the 902-928 MHz entry with a plausible bandwidth: a grounded match, so a
0.92-confidence answer is auto-classified. A continuous carrier 250 kHz
below rides along as context. A second record, whose snippet capture
dropped for low disk, is closed out for review without an LLM call.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from anthropic.types import Message

from agent.band_table import load_band_table
from agent.db_gateway import PendingRecord
from agent.service import AgentSettings, ClassificationAgent
from capture.unknown.energy_trigger import TriggerEvent
from capture.unknown.service import CaptureSettings, process_snippet
from capture.unknown.snippet_assembler import CapturedSnippet
from dsp import synthetic
from schema.records import ClassificationStatus
from storage.snippet_store import LocalSnippetStore

FS = 1e6
TUNED = 915e6
N = 1 << 20  # 1.05 s, the shape of a step-4 snippet
PRE = round(0.1 * FS)  # step 4's default pre-trigger window
ANCHOR = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
ANSWER = {"tag": "ism_902_928:lora", "confidence": 0.92, "reasoning": "125 kHz bursts at 915.2 MHz."}


class RecordingGateway:
    def __init__(self, records):
        self.records, self.submitted = records, []

    def fetch_pending(self, after_id, limit):
        return [r for r in self.records if r.id > after_id][:limit]

    def fetch_newest_with_snippet(self, after_id, limit):
        withs = [r for r in self.records if r.id > after_id and r.iq_snippet_path is not None]
        return sorted(withs, key=lambda r: r.id, reverse=True)[:limit]

    def submit_classification(self, *args):
        self.submitted.append(args)


class RecordingLlm:
    def __init__(self):
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return Message.model_validate(
            {
                "id": "msg", "type": "message", "role": "assistant", "model": kwargs["model"],
                "stop_reason": "tool_use", "stop_sequence": None,
                "usage": {"input_tokens": 900, "output_tokens": 80},
                "content": [{"type": "tool_use", "id": "t", "name": "record_classification", "input": ANSWER}],
            }
        )


def _captured(seed: int) -> CapturedSnippet:
    """What step 4's assembler hands process_snippet: the trigger at 0.1 s."""
    rng = np.random.default_rng(seed)
    iq = (
        synthetic.noise(rng, N, 1e-5)
        + synthetic.tone(N, FS, -250e3, 1e-4)
        + synthetic.gate(synthetic.band_limited(rng, N, FS, 125e3, 200e3, 1e-3), FS, [(0.1, 0.3)])
    )
    trigger_index = 5_000_000 + PRE
    return CapturedSnippet(
        iq=iq,
        power=(np.abs(iq) ** 2).astype(np.float32),
        start_index=trigger_index - PRE,
        start_time=ANCHOR,
        trigger=TriggerEvent(sample_index=trigger_index, time=ANCHOR + timedelta(seconds=0.1), center_freq_hz=TUNED),
        sample_rate=FS,
    )


def test_step4_snippet_to_classification(tmp_path):
    staging, store_root = tmp_path / "staging", tmp_path / "snippets"
    settings = CaptureSettings(center_freq_hz=TUNED, sample_rate=FS, noise_floor_dbfs=-50.0, staging_dir=staging)
    record = process_snippet(_captured(42), settings, "survey-1", "op-1")
    stored = LocalSnippetStore(staging, store_root).adopt(record.metadata.iq_snippet_path)

    gateway = RecordingGateway(
        [
            PendingRecord(
                41,
                stored,
                record.metadata.sample_rate,
                record.identifier.center_freq,
                record.signal.peak_power,
                record.metadata.snippet_duration_ms,
            ),
            PendingRecord(42, None, FS, TUNED, -18.0, None, None, "low_disk"),
        ]
    )
    llm = RecordingLlm()
    agent = ClassificationAgent(
        gateway, llm, AgentSettings(snippet_root=store_root), load_band_table().entries
    )
    agent.check_snippet_root()  # the startup probe reads record 41's snippet
    assert agent.run_batch() == 2

    assert gateway.submitted == [
        (41, ClassificationStatus.AUTO_CLASSIFIED, "ism_902_928:lora", 0.92, ANSWER["reasoning"]),
        (
            42,
            ClassificationStatus.NEEDS_REVIEW,
            None,
            0.0,
            "No IQ snippet to classify: capture dropped it (quality_flags.snippet_dropped = 'low_disk').",
        ),
    ]
    (request,) = llm.requests
    prompt = request["messages"][0]["content"]
    payload = json.loads(prompt[prompt.index("{") :])
    assert payload["capture"]["noise_reference"] == "pre_trigger"  # step 4's annotation
    assert payload["primary_emitter"]["center_mhz"] == pytest.approx(915.2, abs=0.002)
    assert payload["primary_emitter"]["occupied_bandwidth_khz"] == pytest.approx(125, rel=0.05)
    assert payload["primary_emitter"]["duty_cycle"] == pytest.approx(0.3 / (N / FS), abs=0.01)
    assert [(m["id"], m["grounded"]) for m in payload["band_table_matches"]] == [("ism_902_928", True)]
    assert payload["other_emitters"][0]["center_mhz"] == pytest.approx(914.75, abs=0.002)
    assert payload["other_emitters"][0]["present_before_trigger"] is True
    assert payload["capture"]["reduced_confidence"] == []
    assert str(store_root) not in prompt and "survey" not in prompt
```
<!-- check: t12_e2e -->
Run: `python -m pytest tests/agent/test_end_to_end.py -q`
Expected: PASS, `1 passed` in about 1 s. It runs step 4's own process_snippet.


- [ ] **Step 2: Add the real-PostgreSQL closure test**

In `tests/storage/test_agent_boundary_pg.py`, extend the imports:

<!-- edit: tests/storage/test_agent_boundary_pg.py -->
Replace

```python
import pytest

```

with

```python
import numpy as np
import pytest
from anthropic.types import Message

```

<!-- edit: tests/storage/test_agent_boundary_pg.py -->
Replace

```python
from agent.db_gateway import (
```

with

```python
from agent.band_table import load_band_table
from agent.db_gateway import (
```

<!-- edit: tests/storage/test_agent_boundary_pg.py -->
Replace

```python
    connect_gateway,
)

```

with

```python
    connect_gateway,
)
from agent.service import AgentSettings, ClassificationAgent
from capture.unknown.snippet_writer import write_sigmf_snippet
from dsp import synthetic

```

<!-- edit: tests/storage/test_agent_boundary_pg.py -->
Replace

```python
from storage.repository import save_record

```

with

```python
from storage.repository import save_record
from storage.snippet_store import LocalSnippetStore

```

Append:

<!-- append: tests/storage/test_agent_boundary_pg.py -->
```python


def test_agent_classifies_a_real_row_through_the_boundary(boundary, tmp_path):
    """The whole agent against real PostgreSQL: a stored step-4 snippet, a
    pending row, the agent login role, a fake LLM; the row ends up
    auto_classified and nothing else in it changes. A second pending row
    whose snippet capture dropped is closed out as needs_review."""
    rng = np.random.default_rng(42)
    n = 1 << 18
    iq = synthetic.noise(rng, n, 1e-5) + synthetic.gate(
        synthetic.band_limited(rng, n, 1e6, 125e3, 200e3, 1e-3), 1e6, [(0.05, 0.1)]
    )
    staging, store_root = tmp_path / "staging", tmp_path / "snippets"
    staged = write_sigmf_snippet(
        iq, staging, 1e6, 915e6, datetime(2026, 10, 3, 12, tzinfo=timezone.utc), trigger_offset=50_000
    )
    stored = LocalSnippetStore(staging, store_root).adopt(str(staged))
    watermark = _watermark(boundary)
    record_id = _insert(
        boundary, Modality.UNKNOWN, ClassificationStatus.UNCLASSIFIED, path=stored, sample_rate=1e6
    )
    before = _metadata(boundary, record_id)
    dropped_id = _insert(
        boundary, Modality.UNKNOWN, None, path=None, sample_rate=1e6, flags={"snippet_dropped": "low_disk"}
    )
    answer = {"tag": "ism_902_928:lora", "confidence": 0.92, "reasoning": "125 kHz at 915.2 MHz."}

    class FakeLlm:
        messages = None

        def create(self, **kwargs):
            return Message.model_validate(
                {
                    "id": "m", "type": "message", "role": "assistant", "model": kwargs["model"],
                    "stop_reason": "tool_use", "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [{"type": "tool_use", "id": "t", "name": "record_classification", "input": answer}],
                }
            )

    llm = FakeLlm()
    llm.messages = llm
    gateway = connect_gateway(_url(boundary.agent_url), boundary.agent_role)
    try:
        agent = ClassificationAgent(
            gateway,
            llm,
            AgentSettings(snippet_root=store_root),
            load_band_table().entries,
            start_after_id=watermark,  # rows other tests left pending are not this test's
        )
        agent.check_snippet_root()
        assert agent.run_batch() == 2
    finally:
        gateway.close()
    assert _metadata(boundary, record_id) == {
        **before,
        "classification_status": "auto_classified",
        "tag": "ism_902_928:lora",
        "confidence": 0.92,
        "reasoning": "125 kHz at 915.2 MHz.",
    }
    dropped = _metadata(boundary, dropped_id)
    assert (dropped["classification_status"], dropped["tag"], dropped["confidence"]) == ("needs_review", None, 0.0)
    assert "snippet_dropped = 'low_disk'" in dropped["reasoning"]
```
<!-- check: t12_pg -->
Run with PostgreSQL: `python -m pytest tests/storage/test_agent_boundary_pg.py -q -W error`
Expected: PASS, `60 passed, 1 skipped`, with no warnings.


- [ ] **Step 3: Run the full suite, with and without PostgreSQL**

<!-- check: t12_full_pg -->
Run with PostgreSQL: `python -m pytest -q`
Expected: `exit 0`. The skips are the cellular CellSearch binary test, the live LLM test, and the pg_maintain case before PostgreSQL 17.

<!-- check: t12_full_nopg -->
Run: `python -m pytest -q`
Expected: `exit 0`. The PostgreSQL tests skip.


The `-W error` gate covers this plan's files only.
`tests/storage/test_repository.py` already fails under `-W error` at the
`9615218` base, with unclosed SQLite connections.

<!-- check: t12_werror -->
Run with PostgreSQL: `python -m pytest tests/agent tests/dsp tests/schema tests/test_no_tag_branching.py tests/storage/test_agent_boundary.py tests/storage/test_agent_boundary_pg.py -q -W error`
Expected: `exit 0`, with no warnings raised as errors.


- [ ] **Step 4: Commit, then stop the throwaway PostgreSQL**

<!-- run -->
```bash
git add tests/agent/test_end_to_end.py tests/storage/test_agent_boundary_pg.py
git commit -m "test: add end-to-end closure tests for the classification agent"
```

```bash
distrobox-host-exec podman stop sdr-part4-pg
```

---

## What this plan deliberately does not cover

- **TorchSig training and Jetson inference.** Only the `ModulationClassifier` seam ships.
  A real model will want the channelized primary region: `dsp.features.channelize`
  exists for that. Today the seam passes the raw capture, as the spec's signature says.
- **Library matching against SigMF/RadioML corpora, and frequency-hop detection.**
- **Live verification of the LLM call.** `claude-sonnet-5-5`, forced `tool_choice`,
  `thinking: disabled` and the real refusal behavior are untested against the API.
  Before trusting `auto_classified`, run the opt-in test in Task 9.
- **Building the container.** The deployment is decided, and `agent/README.md`
  documents it. This plan does not build the image, the manifests, the egress firewall or
  the memory limit:
  - same uid as ingest;
  - the store read-only at the identical resolved path;
  - no ingest socket;
  - egress only to the database and `api.anthropic.com`;
  - about 1 GiB of memory.
- **The self floor on real hardware.** With a quiet pre-trigger reference, the per-bin
  floor follows the receiver's roll-off. Without one, the percentile self floor assumes
  flat noise, and a roll-off of more than about 6 dB reads as a false wide region
  (measured with a synthetic 15 dB roll-off). Continuous emitters, which retrigger and
  fill their own reference, always take this path. The AD9361's real roll-off needs a
  recorded bladeRF capture to check.
- **More than one agent worker.** The atomic pending check makes a duplicate submit fail
  safely (`RecordNotPending`), but there is no work-claiming protocol.
- **Rescanning held-back records.** The cursor only moves forward. A record left
  pending is retried only after a restart from a lower `--start-after-id`. That covers:
  - unreadable snippets, empty spectra and analysis crashes;
  - held records whose release never came: outside-store snippets, invalid answers, or
    a timed-out write.
- **A schema migration tool.** The boundary is installed by the admin CLI, and
  `init_db`'s `create_all` still owns the table.
- **Packaging.** `pip wheel .` already fails at base on `license = "MIT"` with setuptools
  70.2. That fix belongs elsewhere.
- **Profiling on the Jetson Orin Nano.** The 1.00 s / 385 MB per 1 s/20 MS/s snippet
  were measured on x86.
- **Regions other than the US, ULS licensee lookups, a full §2.106 table, a review UI,
  and viz changes for `needs_review`.**
