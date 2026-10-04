# Signal Classification Agent (Part 4) Design

Scope: Build Sequencing step 5 (`docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md`
§9, §10). It triages `modality: "unknown"` records produced by step 4
(`docs/superpowers/specs/2026-08-11-unknown-signal-capture-design.md`, implemented on
branch `worktree-unknown-signal-capture`) and writes a classification tag, confidence
and reasoning back into that record's own metadata, and nothing else.

Revision 2 (2026-10-03) incorporates an adversarial critique of revision 1. The
main changes:
- a hardened DB boundary (search_path, function owner, `security_barrier`,
  connect-time role check, atomic pending check);
- a signal-region segmentation step replacing the whole-capture spectral centroid;
- a real grounding guard;
- import allowlisting instead of denylisting;
- revised failure handling and a revised test harness.

## Decisions taken with the user (2026-10-03)

1. **The boundary is enforced by a locked-down DB function and a filtered view, not
   column GRANTs.** The architecture doc's "UPDATE limited to
   `metadata.classification_status/tag/confidence/reasoning`" can't be expressed with
   Postgres column privileges, because those are keys inside one `JSON` column
   (`storage/models.py: metadata_`). The agent role instead has **no** privilege on
   `survey_records` itself.
2. **v1 analysis uses deterministic features, a curated band table, and an LLM. The
   modulation classifier is a pluggable stub.** TorchSig training on the DGX Spark
   can't happen in this environment. The classifier is an interface whose v1
   implementation reports "unavailable".
3. **LLM: `claude-sonnet-5-5`**, configurable.

Everything else here is a drafting default, open to revision in planning if a
build-and-run step contradicts it.

## Pipeline (per record)

```
agent_pending_unknown view ─► load SigMF snippet (contained, capped, no checksum pass)
                                   │
                                   ▼
          segment occupied regions in a Welch PSD (dsp/) ─► primary region
                                   │
                                   ▼
     mix+filter+decimate to primary region ─► time-domain features on that region
                                   │
                                   ▼
          curated band table: grounded match? (center inside band AND OBW plausible)
                                   │
                                   ▼
               modulation classifier (v1: UnavailableClassifier → None)
                                   │
                                   ▼
               Claude: one forced `record_classification` tool call
                                   │
                                   ▼
        routing (pure) ─► classify_unknown(): auto_classified | needs_review
```

1. **Fetch** from the `agent_pending_unknown` view, ordered by `id`. The agent
   remembers the highest id it has seen and fetches only higher ids
   (`WHERE id > $last ORDER BY id LIMIT n`), so a few stuck rows at the front can't
   block the rest of the backlog.
2. **Load the snippet.** The path is DB data, so it's untrusted. It must resolve
   (`Path.resolve()`, symlinks followed) under the configured snippet-store root,
   which is the directory `storage/snippet_store.py: LocalSnippetStore` adopts
   into. The reader uses `sigmffile.fromfile(..., skip_checksum=True)`, because the
   default SHA-512 pass reads the whole file, and then
   `read_samples(start, count)` with a hard cap of **2 s of samples** at the
   record's sample rate. The agent's OS user mounts or opens the store read-only
   (see Deployment).
3. **Segment** (new pure-numpy helpers added to `dsp/`, next to the existing
   `occupied_bandwidth_hz`):
   - Compute a Welch PSD over the frames that are above the noise floor.
   - Subtract the median noise floor and mask the DC/LO bin.
   - Find contiguous above-floor regions.
   - Pick the **primary region** with the largest integrated excess power.
   - Its centroid plus the tuned `center_freq` gives the absolute
     `signal_center_hz`; its 99%-power width is `obw_hz`.
   - Up to 3 other regions (center, OBW, relative power) go along as context.
   - A step-4 snippet can contain several emitters, since the trigger is
     level-based and the capture is up to ~56 MHz wide. Neither the
     whole-capture centroid nor the whole-capture time-domain statistics mean
     anything on that composite.
4. **Time-domain features**, computed after mixing the primary region to baseband,
   FFT-mask filtering it to `obw_hz`, and decimating:
   - peak-to-average power ratio;
   - duty cycle (fraction of frames above floor + 6 dB), burst count, mean burst
     duration;
   - spectral flatness;
   - a rough symbol-rate estimate from the strongest line in the envelope (|x|²)
     spectrum, or `None` if no line clears a peak-to-median threshold. A constant
     envelope (FM, CPM, rectangular PSK) legitimately yields `None`;
   - an SNR estimate.
5. **Curated band table** (`agent/data/band_table_us.json`): about 20 well-known
   allocations inside 47 MHz–6 GHz that an urban drive survey will actually hit.
   Examples: FM broadcast, airband, NOAA weather, FRS/GMRS, the 902–928 MHz ISM band,
   LTE bands 2/4/12/13/66/71, ADS-B 1090, GNSS L1, 2.4 GHz ISM, CBRS, U-NII-1/3.
   - Each entry has `start_hz`, `end_hz`, `service`, `typical_signals`, and
     `expected_obw_hz: [min, max]`.
   - Each entry also needs a **required citation** to the eCFR section (47 CFR
     §2.106 or the specific rule part). Planning verifies every edge against the
     eCFR text, and an entry without a verifiable citation is dropped.
   - A match is **grounded** when `signal_center_hz` lies inside the band *and*
     `obw_hz` lies within `expected_obw_hz`.
   - All overlapping entries go to the LLM, each marked grounded or not.
6. **Modulation classifier**: a `ModulationClassifier` protocol with
   `predict(iq, sample_rate) -> ModulationPrediction | None`. In v1,
   `UnavailableClassifier` always returns `None`. The trained TorchSig model plugs in
   here later.
7. **LLM**: a single `messages.create` with an explicit `max_tokens`, thinking off
   (it's incompatible with a forced `tool_choice`), `tools=[record_classification]`,
   and `tool_choice` forced to that tool.
   - **Tool schema:** `tag`, matching `^[a-z0-9][a-z0-9_.:-]{0,63}$`;
     `confidence` in [0, 1]; `reasoning`, at most 2000 chars. Alternative
     identities, if the model offers any, go inside `reasoning`; there's no
     separate field.
   - **Treated as validation failures:** `stop_reason` of `max_tokens` or
     `refusal`, no tool_use block, or a block that fails pydantic validation.
   - **No prompt caching:** the constant system prompt is likely below the minimum
     cacheable size, so caching buys nothing.
8. **Routing** (pure function):
   - `auto_classified` requires `confidence >= 0.85` **and** grounding: either a
     grounded band match, or a non-None modulation prediction.
   - Everything else is `needs_review`.
   - In v1 the classifier is always None, so auto-classification requires a
     grounded band match. That's intended.
9. **Write** through `classify_unknown(...)`, the only write path.

## Schema change

Add `ClassificationStatus.NEEDS_REVIEW = "needs_review"`. This is additive: viz,
ingest and storage don't branch on the status today. The architecture doc says
low-confidence results are "flagged for human review with reasoning attached", and
the enum has no state for that. A human resolves `needs_review` to
`manually_tagged`. **`needs_review` is terminal for the agent**, so it is only
assigned when a result is a genuine judgment about the record. It is never assigned
for systemic faults (see Failure handling). A `needs_review` write may carry a NULL
tag.

The architecture doc (§4 wording and §9 "Hard-boundary enforcement") and
`agent/README.md` still describe column grants. This branch updates both to describe
the function-and-view boundary.

## Structural boundary

### Database

`storage/sql/agent_boundary.sql` is idempotent. It is applied by
`storage/agent_boundary.py: install_agent_boundary(admin_engine)` through an admin
CLI, and **never** at ingest or agent startup. It does the following:

- **Roles.** `surveytool_agent` is NOLOGIN and is the agent's privilege set.
  Operators create a login role `IN ROLE surveytool_agent` with a password from
  their own secret store; nothing goes in the repo. `surveytool_classifier_owner`
  is NOLOGIN, holds only `SELECT` and `UPDATE (metadata)` on
  `public.survey_records`, and **owns** the function and the view. A superuser is
  never the definer.
- **Revokes.**
  - `REVOKE ALL ON public.survey_records FROM surveytool_agent`.
  - `REVOKE CREATE ON SCHEMA public FROM PUBLIC`, stated explicitly, because
    clusters upgraded from before PG15 keep the old default.
  - `REVOKE TEMPORARY ON DATABASE <db> FROM PUBLIC`.
- **Pending predicate.** It is defined once, as an `IMMUTABLE` SQL function:
  `modality = 'unknown' AND coalesce(metadata->>'classification_status',
  'unclassified') = 'unclassified'`. The view and `classify_unknown` both use it.
- **The view.** `public.agent_pending_unknown WITH (security_barrier = true)`
  projects only what the agent needs: `id`, `iq_snippet_path`, `sample_rate`,
  `center_freq`, `peak_power`, `snippet_duration_ms`. There's no location, no
  survey/operator ID, and nothing from other modalities. It has `GRANT SELECT` to
  `surveytool_agent`.
- **The function.** `public.classify_unknown(p_record_id, p_status, p_tag,
  p_confidence, p_reasoning)` is `SECURITY DEFINER` with
  `SET search_path = pg_catalog, pg_temp`. That puts `pg_temp` last explicitly;
  otherwise it is implicitly searched first. All object references inside are
  schema-qualified. `EXECUTE` is revoked from `PUBLIC` and granted to
  `surveytool_agent`.
  - **Arguments** are validated: the status is `auto_classified` or
    `needs_review`; `0 <= confidence <= 1`; the tag matches the regex, or is NULL
    only when the status is `needs_review`; reasoning is at most 4000 chars.
  - **The update is atomic:** `UPDATE public.survey_records SET metadata =
    (metadata::jsonb || jsonb_build_object(...))::json WHERE id = p_record_id AND
    <pending predicate> RETURNING id`, and it raises if no row comes back. A human
    tag committed while the LLM call was in flight therefore makes the agent's
    write fail instead of overwriting it.
- **Connect-time self-check.** At startup, `agent/db_gateway.py` refuses to run if
  any of these hold:
  - the current role is a superuser (`rolsuper`), or has `BYPASSRLS` or
    `CREATEROLE`;
  - `pg_has_role(current_user, 'surveytool_agent', 'MEMBER')` is false;
  - `has_table_privilege(current_user, 'public.survey_records', p)` is true for any
    of `SELECT`/`INSERT`/`UPDATE`/`DELETE`/`TRUNCATE`.

  Connecting with the compose `surveytool` superuser URL therefore fails loudly,
  instead of running without the boundary. The agent also refuses non-Postgres
  URLs.

### Code

- **Import allowlist, enforced by an AST test over every module in `agent/`.**
  Allowed are a fixed set of stdlib modules (`dataclasses`, `typing`, `enum`,
  `json`, `logging`, `math`, `pathlib`, `os` (env only), `signal`, `time`,
  `datetime`, `argparse`, `re`, `collections`, `functools`) plus `numpy`, `sigmf`,
  `anthropic`, `pydantic`, `sqlalchemy`, `psycopg`, `dsp`, `schema`, and `agent`
  itself.
- **Explicitly forbidden** (the same AST test): `subprocess`, `socket`, `ctypes`,
  `importlib`, `multiprocessing`, any `__import__`/`exec`/`eval` call, and
  `os.system`/`os.exec*`/`os.spawn*`/`os.popen`. These rule out invoking the
  cellular scanner, connecting to the ingest socket, and dynamic imports.
- All DB access goes through `agent/db_gateway.py`. It exposes exactly
  `fetch_pending(after_id, limit)` and `submit_classification(...)`, as SQL
  against the view and the function only.
- **Architecture rule, written into the architecture doc §1:** no module anywhere
  may read `metadata.tag` or `classification_status` to decide whether to start,
  retune or chain a capture/decode. An AST test over `capture/` and `ingest/`
  catches regressions. It flags any reference to `tag` or
  `classification_status` except an explicit allowlist, which starts as the step-4
  normalizer's single write of `UNCLASSIFIED`.

### LLM

- The model gets no tools except the forced output tool, so it can't take
  actions.
- The prompt contains only numeric features, region summaries, curated band-table
  entries and the classifier output.
- It contains no SigMF free-text fields: author/description are untrusted and a
  prompt-injection surface. It also contains no location, survey or operator IDs.

### Deployment (documented, not code-enforced here)

- The agent runs as a separate OS user, or in a separate container, with
  read-only access to the snippet store.
- It has no access to the ingest socket, which is `0600` and owned by the ingest
  user, so the agent can't inject records even with a code-level bypass.

## Failure handling

The service follows the capture services' pattern: per-record errors are isolated
and logged, and never fatal to the loop.

- **Snippet path outside the store root** (a containment violation): the record
  goes to `needs_review` with a NULL tag, and the reason goes in the reasoning.
  This is a judgment about the record itself.
- **Snippet file missing or unreadable:** the record stays pending.
  - **Halt condition:** after **N consecutive** such failures (default 5) the agent
    stops with a clear error. That situation is a systemic fault (an unmounted
    disk, a wrong root, a different host path), and the backlog must not be
    mass-marked.
- **Transient API or network error:** the record stays pending.
  - **Retry:** exponential backoff.
  - **Per-record cap:** an in-memory attempt cap (default 3). Once exceeded, the
    record goes to `needs_review` with a NULL tag, so it doesn't cycle forever.
- **Model output fails validation:** the record goes to `needs_review` with
  confidence 0, a NULL tag, and the reasoning naming the failure.
- **`ANTHROPIC_API_KEY` missing, or the role self-check fails:** fail fast at
  startup.
- **Spend:** a configurable **daily token budget**. When it's exhausted, the agent
  pauses until the UTC day rolls over. Per-minute rate limiting is unnecessary:
  step 4's 30 s per-frequency cooldown already caps input at about 2 records per
  minute per tuned frequency.

## Testing

- **Segmentation and features on synthetic IQ with known answers:**
  - a tone;
  - a band-limited noise burst of known bandwidth;
  - **RRC-shaped** BPSK (β≈0.35, at least 4 samples/symbol) at a known symbol
    rate; a rectangular-pulse PSK has a constant envelope and no |x|² line;
  - OOK at a known duty cycle;
  - **two-emitter** captures: a continuous carrier plus a burst at a different
    offset. The test asserts that the primary region and its features belong to
    the stronger emitter, and that the other emitter shows up as context.

  All assertions use explicit tolerances.
- **Band table:**
  - a data test: every entry has a citation, `start < end`, lies within 47 MHz–6
    GHz, and has a sane `expected_obw_hz`;
  - grounded vs. ungrounded matching at the edges;
  - overlapping entries.
- **Routing:** threshold boundaries, the grounding rule, and a NULL tag only with
  `needs_review`.
- **LLM:** an injected fake client, no network. The tests cover:
  - the forced tool-call request shape, including `max_tokens`;
  - a malformed tool input, a missing tool_use block, and `stop_reason` of
    `max_tokens` or `refusal`, each giving `needs_review`;
  - an API exception, which leaves the row pending, until the attempt cap is hit
    and it becomes `needs_review`.

  An optional live smoke test is skipped unless `ANTHROPIC_API_KEY` and
  `SURVEYTOOL_LIVE_LLM_TEST=1` are both set.
- **DB boundary: real PostgreSQL only.**
  - `tests/storage/test_agent_boundary_pg.py` is skipped unless
    `SURVEYTOOL_TEST_PG_URL` (an admin URL) is set.
  - During implementation it runs against a **throwaway**, fully qualified
    `docker.io/postgis/postgis:16-3.4` container on a non-default port. It is
    never run against the compose dev volume. Each run gets its own database and
    uniquely suffixed role names, because roles are cluster-global.
  - Started via `distrobox-host-exec podman`, because the interactive `podman`
    alias isn't available to non-interactive shells.
  - `psycopg[binary]` is added to `pyproject.toml`.
  - Connected as the agent login role, the test asserts that:
    - every direct operation on `survey_records` fails;
    - the view shows only pending unknown rows and only the projected columns;
    - a leaky-qual probe can't observe non-unknown rows through the view (a
      temp function is blocked by the TEMP revoke, and a failing cast on a
      hidden column isn't reachable);
    - `classify_unknown` succeeds on a pending unknown row, and every other
      metadata key is unchanged, compared by `json.loads` equality, not bytes,
      since the jsonb round-trip reorders keys;
    - it raises on a wifi row, a `manually_tagged` row, a row tagged by a
      concurrent "human" UPDATE made between fetch and submit, a bad status,
      confidence 1.5, a bad tag, a NULL tag with `auto_classified`, and
      over-long reasoning;
    - the connect-time self-check rejects a superuser URL.
  - A SQLite run of the suite still passes, with these skipped.
- **Import allowlist and forbidden-call AST test**, mutation-checked: an injected
  `import subprocess` makes it fail.
- **The `capture/`/`ingest/` "no branching on tag" AST test**, using the allowlist
  described under Code.
- **End-to-end**, with no hardware or network: a synthetic step-4 snippet is
  adopted into the store, read through the gateway-shaped interface, then goes
  through the features and band table to a fake LLM, ending in specific
  `submit_classification` arguments.

## Out of scope for this plan

- **TorchSig training and Jetson inference.** Only the classifier interface ships.
- **Library matching against SigMF/RadioML corpora** (architecture §9 step 2). No
  corpus exists in this repo yet.
- **Frequency-hop pattern detection.**
- **Regions other than US, FCC ULS licensee lookups, and a full §2.106 table.** The
  curated table is deliberately small.
- **A human review UI, and viz changes for `needs_review` and tags.**
- **More than one agent worker.** The atomic pending check makes duplicate submits
  fail safely, but there is no work-claiming protocol.
- **Enforcing the OS-user/container separation.** It's documented only.
