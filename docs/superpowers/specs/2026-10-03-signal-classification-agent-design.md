# Signal Classification Agent (Part 4) Design

Scope: Build Sequencing step 5 (`docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md`
§9, §10). It triages `modality: "unknown"` records produced by step 4
(`docs/superpowers/specs/2026-08-11-unknown-signal-capture-design.md`) and writes a
classification tag, confidence and reasoning back into that record's own metadata,
and nothing else.

## Decisions taken with the user (2026-10-03)

1. **The boundary is enforced by a locked-down DB function, not column GRANTs.** The
   architecture doc says the agent role gets "UPDATE privilege limited to
   `metadata.classification_status/tag/confidence/reasoning`". Those are keys inside
   one `JSON` column (`storage/models.py: metadata_`), so Postgres column privileges
   can't express that. Instead the agent role has **no** privilege on
   `survey_records` at all. It gets `SELECT` on one filtered view and `EXECUTE` on
   one `SECURITY DEFINER` function, which validates its inputs and updates exactly
   those four keys on pending `unknown` rows. This boundary is narrower than the
   original column grant: it also enforces modality, value ranges, and "a human tag
   is never overwritten".
2. **v1 analysis uses deterministic features, a static band plan, and an LLM. The
   modulation classifier is a pluggable stub.** TorchSig training on the DGX Spark
   can't happen in this environment. The classifier is an interface whose v1
   implementation reports "unavailable"; the trained model plugs in later without
   touching the rest of the pipeline.
3. **LLM: `claude-sonnet-5-5`**, configurable. The design doc's choice ("bounded tool
   use with structured output") is unchanged; only the model ID is current.

Everything below that isn't one of these three decisions is a default chosen while
drafting. These are flagged as such where they matter.

## Pipeline (per record)

```
agent_pending_unknown view ──► load SigMF snippet ──► features (pure numpy)
        (DB, read-only)          (path-contained)          │
                                                           ▼
                            band-plan lookup at the *signal's* frequency
                                                           │
                                                           ▼
                            modulation classifier (v1: unavailable stub)
                                                           │
                                                           ▼
                  Claude: one forced `record_classification` tool call
                                                           │
                                                           ▼
            routing (pure): auto_classified | needs_review ──► classify_unknown()
```

1. **Fetch** pending rows from the `agent_pending_unknown` view. That covers
   `modality = 'unknown'` rows whose `classification_status` is `unclassified` or
   null, batch-limited.
2. **Load the snippet** from `metadata.iq_snippet_path` with `sigmf`. That path is
   DB data, so it's treated as untrusted: it must resolve (after `Path.resolve()`)
   under the configured snippet root, and the number of samples read is capped. If
   either check fails, the record goes to `needs_review` with that reason and the
   file is never opened.
3. **Features**, pure numpy, reusing the step-4 `dsp/` package:
   - 99%-power occupied bandwidth;
   - spectral-centroid offset, giving the absolute `signal_center_hz` (tuned center
     plus offset). The capture is up to ~56 MHz wide, so the tuned center frequency
     is **not** the signal's frequency, and the band plan must use the computed
     one;
   - peak-to-average power ratio;
   - duty cycle (fraction of windows above noise floor + 6 dB), burst count and
     mean burst duration;
   - spectral flatness;
   - a rough symbol-rate estimate from the strongest line in the envelope (|x|²)
     spectrum, or `None` when no line clears a peak-to-median threshold;
   - an SNR estimate.
4. **Band plan**: every curated allocation overlapping
   `[signal_center_hz ± occupied_bw/2]`.
5. **Modulation classifier**: `ModulationClassifier` protocol
   (`predict(samples, sample_rate) -> ModulationPrediction | None`). In v1,
   `UnavailableClassifier` always returns `None`.
6. **LLM**: a single `messages.create`, with `tools=[record_classification]` and
   `tool_choice` forced to that tool. Output schema:
   - `tag`, matching `^[a-z0-9][a-z0-9_.:-]{0,63}$`, e.g. `lte_downlink`,
     `ism_900_lora_css`, `unknown_narrowband`;
   - `confidence` in [0, 1];
   - `reasoning`, at most 2000 chars;
   - `candidate_identities`, an optional list.

   The system prompt is constant and cached (`cache_control`).
7. **Routing** is a pure function. `confidence >= threshold` (default **0.85**, a
   drafting default) gives `auto_classified`; anything else gives `needs_review`.
   **Grounding guard** (drafting default): if the band plan returned no match *and*
   the modulation classifier returned `None`, the result is `needs_review` whatever
   the LLM's confidence. A confident answer with nothing to ground it is not
   trusted for auto-classification.
8. **Write** through `classify_unknown(...)`. Nothing else can be written.

## Schema change

Add `ClassificationStatus.NEEDS_REVIEW = "needs_review"`. This is additive. The
architecture doc says low-confidence results are "flagged for human review with
reasoning attached", but the enum has no state for that. Without one, a
low-confidence record would either look untouched (and get re-sent to the LLM on
every cycle) or be indistinguishable from a confident one. A human resolves
`needs_review` to `manually_tagged`.

## Structural boundary

**Database** (`storage/sql/agent_boundary.sql`, idempotent; applied by
`storage/agent_boundary.py: install_agent_boundary(admin_engine)` and an admin CLI,
**never** at ingest or agent startup). It contains:

- `CREATE ROLE surveytool_agent NOLOGIN`. Operators create a login role `IN ROLE
  surveytool_agent` with a password from their own secret store. No credentials go
  in the repo, and `docker-compose.yml`'s dev password is not reused.
- `REVOKE ALL ON survey_records FROM surveytool_agent`.
- View `agent_pending_unknown(id, timestamp, identifier, signal, metadata)`,
  filtered as in step 1, with `GRANT SELECT` to the agent role. It omits
  `lat`/`lon`/`survey_id`/`operator_id`, which the agent doesn't need.
- `classify_unknown(p_record_id, p_status, p_tag, p_confidence, p_reasoning)`. It is
  `SECURITY DEFINER` with `SET search_path = pg_catalog, public`. `EXECUTE` is
  revoked from `PUBLIC` and granted to the agent role. It raises unless:
  - the row exists, has `modality = 'unknown'`, and is still pending;
  - `p_status` is one of `auto_classified` or `needs_review`;
  - `0 <= p_confidence <= 1`;
  - the tag matches the regex above;
  - the reasoning is at most 4000 chars.

  It updates only those four keys (`(metadata::jsonb || jsonb_build_object(...))::json`).
  `manually_tagged` rows are never pending, so the agent can never overwrite a
  human's tag.

**Code**:
- `agent/` never imports `capture.*`, `ingest.*` or `storage.repository`. A test
  enforces this by AST-parsing every module under `agent/`.
- All DB access goes through `agent/db_gateway.py`, which exposes exactly
  `fetch_pending(limit)` and `submit_classification(...)` as raw SQL against the
  view and function.
- The agent service refuses to start on a non-Postgres database URL. SQLite can't
  express the boundary, and running without it is not an option.

**LLM**: the model gets no tools except the forced output tool, so it can't take
actions. The prompt contains only:
- numeric features;
- curated band-plan entries;
- classifier output.

It contains no SigMF free-text fields (author/description: untrusted, a
prompt-injection surface) and no location, survey or operator IDs (not needed;
keeps survey data off a third-party API).

## Band plan

`agent/data/band_plan_us.json` is a curated static table, shipped as package data,
covering only 47 MHz to 6 GHz (the bladeRF xA9 tuning range). Each entry has
`start_hz`, `end_hz`, `service`, `typical_signals`, and a **required citation**:
47 CFR §2.106, or the specific rule part, e.g. §15.247 or Part 27. The
implementation plan must check every band edge against the eCFR text. Entries
without a verifiable citation are dropped, not guessed. Region is a config key, and
v1 ships `US` only.

## Failure handling

The service pattern matches the capture services: per-record errors are isolated
and logged, and never fatal to the loop.
- **Transient API or network errors** leave the row pending, and it is retried next
  cycle with backoff.
- **Model output that fails validation** gives `needs_review` with confidence 0 and
  reasoning naming the validation failure.
- **Config**: `ANTHROPIC_API_KEY` missing means fail fast at startup.
- **Cost**: `max_records_per_minute` caps API spend, since a busy band could produce
  many snippets.

## Testing

- Features are tested on synthetic IQ with known answers: a tone, a band-limited
  noise burst at a known bandwidth, BPSK at a known symbol rate, and OOK at a known
  duty cycle. Assertions use explicit tolerances.
- Band-plan lookup: edge overlaps, multiple matches, and no match. A data test
  checks every entry has a citation and `start < end` within range.
- Routing: threshold boundary values and the grounding guard.
- LLM: an injected fake client, with no network. The tests cover the forced
  tool-call shape, a malformed tool output becoming `needs_review`, and an API
  exception leaving the row pending. An optional live smoke test is skipped unless
  `ANTHROPIC_API_KEY` and `SURVEYTOOL_LIVE_LLM_TEST=1` are both set.
- **DB boundary: real Postgres only.** `tests/storage/test_agent_boundary_pg.py` is
  skipped unless `SURVEYTOOL_TEST_PG_URL` is set. It is run against the repo's
  `docker-compose.yml` PostGIS service during implementation. Connected as the
  agent role, it asserts that:
  - direct `SELECT`/`UPDATE`/`INSERT`/`DELETE` on `survey_records` all fail;
  - the view shows only pending unknown rows;
  - `classify_unknown` succeeds on a pending unknown row and leaves every other
    metadata key (`iq_snippet_path`, `sample_rate`, …) byte-identical;
  - it raises on a wifi row, a `manually_tagged` row, a bad status, confidence 1.5,
    a bad tag, and over-long reasoning.

  A SQLite run of the suite still passes, with the PG tests skipped.
- Import-boundary AST test (above).
- End-to-end, without hardware or network: a step-4 synthetic snippet goes through
  the record, the feature/band-plan path and a fake LLM, ending in
  `submit_classification` arguments.

## Dependency on step 4

Branch `worktree-signal-classification-agent` is stacked on
`worktree-unknown-signal-capture`. It needs that branch's `dsp/` package, the `sigmf`
dependency, the snippet-store root, and the normalizer setting
`classification_status = unclassified`. Its PR targets that branch until step 4
merges.

## Out of scope for this plan

- **TorchSig training and Jetson inference deployment.** Only the classifier
  interface ships.
- **Library matching against SigMF/RadioML corpora** (architecture §9 step 2). No
  corpus exists in this repo yet.
- **Frequency-hop pattern detection.** It needs time-frequency tracking across
  snippets.
- **Regions other than US, and FCC ULS licensee lookups.**
- **A human review UI, and viz changes to show `needs_review` and tags.**
- **More than one agent worker.** `classify_unknown` only accepts pending rows, so a
  duplicate submit fails cleanly, but there is no work-claiming protocol.
