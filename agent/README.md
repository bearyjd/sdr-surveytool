# agent

Part 4 signal classification. Triages `modality: "unknown"` records: it reads the
record's SigMF snippet, splits the spectrum into occupied regions, measures the strongest
one (bandwidth, center, duty cycle, bursts, PAPR, flatness, symbol rate), matches it
against a small curated US band table with eCFR citations, and asks Claude for one
forced `record_classification` tool call. Routing writes `auto_classified` only when the
confidence is >= 0.85 and the tag's band prefix (`ism_902_928` in `ism_902_928:lora`) is
a band-table entry the signal is grounded in; everything else goes to `needs_review`.
The noise reference is the snippet's pre-trigger, below the trigger threshold step 4
records in the SigMF. Any `reduced_confidence` reason keeps every band match ungrounded
and caps routing at `needs_review`: NaN or inf samples, an unreliable bandwidth, or no
quiet pre-trigger reference at all (`no_quiet_noise_reference`). The last covers every
always-on emitter, such as an LTE downlink, which fills its own pre-trigger: by default
those records always go to review.

### `--allow-self-floor-grounding` (off by default)

Without a quiet reference the agent estimates a flat noise floor from the capture itself,
which is blind to the receiver's filter roll-off. With this flag, a primary region
measured that way may ground and be auto-classified unless it reaches the outer 15% of
the band on either side (`edge_region_unreliable`, still `needs_review`). On synthetic
noise rolling off 15-20 dB, the false region the roll-off creates always reaches that
zone. At a milder 8-12 dB roll-off it need not: 20 of 259 wrong self-floor primaries were
interior regions inflated to 31-360 kHz, which could ground with a wrong bandwidth. Enable
the flag only after recorded bladeRF captures, at the sample rates and analog bandwidths
in use, confirm the receiver's roll-off is steep enough.
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
