# agent

Part 4 signal classification. Triages `modality: "unknown"` records: it reads the
record's SigMF snippet, splits the spectrum into occupied regions, measures the strongest
one (bandwidth, center, duty cycle, bursts, PAPR, flatness, symbol rate), matches it
against a small curated US band table with eCFR citations, and asks Claude for one
forced `record_classification` tool call. Routing writes `auto_classified` only when the
tag has the form `<band-id>:<signal>`, the confidence is >= 0.85, and the band id
(`ism_902_928` in `ism_902_928:lora`) is a band-table entry the signal is grounded in
(its whole occupied span inside the band, give or take one fine bin, at a plausible OBW),
and the signal is one that entry is known for (every word of it appears in one of the
entry's `typical_signals`: `lora` or `lorawan` for 902-928 MHz, not `lte`);
everything else goes to `needs_review`. In v1 only the band table grounds: a modulation
classifier's label is shown to the model but never grounds a tag by itself.
The noise reference is the snippet's pre-trigger, below the trigger threshold step 4
records in the SigMF. Any `reduced_confidence` reason keeps every band match ungrounded
and caps routing at `needs_review`: NaN or inf samples, a truncated snippet (only the
first 2 s or 2^25 samples analysed), an unreliable bandwidth, a
primary less than 10 dB above the noise (`low_snr`), a width within 4 fine bins of the
resolution (`obw_unresolved`: a bare carrier), or no quiet pre-trigger reference at all
(`no_quiet_noise_reference`). The last covers every
always-on emitter, such as an LTE downlink, which fills its own pre-trigger: by default
those records always go to review.

The tag, confidence and reasoning go back into the record's own metadata, and nothing
else does. A record whose snippet ingest rejected or capture dropped (`iq_snippet_path`
NULL, `quality_flags.snippet_rejected` / `snippet_dropped` set) is closed out as
`needs_review` without an LLM call.

The modulation classifier is a seam (`classifier.py`): v1 ships `UnavailableClassifier`;
the TorchSig model trained offline on the DGX Spark plugs in later for Jetson inference.

## `--allow-self-floor-grounding` (off by default)

Without a quiet reference the agent estimates a flat noise floor from the capture itself,
which is blind to the receiver's filter roll-off. With this flag, a primary region
measured that way may ground and be auto-classified unless it reaches the outer 15% of
the band on either side (`edge_region_unreliable`, still `needs_review`). On synthetic
noise rolling off 15-20 dB, the false region the roll-off creates always reaches that
zone. At a milder 8-12 dB roll-off it need not: 20 of 259 wrong self-floor primaries were
interior regions inflated to 31-360 kHz, which could ground with a wrong bandwidth. Enable
the flag only after recorded bladeRF captures, at the sample rates and analog bandwidths
in use, confirm the receiver's roll-off is steep enough.

## Failure policy

`needs_review` is a judgment about one record, never the result of a systemic fault, and
the LLM answers at most once per record per run. Two consequences are by design:

- **Fast captures are never auto-classified.** The reader keeps at most 2 s and at most
  2^25 samples. A step-4 snippet lasts about 1 s (0.1 s before the trigger, 0.9 s after),
  so above ~33.55 MS/s (2^25 samples per second) every snippet is cut, carries the
  `truncated` reason, and goes to `needs_review`.
- **Five genuinely missing snippets in a row halt.** The agent cannot tell five records
  whose files were deleted from a stale or unmounted copy of the store, so it stops with
  them unmarked. Clean up or restore the files, then restart.

Every outcome falls in one class:

| Class | Cases | Action |
|---|---|---|
| Per-record judgment | malformed record (`malformed_record`); no snippet (the `snippet_rejected` / `snippet_dropped` flag named); snippet outside the store (`snippet_outside_store`); snippet corrupt while the store root is healthy (`snippet_unreadable`); snippet missing once another snippet has read after it (`snippet_missing`; held unmarked until then); no region even against the self floor (`no_occupied_region`); an analysis that fails twice (`analysis_failed`) | `needs_review` at once, NULL tag, confidence 0, the reason named, no LLM call, never counted toward a halt |
| Transient per-record | a transient read error (EIO, EAGAIN, EINTR, ETIMEDOUT, ESTALE, ENOMEM, EBUSY); a write timeout (`statement_timeout`, lock timeout); a transient API error (connection, 408, 409, 429, >= 500) | kept, with its verdict if one was decided, in an in-run deferred set that later batches retry by id, whatever the cursor, once due: 2, 4, 8, ... s after the last attempt (capped at 300 s). While only not-yet-due records remain, the loop sleeps its poll. A record the model has answered is never asked again. An API error also backs off 2, 4, ... s (capped at 300 s) before the next batch. After 10 attempts the record is left pending for the next run |
| Systemic: halt | the store root missing, not a directory, unreadable, or empty while records point into it; 5 missing snippets in a row with none read in between (a stale copy of the store); 5 batches in a row failing outside any record (the database unreachable); 5 invalid model answers in a row; a non-retryable API error; a write the database rejects | halt with nothing marked for it, exit status 3 (never auto-restarted, below); the message names the cause |

A row whose JSON holds a `\u0000` escape (only a writer bypassing the schema's NUL check
can store one) makes every `->>` on it fail, so the view leaves it out and
`classify_unknown` refuses it; a write that still meets one (SQLSTATE 22P05) skips that
record. Such rows stay pending until cleaned up by hand; find them with
`SELECT id FROM survey_records WHERE metadata::text || identifier::text || signal::text ~ '(^|[^\\])(\\\\)*\\u0000';`
(standard_conforming_strings on; a `\u0000` preceded by an even number of backslashes).

An invalid model answer (validation failure, refusal, `max_tokens`) is held until a later
answer validates, then goes to review. A spent daily token budget pauses until UTC
midnight. At startup the agent checks the store root and reads the newest pending
snippets until one reads; if none does, it halts. One broken snippet among readable ones
never blocks startup.

### The daily token budget

`--daily-token-budget` (default 2,000,000) bounds what the agent spends per UTC day. Each
call reserves its worst case before it is made: its estimated input (system prompt, tool
schema and message, at 3 characters per token, an overestimate) plus `--max-tokens`. The
reservation is reconciled to the billed usage when the call returns; a call that fails or
times out keeps its reservation, since it may still have been billed. A call whose
reservation does not fit waits for UTC midnight; the day's first call always goes ahead.
The CLI refuses a budget smaller than one reservation (about 3,000 tokens plus
`--max-tokens`), which would allow about one call a day.
The budget lives in the process: a restart starts it over, so a crash loop could spend it
again and again. That is why halts exit with status 3 and must never be auto-restarted
(below), and why other restarts should be slow (`RestartSec`).

## The boundary

The agent is a dead end in the pipeline: it never triggers, chains into, or hands data
to the cellular/WiFi/BT decode modules.

- **The container is the boundary.** The agent runs in its own container as the same uid
  as ingest (step 4 keeps the snippet store `0700` and every snippet `0600`, owned by that
  uid). The store is bind-mounted **read-only at the identical resolved path ingest uses**:
  records hold absolute `resolve()`d paths, and the agent refuses to start if the root is
  unusable or none of the newest pending snippets reads under `--snippet-store-dir`.
  The ingest socket is **not**
  mounted. Network egress is limited to the database and `api.anthropic.com`. Give it a
  memory limit of about 1 GiB. The worst case is a snippet at the 2^25-sample read cap
  whose primary region fills the band (nothing is decimated): the read plus the analysis
  then peak at 584 MB RSS, the 256 MiB of IQ, one channelized copy of the same size, and
  bounded batches (a 1 MHz region at the same size: 454 MB).
- **The database.** The agent's login role (`IN ROLE surveytool_agent`, and in nothing
  else) has no privilege on `survey_records`; it reads the `agent_pending_unknown` view and
  writes only through the `classify_unknown` function (`storage/sql/agent_boundary.sql`,
  installed by `sdr-agent-boundary`). At startup the agent checks an allowlist against the
  login itself (`session_user`): the session must run as the login (no `SET ROLE`), the
  login must have no role settings (`ALTER ROLE ... SET`), no membership but that role, no
  dangerous role attribute, no privilege on the table, no CREATE anywhere, no TEMPORARY on
  the survey database, and no other executable SECURITY DEFINER function. It also checks the
  definer: the owner role (NOINHERIT, set by the installer) belongs to no role, holds
  nothing on `survey_records` beyond SELECT and UPDATE (metadata), and owns the view and
  both functions, which have no other overloads. The installer only touches the survey
  database. On a default cluster PUBLIC keeps TEMPORARY and CONNECT on `postgres` and the
  templates; the agent starts anyway and logs a warning naming them. Recommended
  hardening: `REVOKE TEMPORARY, CONNECT ON DATABASE postgres FROM PUBLIC;`, and likewise
  for each other database the login should not reach. The installer also revokes
  the large-object writers (`lo_create`, `lo_creat`, `lo_from_bytea`, `lo_import`,
  `lo_open`, `lo_put`) from PUBLIC, the one write path that needs no table privilege, and
  the self-check refuses a login that can still execute one.
- **Accepted leak: row counts.** The agent role can read `pg_stat_*` counters and run
  `EXPLAIN` on the view, which reveal approximate row counts of `survey_records` (no
  values). That is accepted.
- **What PostgreSQL cannot prevent.** Any role may change its own password and its own
  session defaults (`ALTER ROLE <itself> ... SET`). The self-check refuses role settings
  at the next start, but cannot see a password change. Alert on changes to `pg_authid`
  and `pg_db_role_setting` for the agent login, and give the login a `CONNECTION LIMIT`
  (2: the agent and its advisory-lock session).
- **The LLM.** The model has exactly one tool, the forced output tool, and sees only
  numeric features and band-table entries: no SigMF free text, location, survey or
  operator IDs.
- **Tripwires, not walls.** `tests/agent/test_import_boundary.py` (an import allowlist
  and banned calls over `agent/`, `dsp/` and `schema/`) and `tests/test_no_tag_branching.py`
  (no reference to tags or classification status in `capture/` or `ingest/`) catch obvious
  regressions in review. They do not stop a determined bypass; the container does.

## Running

```bash
# Once, as a database administrator. The URL comes from the environment, never argv
# (visible to every local user); leave the password out and keep it in ~/.pgpass or a
# libpq service file.
export SURVEYTOOL_ADMIN_DATABASE_URL=postgresql://postgres@localhost:5432/surveytool
sdr-agent-boundary          # preview: lists what it would revoke from PUBLIC, and who relies on it
sdr-agent-boundary --apply  # install
# Create the agent's login role yourself, then set its password interactively, so it
# never appears in a command line, shell history or the server log:
psql "$SURVEYTOOL_ADMIN_DATABASE_URL" \
    -c 'CREATE ROLE surveytool_agent_login LOGIN CONNECTION LIMIT 2 IN ROLE surveytool_agent'
psql "$SURVEYTOOL_ADMIN_DATABASE_URL" -c '\password surveytool_agent_login'

# A database that is not local (unix socket, localhost, 127.0.0.1, ::1) must be reached
# over TLS: the agent refuses sslmode disable, allow and prefer (prefer silently falls
# back to clear text). It judges what libpq will connect with: every entry of the host
# and hostaddr lists (a query host= overrides the URL's host), and PGHOST, PGHOSTADDR and
# PGSSLMODE for whatever the URL leaves out. A connection service (service= or PGSERVICE)
# is refused, since its pg_service.conf entry could set any of them; put them in the URL.
# verify-full is recommended: require and verify-ca encrypt, but only verify-full checks
# the server's name, so a host that can redirect the connection cannot impersonate the
# database.
export SURVEYTOOL_AGENT_DATABASE_URL=postgresql://surveytool_agent_login:...@localhost:5432/surveytool
# remote: postgresql://surveytool_agent_login:...@db.example.net/surveytool?sslmode=verify-full
export ANTHROPIC_API_KEY=...
sdr-agent --snippet-store-dir /absolute/path/of/ingest/data/snippets
```

### Regranting what the boundary revokes from PUBLIC

`--apply` revokes from PUBLIC: every privilege on `survey_records`, CREATE on schema
`public`, TEMPORARY on the survey database, and EXECUTE on the large-object writers. The
preview names each login role that holds one of these only through PUBLIC. Grant each
such role what it actually needs, explicitly, before or right after `--apply`; for
example, for ingest (if it is not the table owner) and a migration role:

```sql
GRANT SELECT, INSERT, UPDATE ON public.survey_records TO surveytool_ingest;
GRANT USAGE ON SEQUENCE public.survey_records_id_seq TO surveytool_ingest;
GRANT TEMPORARY ON DATABASE surveytool TO surveytool_ingest;   -- only if it uses temp tables
GRANT CREATE ON SCHEMA public TO surveytool_migrations;
GRANT EXECUTE ON FUNCTION lo_create(oid), lo_open(oid, integer), lo_put(oid, bigint, bytea)
    TO surveytool_blobs;                                        -- only if it uses large objects
```

Never grant any of these to `surveytool_agent` or to the agent login: the self-check
refuses it.

Each record's id is logged before its snippet is read. If a snippet crashes the agent,
the last log line names the record; restart with `--start-after-id <id>` to skip it once
the cause is understood.

### Halts are terminal

A halt (the systemic class above) logs `Agent halted: <cause>` and exits with status
**3**. Never restart on it automatically: the fault would recur, and the daily token
budget, which lives in the process, would start over with every restart. Other exits (a
crash, a lost database at startup) may restart, slowly.

### A hardened systemd unit

The container described under "The boundary" remains the recommended boundary: its PID
namespace does not depend on the host's systemd version, and its network policy limits
egress by host. This unit is narrower than the container, not equivalent to it (see the
notes after it). Each line says what it enforces:

```ini
[Unit]
Description=sdr-surveytool Part 4 classification agent
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=/usr/local/bin/sdr-agent --snippet-store-dir /srv/surveytool/snippets
# The uid that owns the 0700 store and its 0600 snippets (step 4's ingest uid).
User=surveytool-ingest
# SURVEYTOOL_AGENT_DATABASE_URL and ANTHROPIC_API_KEY; the file is root-owned, 0600.
EnvironmentFile=/etc/surveytool/agent.env
# No privilege gain through setuid/setgid binaries or file capabilities.
NoNewPrivileges=yes
# No capabilities at all, not even the bounding set to regain them from.
CapabilityBoundingSet=
# The whole filesystem is read-only to the agent...
ProtectSystem=strict
# ...and the store explicitly so (it never writes there).
ReadOnlyPaths=/srv/surveytool/snippets
# No /home, /root or /run/user; a private /tmp and /dev (no device files).
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
# TCP/IP only. No AF_UNIX: the ingest socket (and any other local socket) is
# unreachable, so the database must be reached over TCP (127.0.0.1 locally).
RestrictAddressFamilies=AF_INET AF_INET6
# Its own PID namespace (systemd 257 or later): /proc shows only the agent's own
# processes, so ingest's /proc entries (command line, environment, open files, memory)
# are out of reach. The agent is PID 1 there and handles SIGTERM itself once started.
PrivatePIDs=yes
# Other users' processes are hidden from /proc. Ingest shares this uid, so this alone
# does not hide ingest: PrivatePIDs= does.
ProtectProc=invisible
# /proc holds only process directories: no /proc/sys, /proc/net or /proc/cpuinfo.
ProcSubset=pid
# A seccomp allow-list: the system calls a typical service needs, minus the debugging
# ones (ptrace, perf_event_open, pidfd_getfd). Any other call kills the agent
# (status=31/SYS in the journal).
SystemCallFilter=@system-service
SystemCallFilter=~@debug
# Every unit's systemd credentials (LoadCredential=), ingest's database URL included.
# The "-" lets the unit start on a host where the directory does not exist.
InaccessiblePaths=-/run/credentials
# The worst case peaks at 584 MB (above); past 1 GiB the kernel kills it.
MemoryMax=1G
# Crashes restart, slowly; a halt (status 3) never does.
Restart=on-failure
RestartSec=60
RestartPreventExitStatus=3

[Install]
WantedBy=multi-user.target
```

What the unit does not cover:

- **PID isolation needs systemd 257.** An older systemd logs `Unknown key name
  'PrivatePIDs'` and starts the agent without it. Nothing in the unit then separates the
  agent from ingest, since they share a uid: through `/proc/<ingest pid>` the agent can
  read ingest's command line and environment, its credentials (via `root/`), and its
  memory where `kernel.yama.ptrace_scope` is 0. Check with `systemctl show -p PrivatePIDs
  sdr-agent.service` (expect `PrivatePIDs=yes`); on an older host, run the agent in the
  container.
- **Ingest's database password.** It must never go on ingest's command line, which every
  local user can read (ingest warns when `--database-url` carries one). Give it to ingest
  as a credential: `LoadCredential=database_url:/etc/surveytool/ingest-database-url` in
  ingest's unit, the file root-owned and `0600` (systemd reads it, the shared uid cannot),
  or `SURVEYTOOL_DATABASE_URL` from a root-owned `0600` `EnvironmentFile=`. With
  `PrivatePIDs=` and `InaccessiblePaths=-/run/credentials` above, neither reaches the
  agent. Leave `InaccessiblePaths=` out of ingest's unit: it would hide ingest's own
  credentials.
- **Egress by host name.** Restrict it to the database and `api.anthropic.com` with a
  firewall or an egress proxy, as the container's network policy does.
  (`IPAddressAllow=` takes addresses, and the API's change.)

One agent runs per database: it holds a session advisory lock for its lifetime, and a
second instance refuses to start, naming the lock and the query that finds its holder
(`pg_locks` joined to `pg_stat_activity`). The agent checks the lock before every batch; if
its lock session dies, it halts (status 3) rather than race a successor. Any role that can
connect to the database can take the key first and block startup (lock-key squatting).
The refusal names the query that finds the holder, which you can also run yourself:

```sql
SELECT a.pid, a.usename, a.client_addr, a.backend_start
  FROM pg_locks AS l JOIN pg_stat_activity AS a USING (pid)
 WHERE l.locktype = 'advisory' AND l.classid = 1396986433 AND l.objid = 1195724372 AND l.granted;
```

Terminate that session (`pg_terminate_backend(pid)`) if it is not an agent; revoking CONNECT
from PUBLIC (above) limits who can squat. SIGTERM stops the agent at once, even mid-sleep
(a poll, a backoff or a budget pause).
The Anthropic client is pinned to `https://api.anthropic.com`; `ANTHROPIC_BASE_URL` is
ignored.
