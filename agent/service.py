# agent/service.py
"""The Part 4 classification agent: fetch, analyse, ask, route, write back.

Two invariants:
- needs_review is a judgment about one record, never the consequence of a
  systemic fault;
- the LLM answers at most once per record per process run. A decision that
  could not be written is kept and written again, never re-asked. (A call
  that raised before returning a response is not an answer.)

Every outcome falls in one of three classes (agent/README.md has the table):
- per-record judgment: needs_review at once, NULL tag, a reason naming it,
  no LLM call, never counted toward a halt. A malformed record, no snippet
  (rejected or dropped), a snippet outside the store, a snippet missing or
  corrupt while the store root is healthy, no occupied region even against
  the self floor, an analysis that fails twice;
- transient per-record: the record, with its verdict if one was decided, is
  kept in an in-run deferred set that later batches retry by id, so the
  advancing cursor never strands it. A transient read error, a write
  timeout, a transient API error (which also backs off, capped, before the
  next batch). After max_attempts_per_record attempts the record is left
  pending for the next run;
- systemic: halt, with nothing marked for it. The store root unhealthy
  (missing, not a directory, unreadable, or empty while records point into
  it), a run of failed batches (the database unreachable), a run of invalid
  model answers, a non-retryable API error, a write the database rejects.

Two outcomes sit outside the classes: an invalid model answer is held until
a later answer validates (then it goes to review), and a spent daily token
budget pauses until the UTC day rolls over.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import anthropic

from agent.analysis import SnippetAnalysis, analyse_snippet
from agent.band_table import DEFAULT_BAND_TABLE, BandEntry, load_band_table
from agent.classifier import ModulationClassifier, UnavailableClassifier
from agent.db_gateway import (
    DEFAULT_AGENT_ROLE,
    AgentAlreadyRunning,
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
from agent.snippet_reader import Snippet, SnippetOutsideStore, SnippetUnreadable, read_snippet

logger = logging.getLogger(__name__)

_SDK_MAX_RETRIES = 2  # inside each of our calls, before our own backoff
_SDK_TIMEOUT_SECONDS = 60.0
# Pinned: ANTHROPIC_BASE_URL in the environment must not redirect the agent.
_API_BASE_URL = "https://api.anthropic.com"
# A halt's exit status. Never auto-restart on it (RestartPreventExitStatus,
# agent/README.md): the fault would recur, and the daily token budget, which
# lives in the process, would start over.
EXIT_HALTED = 3
_STARTUP_PROBES = 5
_MAX_BACKOFF_EXPONENT = 30  # 2 s * 2**30 is far past any cap; 2**2000 would overflow a float


class SystemicFault(RuntimeError):
    """Not about any one record: halt rather than mark the backlog."""


class _Backoff(Exception):
    """A transient API error: the record is deferred; wait before the next batch."""

    def __init__(self, delay: float) -> None:
        super().__init__(delay)
        self.delay = delay


@dataclass(frozen=True)
class _Deferred:
    attempts: int  # transient failures so far
    verdict: Decision | None  # decided and awaiting a successful write; None: decide again


def backoff_delay(streak: int, base_seconds: float, cap_seconds: float) -> float:
    """base * 2**(streak - 1), capped; the exponent is clamped first."""
    return min(base_seconds * 2.0 ** min(streak - 1, _MAX_BACKOFF_EXPONENT), cap_seconds)


def store_root_problem(root: Path, expect_content: bool = False) -> str | None:
    """Why the snippet store root cannot be used, or None. With
    expect_content, an empty root (an unmounted store's mount point) is a
    problem too: records point into it."""
    if not root.is_absolute():
        return f"--snippet-store-dir must be absolute, got {str(root)!r}"
    try:
        empty = next(root.iterdir(), None) is None
    except FileNotFoundError:
        return f"{root} does not exist"
    except NotADirectoryError:
        return f"{root} is not a directory"
    except PermissionError:
        return f"permission denied on {root}"
    except OSError as exc:
        return f"cannot list {root}: {exc}"
    if expect_content and empty:
        return f"{root} is empty although pending records point into it: is the store mounted?"
    return None


def _under(path: str, root: Path) -> bool:
    """Lexically, without touching the file: is `path` inside `root`?"""
    candidate = Path(path)
    return candidate.is_absolute() and Path(*candidate.parts).is_relative_to(root.resolve())


@dataclass(frozen=True)
class AgentSettings:
    snippet_root: Path
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    batch_size: int = 20
    max_attempts_per_record: int = 10  # transient failures before a record waits for the next run
    max_consecutive_bad_outputs: int = 5
    daily_token_budget: int = 2_000_000
    backoff_seconds: float = 2.0
    max_backoff_seconds: float = 300.0
    # Opt-in: interior regions measured against the self floor may ground
    # (agent.analysis). Off until real bladeRF captures confirm the roll-off.
    allow_self_floor_grounding: bool = False


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
    """Stateful by necessity: it carries the fetch cursor, the deferred and
    held records, the decisions not yet written and the day's token spend
    from one record to the next."""

    def __init__(
        self,
        gateway: AgentGateway,
        client: MessagesClient,
        settings: AgentSettings,
        bands: tuple[BandEntry, ...],
        classifier: ModulationClassifier | None = None,
        sleep: Callable[[float], object] = time.sleep,
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
        self._deferred: dict[int, _Deferred] = {}
        self._given_up: set[int] = set()  # past the attempt cap: pending until the next run
        self._bad_outputs: list[tuple[int, Decision]] = []
        self._transient_streak = 0
        self._retry_in: float | None = None
        self._budget_day = now().date()
        self._tokens_today = 0

    def check_snippet_root(self) -> None:
        """Before touching any record: the store root must be healthy, and
        the newest pending snippet paths must lie under it (lexically: a
        broken snippet is a judgment about its record, not a reason to
        refuse to start). Step 4 stores resolve()d absolute paths, so a
        store mounted at another path would put every snippet 'outside'."""
        root = self._settings.snippet_root
        problem = store_root_problem(root)
        if problem is not None:
            raise SystemicFault(f"The snippet store is unusable: {problem}")
        newest = self._gateway.fetch_newest_with_snippet(self._last_id, _STARTUP_PROBES)
        paths = [record.iq_snippet_path for record in newest if record.iq_snippet_path and record.malformed is None]
        if paths and not any(_under(path, root) for path in paths):
            raise SystemicFault(
                f"None of the newest pending snippets is under {root} (e.g. {paths[0][:200]!r}). "
                "Mount the snippet store read-only at the identical resolved path ingest uses."
            )

    def take_retry_delay(self) -> float | None:
        """Seconds to wait before the next batch after a transient API
        error, or None."""
        delay, self._retry_in = self._retry_in, None
        return delay

    def run_batch(self) -> int:
        """Retry the deferred records by id, then process the next pending
        records above the cursor; returns how many records were looked at.
        Raises SystemicFault to halt."""
        looked_at = 0
        try:
            looked_at += self._retry_deferred()
            records = self._gateway.fetch_pending(self._last_id, self._settings.batch_size)
            looked_at += len(records)
            for record in records:
                if record.id not in self._deferred and record.id not in self._given_up:
                    try:
                        self._process(record)
                    except _Backoff:
                        self._last_id = record.id  # deferred: retried by id, not by the cursor
                        raise
                # Any other exception leaves the cursor before the record, so
                # it is fetched again (a verdict already decided is deferred).
                self._last_id = record.id
        except _Backoff as backoff:
            self._retry_in = backoff.delay
        return looked_at

    def _retry_deferred(self) -> int:
        if not self._deferred:
            return 0
        still_pending = {record.id: record for record in self._gateway.fetch_by_ids(sorted(self._deferred))}
        for record_id in sorted(self._deferred):
            if record_id not in still_pending:  # written meanwhile, or a human tagged it
                del self._deferred[record_id]
                continue
            verdict = self._deferred[record_id].verdict
            if verdict is not None:
                self._write(record_id, verdict)
            else:
                self._process(still_pending[record_id])
        return len(still_pending)

    def _process(self, record: PendingRecord) -> None:
        logger.info("Record %d: snippet %s", record.id, record.iq_snippet_path)
        verdict = self._judge(record)
        if verdict is not None:
            self._write(record.id, verdict)

    def _judge(self, record: PendingRecord) -> Decision | None:
        """The verdict to write now, or None when the record is deferred or
        held."""
        if record.malformed is not None:
            return needs_review(f"malformed_record: {record.malformed}")
        if record.iq_snippet_path is None:
            return needs_review(_no_snippet_reason(record))
        try:
            snippet = read_snippet(
                record.iq_snippet_path, self._settings.snippet_root, record.sample_rate, record.center_freq
            )
        except SnippetOutsideStore as exc:
            self._require_healthy_root(expect_content=False)
            return needs_review(f"snippet_outside_store: {exc}")
        except SnippetUnreadable as exc:
            if exc.transient:
                self._defer(record.id, str(exc))
                return None
            self._require_healthy_root(expect_content=True)
            return needs_review(f"snippet_unreadable: {exc}")
        analysis = self._analyse(record.id, snippet)
        if isinstance(analysis, Decision):
            return analysis
        result = self._ask(record.id, build_user_message(analysis))
        if result.classification is None:
            self._hold_bad_output(record.id, needs_review(f"Model output failed validation: {result.failure}"))
            return None
        decision = route(
            result.classification,
            analysis.grounded_band_ids,
            analysis.modulation_label,
            analysis.reduced_confidence,
        )
        self._remember(record.id, decision)  # before releasing held answers can fail
        self._release_bad_outputs()
        return decision

    def _analyse(self, record_id: int, snippet: Snippet) -> SnippetAnalysis | Decision:
        """The analysis, or the judgment it leads to. A failure is retried
        once; a second one is about the record (the analysis is
        deterministic)."""
        try:
            analysis = self._analyse_once(snippet)
        except Exception:
            logger.exception("Analysis failed for record %d; retrying once", record_id)
            try:
                analysis = self._analyse_once(snippet)
            except Exception as exc:
                logger.exception("Analysis failed again for record %d", record_id)
                return needs_review(f"analysis_failed: {exc!r}")
        if analysis.primary is None:
            return needs_review(
                "no_occupied_region: nothing stood above the noise floor, the self floor included"
            )
        return analysis

    def _analyse_once(self, snippet: Snippet) -> SnippetAnalysis:
        return analyse_snippet(snippet, self._bands, self._classifier, self._settings.allow_self_floor_grounding)

    def _require_healthy_root(self, expect_content: bool) -> None:
        problem = store_root_problem(self._settings.snippet_root, expect_content)
        if problem is not None:
            raise SystemicFault(f"The snippet store is unusable: {problem}. No record was marked for it.")

    def _remember(self, record_id: int, verdict: Decision) -> None:
        entry = self._deferred.get(record_id)
        self._deferred[record_id] = _Deferred(0 if entry is None else entry.attempts, verdict)

    def _defer(self, record_id: int, why: str) -> None:
        """A transient failure: keep the record (and any verdict) for a later
        batch, up to max_attempts_per_record attempts."""
        entry = self._deferred.get(record_id) or _Deferred(0, None)
        attempts = entry.attempts + 1
        if attempts >= self._settings.max_attempts_per_record:
            self._deferred.pop(record_id, None)
            self._given_up.add(record_id)
            logger.error(
                "Record %d left pending after %d attempts (last: %s); not retried again this run",
                record_id, attempts, why,
            )
            return
        self._deferred[record_id] = _Deferred(attempts, entry.verdict)
        logger.warning("Record %d deferred (attempt %d): %s", record_id, attempts, why)

    def _hold_bad_output(self, record_id: int, verdict: Decision) -> None:
        self._bad_outputs.append((record_id, verdict))
        logger.warning("Record %d held pending: %s", record_id, verdict.reasoning)
        if len(self._bad_outputs) >= self._settings.max_consecutive_bad_outputs:
            ids = [held for held, _ in self._bad_outputs]
            raise SystemicFault(
                f"{len(ids)} model answers in a row were refused or failed validation (records "
                f"{ids}; last: {verdict.reasoning}). Check the model id, the tool schema and "
                "the prompt; they stay pending."
            )

    def _release_bad_outputs(self) -> None:
        """An answer just validated, so the model works: each held invalid
        answer was about its own record, and goes to review."""
        while self._bad_outputs:
            record_id, verdict = self._bad_outputs.pop(0)
            self._remember(record_id, verdict)
            self._write(record_id, verdict)

    def _ask(self, record_id: int, user_message: str) -> LlmResult:
        self._wait_for_budget()
        try:
            result = request_classification(
                self._client, SYSTEM_PROMPT, user_message, self._settings.model, self._settings.max_tokens
            )
        except Exception as exc:
            if not is_transient(exc):
                raise SystemicFault(f"Non-retryable Anthropic API error on record {record_id}: {exc!r}") from exc
            self._transient_streak += 1
            delay = backoff_delay(
                self._transient_streak, self._settings.backoff_seconds, self._settings.max_backoff_seconds
            )
            self._defer(record_id, f"transient API error {exc!r}; next batch in {delay:.0f} s")
            raise _Backoff(delay) from exc
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

    def _write(self, record_id: int, verdict: Decision) -> None:
        """Write a verdict. It stays in the deferred set until the database
        accepts it (or a human got there first), so no failure loses it."""
        self._remember(record_id, verdict)
        try:
            self._gateway.submit_classification(
                record_id, verdict.status, verdict.tag, verdict.confidence, verdict.reasoning
            )
        except RecordNotPending:
            logger.info("Record %d was no longer pending (a human got there first); skipped", record_id)
        except SubmitTimedOut as exc:
            self._defer(record_id, str(exc))
            return
        except SubmitRejected as exc:
            raise SystemicFault(f"The database rejected a write, a bug in the agent: {exc}") from exc
        else:
            logger.info(
                "Record %d -> %s (%s, %.2f)", record_id, verdict.status.value, verdict.tag, verdict.confidence
            )
        self._deferred.pop(record_id, None)


def run(
    agent: ClassificationAgent,
    poll_seconds: float,
    stopping: Callable[[], bool],
    sleep: Callable[[float], object],
    max_failed_batches: int = 5,
) -> None:
    """Loop until `stopping()`. A batch that fails outside any record (the
    database going away) is retried after `poll_seconds`, and
    max_failed_batches in a row halt, naming the last error; SystemicFault
    propagates."""
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
                raise SystemicFault(
                    f"{failed} batches in a row failed outside any record, so the database (or the "
                    f"connection to it) is failing; last error: {exc!r}"
                ) from exc
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
    parser.add_argument(
        "--allow-self-floor-grounding",
        action="store_true",
        help="Let a snippet without a quiet pre-trigger reference (an always-on emitter) be "
        "auto-classified when its primary region is outside the outer 15%% of the band. Off "
        "by default: the self floor is blind to receiver roll-off, and a milder 8-12 dB "
        "roll-off can inflate an interior region's bandwidth unnoticed. Enable only once "
        "recorded bladeRF captures confirm a steep enough roll-off (agent/README.md).",
    )
    return parser.parse_args(argv)


def install_stop_signal() -> threading.Event:
    """An event SIGTERM sets. Waiting on it is how the agent sleeps, so a stop
    request ends a poll, backoff or budget pause at once."""
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    return stop


def make_client(api_key: str) -> MessagesClient:
    client = anthropic.Anthropic(
        api_key=api_key, base_url=_API_BASE_URL, max_retries=_SDK_MAX_RETRIES, timeout=_SDK_TIMEOUT_SECONDS
    )
    # The SDK's overloaded messages.create (streaming variants included) is
    # wider than the one call MessagesClient describes.
    return cast(MessagesClient, client)


def serve(agent: ClassificationAgent, poll_seconds: float, stop: threading.Event) -> None:
    """Probe the store, then loop until `stop` is set. A halt logs its cause
    and exits with EXIT_HALTED."""
    try:
        agent.check_snippet_root()
        run(agent, poll_seconds, stop.is_set, stop.wait)
    except SystemicFault as exc:
        logger.critical("Agent halted: %s", exc)
        raise SystemExit(EXIT_HALTED) from exc


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
    except (AgentAlreadyRunning, BoundaryViolation, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    settings = AgentSettings(
        snippet_root=Path(args.snippet_store_dir),
        model=args.model,
        max_tokens=args.max_tokens,
        daily_token_budget=args.daily_token_budget,
        allow_self_floor_grounding=args.allow_self_floor_grounding,
    )
    stop = install_stop_signal()
    agent = ClassificationAgent(
        gateway,
        make_client(api_key),
        settings,
        load_band_table(Path(args.band_table)).entries,
        sleep=stop.wait,
        start_after_id=args.start_after_id,
    )
    try:
        serve(agent, args.poll_seconds, stop)
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
