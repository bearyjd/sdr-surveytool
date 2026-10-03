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
- malformed (a view value fails db_gateway.parse_pending_row): needs_review
  at once, reason malformed_record; no LLM call; never counts toward a halt;
- no snippet (ingest rejected it or capture dropped it): needs_review at once,
  naming the quality flag; no LLM call; never counts toward a halt;
- snippet outside the store: held; marked needs_review once a later snippet
  is analysed (the store root is then known to be right);
- snippet unreadable, no occupied region, or feature extraction failed: left
  pending (step 4 triggered on energy, so finding none is our failure);
- no quiet pre-trigger reference (an always-on emitter fills its own): the
  self-floor analysis goes to the LLM, but routing caps it at needs_review
  unless allow_self_floor_grounding is set (see agent.analysis);
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
from typing import cast

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
        if record.malformed is not None:
            self._submit(record.id, needs_review(f"malformed_record: {record.malformed}"))
            return
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
        decision = route(
            result.classification,
            analysis.grounded_band_ids,
            analysis.modulation_label,
            analysis.reduced_confidence,
        )
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
            analysis = analyse_snippet(
                snippet, self._bands, self._classifier, self._settings.allow_self_floor_grounding
            )
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
        allow_self_floor_grounding=args.allow_self_floor_grounding,
    )
    client = anthropic.Anthropic(api_key=api_key, max_retries=_SDK_MAX_RETRIES, timeout=_SDK_TIMEOUT_SECONDS)
    agent = ClassificationAgent(
        gateway,
        # The SDK's overloaded messages.create (streaming variants included)
        # is wider than the one call MessagesClient describes.
        cast(MessagesClient, client),
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
