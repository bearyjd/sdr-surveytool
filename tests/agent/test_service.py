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
from agent.service import AgentSettings, ClassificationAgent, SystemicFault, _parse_args, main, run
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


def _snippet_record(store: Path, record_id: int, quiet: bool = False, always_on: float = 0.0) -> PendingRecord:
    """A real step-4 SigMF pair with its 50 ms pre_trigger annotation: a
    125 kHz burst at 915.2 MHz from the trigger on, noise only, or an
    always-on emitter filling `always_on` of the band (and its own pre-trigger)."""
    rng = np.random.default_rng(record_id)
    n = 1 << 18
    iq = synthetic.noise(rng, n, 1e-5)
    if always_on:
        iq = iq + synthetic.band_limited(rng, n, FS, always_on * FS, 0.0, 1e-3)
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


@pytest.mark.parametrize(
    "width, opt_in, status, why",
    [
        (0.1, None, REVIEW, "reduced confidence (no_quiet_noise_reference)"),  # the default
        (0.9, None, REVIEW, "reduced confidence (no_quiet_noise_reference)"),
        (0.1, True, AUTO, None),
        (0.9, True, REVIEW, "reduced confidence (edge_region_unreliable)"),
    ],
)
def test_an_always_on_emitter_is_classified_and_never_trips_a_halt(store, width, opt_in, status, why):
    """An always-on emitter (an LTE downlink) retriggers after every
    cooldown and fills its own pre-trigger, so there is no quiet reference.
    It is still found as the primary region every time -- never 'no
    region' -- and never halts the agent. By default the self floor grounds
    nothing, so it goes to review even with a confident, in-band answer.
    With --allow-self-floor-grounding a central 100 kHz one is
    auto-classified, while one filling 90% of the band reaches the edge
    zone and still goes to review."""
    gateway = FakeGateway([_snippet_record(store, i, always_on=width) for i in range(1, 8)])
    client = ScriptedClient([{**GOOD, "tag": "ism_902_928:lte", "confidence": 0.95}] * 7)
    settings = {} if opt_in is None else {"allow_self_floor_grounding": opt_in}
    assert _agent(gateway, client, store, **settings).run_batch() == 7
    assert len(client.requests) == 7
    assert [(s[0], s[1]) for s in gateway.submitted] == [(i, status) for i in range(1, 8)]
    if why is not None:
        assert all(why in s[4] for s in gateway.submitted)


def test_self_floor_grounding_is_off_unless_the_flag_is_given(tmp_path):
    assert AgentSettings(snippet_root=tmp_path).allow_self_floor_grounding is False
    args = ["--snippet-store-dir", str(tmp_path)]
    assert _parse_args(args).allow_self_floor_grounding is False
    assert _parse_args([*args, "--allow-self-floor-grounding"]).allow_self_floor_grounding is True


def test_an_always_on_emitter_at_the_trigger_threshold_always_closes(store):
    """The reviewer's probe: a 200 kHz always-on emitter 0.5-2 dB above the
    recorded trigger threshold at 20 MS/s. Its quiet frames hold the emitter
    itself, so the reference floor subtracts it; the self-floor fallback
    finds it again. Every record closes (20 seeds), and the agent never halts."""
    fs, n, pre, threshold = 20e6, 1 << 19, 1 << 17, -40.0
    records = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        band_power = 10 ** ((threshold + (0.5, 1.0, 2.0)[seed % 3]) / 10)
        iq = synthetic.noise(rng, n, 1e-5) + synthetic.band_limited(rng, n, fs, 200e3, 1.5e6, band_power - 1e-5)
        path = write_sigmf_snippet(iq, store, fs, TUNED, START, trigger_offset=pre, threshold_dbfs=threshold)
        records.append(PendingRecord(seed + 1, str(path), fs, TUNED, None, None))
    gateway = FakeGateway(records)
    client = ScriptedClient([{**GOOD, "confidence": 0.5}] * 20)
    assert _agent(gateway, client, store, batch_size=20).run_batch() == 20
    assert [s[0] for s in gateway.submitted] == list(range(1, 21))
    assert len(client.requests) == 20


def test_a_malformed_record_is_closed_without_an_llm_call_and_never_halts(store):
    """A record whose view values cannot be parsed is a judgment about that
    record: needs_review, NULL tag, no LLM call, and no halt however many."""
    bad = [PendingRecord(i, None, None, None, None, None, malformed="snippet_duration_ms '3e9' is outside [0, 2147483647]") for i in range(1, 8)]
    gateway = FakeGateway([*bad, _snippet_record(store, 8)])
    client = ScriptedClient([GOOD])
    assert _agent(gateway, client, store).run_batch() == 8
    assert [(s[0], s[1], s[2], s[3]) for s in gateway.submitted] == [(i, REVIEW, None, 0.0) for i in range(1, 8)] + [(8, AUTO, "ism_902_928:lora", 0.9)]
    assert all(s[4].startswith("malformed_record: snippet_duration_ms '3e9'") for s in gateway.submitted[:7])
    assert len(client.requests) == 1


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
