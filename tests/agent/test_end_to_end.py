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
    # Trigger threshold -35 dBFS, which step 4 records in the SigMF: the
    # carrier plus noise (-39.6 dBFS) stays below it, the burst crosses it.
    settings = CaptureSettings(center_freq_hz=TUNED, sample_rate=FS, noise_floor_dbfs=-45.0, staging_dir=staging)
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
    assert payload["capture"]["noise_reference"] == "pre_trigger"  # quiet below the recorded threshold
    assert payload["primary_emitter"]["center_mhz"] == pytest.approx(915.2, abs=0.002)
    assert payload["primary_emitter"]["occupied_bandwidth_khz"] == pytest.approx(125, rel=0.05)
    assert payload["primary_emitter"]["duty_cycle"] == pytest.approx(0.3 / (N / FS), abs=0.01)
    assert [(m["id"], m["grounded"]) for m in payload["band_table_matches"]] == [("ism_902_928", True)]
    assert payload["other_emitters"][0]["center_mhz"] == pytest.approx(914.75, abs=0.002)
    assert payload["other_emitters"][0]["present_before_trigger"] is True
    assert payload["capture"]["reduced_confidence"] == []
    assert str(store_root) not in prompt and "survey" not in prompt
