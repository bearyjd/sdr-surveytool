# tests/capture/unknown/test_flowgraph.py
"""Real GNU Radio scheduler tests. GNU Radio is a system (dnf) package, not
pip-installable, so these skip cleanly where it is absent."""
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("gnuradio")

from gnuradio import blocks  # noqa: E402

from capture.unknown.energy_trigger import TriggerConfig  # noqa: E402
from capture.unknown.flowgraph import CaptureFlowgraph, build_flowgraph  # noqa: E402
from capture.unknown.sample_clock import SampleClock  # noqa: E402
from capture.unknown.snippet_assembler import SnippetAssembler  # noqa: E402

FS = 100_000.0
ANCHOR = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
AVERAGING = 100  # 1 ms
NOISE_POWER = 1e-4  # -40 dBFS
BURST_POWER = 1e-2  # -20 dBFS
CONFIG = TriggerConfig(threshold_dbfs=-30.0, cooldown=timedelta(seconds=3))
PRE, POST = 10_000, 90_000  # 1.0 s snippets


def _signal(n: int, burst_starts: list[int], burst_len: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(3)
    iq = np.sqrt(NOISE_POWER / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    for start in burst_starts:
        iq[start : start + burst_len] += np.sqrt(BURST_POWER) * np.exp(
            2j * np.pi * 0.1 * np.arange(burst_len)
        )
    return iq.astype(np.complex64)


def _assembler() -> SnippetAssembler:
    return SnippetAssembler(
        clock=SampleClock(anchor=ANCHOR, sample_rate=FS),
        center_freq_hz=915e6,
        config=CONFIG,
        pre_trigger_samples=PRE,
        post_trigger_samples=POST,
        last_trigger_at={},
    )


def _run_bounded(flowgraph: CaptureFlowgraph, timeout: float = 30.0) -> None:
    """start() + bounded wait(): a regression that hangs the scheduler fails
    this test instead of hanging the whole suite."""
    waiter = threading.Thread(target=flowgraph.top_block.wait, daemon=True)
    flowgraph.top_block.start()
    waiter.start()
    waiter.join(timeout)
    if waiter.is_alive():
        flowgraph.top_block.stop()
        waiter.join(5)
        pytest.fail(f"flowgraph did not finish within {timeout}s")


def test_flowgraph_captures_burst_aligned_with_source_samples():
    iq = _signal(200_000, [50_000])
    snippets = []
    flowgraph = build_flowgraph(blocks.vector_source_c(iq, False), AVERAGING, _assembler(), snippets.append)
    _run_bounded(flowgraph)

    assert len(snippets) == 1
    snippet = snippets[0]
    # The trailing moving average crosses -30 dBFS ~9% of a window into the burst.
    assert 50_000 <= snippet.trigger.sample_index < 50_000 + AVERAGING
    assert snippet.start_index == snippet.trigger.sample_index - PRE
    np.testing.assert_array_equal(snippet.iq, iq[snippet.start_index : snippet.start_index + PRE + POST])
    # power[i] is the mean |x|^2 of the AVERAGING samples ending at i.
    i = snippet.trigger.sample_index
    expected = np.mean(np.abs(iq[i - AVERAGING + 1 : i + 1]) ** 2)
    assert snippet.power[PRE] == pytest.approx(expected, rel=1e-4)
    assert flowgraph.tap.samples_seen == len(iq)
    assert flowgraph.tap.error is None


def test_flowgraph_suppresses_burst_inside_cooldown():
    # Bursts at 0.5 s, 2.0 s (inside the 3 s cooldown) and 4.0 s (after it).
    iq = _signal(550_000, [50_000, 200_000, 400_000])
    snippets = []
    flowgraph = build_flowgraph(blocks.vector_source_c(iq, False), AVERAGING, _assembler(), snippets.append)
    _run_bounded(flowgraph)

    starts = [s.trigger.sample_index for s in snippets]
    assert len(starts) == 2
    assert 50_000 <= starts[0] < 50_100
    assert 400_000 <= starts[1] < 400_100


def test_tap_failure_ends_the_flowgraph_instead_of_hanging_it():
    """An exception escaping a Python block's work() kills its scheduler
    thread and leaves wait() blocked forever (verified). The tap must turn
    it into WORK_DONE and expose it as .error."""

    class ExplodingAssembler:
        def process(self, iq, power, start_index):
            raise ValueError("assembler bug")

    flowgraph = build_flowgraph(
        blocks.vector_source_c(_signal(100_000, []), False), AVERAGING, ExplodingAssembler(), lambda s: None
    )
    _run_bounded(flowgraph, timeout=10.0)
    assert isinstance(flowgraph.tap.error, ValueError)
