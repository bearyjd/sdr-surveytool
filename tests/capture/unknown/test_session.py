"""_open_session against the real GNU Radio scheduler; only the SDR source
itself (_build_soapy_source) is replaced, by a finite vector source."""
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("gnuradio")

from gnuradio import blocks, gr  # noqa: E402

from capture.unknown import service  # noqa: E402
from capture.unknown.service import CaptureSettings  # noqa: E402

FS = 100_000.0


def _settings(tmp_path, **overrides) -> CaptureSettings:
    return CaptureSettings(
        **{
            "center_freq_hz": 915e6,
            "sample_rate": FS,
            "noise_floor_dbfs": -40.0,
            "staging_dir": tmp_path,
            "stall_seconds": 0.2,
            **overrides,
        }
    )


def _iq_with_burst_at(start: int, n: int = 200_000) -> np.ndarray:
    rng = np.random.default_rng(5)
    iq = np.sqrt(1e-4 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    iq[start : start + 20_000] += 0.1
    return iq.astype(np.complex64)


@pytest.fixture
def top_block_calls(monkeypatch):
    calls = []
    for name in ("start", "stop", "wait"):
        real = getattr(gr.top_block, name)

        def spy(self, *args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(gr.top_block, name, spy)
    return calls


def test_open_session_streams_snippets_then_stalls_and_tears_down(tmp_path, monkeypatch, top_block_calls):
    opened_at = []

    def fake_source(settings):
        opened_at.append(datetime.now(timezone.utc))
        return blocks.vector_source_c(_iq_with_burst_at(50_000), False)

    monkeypatch.setattr(service, "_build_soapy_source", fake_source)
    settings = _settings(tmp_path)

    with service._open_session(settings, {}, Counter()) as snippets:
        snippet = next(snippets)
        # The finite source runs dry, so samples stop arriving: the stall
        # watchdog must end the session rather than wait forever.
        with pytest.raises(RuntimeError, match="stalled"):
            next(snippets)

    assert 50_000 <= snippet.trigger.sample_index < 50_000 + settings.samples(settings.averaging_seconds)
    assert len(snippet.iq) == settings.samples(0.1) + settings.samples(0.9)
    # The anchor (time of sample 0) is read after the device opened.
    anchor = snippet.trigger.time - timedelta(seconds=snippet.trigger.sample_index / FS)
    assert anchor >= opened_at[0]
    assert top_block_calls == ["start", "stop", "wait"]


def test_open_session_tears_down_a_flowgraph_whose_start_fails(tmp_path, monkeypatch, top_block_calls):
    monkeypatch.setattr(
        service, "_build_soapy_source", lambda settings: blocks.vector_source_c(np.zeros(10, np.complex64), False)
    )

    def failing_start(self):
        top_block_calls.append("start")
        raise RuntimeError("start failed")

    monkeypatch.setattr(gr.top_block, "start", failing_start)
    with pytest.raises(RuntimeError, match="start failed"):
        with service._open_session(_settings(tmp_path), {}, Counter()):
            pass  # pragma: no cover
    assert top_block_calls == ["start", "stop", "wait"]
