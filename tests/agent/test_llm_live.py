# tests/agent/test_llm_live.py
"""Optional live smoke test: the only check of the model id, the forced
tool_choice and thinking=disabled against the real API. Skipped unless
ANTHROPIC_API_KEY is set AND SURVEYTOOL_LIVE_LLM_TEST=1 (it costs tokens)."""

import os

import anthropic
import numpy as np
import pytest

from agent.analysis import analyse_snippet
from agent.band_table import load_band_table
from agent.classifier import UnavailableClassifier
from agent.llm import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, request_classification
from agent.prompt import SYSTEM_PROMPT, build_user_message
from agent.snippet_reader import Snippet
from dsp import synthetic

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("SURVEYTOOL_LIVE_LLM_TEST") == "1"),
    reason="live LLM test: set ANTHROPIC_API_KEY and SURVEYTOOL_LIVE_LLM_TEST=1",
)


def test_live_forced_tool_call_returns_a_valid_classification():
    rng = np.random.default_rng(1)
    n, fs = 1 << 18, 1e6
    iq = synthetic.noise(rng, n, 1e-5) + synthetic.gate(
        synthetic.band_limited(rng, n, fs, 125e3, 200e3, 1e-3), fs, [(0.05, 0.1)]
    )
    analysis = analyse_snippet(
        Snippet(iq=iq, sample_rate=fs, center_freq_hz=915e6, truncated=False, pre_trigger_samples=50_000),
        load_band_table().entries,
        UnavailableClassifier(),
    )
    client = anthropic.Anthropic(max_retries=2, timeout=60.0)
    result = request_classification(
        client, SYSTEM_PROMPT, build_user_message(analysis), DEFAULT_MODEL, DEFAULT_MAX_TOKENS
    )
    assert result.classification is not None, result.failure
    assert result.tokens_used > 0
