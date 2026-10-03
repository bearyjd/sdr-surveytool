# tests/agent/test_llm.py
"""The request goes through the real anthropic SDK (0.107) over an httpx
MockTransport, so the wire format and the response parsing are the SDK's
own: no network, no API key."""

import json

import anthropic
import httpx
import pytest
from anthropic.types import Message

from agent.llm import (
    RECORD_CLASSIFICATION_TOOL,
    TOOL_NAME,
    Classification,
    is_transient,
    parse_response,
    request_classification,
)

MODEL = "claude-sonnet-5-5"


def _message(stop_reason: str, content: list[dict]) -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 800, "output_tokens": 90},
        "content": content,
    }


def _tool_use(tool_input: dict, name: str = TOOL_NAME) -> dict:
    return {"type": "tool_use", "id": "toolu_01", "name": name, "input": tool_input}


GOOD = {"tag": "ism_902_928:lora", "confidence": 0.9, "reasoning": "Chirp-like, 125 kHz."}


def _client(handler) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_request_is_one_forced_tool_call_with_thinking_off():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_message("tool_use", [_tool_use(GOOD)]))

    result = request_classification(_client(handler), "system text", "user text", MODEL, 1024)
    (body,) = sent
    assert body == {
        "model": MODEL,
        "max_tokens": 1024,
        "system": "system text",
        "messages": [{"role": "user", "content": "user text"}],
        "tools": [RECORD_CLASSIFICATION_TOOL],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "thinking": {"type": "disabled"},
    }
    assert result.classification == Classification(**GOOD)
    assert result.failure is None and result.tokens_used == 890


def test_tool_schema_matches_the_db_contract():
    schema = RECORD_CLASSIFICATION_TOOL["input_schema"]
    assert schema["required"] == ["tag", "confidence", "reasoning"]
    assert schema["properties"]["tag"]["pattern"] == "^[a-z0-9][a-z0-9_.:-]{0,63}$"
    assert schema["properties"]["reasoning"]["maxLength"] == 2000
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "message, failure",
    [
        (_message("max_tokens", [_tool_use({"tag": "x"})]), "stop_reason was max_tokens"),
        (_message("refusal", []), "stop_reason was refusal"),
        (_message("end_turn", [{"type": "text", "text": "It is LoRa."}]), "got 0"),
        (_message("tool_use", [_tool_use(GOOD, name="something_else")]), "got 0"),
        (_message("tool_use", [_tool_use(GOOD), _tool_use(GOOD)]), "got 2"),
        (_message("tool_use", [_tool_use({**GOOD, "tag": "Not A Tag"})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "tag": "abc\n"})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "confidence": 1.5})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "confidence": "0.9"})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "reasoning": "x" * 2001})]), "validation"),
        (_message("tool_use", [_tool_use({"tag": "x", "confidence": 0.5})]), "validation"),
        (_message("tool_use", [_tool_use({**GOOD, "extra": 1})]), "validation"),
    ],
)
def test_failure_shapes_are_validation_failures(message, failure):
    result = parse_response(Message.model_validate(message))
    assert result.classification is None
    assert failure in result.failure
    assert result.tokens_used == 890
    assert len(result.failure) < 1000  # never echoes the (untrusted, long) input


def test_null_tag_is_a_valid_classification():
    result = parse_response(Message.model_validate(_message("tool_use", [_tool_use({**GOOD, "tag": None})])))
    assert result.classification.tag is None


@pytest.mark.parametrize(
    "status, transient",
    [(408, True), (409, True), (429, True), (500, True), (503, True), (529, True),
     (400, False), (401, False), (403, False), (404, False), (413, False), (422, False)],
)
def test_transient_statuses_match_the_sdks_own_retry_rule(status, transient):
    def handler(request):
        return httpx.Response(status, json={"type": "error", "error": {"type": "x", "message": "m"}})

    with pytest.raises(anthropic.APIStatusError) as excinfo:
        request_classification(_client(handler), "s", "u", MODEL, 16)
    assert is_transient(excinfo.value) is transient


def test_connection_errors_and_timeouts_are_transient():
    def refuse(request):
        raise httpx.ConnectError("refused")

    def stall(request):
        raise httpx.ReadTimeout("slow")

    for handler in (refuse, stall):
        with pytest.raises(anthropic.APIConnectionError) as excinfo:
            request_classification(_client(handler), "s", "u", MODEL, 16)
        assert is_transient(excinfo.value)
    assert not is_transient(ValueError("a bug"))
