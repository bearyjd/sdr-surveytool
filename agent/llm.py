# agent/llm.py
"""One forced `record_classification` tool call per record, and nothing else.

The model gets exactly one tool, the forced output tool, so it can never
take an action. Thinking is explicitly disabled: it cannot be combined with
a forced tool_choice. No prompt caching: the constant system prompt is
below the minimum cacheable size, so caching would buy nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
from anthropic.types import Message
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 1024
# One forced record_classification call (reasoning at most 2000 characters)
# needs well under 1000 output tokens: below 256 a full answer cannot fit,
# and 8192, far under any current model's output limit, bounds the budget
# reservation each call makes.
MAX_TOKENS_RANGE = (256, 8192)
TOOL_NAME = "record_classification"
TAG_PATTERN = r"^[a-z0-9][a-z0-9_.:-]{0,63}$"  # also enforced by classify_unknown
MAX_REASONING_CHARS = 2000

RECORD_CLASSIFICATION_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Record your classification of the unknown RF emission described in the "
        "user message. This is your only output; call it exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tag": {
                "type": ["string", "null"],
                "pattern": TAG_PATTERN,
                "description": (
                    "Lower-case identity label, e.g. 'ism_902_928:lora' or 'fm_broadcast:wbfm': "
                    "a band table id, then a signal drawn from that entry's typical_signals. "
                    "null if you cannot propose any identity."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Probability that the tag is correct.",
            },
            "reasoning": {
                "type": "string",
                "maxLength": MAX_REASONING_CHARS,
                "description": "The evidence for the tag, and any alternative identities.",
            },
        },
        "required": ["tag", "confidence", "reasoning"],
        "additionalProperties": False,
    },
}


class Classification(BaseModel):
    """The validated tool input. Anything else the model sends is a failure."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tag: str | None = Field(pattern=TAG_PATTERN)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=MAX_REASONING_CHARS)


@dataclass(frozen=True)
class LlmResult:
    classification: Classification | None  # None: the response failed validation
    failure: str | None  # why it failed, for the needs_review reasoning
    tokens_used: int


class _Messages(Protocol):
    def create(self, **kwargs: Any) -> Message: ...


class MessagesClient(Protocol):
    """The slice of anthropic.Anthropic the agent uses (tests inject fakes)."""

    @property
    def messages(self) -> _Messages: ...


def build_request(system: str, user_message: str, model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_message}],
        "tools": [RECORD_CLASSIFICATION_TOOL],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "thinking": {"type": "disabled"},
    }


def parse_response(message: Message) -> LlmResult:
    tokens = message.usage.input_tokens + message.usage.output_tokens
    if message.stop_reason in ("max_tokens", "refusal"):
        return LlmResult(None, f"stop_reason was {message.stop_reason}", tokens)
    calls = [block for block in message.content if block.type == "tool_use" and block.name == TOOL_NAME]
    if len(calls) != 1:
        return LlmResult(None, f"expected one {TOOL_NAME} call, got {len(calls)}", tokens)
    try:
        classification = Classification.model_validate(calls[0].input)
    except ValidationError as exc:
        return LlmResult(None, f"tool input failed validation: {exc.errors(include_url=False, include_input=False)}", tokens)
    return LlmResult(classification, None, tokens)


def request_classification(
    client: MessagesClient, system: str, user_message: str, model: str, max_tokens: int
) -> LlmResult:
    """One API call. API exceptions propagate: see is_transient."""
    return parse_response(client.messages.create(**build_request(system, user_message, model, max_tokens)))


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: a network failure or timeout, or a status the SDK
    itself retries (408, 409, 429, >= 500). Matched on the status code
    because 529 raises OverloadedError, which is not an InternalServerError
    and is not exported by anthropic 0.107."""
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    return isinstance(exc, anthropic.APIStatusError) and (
        exc.status_code in (408, 409, 429) or exc.status_code >= 500
    )
