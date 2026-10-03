# agent/routing.py
"""Confidence-based routing (pure). auto_classified needs a tag of the form
<band-id>:<signal> (a non-empty signal), confidence >= 0.85, no
reduced-confidence reason (agent.analysis; e.g. edge_region_unreliable), AND
grounding of that very tag: its band id is a grounded band-table entry, and
its signal is one that entry is known for (every word of it appears in one
of the entry's typical_signals, case and punctuation aside). In v1 only the
band table grounds; a modulation classifier's label never does by itself.
Everything else is needs_review, which is terminal for the agent."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agent.llm import Classification
from schema.records import ClassificationStatus

AUTO_CLASSIFY_CONFIDENCE = 0.85
MAX_REASONING_CHARS = 4000  # classify_unknown rejects anything longer


@dataclass(frozen=True)
class Decision:
    status: ClassificationStatus
    tag: str | None
    confidence: float
    reasoning: str


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _typical(signal: str, typical_signals: Sequence[str]) -> bool:
    """Every word of `signal` appears in one typical-signal description."""
    words = _words(signal)
    return bool(words) and any(words <= _words(described) for described in typical_signals)


def route(
    classification: Classification,
    grounded_bands: Mapping[str, Sequence[str]],
    reduced_confidence: tuple[str, ...] = (),
) -> Decision:
    """`grounded_bands` maps each grounded band-table id to its typical_signals."""
    tag = classification.tag
    band, _, signal = (tag or "").partition(":")
    if tag is None:
        why = "the model proposed no tag"
    elif reduced_confidence:
        why = f"reduced confidence ({', '.join(reduced_confidence)})"
    elif classification.confidence < AUTO_CLASSIFY_CONFIDENCE:
        why = f"confidence {classification.confidence:.2f} is below {AUTO_CLASSIFY_CONFIDENCE:.2f}"
    elif not signal:
        why = f"the tag {tag!r} is not <band-id>:<signal>"
    elif band not in grounded_bands:
        why = (
            f"the tag's band prefix {band!r} is not a grounded band-table entry "
            f"({', '.join(sorted(grounded_bands)) or 'none'})"
        )
    elif not _typical(signal, grounded_bands[band]):
        why = (
            f"the tag's signal {signal!r} is not among {band}'s typical signals "
            f"({'; '.join(grounded_bands[band])})"
        )
    else:
        return Decision(
            ClassificationStatus.AUTO_CLASSIFIED,
            classification.tag,
            classification.confidence,
            classification.reasoning,
        )
    return Decision(
        ClassificationStatus.NEEDS_REVIEW,
        classification.tag,
        classification.confidence,
        f"{classification.reasoning}\n\nRouted to needs_review: {why}.",
    )


def needs_review(reason: str) -> Decision:
    """A judgment about the record that produced no usable classification:
    NULL tag, zero confidence, the reason as the reasoning."""
    return Decision(ClassificationStatus.NEEDS_REVIEW, None, 0.0, reason[:MAX_REASONING_CHARS])
