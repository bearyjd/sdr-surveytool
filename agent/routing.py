# agent/routing.py
"""Confidence-based routing (pure). auto_classified needs a tag, confidence
>= 0.85, no reduced-confidence reason (agent.analysis; e.g.
edge_region_unreliable), AND grounding of that very tag: its band prefix
(the part before the first ':') is the id of a grounded band-table entry, or
its last part is the modulation classifier's label. Everything else is
needs_review, which is terminal for the agent."""

from __future__ import annotations

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


def route(
    classification: Classification,
    grounded_band_ids: frozenset[str],
    modulation_label: str | None,
    reduced_confidence: tuple[str, ...] = (),
) -> Decision:
    tag = classification.tag
    if tag is None:
        why = "the model proposed no tag"
    elif reduced_confidence:
        why = f"reduced confidence ({', '.join(reduced_confidence)})"
    elif classification.confidence < AUTO_CLASSIFY_CONFIDENCE:
        why = f"confidence {classification.confidence:.2f} is below {AUTO_CLASSIFY_CONFIDENCE:.2f}"
    elif not grounded_band_ids and modulation_label is None:
        why = "no grounded band-table match and no modulation prediction"
    elif tag.split(":")[0] not in grounded_band_ids and tag.split(":")[-1] != modulation_label:
        why = (
            f"the tag's band prefix {tag.split(':')[0]!r} is not a grounded band-table entry "
            f"({', '.join(sorted(grounded_band_ids)) or 'none'})"
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
