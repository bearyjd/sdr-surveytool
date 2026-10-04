# agent/routing.py
"""Confidence-based routing (pure). auto_classified needs a tag of the form
<band-id>:<signal> (a non-empty signal), confidence >= 0.85, no
reduced-confidence reason (agent.analysis; e.g. edge_region_unreliable), AND
grounding of that very tag: its band id is a grounded band-table entry, and
its signal is one that entry is known for: lower-cased, with -, _, . and
spaces stripped, it equals a word or a run of consecutive words of one of
the entry's typical_signals, treated the same way (wifi matches "Wi-Fi",
adsb "ADS-B", 80211n "802.11n"). In v1 only the band table grounds; a
modulation classifier's label never does by itself. Everything else is
needs_review, which is terminal for the agent."""

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


_CLAUSES = re.compile(r"[/(),;:]")  # a typical-signal description's parts
_JOINERS = re.compile(r"[-_.:\s]+")  # stripped on both sides before comparing
_MAX_PHRASE_WORDS = 4
# Words that name no signal on their own ("FHSS and FSK", "1 Mbit/s").
_FILLER = {"and", "or", "on", "of", "the", "in", "for", "with", "to", "s", "mhz", "khz", "band", "channels"}


def _normalized(text: str) -> str:
    return _JOINERS.sub("", text.lower())


def _forms(described: str) -> set[str]:
    """Each word of a typical-signal description, and each run of up to 4
    consecutive words within one of its clauses, normalized. A lone filler
    word or number is not a form."""
    forms = set()
    for clause in _CLAUSES.split(described):
        words = clause.split()
        for start in range(len(words)):
            for stop in range(start + 1, min(start + _MAX_PHRASE_WORDS, len(words)) + 1):
                form = _normalized("".join(words[start:stop]))
                if stop - start > 1 or (form not in _FILLER and not form.isdigit()):
                    forms.add(form)
    return forms


def _typical(signal: str, typical_signals: Sequence[str]) -> bool:
    """`signal`, normalized, is a form of one typical-signal description."""
    wanted = _normalized(signal)
    return bool(wanted) and any(wanted in _forms(described) for described in typical_signals)


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
