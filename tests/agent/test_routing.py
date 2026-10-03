# tests/agent/test_routing.py
import pytest

from agent.analysis import EDGE_REGION_UNRELIABLE
from agent.llm import Classification
from agent.routing import AUTO_CLASSIFY_CONFIDENCE, needs_review, route
from schema.records import ClassificationStatus

GROUNDED = {
    "ism_902_928": ("LoRa / LoRaWAN chirp spread spectrum", "FHSS and FSK telemetry, smart meters", "802.15.4g")
}


def _c(tag="ism_902_928:lora", confidence=0.9, reasoning="Because.") -> Classification:
    return Classification(tag=tag, confidence=confidence, reasoning=reasoning)


def test_threshold_is_085():
    assert AUTO_CLASSIFY_CONFIDENCE == 0.85


def test_confident_tag_under_a_grounded_band_is_auto_classified_verbatim():
    decision = route(_c(confidence=0.85), GROUNDED)
    assert decision.status is ClassificationStatus.AUTO_CLASSIFIED
    assert (decision.tag, decision.confidence, decision.reasoning) == ("ism_902_928:lora", 0.85, "Because.")


@pytest.mark.parametrize(
    "classification, grounded, why",
    [
        (_c(confidence=0.8499), GROUNDED, "below 0.85"),
        (_c(confidence=1.0), {}, "'ism_902_928' is not a grounded"),
        (_c(tag=None, confidence=1.0), GROUNDED, "proposed no tag"),
        # A confident tag naming a band the signal is not grounded in.
        (_c(tag="pcs_downlink:lte", confidence=0.99), GROUNDED, "'pcs_downlink' is not a grounded"),
        # Bare or empty-suffixed tags carry no signal identity.
        (_c(tag="ism_902_928", confidence=0.99), GROUNDED, "is not <band-id>:<signal>"),
        (_c(tag="ism_902_928:", confidence=0.99), GROUNDED, "is not <band-id>:<signal>"),
        (_c(tag="lora", confidence=0.99), GROUNDED, "is not <band-id>:<signal>"),
        # In v1 a modulation label never grounds a tag by itself.
        (_c(tag="unknown_band:lora", confidence=0.99), {}, "'unknown_band' is not a grounded"),
        # Codex H4: the signal must be one the grounded band is known for.
        (_c(tag="ism_902_928:zigbee", confidence=0.99), GROUNDED, "'zigbee' is not among ism_902_928's typical signals"),
        (_c(tag="ism_902_928:lte", confidence=0.99), GROUNDED, "'lte' is not among ism_902_928's typical signals"),
    ],
)
def test_everything_else_needs_review_keeping_the_models_answer(classification, grounded, why):
    decision = route(classification, grounded)
    assert decision.status is ClassificationStatus.NEEDS_REVIEW
    assert (decision.tag, decision.confidence) == (classification.tag, classification.confidence)
    assert decision.reasoning.startswith("Because.") and why in decision.reasoning


@pytest.mark.parametrize("reasons", [(EDGE_REGION_UNRELIABLE,), ("non_finite_samples", "bandwidth_unreliable")])
def test_any_reduced_confidence_reason_caps_at_needs_review(reasons):
    """A self-floor primary in the band-edge zone may be the receiver's
    roll-off, not a signal: even a grounded band cannot auto-classify it,
    nor any other reason's record."""
    decision = route(_c(confidence=0.99), GROUNDED, reasons)
    assert decision.status is ClassificationStatus.NEEDS_REVIEW
    assert f"reduced confidence ({', '.join(reasons)})" in decision.reasoning


@pytest.mark.parametrize("signal", ["lora", "lorawan", "LoRa", "fsk-telemetry", "smart_meters", "802.15.4g"])
def test_a_signal_named_in_the_grounded_bands_typical_signals_auto_classifies(signal):
    """Normalized: lower case, words of one typical-signal entry."""
    decision = route(_c(tag=f"ism_902_928:{signal.lower()}", confidence=0.9), GROUNDED)
    assert decision.status is ClassificationStatus.AUTO_CLASSIFIED


def test_routed_reasoning_fits_the_db_limit():
    decision = route(_c(confidence=0.1, reasoning="x" * 2000), {})
    assert len(decision.reasoning) <= 4000


def test_needs_review_has_null_tag_zero_confidence_and_bounded_reason():
    decision = needs_review("y" * 5000)
    assert decision.status is ClassificationStatus.NEEDS_REVIEW
    assert (decision.tag, decision.confidence) == (None, 0.0)
    assert len(decision.reasoning) == 4000
