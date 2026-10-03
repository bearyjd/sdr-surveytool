# tests/agent/test_routing.py
import pytest

from agent.analysis import EDGE_REGION_UNRELIABLE
from agent.llm import Classification
from agent.routing import AUTO_CLASSIFY_CONFIDENCE, needs_review, route
from schema.records import ClassificationStatus

GROUNDED = frozenset({"ism_902_928"})


def _c(tag="ism_902_928:lora", confidence=0.9, reasoning="Because.") -> Classification:
    return Classification(tag=tag, confidence=confidence, reasoning=reasoning)


def test_threshold_is_085():
    assert AUTO_CLASSIFY_CONFIDENCE == 0.85


def test_confident_tag_under_a_grounded_band_is_auto_classified_verbatim():
    decision = route(_c(confidence=0.85), GROUNDED, None)
    assert decision.status is ClassificationStatus.AUTO_CLASSIFIED
    assert (decision.tag, decision.confidence, decision.reasoning) == ("ism_902_928:lora", 0.85, "Because.")


def test_a_modulation_prediction_grounds_a_tag_ending_in_its_label():
    decision = route(_c(tag="unknown_band:lora", confidence=0.9), frozenset(), "lora")
    assert decision.status is ClassificationStatus.AUTO_CLASSIFIED


@pytest.mark.parametrize(
    "classification, grounded, modulation, why",
    [
        (_c(confidence=0.8499), GROUNDED, None, "below 0.85"),
        (_c(confidence=1.0), frozenset(), None, "no grounded band-table match"),
        (_c(tag=None, confidence=1.0), GROUNDED, None, "proposed no tag"),
        # A confident tag naming a band the signal is not grounded in.
        (_c(tag="pcs_downlink:lte", confidence=0.99), GROUNDED, None, "'pcs_downlink' is not a grounded"),
        (_c(tag="lora", confidence=0.99), GROUNDED, None, "'lora' is not a grounded"),
        (_c(tag="unknown_band:fsk", confidence=0.99), frozenset(), "lora", "'unknown_band' is not a grounded"),
    ],
)
def test_everything_else_needs_review_keeping_the_models_answer(classification, grounded, modulation, why):
    decision = route(classification, grounded, modulation)
    assert decision.status is ClassificationStatus.NEEDS_REVIEW
    assert (decision.tag, decision.confidence) == (classification.tag, classification.confidence)
    assert decision.reasoning.startswith("Because.") and why in decision.reasoning


@pytest.mark.parametrize("reasons", [(EDGE_REGION_UNRELIABLE,), ("non_finite_samples", "bandwidth_unreliable")])
def test_any_reduced_confidence_reason_caps_at_needs_review(reasons):
    """A self-floor primary in the band-edge zone may be the receiver's
    roll-off, not a signal: neither a grounded band nor a matching
    modulation label can auto-classify it, nor any other reason's record."""
    for grounded, modulation in ((GROUNDED, None), (frozenset(), "lora")):
        decision = route(_c(confidence=0.99), grounded, modulation, reasons)
        assert decision.status is ClassificationStatus.NEEDS_REVIEW
        assert f"reduced confidence ({', '.join(reasons)})" in decision.reasoning


def test_routed_reasoning_fits_the_db_limit():
    decision = route(_c(confidence=0.1, reasoning="x" * 2000), frozenset(), None)
    assert len(decision.reasoning) <= 4000


def test_needs_review_has_null_tag_zero_confidence_and_bounded_reason():
    decision = needs_review("y" * 5000)
    assert decision.status is ClassificationStatus.NEEDS_REVIEW
    assert (decision.tag, decision.confidence) == (None, 0.0)
    assert len(decision.reasoning) == 4000
