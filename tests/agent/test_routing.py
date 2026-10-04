# tests/agent/test_routing.py
import re

import pytest

from agent.analysis import EDGE_REGION_UNRELIABLE
from agent.band_table import load_band_table
from agent.llm import RECORD_CLASSIFICATION_TOOL, Classification
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


SHIPPED = {entry.id: entry.typical_signals for entry in load_band_table().entries}


@pytest.mark.parametrize(
    "tag",
    [
        "fm_broadcast:wbfm", "fm_broadcast:wideband_fm", "ism_2400:wifi", "ism_2400:802.11n", "ism_2400:ble",
        "adsb_1090:adsb", "adsb_1090:ppm", "frs_gmrs_462:nfm", "ham_2m:nfm", "uhf_tv:8vsb", "uhf_tv:ofdm",
        "pcs_downlink:lte", "pcs_downlink:nr", "lte_b71_600_downlink:nr", "unii_5725_5850:802.11ac",
        "marine_vhf:ais", "airband_vhf:am", "ism_902_928:lorawan",
    ],
)
def test_natural_tags_ground_against_the_shipped_typical_signals(tag):
    """The reviewer's list: lower case, with -, _, . and spaces stripped on
    both sides, so wifi matches Wi-Fi, adsb ADS-B and 80211n 802.11n."""
    band = tag.split(":")[0]
    decision = route(_c(tag=tag, confidence=0.9), {band: SHIPPED[band]})
    assert decision.status is ClassificationStatus.AUTO_CLASSIFIED, decision.reasoning


@pytest.mark.parametrize("tag", ["ism_902_928:lte", "ism_902_928:and", "adsb_1090:1", "fm_broadcast:mhz"])
def test_a_signal_the_band_is_not_known_for_still_needs_review(tag):
    band = tag.split(":")[0]
    assert route(_c(tag=tag, confidence=0.99), {band: SHIPPED[band]}).status is ClassificationStatus.NEEDS_REVIEW


def test_the_tool_schemas_own_examples_auto_classify():
    """The model copies the schema's examples; they must be tags that pass."""
    description = RECORD_CLASSIFICATION_TOOL["input_schema"]["properties"]["tag"]["description"]
    examples = re.findall(r"'([a-z0-9_]+:[a-z0-9_.-]+)'", description)
    assert examples
    for tag in examples:
        band = tag.split(":")[0]
        assert route(_c(tag=tag, confidence=0.9), {band: SHIPPED[band]}).status is ClassificationStatus.AUTO_CLASSIFIED
