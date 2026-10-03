# tests/agent/test_analysis_prompt.py
import json

import numpy as np
import pytest

from agent.analysis import EDGE_REGION_UNRELIABLE, NO_QUIET_NOISE_REFERENCE, analyse_snippet
from agent.band_table import load_band_table
from agent.classifier import ModulationPrediction, UnavailableClassifier
from agent.prompt import build_user_message
from agent.snippet_reader import Snippet
from dsp import synthetic

FS = 1e6
N = 1 << 18
TUNED = 915e6
BANDS = load_band_table().entries


PRE = round(0.05 * FS)  # the pre-trigger: everything before the burst


def _snippet(
    iq: np.ndarray, pre_trigger: int = PRE, non_finite: int = 0, threshold: float | None = None
) -> Snippet:
    return Snippet(
        iq=iq, sample_rate=FS, center_freq_hz=TUNED, truncated=False,
        pre_trigger_samples=pre_trigger, non_finite_samples=non_finite, trigger_threshold_dbfs=threshold,
    )


def _two_emitters() -> np.ndarray:
    """125 kHz burst at 915.2 MHz (primary) + carrier at 914.75 MHz (context)."""
    rng = np.random.default_rng(20)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 125e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    return synthetic.tone(N, FS, -250e3, 1e-4) + burst + synthetic.noise(rng, N, 1e-5)


class _AlwaysPredicts:
    def predict(self, iq, sample_rate):
        return ModulationPrediction(label="lora", confidence=0.7)


def test_analysis_places_the_primary_in_absolute_frequency_and_grounds_it():
    analysis = analyse_snippet(_snippet(_two_emitters()), BANDS, UnavailableClassifier())
    assert analysis.signal_center_hz == pytest.approx(915.2e6, abs=2e3)
    assert analysis.primary.obw_hz == pytest.approx(125e3, rel=0.05)
    (context,) = analysis.context
    assert context.center_hz == pytest.approx(914.75e6, abs=1e3)
    assert context.power_relative_to_primary_db == pytest.approx(-10.0, abs=0.5)
    assert context.present_before_trigger is True  # found in the pre-trigger reference
    assert (analysis.noise_reference, analysis.reduced_confidence) == ("pre_trigger", ())
    assert [(m.entry.id, m.grounded) for m in analysis.band_matches] == [("ism_902_928", True)]
    assert analysis.modulation is None and analysis.grounded_band_ids == {"ism_902_928"}


def test_without_a_quiet_reference_nothing_grounds_by_default():
    """The self floor is blind to receiver roll-off, so by default nothing
    measured against it grounds, however central."""
    analysis = analyse_snippet(_snippet(_two_emitters(), pre_trigger=0), BANDS, UnavailableClassifier())
    assert (analysis.noise_reference, analysis.reduced_confidence) == ("self", (NO_QUIET_NOISE_REFERENCE,))
    assert analysis.signal_center_hz == pytest.approx(915.2e6, abs=2e3)
    assert [(m.entry.id, m.grounded) for m in analysis.band_matches] == [("ism_902_928", False)]
    assert analysis.context[0].present_before_trigger is None  # no reference to tell


def test_opted_in_an_interior_self_floor_primary_grounds():
    analysis = analyse_snippet(
        _snippet(_two_emitters(), pre_trigger=0), BANDS, UnavailableClassifier(), allow_self_floor_grounding=True
    )
    assert (analysis.noise_reference, analysis.reduced_confidence) == ("self", ())
    assert analysis.grounded_band_ids == {"ism_902_928"}


def test_on_colored_noise_without_a_reference_the_edge_zone_grounds_nothing():
    """The self floor reads a 15 dB receiver roll-off as one region some
    600 kHz wide that swallows the 20 kHz burst. Even opted in, it reaches
    the band-edge zone, so it is flagged and grounds nothing; with the
    pre-trigger reference the same capture grounds the burst itself."""
    rng = np.random.default_rng(24)
    burst = synthetic.band_limited(rng, N, FS, 20e3, 100e3, 1e-4)
    burst[:PRE] = 0
    iq = synthetic.colored_noise(rng, N, FS, 1e-5, 0.6, 15.0) + burst
    blind = analyse_snippet(_snippet(iq, pre_trigger=0), BANDS, UnavailableClassifier(), allow_self_floor_grounding=True)
    assert blind.primary.obw_hz > 500e3
    assert (blind.noise_reference, blind.reduced_confidence) == ("self", (EDGE_REGION_UNRELIABLE,))
    assert [(m.entry.id, m.grounded) for m in blind.band_matches] == [("ism_902_928", False)]
    referenced = analyse_snippet(_snippet(iq), BANDS, UnavailableClassifier())
    assert referenced.primary.obw_hz == pytest.approx(20e3, rel=0.15)
    assert (referenced.reduced_confidence, referenced.grounded_band_ids) == ((), {"ism_902_928"})


def test_the_edge_zone_applies_only_without_a_reference():
    """A real 125 kHz burst at +420 kHz reaches the zone (beyond 0.35 fs).
    The self floor cannot vouch for it, even opted in; the pre-trigger
    reference can."""
    rng = np.random.default_rng(25)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 125e3, 420e3, 1e-3), FS, [(0.05, 0.1)])
    iq = burst + synthetic.noise(rng, N, 1e-5)
    blind = analyse_snippet(_snippet(iq, pre_trigger=0), BANDS, UnavailableClassifier(), allow_self_floor_grounding=True)
    assert blind.reduced_confidence == (EDGE_REGION_UNRELIABLE,)
    assert analyse_snippet(_snippet(iq), BANDS, UnavailableClassifier()).grounded_band_ids == {"ism_902_928"}


def test_the_recorded_trigger_threshold_reaches_segmentation():
    """An emitter already on below the trigger threshold plus a weaker
    burst: with step 4's recorded threshold the burst is the primary; with
    the 3 dB fallback the self floor makes the older emitter the primary."""
    pre = round(0.025 * FS)
    rng = np.random.default_rng(20)
    burst = synthetic.band_limited(rng, N, FS, 50e3, 200e3, 5e-5)
    burst[:pre] = 0
    iq = synthetic.noise(rng, N, 1e-5) + synthetic.band_limited(rng, N, FS, 20e3, -250e3, 1e-4) + burst
    recorded = analyse_snippet(_snippet(iq, pre_trigger=pre, threshold=-38.5), BANDS, UnavailableClassifier())
    assert recorded.noise_reference == "pre_trigger"
    assert recorded.signal_center_hz == pytest.approx(915.2e6, abs=2e3)
    fallback = analyse_snippet(_snippet(iq, pre_trigger=pre), BANDS, UnavailableClassifier())
    assert fallback.noise_reference == "self"
    assert fallback.signal_center_hz == pytest.approx(914.75e6, abs=2e3)


@pytest.mark.parametrize("in_band_snr_db, grounded", [(9.0, False), (12.0, True)])
def test_a_low_snr_primary_grounds_nothing(in_band_snr_db, grounded):
    """The reviewer's 1 MS/s probe: a 125 kHz burst at 915.2 MHz only 9 dB
    above the noise reads a plausible OBW and grounded in 902-928 MHz.
    Below 10 dB region SNR its measurements are too noisy to vouch for."""
    rng = np.random.default_rng(0)
    burst = synthetic.gate(
        synthetic.band_limited(rng, N, FS, 125e3, 200e3, 1.25e-6 * 10 ** (in_band_snr_db / 10)), FS, [(0.05, 0.1)]
    )
    analysis = analyse_snippet(_snippet(burst + synthetic.noise(rng, N, 1e-5)), BANDS, UnavailableClassifier())
    assert analysis.noise_reference == "pre_trigger"
    assert ("low_snr" in analysis.reduced_confidence) is not grounded
    assert analysis.grounded_band_ids == ({"ism_902_928"} if grounded else frozenset())


def test_non_finite_samples_reduce_confidence():
    analysis = analyse_snippet(_snippet(_two_emitters(), non_finite=3), BANDS, UnavailableClassifier())
    assert analysis.reduced_confidence == ("non_finite_samples",)
    assert analysis.grounded_band_ids == frozenset()


def test_an_always_on_emitter_is_still_the_primary():
    """An always-on emitter fills its own pre-trigger: no quiet reference,
    so the self floor finds it, never nothing."""
    rng = np.random.default_rng(23)
    iq = synthetic.band_limited(rng, N, FS, 0.9 * FS, 0.0, 1e-3) + synthetic.noise(rng, N, 1e-5)
    analysis = analyse_snippet(_snippet(iq), BANDS, UnavailableClassifier())
    assert analysis.primary.obw_hz == pytest.approx(0.9 * FS, rel=0.03)
    assert (analysis.noise_reference, analysis.reduced_confidence) == ("self", (NO_QUIET_NOISE_REFERENCE,))


def test_out_of_table_signal_is_ungrounded_unless_the_classifier_predicts():
    snippet = Snippet(iq=_two_emitters(), sample_rate=FS, center_freq_hz=433.92e6, truncated=False, pre_trigger_samples=PRE)
    unpredicted = analyse_snippet(snippet, BANDS, UnavailableClassifier())
    assert (unpredicted.grounded_band_ids, unpredicted.modulation_label) == (frozenset(), None)
    analysis = analyse_snippet(snippet, BANDS, _AlwaysPredicts())
    assert analysis.band_matches == () and analysis.modulation_label == "lora"


def test_an_unreliable_bandwidth_grounds_nothing():
    """A 125 kHz signal straddling the +-fs/2 edge of a 914.5 MHz tuning
    sits at 915.0 MHz inside 902-928 MHz, but touching both band edges makes
    its bandwidth unreliable, so the match is listed, not grounded."""
    rng = np.random.default_rng(22)
    burst = synthetic.band_limited(rng, N, FS, 125e3, FS / 2, 1e-3)
    burst[:PRE] = 0
    analysis = analyse_snippet(
        Snippet(iq=burst + synthetic.noise(rng, N, 1e-5), sample_rate=FS, center_freq_hz=914.5e6, truncated=False, pre_trigger_samples=PRE),
        BANDS,
        UnavailableClassifier(),
    )
    assert analysis.noise_reference == "pre_trigger"
    assert not analysis.primary.bandwidth_reliable
    assert analysis.reduced_confidence == ("bandwidth_unreliable",)
    assert [(m.entry.id, m.grounded) for m in analysis.band_matches] == [("ism_902_928", False)]
    assert analysis.grounded_band_ids == frozenset()


def test_empty_spectrum_has_no_primary():
    rng = np.random.default_rng(21)
    analysis = analyse_snippet(_snippet(synthetic.noise(rng, N, 1e-5)), BANDS, UnavailableClassifier())
    assert analysis.primary is None and analysis.band_matches == () and not analysis.grounded_band_ids
    with pytest.raises(ValueError):
        build_user_message(analysis)


def test_prompt_holds_only_numeric_features_and_curated_entries():
    message = build_user_message(analyse_snippet(_snippet(_two_emitters()), BANDS, UnavailableClassifier()))
    payload = json.loads(message[message.index("{") :])
    assert set(payload) == {"capture", "primary_emitter", "other_emitters", "band_table_matches", "modulation_classifier"}
    assert payload["primary_emitter"]["center_mhz"] == pytest.approx(915.2, abs=0.002)
    assert payload["primary_emitter"]["symbol_rate_khz"] is None
    assert payload["band_table_matches"][0]["id"] == "ism_902_928"
    assert payload["band_table_matches"][0]["grounded"] is True
    assert payload["other_emitters"][0]["resolution_khz"] == pytest.approx(FS / 1024 / 1e3, rel=1e-4)
    assert set(payload["capture"]) == {
        "tuned_center_mhz", "sample_rate_msps", "analysed_seconds", "truncated", "noise_reference",
        "reduced_confidence",
    }
    assert (payload["capture"]["noise_reference"], payload["capture"]["reduced_confidence"]) == ("pre_trigger", [])
    assert payload["other_emitters"][0]["present_before_trigger"] is True
    assert set(payload["primary_emitter"]) == {
        "center_mhz", "occupied_bandwidth_khz", "snr_db", "duty_cycle", "burst_count", "mean_burst_ms",
        "papr_db", "spectral_flatness", "symbol_rate_khz", "analysis_rate_msps", "bandwidth_reliable",
    }
    for forbidden in ("survey", "operator", "latitude", "longitude", "author", "description", "path", "sigmf"):
        assert forbidden not in message.lower()
