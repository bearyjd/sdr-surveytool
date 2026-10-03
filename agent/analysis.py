# agent/analysis.py
"""Everything the agent measures about one snippet, before the LLM sees it.

Segmentation picks the primary region: the burst that triggered capture,
measured against the snippet's quiet pre-trigger reference when there is
one. Features are measured on that region alone. Up to three other
emitters go along as context: those already on before the trigger (found in
the reference) and any other burst-time regions. The primary's absolute
center and fine OBW are matched against the curated band table; the
modulation classifier gets the channelized primary region, and in v1 its
label is shown to the LLM but never grounds a tag. Anything that makes the
measurements less trustworthy is listed in reduced_confidence, keeps every
band match ungrounded, and caps routing at needs_review.

Without a quiet reference the self floor is blind to receiver roll-off, so
by default nothing measured against it grounds (no_quiet_noise_reference).
allow_self_floor_grounding (opt-in, off until real bladeRF captures confirm
a steep enough roll-off) lets an interior primary ground; one reaching the
band-edge zone still cannot (edge_region_unreliable). Even then a milder
8-12 dB roll-off can widen an interior primary unnoticed (measured: 20 of
259 wrong self-floor primaries escape the zone).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent.band_table import BandEntry, BandMatch, match_bands
from agent.classifier import ModulationClassifier, ModulationPrediction
from agent.snippet_reader import Snippet
from dsp.features import RegionFeatures, channelize_region, region_features
from dsp.segmentation import (
    MAX_CONTEXT_REGIONS,
    NFFT,
    Segmentation,
    SpectralRegion,
    segment_spectrum,
    touches_edge_zone,
)

NO_QUIET_NOISE_REFERENCE = "no_quiet_noise_reference"
EDGE_REGION_UNRELIABLE = "edge_region_unreliable"
LOW_SNR = "low_snr"
OBW_UNRESOLVED = "obw_unresolved"
# A bare carrier reads 3-5 fine bins (the Hann main lobe). A width within 4
# bins only bounds the true width from above, so it cannot meet a band's
# expected minimum: at 30-56 MS/s a CW read 2.8-7.3 kHz and grounded FRS.
_UNRESOLVED_FINE_BINS = 4
# Below this region SNR the OBW and center are too noisy to ground a band:
# a 125 kHz burst 9 dB above the noise read 9.0 dB and grounded in 902-928 MHz.
_MIN_GROUNDING_SNR_DB = 10.0


@dataclass(frozen=True)
class ContextRegion:
    center_hz: float  # absolute
    obw_hz: float  # coarse: resolution is sample_rate / NFFT
    power_relative_to_primary_db: float
    # True: already on in the pre-trigger reference; False: new during the
    # burst; None: no reference to tell.
    present_before_trigger: bool | None


@dataclass(frozen=True)
class SnippetAnalysis:
    tuned_center_hz: float
    sample_rate: float
    noise_reference: str  # the floor's source: "pre_trigger" or "self" (dsp.segmentation)
    reduced_confidence: tuple[str, ...]  # why the measurements are less trustworthy
    analysed_seconds: float
    truncated: bool
    coarse_resolution_hz: float
    primary: RegionFeatures | None  # None: no occupied region at all
    signal_center_hz: float | None  # absolute center of the primary region
    context: tuple[ContextRegion, ...]
    band_matches: tuple[BandMatch, ...]
    modulation: ModulationPrediction | None

    @property
    def grounded_band_ids(self) -> frozenset[str]:
        """Ids of the band-table entries the primary is grounded in. Routing
        accepts a tag only under one of these prefixes."""
        return frozenset(match.entry.id for match in self.band_matches if match.grounded)

    @property
    def modulation_label(self) -> str | None:
        return None if self.modulation is None else self.modulation.label


def analyse_snippet(
    snippet: Snippet,
    bands: tuple[BandEntry, ...],
    classifier: ModulationClassifier,
    allow_self_floor_grounding: bool = False,
) -> SnippetAnalysis:
    reference = snippet.iq[: snippet.pre_trigger_samples] if snippet.pre_trigger_samples else None
    segmentation = segment_spectrum(snippet.iq, snippet.sample_rate, reference, snippet.trigger_threshold_dbfs)
    primary_region = segmentation.primary
    primary = None if primary_region is None else region_features(snippet.iq, segmentation, primary_region)
    reasons = _reduced_confidence(
        segmentation, snippet.non_finite_samples, primary_region, primary, allow_self_floor_grounding
    )
    context = () if primary_region is None else _context(snippet, segmentation, primary_region)
    signal_center: float | None = None
    matches: tuple[BandMatch, ...] = ()
    if primary is not None:
        signal_center = snippet.center_freq_hz + primary.center_offset_hz
        # Any reduced-confidence reason keeps every match ungrounded: an
        # unreliable bandwidth makes the OBW plausibility check meaningless.
        matches = tuple(
            BandMatch(match.entry, match.grounded and not reasons)
            for match in match_bands(bands, signal_center, primary.obw_hz)
        )
    return SnippetAnalysis(
        tuned_center_hz=snippet.center_freq_hz,
        sample_rate=snippet.sample_rate,
        noise_reference=segmentation.floor_source,
        reduced_confidence=reasons,
        analysed_seconds=len(snippet.iq) / snippet.sample_rate,
        truncated=snippet.truncated,
        coarse_resolution_hz=snippet.sample_rate / NFFT,
        primary=primary,
        signal_center_hz=signal_center,
        context=context,
        band_matches=matches,
        modulation=None if primary_region is None else _predict(classifier, snippet, segmentation, primary_region),
    )


def _predict(
    classifier: ModulationClassifier, snippet: Snippet, segmentation: Segmentation, region: SpectralRegion
) -> ModulationPrediction | None:
    """The classifier sees the primary emitter alone, at baseband."""
    baseband, rate, _, _ = channelize_region(snippet.iq, segmentation, region)
    return classifier.predict(baseband, rate)


def _reduced_confidence(
    segmentation: Segmentation,
    non_finite: int,
    primary_region: SpectralRegion | None,
    primary: RegionFeatures | None,
    allow_self_floor_grounding: bool,
) -> tuple[str, ...]:
    reasons = []
    if segmentation.floor_source == "self":
        if not allow_self_floor_grounding:
            reasons.append(NO_QUIET_NOISE_REFERENCE)  # the self floor is blind to roll-off
        elif primary_region is not None and touches_edge_zone(primary_region, segmentation.sample_rate):
            reasons.append(EDGE_REGION_UNRELIABLE)  # may be the receiver's roll-off, not a signal
    if non_finite:
        reasons.append("non_finite_samples")
    if primary is not None and not primary.bandwidth_reliable:
        reasons.append("bandwidth_unreliable")
    if primary is not None and primary.snr_db < _MIN_GROUNDING_SNR_DB:
        reasons.append(LOW_SNR)
    if primary is not None and primary.obw_hz <= _UNRESOLVED_FINE_BINS * primary.fine_resolution_hz:
        reasons.append(OBW_UNRESOLVED)
    return tuple(reasons)


def _context(snippet: Snippet, segmentation: Segmentation, primary: SpectralRegion) -> tuple[ContextRegion, ...]:
    """The strongest other emitters: already on before the trigger, or new
    during the burst (unknown without a reference)."""
    during = None if segmentation.floor_source == "self" else False
    candidates = [(region, during) for region in segmentation.regions[1:]]
    candidates += [(region, True) for region in segmentation.before_trigger]
    candidates.sort(key=lambda candidate: candidate[0].excess_power, reverse=True)
    return tuple(
        ContextRegion(
            center_hz=snippet.center_freq_hz + region.center_offset_hz,
            obw_hz=region.obw_hz,
            power_relative_to_primary_db=10 * math.log10(region.excess_power / primary.excess_power),
            present_before_trigger=before,
        )
        for region, before in candidates[:MAX_CONTEXT_REGIONS]
    )
