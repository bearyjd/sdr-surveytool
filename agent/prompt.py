# agent/prompt.py
"""What the LLM is told. Numeric features, region summaries, curated
band-table entries and the classifier output only: never SigMF free text
(author/description are an injection surface), never location, survey or
operator IDs."""

from __future__ import annotations

import json

from agent.analysis import SnippetAnalysis

SYSTEM_PROMPT = """\
You classify unknown RF emissions captured by a passive drive-survey receiver in the US.
You receive measured features of one capture and the curated band-table entries its
frequency falls in. Call record_classification exactly once.

- tag: the most likely identity, lower case, "<band-table id>:<signal>", e.g.
  "ism_902_928:lora" or "pcs_downlink:lte". Only a tag prefixed with the id of a grounded
  entry can be accepted without human review. Use null if you cannot propose one.
- confidence: the probability your tag is right. Above 0.85 only when the features agree
  with a grounded band-table entry. Spectral features alone rarely justify high confidence.
  bandwidth_reliable false means the occupied bandwidth could not be measured (the signal
  fills the capture or wraps its edge). reduced_confidence lists why the measurements are
  less trustworthy, and no tag is accepted without review while it is non-empty.
  no_quiet_noise_reference means the noise floor was estimated from the capture itself
  (noise_reference "self"), which is blind to the receiver's roll-off;
  edge_region_unreliable means that, in that case, the primary emitter reaches the outer
  band edges, where the receiver's noise roll-off can pose as a signal.
  Other emitters with present_before_trigger true were already on before the capture
  triggered; the primary emitter is the one that triggered it.
- reasoning: the evidence, and any alternative identities you considered.

Your output is a label stored for human review. It never triggers any other action."""

_SIGNIFICANT = 6


def _round(value: float | None) -> float | None:
    return None if value is None else float(f"{value:.{_SIGNIFICANT}g}")


def build_user_message(analysis: SnippetAnalysis) -> str:
    primary, center_hz = analysis.primary, analysis.signal_center_hz
    if primary is None or center_hz is None:
        raise ValueError("No primary region: there is nothing to classify")
    payload = {
        "capture": {
            "tuned_center_mhz": _round(analysis.tuned_center_hz / 1e6),
            "sample_rate_msps": _round(analysis.sample_rate / 1e6),
            "analysed_seconds": _round(analysis.analysed_seconds),
            "truncated": analysis.truncated,
            "noise_reference": analysis.noise_reference,
            "reduced_confidence": list(analysis.reduced_confidence),
        },
        "primary_emitter": {
            "center_mhz": _round(center_hz / 1e6),
            "occupied_bandwidth_khz": _round(primary.obw_hz / 1e3),
            "snr_db": _round(primary.snr_db),
            "duty_cycle": _round(primary.duty_cycle),
            "burst_count": primary.burst_count,
            "mean_burst_ms": _round(None if primary.mean_burst_s is None else primary.mean_burst_s * 1e3),
            "papr_db": _round(primary.papr_db),
            "spectral_flatness": _round(primary.spectral_flatness),
            "symbol_rate_khz": _round(None if primary.symbol_rate_hz is None else primary.symbol_rate_hz / 1e3),
            "analysis_rate_msps": _round(primary.analysis_rate_hz / 1e6),
            "bandwidth_reliable": primary.bandwidth_reliable,
        },
        "other_emitters": [
            {
                "center_mhz": _round(region.center_hz / 1e6),
                "occupied_bandwidth_khz": _round(region.obw_hz / 1e3),
                "resolution_khz": _round(analysis.coarse_resolution_hz / 1e3),
                "power_relative_to_primary_db": _round(region.power_relative_to_primary_db),
                "present_before_trigger": region.present_before_trigger,
            }
            for region in analysis.context
        ],
        "band_table_matches": [
            {
                "id": match.entry.id,
                "service": match.entry.service,
                "range_mhz": [_round(match.entry.start_hz / 1e6), _round(match.entry.end_hz / 1e6)],
                "expected_occupied_bandwidth_khz": [
                    _round(match.entry.expected_obw_hz[0] / 1e3),
                    _round(match.entry.expected_obw_hz[1] / 1e3),
                ],
                "typical_signals": list(match.entry.typical_signals),
                "citation": match.entry.citation,
                "grounded": match.grounded,
            }
            for match in analysis.band_matches
        ],
        "modulation_classifier": None
        if analysis.modulation is None
        else {"label": analysis.modulation.label, "confidence": _round(analysis.modulation.confidence)},
    }
    return (
        "Features of one unknown-signal capture. Frequencies are absolute; powers are "
        "uncalibrated (dBFS-relative).\n" + json.dumps(payload, indent=1)
    )
