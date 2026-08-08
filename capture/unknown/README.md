# capture/unknown

Wideband unknown-signal detection: a custom GNU Radio flowgraph (via `gr-soapy` on the
bladeRF xA9) doing threshold-based energy detection, 1-2 second triggered IQ snippet
capture with a cooldown, and metadata (center_freq, bandwidth_estimate, peak_power,
snr, iq_snippet_path) written into the unified record schema (`modality: "unknown"`).

Records emitted here are the input to the Part 4 agentic characterization pipeline in
`agent/`.

**Status**: not yet implemented.
