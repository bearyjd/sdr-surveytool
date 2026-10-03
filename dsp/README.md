# dsp

Shared, numpy-only signal-processing helpers. No I/O, no GNU Radio, no scipy, and no
imports from `capture/`.

- `spectral.py`: dBFS power statistics and occupied bandwidth (used by `capture/unknown`).
- `segmentation.py`: Welch PSD over active frames, split into occupied regions (Part 4).
  The primary, the burst that triggered capture, is measured against
  `spectral.noise_floor_psd` of the snippet's pre-trigger frames that are quiet below
  the recorded trigger threshold (`spectral.quiet_reference`). Emitters already on are
  found in that reference against a local median floor, as context. Without a quiet
  reference, a percentile self floor is used; `touches_edge_zone` marks regions in the
  outer 15% of the band, which that flat floor cannot tell from receiver roll-off. The
  band is circular (a signal straddling +-fs/2 is one region), like
  `features.channelize`.
- `features.py`: channelize one region and measure it: fine OBW and center, duty cycle,
  bursts, PAPR, spectral flatness, symbol rate (Part 4).
- `synthetic.py`: deterministic test signals with known answers (tone, band-limited noise,
  colored receiver noise, RRC BPSK, on-off gating).

Memory discipline: everything works in complex64/float32 batches of about 1 MiB, whatever
the capture length. The one exception is the channelizer's fixed 65536-sample block,
which is transformed in complex128 (also 1 MiB): a float32 FFT of that length leaves
round-off spurs that read as symbol-rate lines.
