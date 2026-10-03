# dsp/features.py
"""Time-domain and fine-spectral features of one occupied region (numpy only).

The region (dsp.segmentation) is first channelized: shifted to baseband,
FFT-mask filtered to its band plus a margin, and decimated. Every feature is
then measured on that narrowband signal alone, so a second emitter elsewhere
in the capture cannot leak into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dsp.segmentation import (
    BATCH_SAMPLES,
    NFFT,
    Segmentation,
    SpectralRegion,
    active_frame_mask,
    frame_powers,
    occupied_span,
    runs,
    welch_psd,
)

CHANNEL_BLOCK = 1 << 16  # channelizer FFT block; 50% overlap, middle half kept
_OVERSAMPLE = 4.0  # decimated rate >= 4x the region width: |x|^2 cannot alias
_MARGIN = 1.5  # passband = region width x 1.5, at least a few coarse bins
_ROLLOFF = 0.25  # filter skirt, as a fraction of the half passband
_MIN_PASSBAND_BINS = 4
_FINE_NFFT = 1024
_ON_THRESHOLD = 4.0  # duty cycle counts frames 6 dB above the region's noise
_MIN_FRAME = 32  # samples per duty-cycle frame (power estimate within ~18%)
_FRAME_SECONDS = 100e-6
_PAPR_PERCENTILE = 99.9
_MIN_FLATNESS_BINS = 16  # a tone's 3-bin span would read as flat
_ENVELOPE_NFFT = 4096
_MIN_ENVELOPE_NFFT = 256  # shorter on-runs carry no usable line
_MIN_ENVELOPE_FRAMES = 8
_LINE_THRESHOLD = 10.0  # symbol-rate line must stand 10 dB above the median
_MIN_LINE_BIN = 8  # skip the envelope's DC skirt
_LINE_WINDOW = 32  # bins either side that define the local continuum
# Measured: below ~12 dB in-region SNR the line is lost and spurious ones win.
_MIN_LINE_SNR_DB = 13.0


@dataclass(frozen=True)
class RegionFeatures:
    center_offset_hz: float  # refined on the decimated signal, from the tuned center
    obw_hz: float  # 99%-power occupied bandwidth at fine resolution
    papr_db: float  # 99.9th-percentile |x|^2 over the mean, in on-frames
    duty_cycle: float  # fraction of frames 6 dB above the region's noise
    burst_count: int
    mean_burst_s: float | None
    spectral_flatness: float  # Wiener entropy of the occupied bins, 0..1
    symbol_rate_hz: float | None  # strongest |x|^2 line, or None
    snr_db: float
    analysis_rate_hz: float  # decimated sample rate the features were measured at
    bandwidth_reliable: bool  # the region's own judgment (dsp.segmentation)
    fine_resolution_hz: float  # one fine-PSD bin: obw_hz is a multiple of it


def channelize(
    iq: np.ndarray, sample_rate: float, center_offset_hz: float, passband_hz: float
) -> tuple[np.ndarray, float, float]:
    """Shift `center_offset_hz` to DC, keep `passband_hz` (raised-cosine
    edges), and decimate by FFT-bin selection, block by block with 50%
    overlap (the middle half of each block is kept, so circular-convolution
    wrap-around is discarded). Working memory is one block whatever the
    capture length. Each block is transformed in complex128 (1 MiB, the same
    as one complex64 batch elsewhere): measured, a float32 FFT of this length
    leaves structured round-off spurs at multiples of rate/16 that the
    symbol-rate detector reads as lines (14 of 100 clean tones). The output,
    and everything downstream, is complex64/float32. Returns (complex64 baseband aligned with `iq`, its sample
    rate, the filter's noise-equivalent bandwidth in Hz). Power is preserved:
    a signal inside the passband keeps its mean |x|^2."""
    block = CHANNEL_BLOCK
    kept = 1 << max(4, math.ceil(math.log2(block * min(1.0, _OVERSAMPLE * passband_hz / sample_rate))))
    out_rate = sample_rate * kept / block
    shift = round(center_offset_hz / sample_rate * block)
    bins = np.arange(-kept // 2, kept // 2)
    taper = _passband(bins * sample_rate / block, passband_hz).astype(np.float32)
    hop, quarter = block // 2, kept // 4
    pieces = []
    for index in range(math.ceil(len(iq) / hop)):
        # Block `index` is centred on input samples [index*hop, (index+1)*hop),
        # zero-padded where it runs off either end of the capture.
        start = index * hop - block // 4
        chunk = np.zeros(block, dtype=np.complex128)
        lo, hi = max(start, 0), min(start + block, len(iq))
        chunk[lo - start : hi - start] = iq[lo:hi]
        selected = np.fft.fft(chunk)[(bins + shift) % block] * taper
        # Bin selection restarts the mixer's phase at every block start:
        # rotate it back to one continuous oscillator (hop = block / 2).
        rotation = -1.0 if (shift * index) % 2 else 1.0
        y = np.fft.ifft(np.fft.ifftshift(selected)) * (kept / block) * rotation
        pieces.append(y[quarter : kept - quarter].astype(np.complex64, copy=False))
    baseband = np.concatenate(pieces)[: round(len(iq) * kept / block)]
    noise_bandwidth = float(np.sum(taper.astype(np.float64) ** 2)) * sample_rate / block
    return baseband, out_rate, noise_bandwidth


def _passband(freqs: np.ndarray, passband_hz: float) -> np.ndarray:
    """1 inside +-passband/2, raised-cosine roll-off to 0 over the next
    _ROLLOFF x passband/2. Narrow on purpose: a wide skirt would let a
    neighbouring emitter into the channel."""
    half = passband_hz / 2
    over = np.clip((np.abs(freqs) - half) / (half * _ROLLOFF), 0.0, 1.0)
    return 0.5 * (1 + np.cos(np.pi * over))


def channelize_region(
    iq: np.ndarray, segmentation: Segmentation, region: SpectralRegion
) -> tuple[np.ndarray, float, float, float]:
    """`region` channelized to baseband: (complex64 signal, its rate, the
    filter's noise bandwidth, the passband kept)."""
    coarse_bin = segmentation.sample_rate / NFFT
    passband = max(region.end_offset_hz - region.start_offset_hz, _MIN_PASSBAND_BINS * coarse_bin) * _MARGIN
    y, rate, noise_bandwidth = channelize(iq, segmentation.sample_rate, region.center_offset_hz, passband)
    return y, rate, noise_bandwidth, passband


def region_features(iq: np.ndarray, segmentation: Segmentation, region: SpectralRegion) -> RegionFeatures:
    """Features of `region`, measured on its channelized baseband signal."""
    coarse_bin = segmentation.sample_rate / NFFT
    y, rate, noise_bandwidth, passband = channelize_region(iq, segmentation, region)
    # Noise power the coarse floor puts through the channel filter.
    noise = region.noise_per_bin * noise_bandwidth / coarse_bin

    frame = max(_MIN_FRAME, round(rate * _FRAME_SECONDS))
    powers = frame_powers(y, frame)
    on = powers > noise * _ON_THRESHOLD
    bursts = [stop - start for start, stop in runs(on)]
    on_samples = np.repeat(on, frame)
    on_power = np.abs(y[: len(on_samples)][on_samples]) ** 2 if on.any() else np.abs(y) ** 2
    # On-runs as sample ranges, one frame trimmed off each end (edge transients).
    on_runs = [((start + 1) * frame, (stop - 1) * frame) for start, stop in runs(on) if stop - start > 2]

    fine_nfft = min(_FINE_NFFT, len(y))
    fine_frames = frame_powers(y, fine_nfft)
    psd = welch_psd(y, fine_nfft, active_frame_mask(fine_frames))
    freqs = (np.arange(fine_nfft) - fine_nfft // 2) * rate / fine_nfft
    # The coarse noise density, shaped by the channel filter, per fine bin.
    # (The fine PSD's own median would sit below the in-passband noise,
    # because the filter's roll-off attenuates the band edges.)
    fine_noise = (
        region.noise_per_bin * (rate / fine_nfft) / coarse_bin
        * _passband(freqs, passband) ** 2
    )
    excess = np.clip(psd - fine_noise, 0.0, None)
    low, high = occupied_span(excess) if excess.sum() > 0 else (0, fine_nfft - 1)
    middle, half = (low + high) // 2, max(high - low + 1, _MIN_FLATNESS_BINS) // 2
    occupied = psd[max(middle - half, 0) : middle + half + 1]

    return RegionFeatures(
        center_offset_hz=region.center_offset_hz
        + float(np.sum(freqs * excess) / excess.sum() if excess.sum() > 0 else 0.0),
        obw_hz=(high - low + 1) * rate / fine_nfft,
        papr_db=float(10 * np.log10(np.percentile(on_power, _PAPR_PERCENTILE) / np.mean(on_power))),
        duty_cycle=float(on.mean()),
        burst_count=len(bursts),
        mean_burst_s=float(np.mean(bursts) * frame / rate) if bursts else None,
        spectral_flatness=float(np.exp(np.mean(np.log(occupied))) / np.mean(occupied)),
        symbol_rate_hz=_symbol_rate(y, rate, on_runs) if region.snr_db >= _MIN_LINE_SNR_DB else None,
        snr_db=region.snr_db,
        analysis_rate_hz=rate,
        bandwidth_reliable=region.bandwidth_reliable,
        fine_resolution_hz=rate / fine_nfft,
    )


def _symbol_rate(y: np.ndarray, rate: float, on_runs: list[tuple[int, int]]) -> float | None:
    """Frequency of the most prominent line in the |y|^2 spectrum, or None.

    Envelope frames are taken only from inside on-runs (sample ranges), each
    with its own mean removed: splicing bursts together would put the splice
    pattern, not the modulation, into the spectrum. Prominence is measured
    against the local continuum (median of the _LINE_WINDOW bins either
    side), because any band-limited envelope has a smooth triangular
    continuum that a global median would mistake for a line. A constant
    envelope (tone, FM, rectangular PSK) legitimately has no line.
    """
    longest = max((stop - start for start, stop in on_runs), default=0)
    total = sum(stop - start for start, stop in on_runs)
    # Largest power of two that still fits a run and averages enough frames
    # to keep the continuum's own fluctuations far below the threshold.
    fit = min(longest, total // _MIN_ENVELOPE_FRAMES)
    if fit < _MIN_ENVELOPE_NFFT:
        return None
    nfft = min(_ENVELOPE_NFFT, 1 << int(math.log2(fit)))
    starts = [s for start, stop in on_runs for s in range(start, stop - nfft + 1, nfft)]
    window = np.hanning(nfft).astype(np.float32)
    step = max(1, BATCH_SAMPLES // nfft)
    spectrum = np.zeros(nfft // 2 + 1)
    for i in range(0, len(starts), step):
        frames = np.abs(np.stack([y[s : s + nfft] for s in starts[i : i + step]])) ** 2
        frames -= frames.mean(axis=1, keepdims=True)
        spectrum += np.sum(np.abs(np.fft.rfft(frames * window, axis=1)) ** 2, axis=0)
    spectrum /= len(starts)
    padded = np.pad(spectrum, _LINE_WINDOW, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * _LINE_WINDOW + 1)
    prominence = spectrum / np.maximum(np.median(windows, axis=1), 1e-300)
    # Skip the DC skirt, and the top edge, where the channel filter's
    # roll-off reflects into the envelope.
    prominence[:_MIN_LINE_BIN] = 0.0
    prominence[-_LINE_WINDOW:] = 0.0
    index = int(np.argmax(prominence))
    if prominence[index] < _LINE_THRESHOLD:
        return None
    a, b, c = np.log(spectrum[index - 1 : index + 2])  # parabolic peak interpolation
    return float((index + 0.5 * (a - c) / (a - 2 * b + c)) * rate / nfft)
