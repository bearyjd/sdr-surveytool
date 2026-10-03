# dsp/spectral.py
"""Shared numpy-only DSP helpers: no I/O, no GNU Radio, no imports from
capture/. Used by capture/unknown for snippet metadata and meant for reuse
by the Part 4 agent's feature extraction, which may never import capture/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# All powers are dBFS: 10*log10 of linear |x|^2 power where a full-scale
# complex sample (|x| = 1.0, Soapy's CF32 scaling) is 0 dBFS. Uncalibrated:
# not dBm, since no RF power calibration exists for the bladeRF front end.
_SILENCE_FLOOR = 1e-20  # -200 dBFS; keeps log10 finite on all-zero input

_NFFT = 1024
_FRAMES_PER_BATCH = 128  # 128 x 1024 complex64 = 1 MiB per batch
_OCCUPIED_POWER_FRACTION = 0.99
_RELIABLE_FRACTION_OF_BAND = 0.9
# Noise reference (noise_floor_psd): below this many frames the per-bin floor
# is too noisy to trust and occupied_bandwidth falls back to a flat median.
_REFERENCE_MIN_FRAMES = 8
_REFERENCE_GROUPS = 16
_FLOOR_SMOOTHING_BINS = 9


def dbfs(linear_power: float) -> float:
    return 10.0 * math.log10(max(linear_power, _SILENCE_FLOOR))


def peak_power_dbfs(power: np.ndarray) -> float:
    """Peak of the moving-average power (not raw per-sample |x|^2)."""
    return dbfs(float(np.max(power)))


def mean_burst_power_dbfs(power: np.ndarray, threshold_dbfs: float) -> float:
    """Mean of the moving-average power over the samples at or above the
    trigger threshold, i.e. the burst itself, excluding the quiet pre/post
    padding. Falls back to the peak if nothing crosses (cannot happen for an
    assembler snippet, whose trigger sample always does)."""
    above = power[power >= 10.0 ** (threshold_dbfs / 10.0)]
    return dbfs(float(np.mean(above)) if above.size else float(np.max(power)))


@dataclass(frozen=True)
class OccupiedBandwidth:
    hz: float
    # False when the burst fills more than 90% of the capture band or wraps
    # around its edges: the edge-to-edge span is then not its bandwidth.
    reliable: bool


def occupied_bandwidth_hz(
    iq: np.ndarray,
    sample_rate: float,
    threshold_dbfs: float,
    *,
    reference_iq: np.ndarray | None = None,
) -> float:
    """99%-power occupied bandwidth of the burst inside a snippet; see
    occupied_bandwidth() for the method and for whether to trust it."""
    return occupied_bandwidth(iq, sample_rate, threshold_dbfs, reference_iq=reference_iq).hz


def occupied_bandwidth(
    iq: np.ndarray,
    sample_rate: float,
    threshold_dbfs: float,
    *,
    reference_iq: np.ndarray | None = None,
) -> OccupiedBandwidth:
    """99%-power occupied bandwidth of the burst in `iq`.

    Welch-style PSD over only the FFT frames whose mean power reaches the
    trigger threshold, minus a per-bin noise floor measured from
    `reference_iq`: signal-free samples from the same receiver, normally the
    snippet's pre-trigger portion (see noise_floor_psd). A per-bin floor
    follows the receiver's filter roll-off and coloured noise, and subtracts
    an emitter that was already on (below the trigger threshold) before the
    trigger, so the estimate describes the burst that triggered capture. The floor is raised by one
    standard deviation of the burst PSD's own noise, so estimation noise
    isn't read as signal in the tails. The bandwidth is the span between the
    0.5% and 99.5% points of the remaining power's cumulative sum.
    Resolution is sample_rate / 1024; memory stays a few MiB (float32 batches).

    Without a reference of at least 8 frames, a flat median floor is used
    and the estimate is marked unreliable: it is blind to roll-off and cuts
    the skirts of signals occupying half the band or more.

    Measured over 20 seeds: within -3%..+3% for 10-70 kHz brick-wall and
    70-85% Hann-shaped bursts at 11-20 dB SNR, on white noise and on noise
    through 60%/80%-passband cosine roll-off filters, with 9 or 48 reference
    frames; bursts only a few frames long are overestimated.
    """
    nfft = min(_NFFT, len(iq))
    frames = iq[: (len(iq) // nfft) * nfft].reshape(-1, nfft)
    frame_power = np.concatenate(
        [
            np.mean(np.abs(frames[i : i + _FRAMES_PER_BATCH]) ** 2, axis=1)
            for i in range(0, len(frames), _FRAMES_PER_BATCH)
        ]
    )
    in_burst = np.flatnonzero(frame_power >= 10.0 ** (threshold_dbfs / 10.0))
    if in_burst.size == 0:
        in_burst = np.array([np.argmax(frame_power)])
    psd = _mean_psd(frames, in_burst, _window(nfft))
    has_reference = False
    floor = np.full(nfft, np.median(psd))  # fallback: flat, blind to roll-off
    if reference_iq is not None and len(reference_iq) // nfft >= _REFERENCE_MIN_FRAMES:
        has_reference = True
        floor = noise_floor_psd(reference_iq, nfft) * (1.0 + 1.0 / math.sqrt(len(in_burst)))
    excess = np.clip(np.fft.fftshift(psd - floor), 0.0, None)
    total = excess.sum()
    if total <= 0.0:
        return OccupiedBandwidth(hz=sample_rate / nfft, reliable=False)
    cumulative = np.cumsum(excess) / total
    tail = (1.0 - _OCCUPIED_POWER_FRACTION) / 2.0
    low = int(np.searchsorted(cumulative, tail))
    high = int(np.searchsorted(cumulative, 1.0 - tail))
    hz = (high - low + 1) * sample_rate / nfft
    wraps_band_edges = low == 0 and high >= nfft - 1
    reliable = (
        has_reference
        and hz <= _RELIABLE_FRACTION_OF_BAND * sample_rate
        and not wraps_band_edges
    )
    return OccupiedBandwidth(hz=hz, reliable=reliable)


def noise_floor_psd(reference_iq: np.ndarray, nfft: int = _NFFT) -> np.ndarray:
    """Per-bin noise floor PSD from signal-free reference samples, e.g. a
    snippet's pre-trigger portion (its SigMF "pre_trigger" annotation).

    Returns `nfft` bins in natural np.fft order, scaled like the PSD
    occupied_bandwidth() measures: the mean over frames of
    |FFT(frame x Hann)|^2, so white noise of power P gives P * sum(w^2).
    Robust per bin: the frames are split into up to 16 contiguous groups and
    each bin takes the median of the group means (bias-corrected to the
    mean), so a transient in part of the reference doesn't lift the floor;
    then smoothed over 9 bins (circularly, as the band wraps at +-fs/2) to
    tame estimation noise without blurring a filter roll-off. Works in
    float32 batches, so memory doesn't grow with the reference length.
    """
    frames = reference_iq[: (len(reference_iq) // nfft) * nfft].reshape(-1, nfft)
    if len(frames) == 0:
        raise ValueError(f"reference_iq must hold at least one {nfft}-sample frame")
    window = _window(nfft)
    groups = np.array_split(np.arange(len(frames)), min(_REFERENCE_GROUPS, len(frames)))
    group_means = np.stack([_mean_psd(frames, group, window) for group in groups])
    frames_per_group = len(frames) / len(groups)
    # The median of means of m exponentials sits at (1 - 1/(9m))^3 of their
    # mean (Wilson-Hilferty); dividing by it makes the floor unbiased.
    floor = np.median(group_means, axis=0) / (1.0 - 1.0 / (9.0 * frames_per_group)) ** 3
    return _smooth_circular(floor, _FLOOR_SMOOTHING_BINS)


def _window(nfft: int) -> np.ndarray:
    # float32: a float64 window would promote every frame to complex128.
    return np.hanning(nfft).astype(np.float32)


def _mean_psd(frames: np.ndarray, indices: np.ndarray, window: np.ndarray) -> np.ndarray:
    psd = np.zeros(frames.shape[1])
    for i in range(0, len(indices), _FRAMES_PER_BATCH):
        batch = frames[indices[i : i + _FRAMES_PER_BATCH]] * window
        psd += np.sum(np.abs(np.fft.fft(batch, axis=1)) ** 2, axis=0)
    return psd / len(indices)


def _smooth_circular(values: np.ndarray, width: int) -> np.ndarray:
    if len(values) < width:
        return values
    half = width // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    return np.convolve(padded, np.ones(width) / width, mode="valid")
