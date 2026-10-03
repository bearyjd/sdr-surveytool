# dsp/spectral.py
"""Shared numpy-only DSP helpers: no I/O, no GNU Radio, no imports from
capture/. Used by capture/unknown for snippet metadata and meant for reuse
by the Part 4 agent's feature extraction, which may never import capture/.
"""

from __future__ import annotations

import math

import numpy as np

# All powers are dBFS: 10*log10 of linear |x|^2 power where a full-scale
# complex sample (|x| = 1.0, Soapy's CF32 scaling) is 0 dBFS. Uncalibrated:
# not dBm, since no RF power calibration exists for the bladeRF front end.
_SILENCE_FLOOR = 1e-20  # -200 dBFS; keeps log10 finite on all-zero input

_NFFT = 1024
_FRAMES_PER_BATCH = 128  # 128 x 1024 complex64 = 1 MiB per batch
_OCCUPIED_POWER_FRACTION = 0.99


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


def occupied_bandwidth_hz(iq: np.ndarray, sample_rate: float, threshold_dbfs: float) -> float:
    """99%-power occupied bandwidth of the burst inside a snippet.

    Welch-style PSD over only the FFT frames whose mean power reaches the
    trigger threshold (so quiet pre/post padding doesn't dilute the burst),
    minus a per-bin noise floor (the median bin, valid while the burst
    occupies under half the capture bandwidth). The bandwidth is the span
    between the 0.5% and 99.5% points of the remaining power's cumulative
    sum, i.e. 0.5% trimmed from each tail. Resolution is sample_rate / 1024.
    Frames are processed in fixed-size complex64/float32 batches, so working
    memory stays a few MiB whatever the snippet length.
    Verified on synthetic band-limited bursts: within +-5% at >= 15 dB SNR,
    up to +24% at the 10 dB trigger margin; bursts shorter than one
    1024-sample frame are overestimated several-fold.
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
    # float32: a float64 window would promote every frame to complex128.
    window = np.hanning(nfft).astype(np.float32)
    psd = np.zeros(nfft)
    for i in range(0, len(in_burst), _FRAMES_PER_BATCH):
        batch = frames[in_burst[i : i + _FRAMES_PER_BATCH]] * window
        psd += np.sum(np.abs(np.fft.fft(batch, axis=1)) ** 2, axis=0)
    psd /= len(in_burst)
    excess = np.clip(np.fft.fftshift(psd) - np.median(psd), 0.0, None)
    total = excess.sum()
    if total <= 0.0:
        return sample_rate / nfft
    cumulative = np.cumsum(excess) / total
    tail = (1.0 - _OCCUPIED_POWER_FRACTION) / 2.0
    low = int(np.searchsorted(cumulative, tail))
    high = int(np.searchsorted(cumulative, 1.0 - tail))
    return (high - low + 1) * sample_rate / nfft
