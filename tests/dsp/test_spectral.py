# tests/dsp/test_spectral.py
import math
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from dsp.spectral import (
    OccupiedBandwidth,
    dbfs,
    mean_burst_power_dbfs,
    occupied_bandwidth,
    occupied_bandwidth_hz,
    peak_power_dbfs,
)

FS = 100_000.0
NOISE_POWER = 1e-4  # -40 dBFS
THRESHOLD_DBFS = -30.0


def _noise(rng: np.random.Generator, n: int, power: float) -> np.ndarray:
    scale = math.sqrt(power / 2)
    return (scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


def _band_limited(rng: np.random.Generator, n: int, power: float, bandwidth_hz: float, offset_hz: float) -> np.ndarray:
    """Brick-wall band-limited complex noise: a stand-in for an unknown
    modulated signal with a known true occupied bandwidth."""
    spectrum = np.fft.fft(_noise(rng, n, 1.0))
    freqs = np.fft.fftfreq(n, 1 / FS)
    spectrum[np.abs(freqs - offset_hz) > bandwidth_hz / 2] = 0
    x = np.fft.ifft(spectrum)
    return (x * math.sqrt(power / np.mean(np.abs(x) ** 2))).astype(np.complex64)


def _hann_shaped(rng: np.random.Generator, n: int, power: float, occupancy: float) -> tuple:
    """Noise with a raised-cosine (Hann-shaped) spectrum over `occupancy` of
    the band: unlike a brick wall, its skirts sit below its median bin, so a
    median noise floor eats into them. Returns the burst and its true 99%
    occupied bandwidth."""
    freqs = np.fft.fftfreq(n, 1 / FS)
    half = occupancy * FS / 2
    shape = np.where(np.abs(freqs) <= half, np.cos(np.pi * freqs / (2 * half)) ** 2, 0.0)
    x = np.fft.ifft(np.fft.fft(_noise(rng, n, 1.0)) * np.sqrt(shape))
    order = np.argsort(freqs)
    cumulative = np.cumsum(shape[order]) / shape.sum()
    true_bw = freqs[order][np.searchsorted(cumulative, 0.995)] - freqs[order][np.searchsorted(cumulative, 0.005)]
    return (x * math.sqrt(power / np.mean(np.abs(x) ** 2))).astype(np.complex64), true_bw


def _snippet_with(burst: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """1 s snippet: 0.1 s quiet, burst, quiet tail -- the shape the assembler
    produces (pre-trigger history, trigger, post-trigger)."""
    before = _noise(rng, 10_000, NOISE_POWER)
    during = burst + _noise(rng, len(burst), NOISE_POWER)
    after = _noise(rng, 100_000 - 10_000 - len(burst), NOISE_POWER)
    return np.concatenate([before, during, after])


def test_dbfs_full_scale_is_zero_and_floors_silence():
    assert dbfs(1.0) == 0.0
    assert dbfs(1e-4) == pytest.approx(-40.0)
    assert math.isfinite(dbfs(0.0))


def test_peak_power_is_max_of_moving_average_power():
    power = np.array([1e-4, 1e-2, 1e-3], dtype=np.float32)
    assert peak_power_dbfs(power) == pytest.approx(-20.0, abs=1e-4)


def test_mean_burst_power_averages_only_above_threshold_samples():
    power = np.array([1e-4] * 8 + [1e-2, 1e-2], dtype=np.float32)
    assert mean_burst_power_dbfs(power, THRESHOLD_DBFS) == pytest.approx(-20.0, abs=1e-4)


@pytest.mark.parametrize("bandwidth_hz, offset_hz", [(5_000, -20_000), (20_000, 10_000), (60_000, 0)])
def test_occupied_bandwidth_of_band_limited_burst(bandwidth_hz, offset_hz):
    rng = np.random.default_rng(7)
    # 20 dB above the noise floor, 0.2 s long inside the 1 s snippet.
    burst = _band_limited(rng, 20_000, NOISE_POWER * 100, bandwidth_hz, offset_hz)
    estimate = occupied_bandwidth_hz(_snippet_with(burst, rng), FS, THRESHOLD_DBFS)
    assert estimate == pytest.approx(bandwidth_hz, rel=0.15)


@pytest.mark.parametrize("occupancy, tolerance", [(0.5, 0.05), (0.7, 0.05), (0.85, 0.10)])
def test_occupied_bandwidth_of_a_shaped_wideband_signal(occupancy, tolerance):
    """The noise floor is the 10th-percentile PSD bin scaled to the noise
    mean, which stays a noise-only statistic up to 90% occupancy. A median
    floor sits inside such a signal and cuts its skirts: -21% at 70%,
    -31% at 85% (measured)."""
    rng = np.random.default_rng(7)
    burst, true_bw = _hann_shaped(rng, 20_000, NOISE_POWER * 100, occupancy)
    estimate = occupied_bandwidth(_snippet_with(burst, rng), FS, THRESHOLD_DBFS)
    assert estimate.hz == pytest.approx(true_bw, rel=tolerance)
    assert estimate.reliable


def test_a_signal_filling_more_than_90_percent_of_the_band_is_flagged_unreliable():
    rng = np.random.default_rng(7)
    burst = _band_limited(rng, 20_000, NOISE_POWER * 100, 0.95 * FS, 0.0)
    assert not occupied_bandwidth(_snippet_with(burst, rng), FS, THRESHOLD_DBFS).reliable


def test_a_signal_wrapping_around_the_band_edge_is_flagged_unreliable():
    """Centred on +-fs/2, a narrow signal shows up at both ends of the
    shifted spectrum; its edge-to-edge span is not its bandwidth."""
    rng = np.random.default_rng(7)
    burst = _band_limited(rng, 20_000, NOISE_POWER * 100, 10_000, FS / 2)
    assert not occupied_bandwidth(_snippet_with(burst, rng), FS, THRESHOLD_DBFS).reliable


def test_occupied_bandwidth_hz_is_the_estimate_without_its_reliability():
    rng = np.random.default_rng(7)
    iq = _snippet_with(_band_limited(rng, 20_000, NOISE_POWER * 100, 20_000, 10_000), rng)
    assert occupied_bandwidth_hz(iq, FS, THRESHOLD_DBFS) == occupied_bandwidth(iq, FS, THRESHOLD_DBFS).hz
    assert isinstance(occupied_bandwidth(iq, FS, THRESHOLD_DBFS), OccupiedBandwidth)


def test_occupied_bandwidth_of_a_tone_is_a_few_bins():
    rng = np.random.default_rng(7)
    tone = (0.1 * np.exp(2j * np.pi * 12_345 * np.arange(20_000) / FS)).astype(np.complex64)
    estimate = occupied_bandwidth_hz(_snippet_with(tone, rng), FS, THRESHOLD_DBFS)
    assert estimate <= 5 * FS / 1024


def test_occupied_bandwidth_of_silence_is_finite():
    estimate = occupied_bandwidth_hz(np.zeros(4096, dtype=np.complex64), FS, THRESHOLD_DBFS)
    assert math.isfinite(estimate) and estimate > 0


def test_occupied_bandwidth_working_memory_does_not_scale_with_snippet_length():
    """A 1 s snippet is 160 MB of complex64 at 20 MS/s (448 MB at 56 MS/s).
    The estimate used to peak at ~4x its input (float64 window promoting the
    frames to complex128); it must work in fixed-size batches instead."""
    iq = np.full(8_000_000, 0.1 + 0.05j, dtype=np.complex64)  # 61 MiB, all frames in-burst
    tracemalloc.start()
    try:
        occupied_bandwidth_hz(iq, FS, THRESHOLD_DBFS)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < iq.nbytes / 4


def test_dsp_imports_nothing_from_capture_or_gnuradio():
    """agent/ (Part 4) reuses dsp/ and may never import capture/, so dsp/
    must stay free of capture/ and GNU Radio. Fresh interpreter: this test
    session itself has capture/ imported already."""
    probe = (
        "import sys, dsp.spectral; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('capture', 'gnuradio')); "
        "assert not bad, bad"
    )
    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-c", probe], cwd=repo_root, check=True)
