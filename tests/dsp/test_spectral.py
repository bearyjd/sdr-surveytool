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
    noise_floor_psd,
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


PRE = 10_000  # 0.1 s of pre-trigger samples at FS: the noise reference


def _colored_noise(rng: np.random.Generator, n: int, power: float, passband: float) -> np.ndarray:
    """Noise as an SDR front end delivers it: through an anti-alias filter,
    flat over `passband` of the band with a cosine roll-off to the edges."""
    freqs = np.fft.fftfreq(n, 1 / FS)
    edge = passband * FS / 2
    magnitude = np.abs(freqs)
    shape = np.where(magnitude <= edge, 1.0, np.cos(np.pi / 2 * (magnitude - edge) / (FS / 2 - edge)) ** 2)
    x = np.fft.ifft(np.fft.fft(_noise(rng, n, 1.0)) * np.sqrt(shape))
    return (x * math.sqrt(power / np.mean(np.abs(x) ** 2))).astype(np.complex64)


def _split_snippet(rng: np.random.Generator, burst: np.ndarray, passband: float | None = None) -> tuple:
    """(pre-trigger reference, post-trigger samples) of a 1 s snippet with
    `burst` right after the trigger, on white (passband=None) or colored
    noise -- the shape the assembler produces."""
    noise = (
        _noise(rng, 100_000, NOISE_POWER)
        if passband is None
        else _colored_noise(rng, 100_000, NOISE_POWER, passband)
    )
    post = noise[PRE:].copy()
    post[: len(burst)] += burst
    return noise[:PRE], post


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
def test_occupied_bandwidth_of_a_burst_on_white_noise(bandwidth_hz, offset_hz):
    rng = np.random.default_rng(7)
    burst = _band_limited(rng, 20_000, NOISE_POWER * 100, bandwidth_hz, offset_hz)
    reference, post = _split_snippet(rng, burst)
    estimate = occupied_bandwidth(post, FS, THRESHOLD_DBFS, reference_iq=reference)
    assert estimate.hz == pytest.approx(bandwidth_hz, rel=0.15)
    assert estimate.reliable


@pytest.mark.parametrize("passband", [0.8, 0.6])
@pytest.mark.parametrize("bandwidth_hz", [10_000, 50_000, 70_000])
def test_occupied_bandwidth_on_noise_with_a_filter_roll_off(passband, bandwidth_hz):
    """Real SDR noise isn't flat. Any single global floor misreads it: a
    10th-percentile floor lands in the roll-off, so the passband's noise
    reads as signal (a 10 kHz burst measured ~6x too wide, flagged
    reliable). The pre-trigger reference gives a per-bin floor that follows
    the roll-off. The burst is ~11 dB above the noise."""
    rng = np.random.default_rng(7)
    offset = 10_000 if bandwidth_hz < 30_000 else 0
    burst = _band_limited(rng, 20_000, NOISE_POWER * 10**1.1, bandwidth_hz, offset)
    reference, post = _split_snippet(rng, burst, passband=passband)
    estimate = occupied_bandwidth(post, FS, THRESHOLD_DBFS, reference_iq=reference)
    assert estimate.hz == pytest.approx(bandwidth_hz, rel=0.10)
    assert estimate.reliable


@pytest.mark.parametrize("occupancy, tolerance", [(0.5, 0.05), (0.7, 0.05), (0.85, 0.10)])
def test_occupied_bandwidth_of_a_shaped_wideband_signal(occupancy, tolerance):
    """Spectrum skirts below the median bin: a median floor would cut them
    (-21% at 70%, -31% at 85% occupancy, measured)."""
    rng = np.random.default_rng(7)
    burst, true_bw = _hann_shaped(rng, 20_000, NOISE_POWER * 100, occupancy)
    reference, post = _split_snippet(rng, burst)
    estimate = occupied_bandwidth(post, FS, THRESHOLD_DBFS, reference_iq=reference)
    assert estimate.hz == pytest.approx(true_bw, rel=tolerance)
    assert estimate.reliable


def test_an_emitter_already_on_below_the_threshold_is_not_part_of_the_burst():
    """The estimate describes the burst that triggered capture: a carrier
    already on during the pre-trigger reference (below the trigger
    threshold, but ~18 dB above the noise per bin) is subtracted out. A
    median floor counts it, reading 3x too wide."""
    rng = np.random.default_rng(7)
    carrier = _band_limited(rng, 100_000, NOISE_POWER * 10**0.5, 5_000, -30_000)
    reference, post = _split_snippet(rng, _band_limited(rng, 20_000, NOISE_POWER * 100, 20_000, 20_000))
    estimate = occupied_bandwidth(post + carrier[PRE:], FS, THRESHOLD_DBFS, reference_iq=reference + carrier[:PRE])
    assert estimate.hz == pytest.approx(20_000, rel=0.15)


@pytest.mark.parametrize("reference_frames", [None, 5])
def test_without_a_usable_reference_the_median_fallback_is_flagged(reference_frames):
    """No pre-trigger samples (a trigger right at stream start), or too few
    frames to trust per bin: fall back to a flat median floor, which is
    fine on white noise but blind to roll-off and spectral shape."""
    rng = np.random.default_rng(7)
    reference, post = _split_snippet(rng, _band_limited(rng, 20_000, NOISE_POWER * 100, 20_000, 10_000))
    short = None if reference_frames is None else reference[: reference_frames * 1024]
    estimate = occupied_bandwidth(post, FS, THRESHOLD_DBFS, reference_iq=short)
    assert estimate.hz == pytest.approx(20_000, rel=0.15)
    assert not estimate.reliable
    assert not occupied_bandwidth(post, FS, THRESHOLD_DBFS).reliable


def test_a_signal_filling_more_than_90_percent_of_the_band_is_flagged_unreliable():
    rng = np.random.default_rng(7)
    reference, post = _split_snippet(rng, _band_limited(rng, 20_000, NOISE_POWER * 100, 0.95 * FS, 0.0))
    assert not occupied_bandwidth(post, FS, THRESHOLD_DBFS, reference_iq=reference).reliable


def test_a_signal_wrapping_around_the_band_edge_is_flagged_unreliable():
    """Centred on +-fs/2, a narrow signal shows up at both ends of the
    shifted spectrum; its edge-to-edge span is not its bandwidth."""
    rng = np.random.default_rng(7)
    reference, post = _split_snippet(rng, _band_limited(rng, 20_000, NOISE_POWER * 100, 10_000, FS / 2))
    assert not occupied_bandwidth(post, FS, THRESHOLD_DBFS, reference_iq=reference).reliable


def test_occupied_bandwidth_hz_is_the_estimate_without_its_reliability():
    rng = np.random.default_rng(7)
    reference, post = _split_snippet(rng, _band_limited(rng, 20_000, NOISE_POWER * 100, 20_000, 10_000))
    full = occupied_bandwidth(post, FS, THRESHOLD_DBFS, reference_iq=reference)
    assert isinstance(full, OccupiedBandwidth)
    assert occupied_bandwidth_hz(post, FS, THRESHOLD_DBFS, reference_iq=reference) == full.hz
    assert occupied_bandwidth_hz(post, FS, THRESHOLD_DBFS) == occupied_bandwidth(post, FS, THRESHOLD_DBFS).hz


def test_occupied_bandwidth_of_a_tone_is_a_few_bins():
    rng = np.random.default_rng(7)
    tone = (0.1 * np.exp(2j * np.pi * 12_345 * np.arange(20_000) / FS)).astype(np.complex64)
    reference, post = _split_snippet(rng, tone)
    assert occupied_bandwidth(post, FS, THRESHOLD_DBFS, reference_iq=reference).hz <= 5 * FS / 1024


def test_occupied_bandwidth_of_silence_is_finite():
    estimate = occupied_bandwidth_hz(np.zeros(4096, dtype=np.complex64), FS, THRESHOLD_DBFS)
    assert math.isfinite(estimate) and estimate > 0


def test_noise_floor_psd_follows_a_filter_roll_off():
    reference = _colored_noise(np.random.default_rng(7), 50_000, NOISE_POWER, 0.6)
    floor = np.fft.fftshift(noise_floor_psd(reference, 1024))
    assert floor.shape == (1024,)
    in_passband = floor[512 - 200 : 512 + 200].mean()
    near_band_edge = floor[:20].mean()
    assert near_band_edge < in_passband / 10


def test_noise_floor_psd_is_scaled_like_the_measured_psd():
    """Mean over frames of |FFT(frame x Hann)|^2: white noise of power P
    gives P * sum(window^2) per bin. Part 4 relies on this scaling."""
    floor = noise_floor_psd(_noise(np.random.default_rng(7), 200_000, NOISE_POWER), 1024)
    expected = NOISE_POWER * np.sum(np.hanning(1024) ** 2)
    assert np.mean(floor) == pytest.approx(expected, rel=0.05)


def test_noise_floor_psd_shrugs_off_a_transient_in_part_of_the_reference():
    rng = np.random.default_rng(7)
    reference = _noise(rng, 50_000, NOISE_POWER)
    spiky = reference.copy()
    spiky[:4096] += _noise(rng, 4096, NOISE_POWER * 1000)  # 4 of 48 frames
    ratio = noise_floor_psd(spiky, 1024) / noise_floor_psd(reference, 1024)
    assert np.median(ratio) == pytest.approx(1.0, abs=0.15)


def test_occupied_bandwidth_working_memory_does_not_scale_with_snippet_length():
    """A 1 s snippet is 160 MB of complex64 at 20 MS/s (448 MB at 56 MS/s).
    The estimate used to peak at ~4x its input (float64 window promoting the
    frames to complex128); it works in fixed-size batches, reference too."""
    iq = np.full(8_000_000, 0.1 + 0.05j, dtype=np.complex64)  # 61 MiB, all frames in-burst
    reference = np.full(8_000_000, 0.01 + 0.0j, dtype=np.complex64)
    tracemalloc.start()
    try:
        occupied_bandwidth(iq, FS, THRESHOLD_DBFS, reference_iq=reference)
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
