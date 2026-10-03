# tests/dsp/test_spectral.py
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dsp.spectral import (
    dbfs,
    mean_burst_power_dbfs,
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


def test_occupied_bandwidth_of_a_tone_is_a_few_bins():
    rng = np.random.default_rng(7)
    tone = (0.1 * np.exp(2j * np.pi * 12_345 * np.arange(20_000) / FS)).astype(np.complex64)
    estimate = occupied_bandwidth_hz(_snippet_with(tone, rng), FS, THRESHOLD_DBFS)
    assert estimate <= 5 * FS / 1024


def test_occupied_bandwidth_of_silence_is_finite():
    estimate = occupied_bandwidth_hz(np.zeros(4096, dtype=np.complex64), FS, THRESHOLD_DBFS)
    assert math.isfinite(estimate) and estimate > 0


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
