# tests/dsp/test_features.py
"""Features on synthetic IQ with known answers. Every tolerance was set
from a 20-seed sweep (worst case in the comment), with headroom."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dsp import synthetic
from dsp.features import channelize, region_features
from dsp.segmentation import BATCH_SAMPLES, segment_spectrum

FS = 1e6
N = 1 << 18  # 0.262144 s
DURATION = N / FS
NOISE = 1e-5  # -50 dBFS across the band


def _features(iq, fs=FS):
    segmentation = segment_spectrum(iq, fs)
    return region_features(iq, segmentation, segmentation.primary)


def _noise_for_snr(snr_db: float, signal_power: float, signal_bw_hz: float, fs: float = FS) -> float:
    """Total noise power giving `snr_db` inside `signal_bw_hz`."""
    return signal_power / 10 ** (snr_db / 10) * fs / signal_bw_hz


def test_tone():
    rng = np.random.default_rng(10)
    f = _features(synthetic.tone(N, FS, 123_456.0, 1e-3) + synthetic.noise(rng, N, NOISE))
    assert f.center_offset_hz == pytest.approx(123_456.0, abs=20)  # sweep: <= 3.2 Hz
    assert f.obw_hz < 400  # 3 fine bins of 61 Hz
    assert f.symbol_rate_hz is None  # constant envelope
    assert f.spectral_flatness < 0.01  # sweep: <= 0.0003
    assert f.papr_db < 1.0  # sweep: 0.21-0.25 dB
    assert (f.duty_cycle, f.burst_count) == (1.0, 1)


def test_band_limited_noise():
    rng = np.random.default_rng(11)
    f = _features(synthetic.band_limited(rng, N, FS, 50e3, -200e3, 1e-3) + synthetic.noise(rng, N, NOISE))
    assert f.obw_hz == pytest.approx(50e3, rel=0.03)  # sweep: -0.4%..+0.6%
    assert f.center_offset_hz == pytest.approx(-200e3, abs=1_000)  # sweep: <= 388 Hz
    assert f.symbol_rate_hz is None
    assert f.spectral_flatness > 0.95  # sweep: >= 0.987
    assert 7.5 < f.papr_db < 9.5  # Gaussian: sweep 8.2-8.5 dB at the 99.9th percentile


@pytest.mark.parametrize(
    "symbol_rate, offset",
    [(250e3, 100e3), (125e3, 150e3), (50e3, -150e3)],  # 4, 8 and 20 samples/symbol
)
def test_rrc_bpsk_symbol_rate(symbol_rate, offset):
    """RRC (beta 0.35) BPSK has a non-constant envelope, so |x|^2 carries a
    line at the symbol rate. 20 dB SNR inside the signal's bandwidth."""
    rng = np.random.default_rng(12)
    noise = _noise_for_snr(20.0, 1e-3, 1.35 * symbol_rate)
    f = _features(synthetic.rrc_bpsk(rng, N, FS, symbol_rate, 0.35, offset, 1e-3) + synthetic.noise(rng, N, noise))
    assert f.symbol_rate_hz == pytest.approx(symbol_rate, rel=1e-3)  # sweep: <= 4e-5
    assert f.obw_hz == pytest.approx(1.17 * symbol_rate, rel=0.06)  # 99% OBW: 1.16-1.22 x Rs
    assert 3.0 < f.papr_db < 5.0  # sweep: 3.9-4.2 dB


def test_symbol_rate_is_withheld_below_13_db_snr():
    """Measured: at ~10 dB in-band SNR the line is lost and spurious ones
    can win (3/20 seeds gave wrong rates), so it is not estimated at all."""
    rng = np.random.default_rng(13)
    noise = _noise_for_snr(10.0, 1e-3, 1.35 * 50e3)
    f = _features(synthetic.rrc_bpsk(rng, N, FS, 50e3, 0.35, -150e3, 1e-3) + synthetic.noise(rng, N, noise))
    assert f.snr_db < 13.0
    assert f.symbol_rate_hz is None


def test_ook_duty_cycle_and_bursts():
    """Ten 5 ms carrier bursts every 20 ms: duty 0.05 s / 0.262 s."""
    rng = np.random.default_rng(14)
    bursts = [(0.01 + 0.02 * k, 0.005) for k in range(10)]
    iq = synthetic.gate(synthetic.tone(N, FS, 100e3, 1e-3), FS, bursts) + synthetic.noise(rng, N, NOISE)
    f = _features(iq)
    assert f.duty_cycle == pytest.approx(0.05 / DURATION, abs=0.01)  # sweep: +0.0046
    assert f.burst_count == 10
    assert f.mean_burst_s == pytest.approx(0.005, rel=0.05)  # sweep: +2.4% (frame quantization)
    assert f.symbol_rate_hz is None  # a gated tone: no line inside the bursts


def test_gated_rrc_symbol_rate_is_measured_inside_bursts():
    rng = np.random.default_rng(15)
    bursts = [(0.01 + 0.02 * k, 0.005) for k in range(10)]
    iq = synthetic.gate(synthetic.rrc_bpsk(rng, N, FS, 50e3, 0.35, -100e3, 1e-3), FS, bursts)
    f = _features(iq + synthetic.noise(rng, N, NOISE))
    assert f.symbol_rate_hz == pytest.approx(50e3, rel=1e-3)
    assert f.burst_count == 10


def test_two_emitters_features_belong_to_the_primary_only():
    """The continuous carrier must not leak into the burst's duty cycle or
    bandwidth: the primary is channelized before measuring."""
    rng = np.random.default_rng(16)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 50e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    iq = synthetic.tone(N, FS, -250e3, 1e-4) + burst + synthetic.noise(rng, N, NOISE)
    f = _features(iq)
    assert f.center_offset_hz == pytest.approx(200e3, abs=1_000)  # sweep: <= 412 Hz
    assert f.obw_hz == pytest.approx(50e3, rel=0.03)
    assert f.duty_cycle == pytest.approx(0.1 / DURATION, abs=0.01)  # sweep: <= 0.0009
    assert (f.burst_count, f.mean_burst_s) == (1, pytest.approx(0.1, rel=0.02))


def test_narrowband_obw_is_remeasured_at_fine_resolution():
    """At 2 MS/s a coarse bin is 1953 Hz, so a 5 kHz signal spans ~3 bins
    (coarse OBW +95%). The decimated fine PSD recovers it."""
    fs, n = 2e6, 1 << 19
    rng = np.random.default_rng(17)
    iq = synthetic.band_limited(rng, n, fs, 5e3, 300e3, 1e-3) + synthetic.noise(rng, n, NOISE)
    segmentation = segment_spectrum(iq, fs)
    assert segmentation.primary.obw_hz > 1.5 * 5e3
    f = region_features(iq, segmentation, segmentation.primary)
    assert f.obw_hz == pytest.approx(5e3, rel=0.06)  # sweep: +0.1%..+2.5%
    assert f.center_offset_hz == pytest.approx(300e3, abs=300)  # sweep: <= 131 Hz


def test_channelize_preserves_power_alignment_and_length():
    rng = np.random.default_rng(18)
    iq = synthetic.band_limited(rng, N, FS, 20e3, 150e3, 1e-3)
    y, rate, noise_bandwidth = channelize(iq, FS, 150e3, 30e3)
    assert rate == FS * len(y) / N
    assert np.mean(np.abs(y) ** 2) == pytest.approx(1e-3, rel=0.02)
    assert 30e3 < noise_bandwidth < 30e3 * 1.25


def test_ffts_run_in_bounded_batches_without_promotion(monkeypatch):
    """Same memory discipline as dsp.spectral: every FFT input fits in one
    1 MiB batch whatever the capture length, and only the channelizer's
    fixed-size blocks are complex128 (float32 spurs there read as symbol-rate
    lines); everything else stays complex64/float32."""
    rng = np.random.default_rng(19)
    bursts = [(0.01, 0.2)]
    iq = synthetic.gate(synthetic.rrc_bpsk(rng, N, FS, 50e3, 0.35, -100e3, 1e-3), FS, bursts)
    iq = iq + synthetic.noise(rng, N, NOISE)
    seen = []
    for name in ("fft", "ifft", "rfft"):
        original = getattr(np.fft, name)

        def spy(a, *args, _original=original, **kwargs):
            array = np.asarray(a)
            seen.append((sys._getframe(1).f_code.co_name, array.dtype, array.nbytes))
            return _original(a, *args, **kwargs)

        monkeypatch.setattr(np.fft, name, spy)
    f = _features(iq)
    assert f.symbol_rate_hz is not None  # every FFT path ran
    assert {caller for caller, _, _ in seen} == {"welch_psd", "channelize", "_symbol_rate"}
    assert max(nbytes for _, _, nbytes in seen) <= BATCH_SAMPLES * 8
    assert {dtype for caller, dtype, _ in seen if caller != "channelize"} <= {
        np.dtype(np.complex64),
        np.dtype(np.float32),
    }


def test_feature_modules_import_nothing_from_capture_or_gnuradio():
    probe = (
        "import sys, dsp.features, dsp.segmentation, dsp.synthetic; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('capture', 'gnuradio', 'scipy')); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[2], check=True)
