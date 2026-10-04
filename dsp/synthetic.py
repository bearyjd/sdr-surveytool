# dsp/synthetic.py
"""Deterministic synthetic IQ with known answers (numpy only).

Test support shared by tests/dsp and tests/agent: every generator returns
complex64 at a stated mean power (linear |x|^2, full scale = 1.0), so tests
can mix emitters at known relative levels.
"""

from __future__ import annotations

import math

import numpy as np


def _scaled(x: np.ndarray, power: float) -> np.ndarray:
    return (x * math.sqrt(power / np.mean(np.abs(x) ** 2))).astype(np.complex64)


def noise(rng: np.random.Generator, n: int, power: float) -> np.ndarray:
    """Complex white Gaussian noise."""
    scale = math.sqrt(power / 2)
    return (scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


def colored_noise(
    rng: np.random.Generator,
    n: int,
    sample_rate: float,
    power: float,
    flat_fraction: float,
    edge_db: float,
) -> np.ndarray:
    """Noise as a real receiver delivers it: flat over the middle
    `flat_fraction` of the band, then a raised-cosine roll-off down to
    -edge_db at +-sample_rate/2 (the anti-alias filter's skirt)."""
    spectrum = np.fft.fft(noise(rng, n, 1.0))
    edge = np.abs(np.fft.fftfreq(n, 1 / sample_rate)) / (sample_rate / 2)
    roll = np.clip((edge - flat_fraction) / (1.0 - flat_fraction), 0.0, 1.0)
    gain_db = -edge_db * (1.0 - np.cos(np.pi * roll)) / 2.0
    return _scaled(np.fft.ifft(spectrum * 10 ** (gain_db / 20)), power)


def tone(n: int, sample_rate: float, offset_hz: float, power: float) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (math.sqrt(power) * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)


def band_limited(
    rng: np.random.Generator,
    n: int,
    sample_rate: float,
    bandwidth_hz: float,
    offset_hz: float,
    power: float,
) -> np.ndarray:
    """Brick-wall band-limited noise: an unknown modulated signal whose true
    occupied bandwidth is exactly `bandwidth_hz`. Frequency is circular, as
    in a real capture: a signal centred at +-sample_rate/2 straddles the edge."""
    spectrum = np.fft.fft(noise(rng, n, 1.0))
    freqs = np.fft.fftfreq(n, 1 / sample_rate)
    distance = (freqs - offset_hz + sample_rate / 2) % sample_rate - sample_rate / 2
    spectrum[np.abs(distance) > bandwidth_hz / 2] = 0
    return _scaled(np.fft.ifft(spectrum), power)


def rrc_taps(beta: float, samples_per_symbol: int, span_symbols: int = 12) -> np.ndarray:
    """Unit-energy root-raised-cosine pulse, `span_symbols` symbols long."""
    t = np.arange(-span_symbols * samples_per_symbol // 2, span_symbols * samples_per_symbol // 2 + 1)
    t = t / samples_per_symbol
    taps = np.empty(len(t))
    for i, ti in enumerate(t):
        if ti == 0.0:
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            taps[i] = (beta / math.sqrt(2)) * (
                (1 + 2 / np.pi) * math.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * math.cos(np.pi / (4 * beta))
            )
        else:
            taps[i] = (
                math.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * math.cos(np.pi * ti * (1 + beta))
            ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return taps / math.sqrt(np.sum(taps**2))


def rrc_bpsk(
    rng: np.random.Generator,
    n: int,
    sample_rate: float,
    symbol_rate: float,
    beta: float,
    offset_hz: float,
    power: float,
) -> np.ndarray:
    """Random BPSK through an RRC pulse. sample_rate / symbol_rate must be an
    integer >= 4. Unlike rectangular PSK, its envelope is not constant, so
    |x|^2 carries a spectral line at the symbol rate."""
    sps = round(sample_rate / symbol_rate)
    if sps < 4 or not math.isclose(sps * symbol_rate, sample_rate):
        raise ValueError("sample_rate / symbol_rate must be an integer >= 4")
    symbols = rng.choice([-1.0, 1.0], size=n // sps + 1)
    impulses = np.zeros(len(symbols) * sps)
    impulses[::sps] = symbols
    baseband = np.convolve(impulses, rrc_taps(beta, sps), mode="same")[:n]
    return _scaled(baseband * np.exp(2j * np.pi * offset_hz * np.arange(n) / sample_rate), power)


def gate(x: np.ndarray, sample_rate: float, bursts: list[tuple[float, float]]) -> np.ndarray:
    """Zero `x` outside the (start_s, duration_s) bursts: on-off keying."""
    gated = np.zeros_like(x)
    for start_s, duration_s in bursts:
        start = round(start_s * sample_rate)
        stop = start + round(duration_s * sample_rate)
        gated[start:stop] = x[start:stop]
    return gated
