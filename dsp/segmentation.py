# dsp/segmentation.py
"""Find the occupied spectral regions of a capture (numpy only).

A capture can hold several emitters: step 4's trigger is level-based and its
band is up to ~56 MHz wide. Whole-capture statistics then describe nothing in
particular, so the agent first splits the spectrum into contiguous occupied
regions and analyses the strongest one (dsp.features).

The primary region is the burst that triggered capture. When the snippet
carries a usable reference -- the quiet frames (dsp.spectral.quiet_reference)
of its "pre_trigger" samples -- the burst-time spectrum is measured against
step 4's per-bin floor of them (noise_floor_psd), raised to the reference's
own spectrum where that is higher, so an emitter already on before the
trigger leaves no residual to pose as the burst. A frame is quiet below step
4's trigger threshold, as its SigMF records it; for a snippet without one,
3 dB below the snippet's active frames. The emitters already on are found in
the reference itself, against a roll-off-aware local floor (the median of a
sliding 65-bin window), and are context only.

Without at least 8 quiet frames (no annotation, too short, or a continuous
emitter retriggering after its cooldown, which fills its own pre-trigger),
or when nothing stands above the reference (it held the emitter itself),
a self floor is estimated from the burst-time PSD -- a bias-corrected
percentile, never the median, which sits inside any signal occupying half
the band or more and erases it. That floor is flat, so it is blind to
roll-off: a receiver's in-band noise reads as a false region reaching
towards the band edges, and touches_edge_zone() marks the regions that
cannot be told from one.

The band is treated as circular, as dsp.features.channelize treats it: a
signal straddling the +-fs/2 edge is one region, not two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from dsp.spectral import bandwidth_is_reliable, dbfs, noise_floor_psd, quiet_reference

NFFT = 1024
# Work in complex64/float32 batches of this many samples (1 MiB of complex64),
# like dsp.spectral: memory stays a few MiB whatever the capture length, and
# nothing is promoted to complex128.
BATCH_SAMPLES = 1 << 17
_ACTIVE_MARGIN = 2.0  # 3 dB above the quiet-frame level marks an active frame
_MIN_ACTIVE_FRAMES = 16  # fewer than this and the loudest frames are used instead
_REGION_THRESHOLD = 4.0  # 6 dB above the per-bin noise floor
_MERGE_GAP_BINS = 2  # gaps this narrow (the masked DC bin, ripple) are bridged
# Without a recorded trigger threshold, a pre-trigger frame is quiet 3 dB
# below the active frames' mean power (dsp.spectral.quiet_reference then
# needs at least 8 of them).
_QUIET_REFERENCE_RATIO = 2.0
# Self floor on a receiver's roll-off: the in-band noise reads as one false
# region whose outer edge, measured over 20 seeds with 60-90% flat passbands
# and a 15 dB cosine roll-off, reaches 0.373-0.475 fs from the center (0.393+
# at 20 dB). A self-floor region reaching the outer 15% of the band on either
# side (beyond 0.35 fs) is therefore indistinguishable from it; 10% would
# miss the 60% and 70% flat cases. Milder roll-offs (8-12 dB) can instead
# widen a central region without reaching the zone: no width catches those,
# and only a real receiver's measured roll-off can say how often they occur.
EDGE_ZONE_FRACTION = 0.15
# Self floor: the 10th-percentile bin sits inside a signal filling more than
# ~90% of the band; the 2nd-percentile one stays noise-only up to ~98%. When
# they disagree by more than 3 dB the band is crowded and the deep floor is
# used.
_SELF_FLOOR_PERCENTILE = 10.0
_DEEP_FLOOR_PERCENTILE = 2.0
_CROWDED_RATIO = 2.0
# Context floor for the reference: the median of a 65-bin circular window.
# It follows a filter roll-off without lagging a steep skirt (measured: the
# median read no false region in 320 colored-noise references, 60%/80% flat
# passbands rolling off 15 or 25 dB, 24 or 97 frames; a 20th percentile read
# 306 in 40 at 80%/25 dB), while an emitter narrower than ~30 bins stands above it.
_LOCAL_WINDOW_BINS = 65
_LOCAL_PERCENTILE = 50.0
_OCCUPIED_FRACTION = 0.99
MAX_CONTEXT_REGIONS = 3


@dataclass(frozen=True)
class SpectralRegion:
    """One contiguous occupied region. Frequencies are offsets from the tuned
    center; powers are linear (full-scale complex sample = 1.0). start and
    end run continuously, so a region wrapping the band edge has
    end_offset_hz > sample_rate / 2; its center is folded into
    [-sample_rate / 2, sample_rate / 2)."""

    start_offset_hz: float
    end_offset_hz: float
    center_offset_hz: float  # excess-power-weighted centroid
    obw_hz: float  # narrowest span holding 99% of the region's excess power
    excess_power: float  # integrated power above the noise floor
    snr_db: float  # excess power over the noise in the same bins
    # dsp.spectral.bandwidth_is_reliable: False above 90% of the band or when
    # the region touches both band edges.
    bandwidth_reliable: bool
    noise_per_bin: float  # mean floor over the region's bins, in PSD units


@dataclass(frozen=True)
class Segmentation:
    regions: tuple[SpectralRegion, ...]  # during the burst, strongest first: [0] is the primary
    before_trigger: tuple[SpectralRegion, ...]  # emitters already on in the reference: context only
    floor_source: str  # "pre_trigger" (step 4's reference) or "self" (blind to roll-off)
    sample_rate: float
    active_frames: int
    total_frames: int

    @property
    def primary(self) -> SpectralRegion | None:
        return self.regions[0] if self.regions else None


def frame_powers(iq: np.ndarray, frame_len: int) -> np.ndarray:
    """Mean |x|^2 (float32) of each whole frame; a trailing partial frame is dropped."""
    frames = iq[: (len(iq) // frame_len) * frame_len].reshape(-1, frame_len)
    step = max(1, BATCH_SAMPLES // frame_len)
    return np.concatenate(
        [np.mean(np.abs(frames[i : i + step]) ** 2, axis=1) for i in range(0, len(frames), step)]
        or [np.zeros(0, dtype=np.float32)]
    )


def active_frame_mask(powers: np.ndarray) -> np.ndarray:
    """Frames at least 3 dB above the quiet (10th-percentile) frame level.

    None qualify: the emitter is continuous, there are no quiet frames to
    exclude, so every frame is used. Only a few qualify (a burst shorter than
    _MIN_ACTIVE_FRAMES frames): the loudest _MIN_ACTIVE_FRAMES are used, so
    the PSD still averages enough frames to keep noise bins under the region
    threshold.
    """
    active = powers >= np.percentile(powers, 10) * _ACTIVE_MARGIN
    count = int(active.sum())
    if count == 0:
        return np.ones(len(powers), dtype=bool)
    if count < _MIN_ACTIVE_FRAMES:
        active = np.zeros(len(powers), dtype=bool)
        active[np.argsort(powers)[-_MIN_ACTIVE_FRAMES:]] = True
    return active


def psd_scale(nfft: int) -> float:
    """welch_psd's scale relative to dsp.spectral's |FFT(frame x Hann)|^2:
    1 / (sum(w^2) * nfft), so bins sum to the mean power."""
    window = np.hanning(nfft).astype(np.float32)
    return 1.0 / (float(np.sum(window.astype(np.float64) ** 2)) * nfft)


def welch_psd(iq: np.ndarray, nfft: int, frame_mask: np.ndarray) -> np.ndarray:
    """Hann-windowed, fftshifted mean periodogram over the selected frames,
    scaled so its bins sum to the mean power of those frames. Batched in
    complex64; only the nfft-bin accumulator is float64."""
    # float32: a float64 window would promote every frame to complex128.
    window = np.hanning(nfft).astype(np.float32)
    scale = psd_scale(nfft)
    frames = iq[: len(frame_mask) * nfft].reshape(-1, nfft)
    selected = np.flatnonzero(frame_mask)
    step = max(1, BATCH_SAMPLES // nfft)
    total = np.zeros(nfft)
    for i in range(0, len(selected), step):
        batch = frames[selected[i : i + step]] * window
        total += np.sum(np.abs(np.fft.fft(batch, axis=1)) ** 2, axis=0)
    return np.fft.fftshift(total * scale / max(len(selected), 1))


def occupied_span(excess: np.ndarray) -> tuple[int, int]:
    """Indices (low, high), inclusive, of the narrowest span holding 99% of
    `excess`, trimming equal tails."""
    cumulative = np.cumsum(excess) / excess.sum()
    tail = (1.0 - _OCCUPIED_FRACTION) / 2.0
    return int(np.searchsorted(cumulative, tail)), int(np.searchsorted(cumulative, 1.0 - tail))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs, stop exclusive, of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _merged_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in runs(mask):
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def circular_regions(above: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """Bin indices of each occupied region of a circular band, in order.

    Bin len-1 is adjacent to bin 0 (the +-fs/2 edge of an fftshifted PSD).
    The scan starts in the middle of the longest gap, so no region and no
    bridgeable gap is ever cut by the start of the array.
    """
    n = len(above)
    if not above.any():
        return []
    gaps = runs(~above)
    if not gaps:
        return [np.arange(n)]
    longest = max(gaps, key=lambda gap: gap[1] - gap[0])
    if gaps[0][0] == 0 and gaps[-1][1] == n and len(gaps) > 1:  # the gap wraps too
        wrapped = gaps[-1][1] - gaps[-1][0] + gaps[0][1]
        if wrapped > longest[1] - longest[0]:
            longest = (gaps[-1][0], gaps[-1][0] + wrapped)
    if longest[1] - longest[0] <= max_gap:  # every gap bridges: one region
        return [np.arange(n)]
    order = ((longest[0] + longest[1]) // 2 + np.arange(n)) % n
    return [order[start:stop] for start, stop in _merged_runs(above[order], max_gap)]


def self_floor(psd: np.ndarray, frames: int, percentile: float) -> float:
    """Mean noise power per bin of a Welch PSD averaged over `frames` frames,
    from its `percentile`-th bin: each noise bin is a mean of `frames`
    exponentials, so that percentile sits at a known fraction of the noise
    mean (Wilson-Hilferty), and dividing by it removes the low bias."""
    return float(np.percentile(psd, percentile)) / _quantile_fraction(frames, percentile)


def local_floor(psd: np.ndarray, frames: int) -> np.ndarray:
    """Per-bin roll-off-aware floor: the bias-corrected median of a circular
    65-bin window around each bin."""
    half = _LOCAL_WINDOW_BINS // 2
    padded = np.concatenate([psd[-half:], psd, psd[:half]])
    envelope = np.percentile(sliding_window_view(padded, _LOCAL_WINDOW_BINS), _LOCAL_PERCENTILE, axis=1)
    return envelope / _quantile_fraction(frames, _LOCAL_PERCENTILE)


def _quantile_fraction(frames: int, percentile: float) -> float:
    a = 1.0 / (9.0 * frames)
    return (1.0 - a + NormalDist().inv_cdf(percentile / 100.0) * math.sqrt(a)) ** 3


def _bridge_dc(psd: np.ndarray) -> np.ndarray:
    """The DC bin carries the receiver's LO leakage, not an emitter; with a
    Hann window it spreads into DC +-1. Bridge those three bins from DC +-2.
    Only where there is no reference to subtract the leakage instead: the
    self floor, and the context search."""
    dc = len(psd) // 2
    bridged = psd.copy()
    bridged[dc - 1 : dc + 2] = 0.5 * (psd[dc - 2] + psd[dc + 2])
    return bridged


def touches_edge_zone(region: SpectralRegion, sample_rate: float) -> bool:
    """True when the region reaches the outer EDGE_ZONE_FRACTION of the band
    on either side, where a self floor cannot tell it from roll-off."""
    limit = (0.5 - EDGE_ZONE_FRACTION) * sample_rate
    return region.start_offset_hz < -limit or region.end_offset_hz > limit


def _quiet_reference(
    reference_iq: np.ndarray | None, active_power: float, threshold_dbfs: float | None
) -> np.ndarray | None:
    if reference_iq is None:
        return None
    if threshold_dbfs is None:
        threshold_dbfs = dbfs(active_power / _QUIET_REFERENCE_RATIO)
    return quiet_reference(reference_iq, threshold_dbfs, NFFT)


def _self_floor(psd: np.ndarray, frames: int) -> np.ndarray:
    floor = self_floor(psd, frames, _SELF_FLOOR_PERCENTILE)
    deep = self_floor(psd, frames, _DEEP_FLOOR_PERCENTILE)
    return np.full(len(psd), deep if floor > _CROWDED_RATIO * deep else floor)


def _regions(psd: np.ndarray, floor: np.ndarray, sample_rate: float) -> tuple[SpectralRegion, ...]:
    """Occupied regions of `psd` above `floor`, strongest first."""
    floor = np.maximum(floor, 1e-30)  # all-zero input: no regions, no 0/0
    bin_hz = sample_rate / len(psd)
    dc = len(psd) // 2
    excess = np.clip(psd - floor, 0.0, None)
    regions = []
    for bins in circular_regions(psd > floor * _REGION_THRESHOLD, _MERGE_GAP_BINS):
        # Continuous frequencies: a region wrapping the edge runs past +fs/2.
        freqs = (int(bins[0]) + np.arange(len(bins)) - dc) * bin_hz
        region_excess = excess[bins]
        power = float(region_excess.sum())
        low, high = occupied_span(region_excess)
        obw = (high - low + 1) * bin_hz
        center = float(np.sum(freqs * region_excess) / power)
        wraps = bool(np.isin([0, len(psd) - 1], bins).all())
        regions.append(
            SpectralRegion(
                start_offset_hz=float(freqs[0] - bin_hz / 2),
                end_offset_hz=float(freqs[-1] + bin_hz / 2),
                center_offset_hz=(center + sample_rate / 2) % sample_rate - sample_rate / 2,
                obw_hz=obw,
                excess_power=power,
                snr_db=float(10 * np.log10(power / float(floor[bins].sum()))),
                bandwidth_reliable=bandwidth_is_reliable(obw, sample_rate, wraps),
                noise_per_bin=float(floor[bins].mean()),
            )
        )
    return tuple(sorted(regions, key=lambda region: region.excess_power, reverse=True))


def segment_spectrum(
    iq: np.ndarray,
    sample_rate: float,
    reference_iq: np.ndarray | None = None,
    threshold_dbfs: float | None = None,
) -> Segmentation:
    """Split the capture's burst-time spectrum into occupied regions, and
    find the emitters already on in `reference_iq` (the snippet's
    pre-trigger samples, if any, which lie below the trigger threshold
    `threshold_dbfs` when step 4 recorded it)."""
    if len(iq) < NFFT:
        raise ValueError(f"Need at least {NFFT} samples, got {len(iq)}")
    powers = frame_powers(iq, NFFT)
    active = active_frame_mask(powers)
    psd = welch_psd(iq, NFFT, active)
    quiet = _quiet_reference(reference_iq, float(np.mean(powers[active])), threshold_dbfs)
    regions: tuple[SpectralRegion, ...] = ()
    before: tuple[SpectralRegion, ...] = ()
    if quiet is not None:
        # Unmasked: the reference carries the LO leakage at DC, so the floor
        # subtracts it, and a carrier keyed up at DC during the burst stands.
        quiet_frames = len(quiet) // NFFT
        reference_psd = welch_psd(quiet, NFFT, np.ones(quiet_frames, dtype=bool))
        floor = np.maximum(np.fft.fftshift(noise_floor_psd(quiet, NFFT)) * psd_scale(NFFT), reference_psd)
        regions = _regions(psd, floor, sample_rate)
        # The LO leakage is not an emitter: masked for the context search.
        context_psd = _bridge_dc(reference_psd)
        before = _regions(context_psd, local_floor(context_psd, quiet_frames), sample_rate)
    # Nothing new above the reference: its quiet frames held the triggering
    # emitter itself (an always-on emitter hovering at the threshold dips
    # below it), so the reference floor subtracted it. Something triggered
    # capture, so measure against the self floor instead.
    reference_used = bool(regions)
    if not reference_used:
        masked = _bridge_dc(psd)
        regions = _regions(masked, _self_floor(masked, int(active.sum())), sample_rate)
        before = ()
    return Segmentation(
        regions=regions,
        before_trigger=before,
        floor_source="pre_trigger" if reference_used else "self",
        sample_rate=sample_rate,
        active_frames=int(active.sum()),
        total_frames=len(powers),
    )
