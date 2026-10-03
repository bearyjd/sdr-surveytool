# tests/dsp/test_segmentation.py
import math

import numpy as np
import pytest

from dsp import synthetic
from dsp.segmentation import NFFT, circular_regions, segment_spectrum, touches_edge_zone

FS = 1e6
N = 1 << 18  # 0.26 s
NOISE = 1e-5  # -50 dBFS across the whole band
BIN_HZ = FS / NFFT


def test_noise_alone_has_no_regions():
    rng = np.random.default_rng(1)
    segmentation = segment_spectrum(synthetic.noise(rng, N, NOISE), FS)
    assert segmentation.regions == ()
    assert segmentation.primary is None


def test_all_zero_input_has_no_regions():
    assert segment_spectrum(np.zeros(4 * NFFT, dtype=np.complex64), FS).regions == ()


def test_tone_is_one_narrow_region_at_its_offset():
    rng = np.random.default_rng(2)
    iq = synthetic.tone(N, FS, 123_456.0, 1e-3) + synthetic.noise(rng, N, NOISE)
    segmentation = segment_spectrum(iq, FS)
    (region,) = segmentation.regions
    assert region.center_offset_hz == pytest.approx(123_456.0, abs=BIN_HZ)
    assert region.obw_hz <= 5 * BIN_HZ
    # A continuous emitter has no quiet frames: every frame is averaged.
    assert segmentation.active_frames == segmentation.total_frames


def test_band_limited_region_width_and_center():
    rng = np.random.default_rng(3)
    iq = synthetic.band_limited(rng, N, FS, 50e3, -200e3, 1e-3) + synthetic.noise(rng, N, NOISE)
    (region,) = segment_spectrum(iq, FS).regions
    assert region.center_offset_hz == pytest.approx(-200e3, abs=BIN_HZ)
    assert region.obw_hz == pytest.approx(50e3, rel=0.05)
    assert region.start_offset_hz < -225e3 + BIN_HZ and region.end_offset_hz > -175e3 - BIN_HZ


def test_two_emitters_primary_is_the_stronger_and_the_other_is_context():
    """A continuous carrier at -40 dBFS and a 0.1 s band-limited burst at
    -30 dBFS: the burst holds more excess power, so it is primary."""
    rng = np.random.default_rng(4)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 50e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    iq = synthetic.tone(N, FS, -250e3, 1e-4) + burst + synthetic.noise(rng, N, NOISE)
    segmentation = segment_spectrum(iq, FS)
    primary, other = segmentation.regions
    assert primary.center_offset_hz == pytest.approx(200e3, abs=BIN_HZ)
    assert other.center_offset_hz == pytest.approx(-250e3, abs=BIN_HZ)
    assert 10 * math.log10(other.excess_power / primary.excess_power) == pytest.approx(-10.0, abs=0.5)
    # Only the burst's frames (and the loudest neighbours) are averaged.
    assert segmentation.active_frames < segmentation.total_frames / 2


def test_dc_bin_is_not_a_region():
    """The receiver's LO leakage sits in the DC bin; it is masked."""
    rng = np.random.default_rng(5)
    iq = synthetic.tone(N, FS, 0.0, 1e-3) + synthetic.noise(rng, N, NOISE)
    assert segment_spectrum(iq, FS).regions == ()


def test_wide_emitter_centred_on_dc_is_still_one_region():
    rng = np.random.default_rng(6)
    iq = synthetic.band_limited(rng, N, FS, 100e3, 0.0, 1e-3) + synthetic.noise(rng, N, NOISE)
    (region,) = segment_spectrum(iq, FS).regions
    assert region.center_offset_hz == pytest.approx(0.0, abs=2 * BIN_HZ)
    assert region.obw_hz == pytest.approx(100e3, rel=0.05)


@pytest.mark.parametrize("occupancy, in_edge_zone", [(0.6, False), (0.7, True), (0.85, True)])
def test_a_wideband_emitter_survives_the_self_floor(occupancy, in_edge_zone):
    """No reference: a median floor sits inside any signal filling half the
    band and erases it; the percentile self floor does not (measured over 10
    seeds: OBW -0.7%..-1.0% from 55% to 95%). Beyond 0.35 fs it reaches the
    edge zone, where a self floor cannot tell it from receiver roll-off."""
    rng = np.random.default_rng(7)
    iq = synthetic.band_limited(rng, N, FS, occupancy * FS, 20e3, 1e-3) + synthetic.noise(rng, N, NOISE)
    segmentation = segment_spectrum(iq, FS)
    (region,) = segmentation.regions
    assert region.obw_hz == pytest.approx(occupancy * FS, rel=0.03)
    assert segmentation.floor_source == "self" and region.bandwidth_reliable
    assert touches_edge_zone(region, FS) is in_edge_zone


@pytest.mark.parametrize("occupancy, reliable", [(0.9, True), (0.95, False)])
def test_a_crowded_band_is_still_one_region(occupancy, reliable):
    """At ~90% the 10th-percentile floor lands inside the signal; the 2nd
    percentile disagrees by > 3 dB and is used instead. The OBW reads just
    under 90% (reliable) at 90% and over it at 95% (step 4's judgment)."""
    rng = np.random.default_rng(8)
    iq = synthetic.band_limited(rng, N, FS, occupancy * FS, 20e3, 1e-3) + synthetic.noise(rng, N, NOISE)
    (region,) = segment_spectrum(iq, FS).regions
    assert region.obw_hz == pytest.approx(occupancy * FS, rel=0.03)
    assert region.bandwidth_reliable is reliable and touches_edge_zone(region, FS)


def test_two_wideband_emitters_and_a_weak_tone():
    rng = np.random.default_rng(9)
    iq = (
        synthetic.band_limited(rng, N, FS, 300e3, -250e3, 1e-3)
        + synthetic.band_limited(rng, N, FS, 250e3, 220e3, 5e-4)
        + synthetic.tone(N, FS, 30e3, 3e-5)
        + synthetic.noise(rng, N, NOISE)
    )
    wide, other, tone = segment_spectrum(iq, FS).regions
    assert (wide.center_offset_hz, wide.obw_hz) == (pytest.approx(-250e3, abs=BIN_HZ), pytest.approx(300e3, rel=0.03))
    assert (other.center_offset_hz, other.obw_hz) == (pytest.approx(220e3, abs=BIN_HZ), pytest.approx(250e3, rel=0.03))
    assert tone.center_offset_hz == pytest.approx(30e3, abs=BIN_HZ)


def test_a_signal_straddling_the_band_edge_is_one_unreliable_region():
    """+fs/2 and -fs/2 are the same frequency: the region wraps, as it does
    for dsp.features.channelize, and touching both edges makes its
    bandwidth unreliable (dsp.spectral.bandwidth_is_reliable)."""
    rng = np.random.default_rng(10)
    iq = _burst_after_reference(rng, synthetic.band_limited(rng, N, FS, 60e3, FS / 2, 1e-3))
    (region,) = segment_spectrum(iq, FS, iq[:PRE]).regions
    assert abs(region.center_offset_hz) == pytest.approx(FS / 2, abs=BIN_HZ)
    assert region.obw_hz == pytest.approx(60e3, rel=0.05)
    assert region.end_offset_hz > FS / 2 > region.start_offset_hz
    assert not region.bandwidth_reliable


def test_a_signal_near_but_inside_the_edge_stays_reliable():
    rng = np.random.default_rng(11)
    iq = _burst_after_reference(rng, synthetic.band_limited(rng, N, FS, 60e3, 450e3, 1e-3))
    (region,) = segment_spectrum(iq, FS, iq[:PRE]).regions
    assert region.center_offset_hz == pytest.approx(450e3, abs=2 * BIN_HZ) and region.bandwidth_reliable


@pytest.mark.parametrize(
    "above, expected",
    [
        ("..##...", ["2,3"]),
        ("#....##", ["5,6,0"]),  # wraps: the last bins and bin 0 are one region
        ("#.#....", ["0,1,2"]),  # a 1-bin gap is bridged
        ("##.....#.", ["7,8,0,1"]),  # bridged across the wrap
        ("#######", ["0,1,2,3,4,5,6"]),
        (".......", []),
        ("##..##.....", ["0,1", "4,5"]),
    ],
)
def test_circular_regions(above, expected):
    mask = np.array([c == "#" for c in above])
    found = [",".join(str(int(i)) for i in region) for region in circular_regions(mask, max_gap=1)]
    assert found == expected


PRE = 25_000  # a 25 ms pre-trigger reference: 24 frames


def _burst_after_reference(rng, burst):
    """Noise throughout, the burst only after the pre-trigger reference."""
    burst = burst.copy()
    burst[:PRE] = 0
    return synthetic.noise(rng, N, NOISE) + burst


@pytest.mark.parametrize("occupancy, reliable", [(0.6, True), (0.85, True), (0.95, False)])
def test_a_quiet_pre_trigger_reference_supplies_the_floor(occupancy, reliable):
    """Step 4's per-bin floor (dsp.spectral.noise_floor_psd) from the
    snippet's pre-trigger samples: measured -0.7%..-1.0% OBW from 60% to
    95% occupancy over 10 seeds; > 90% is unreliable, as in step 4."""
    rng = np.random.default_rng(12)
    iq = _burst_after_reference(rng, synthetic.band_limited(rng, N, FS, occupancy * FS, 20e3, 1e-3))
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    (region,) = segmentation.regions
    assert segmentation.floor_source == "pre_trigger"
    assert region.obw_hz == pytest.approx(occupancy * FS, rel=0.03)
    assert region.bandwidth_reliable is reliable


@pytest.mark.parametrize(
    "flat, bandwidth, offset",
    [(0.6, 20e3, 100e3), (0.8, 20e3, 100e3), (0.6, 300e3, -50e3), (0.8, 300e3, -50e3)],
)
def test_on_colored_noise_the_reference_floor_measures_the_burst(flat, bandwidth, offset):
    """Receiver noise rolls off 15 dB toward the band edges (an 80% or 60%
    flat passband). The per-bin reference floor follows it: one burst
    region and no false context. Measured over 20 seeds: 20 kHz bursts read
    +7.4% (22 bins) within 455 Hz, 300 kHz bursts -0.7% within 1.1 kHz."""
    rng = np.random.default_rng(13)
    burst = synthetic.band_limited(rng, N, FS, bandwidth, offset, 1e-4)
    burst[:PRE] = 0
    iq = synthetic.colored_noise(rng, N, FS, NOISE, flat, 15.0) + burst
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    (region,) = segmentation.regions
    assert region.center_offset_hz == pytest.approx(offset, abs=2 * BIN_HZ)
    assert region.obw_hz == pytest.approx(bandwidth, rel=0.15 if bandwidth < 1e5 else 0.03)
    assert region.bandwidth_reliable and segmentation.before_trigger == ()


@pytest.mark.parametrize("flat", [0.6, 0.8])
def test_on_colored_noise_the_self_floor_is_fooled_into_the_edge_zone(flat):
    """Without a reference, a flat floor reads the in-band noise of a 15 dB
    roll-off as one false region hundreds of kHz wide, swallowing the burst.
    Its outer edge reaches 0.373-0.475 fs (60-90% flat, 20 seeds), so it
    touches the edge zone, which begins at 0.35 fs."""
    rng = np.random.default_rng(13)
    burst = synthetic.band_limited(rng, N, FS, 20e3, 100e3, 1e-4)
    iq = synthetic.colored_noise(rng, N, FS, NOISE, flat, 15.0) + burst
    primary = segment_spectrum(iq, FS).primary
    assert primary.obw_hz > 500e3 and touches_edge_zone(primary, FS)


@pytest.mark.parametrize("offset, in_edge_zone", [(300e3, False), (-300e3, False), (450e3, True), (-450e3, True)])
def test_the_edge_zone_is_the_outer_15_percent_of_the_band_on_each_side(offset, in_edge_zone):
    rng = np.random.default_rng(21)
    iq = synthetic.band_limited(rng, N, FS, 30e3, offset, 1e-3) + synthetic.noise(rng, N, NOISE)
    assert touches_edge_zone(segment_spectrum(iq, FS).primary, FS) is in_edge_zone


def test_emitters_already_on_are_context_from_the_reference():
    """A carrier and a 20 kHz emitter already on before the trigger, on
    colored noise: found in the reference against its own local floor
    (context only) and absent from the burst regions, where the reference
    floor subtracts them; the primary is the burst."""
    rng = np.random.default_rng(17)
    burst = synthetic.band_limited(rng, N, FS, 20e3, 100e3, 1e-4)
    burst[:PRE] = 0
    iq = (
        synthetic.colored_noise(rng, N, FS, NOISE, 0.8, 15.0)
        + synthetic.tone(N, FS, -250e3, 3e-5)
        + synthetic.band_limited(rng, N, FS, 20e3, 300e3, 3e-5)
        + burst
    )
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    (primary,) = segmentation.regions
    assert primary.center_offset_hz == pytest.approx(100e3, abs=2 * BIN_HZ)
    centers = sorted(region.center_offset_hz for region in segmentation.before_trigger)
    assert centers == [pytest.approx(-250e3, abs=BIN_HZ), pytest.approx(300e3, abs=2 * BIN_HZ)]


def test_a_steep_roll_off_is_not_read_as_context():
    """Noise 25 dB down at the edges of an 80% passband: the median of a
    65-bin window follows the skirt; a lower envelope (20th percentile)
    lags it and reads the slope as emitters (306 false regions in 40)."""
    rng = np.random.default_rng(19)
    burst = synthetic.band_limited(rng, N, FS, 20e3, 100e3, 1e-4)
    burst[:PRE] = 0
    iq = synthetic.colored_noise(rng, N, FS, NOISE, 0.8, 25.0) + burst
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    assert segmentation.before_trigger == ()
    assert segmentation.primary.center_offset_hz == pytest.approx(100e3, abs=2 * BIN_HZ)


def test_an_always_on_emitter_is_the_primary_from_the_self_floor():
    """An always-on emitter (an LTE downlink) fills its own pre-trigger, so
    no reference frame is quiet: the self floor finds it as the primary,
    never no region. Filling 90% of the band, it reaches the edge zone."""
    rng = np.random.default_rng(18)
    iq = synthetic.colored_noise(rng, N, FS, NOISE, 0.8, 15.0) + synthetic.band_limited(rng, N, FS, 0.9 * FS, 0.0, 1e-3)
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    assert segmentation.floor_source == "self"
    assert segmentation.primary.obw_hz == pytest.approx(0.9 * FS, rel=0.03)
    assert touches_edge_zone(segmentation.primary, FS)


def test_the_recorded_trigger_threshold_selects_the_quiet_reference():
    """An emitter already on below the trigger threshold, then a burst that
    adds under 3 dB to the band. Step 4's recorded threshold finds the quiet
    pre-trigger, so the burst is the primary and the emitter is context (20
    of 20 seeds). The 3 dB fallback finds no quiet frame (16 of 20), and its
    self floor makes the older, stronger emitter the primary."""
    rng = np.random.default_rng(20)
    burst = synthetic.band_limited(rng, N, FS, 50e3, 200e3, 5e-5)
    burst[:PRE] = 0
    iq = synthetic.noise(rng, N, NOISE) + synthetic.band_limited(rng, N, FS, 20e3, -250e3, 1e-4) + burst
    recorded = segment_spectrum(iq, FS, iq[:PRE], threshold_dbfs=-38.5)
    assert recorded.floor_source == "pre_trigger"
    assert [region.center_offset_hz for region in recorded.regions] == [pytest.approx(200e3, abs=2 * BIN_HZ)]
    assert [region.center_offset_hz for region in recorded.before_trigger] == [pytest.approx(-250e3, abs=2 * BIN_HZ)]
    fallback = segment_spectrum(iq, FS, iq[:PRE])
    assert fallback.floor_source == "self"
    assert fallback.primary.center_offset_hz == pytest.approx(-250e3, abs=2 * BIN_HZ)


def test_a_recorded_threshold_the_pre_trigger_exceeds_means_no_reference():
    """An always-on emitter that retriggered fills its pre-trigger above the
    threshold, so no frame is quiet and the self floor is used."""
    rng = np.random.default_rng(14)
    iq = synthetic.noise(rng, N, NOISE) + synthetic.band_limited(rng, N, FS, 300e3, -100e3, 1e-3)
    assert segment_spectrum(iq, FS, iq[:PRE], threshold_dbfs=-35.0).floor_source == "self"


def test_a_reference_holding_the_signal_falls_back_to_the_self_floor():
    """A continuous emitter retriggering after its cooldown fills its own
    pre-trigger: subtracting that would erase it. No reference frame is 3 dB
    quieter than the active frames (dsp.spectral.quiet_reference), so the
    self floor is used."""
    rng = np.random.default_rng(14)
    iq = synthetic.noise(rng, N, NOISE) + synthetic.band_limited(rng, N, FS, 300e3, -100e3, 1e-3)
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    (region,) = segmentation.regions
    assert segmentation.floor_source == "self"
    assert region.obw_hz == pytest.approx(300e3, rel=0.03)


def test_the_quiet_frames_of_a_reference_with_a_transient_still_serve():
    """A loud transient in part of the pre-trigger does not spoil the
    reference: its frames fail the quiet gate (dsp.spectral.quiet_reference),
    and step 4's group-median floor would shrug them off anyway."""
    rng = np.random.default_rng(16)
    burst = synthetic.gate(synthetic.band_limited(rng, N, FS, 50e3, 200e3, 1e-3), FS, [(0.05, 0.1)])
    iq = synthetic.noise(rng, N, NOISE) + burst
    iq[: 5 * NFFT] += synthetic.band_limited(rng, 5 * NFFT, FS, 400e3, -100e3, 3e-3)
    segmentation = segment_spectrum(iq, FS, iq[:PRE])
    assert segmentation.floor_source == "pre_trigger"
    assert segmentation.primary.obw_hz == pytest.approx(50e3, rel=0.05)


def test_a_reference_shorter_than_eight_frames_is_not_used():
    rng = np.random.default_rng(15)
    iq = _burst_after_reference(rng, synthetic.band_limited(rng, N, FS, 50e3, 200e3, 1e-3))
    assert segment_spectrum(iq, FS, iq[: 7 * NFFT]).floor_source == "self"


def test_too_short_capture_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        segment_spectrum(np.zeros(NFFT - 1, dtype=np.complex64), FS)
