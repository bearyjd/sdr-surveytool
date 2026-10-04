# tests/agent/test_band_table.py
import pytest
from pydantic import ValidationError

from agent.band_table import BandEntry, load_band_table, match_bands


def _entry(id_: str, start: float, end: float, obw: tuple[float, float] = (10e3, 30e3)) -> BandEntry:
    return BandEntry(
        id=id_,
        start_hz=start,
        end_hz=end,
        service="test service",
        typical_signals=("narrowband FM",),
        expected_obw_hz=obw,
        citation="47 CFR 0.0",
        source="verbatim",
    )


def test_shipped_table_is_valid_and_cited():
    table = load_band_table()
    assert table.region == "US"
    assert 20 <= len(table.entries) <= 30
    for entry in table.entries:
        assert entry.citation.startswith("47 CFR ")
        assert entry.source.strip()
        assert 47e6 <= entry.start_hz < entry.end_hz <= 6e9
        low, high = entry.expected_obw_hz
        # Plausibility must mean something: no range starts at 0 or spans
        # more than ~2.4 decades.
        assert 0 < low < high <= 250 * low, entry.id


def test_shipped_table_ids_include_the_spec_examples():
    ids = {entry.id for entry in load_band_table().entries}
    assert {
        "fm_broadcast",
        "airband_vhf",
        "frs_gmrs_462",
        "ism_902_928",
        "pcs_downlink",
        "aws_downlink",
        "lower700_downlink",
        "upper700_c_downlink",
        "lte_b71_600_downlink",
        "adsb_1090",
        "gnss_rnss_l1",
        "ism_2400",
        "cbrs",
        "unii_5150_5250",
        "unii_5725_5850",
    } <= ids


@pytest.mark.parametrize(
    "start, end, obw",
    [(46e6, 50e6, (1, 2)), (100e6, 100e6, (1, 2)), (100e6, 99e6, (1, 2)), (5.9e9, 6.1e9, (1, 2)), (100e6, 101e6, (2, 1))],
)
def test_entries_outside_47mhz_6ghz_or_inverted_are_rejected(start, end, obw):
    with pytest.raises(ValidationError):
        _entry("bad", start, end, obw)


def test_entry_without_a_cfr_citation_is_rejected():
    with pytest.raises(ValidationError):
        BandEntry.model_validate({**_entry("x", 100e6, 101e6).model_dump(), "citation": "Wikipedia"})


def test_grounded_needs_center_inside_and_plausible_obw():
    bands = (_entry("frs", 462.54e6, 462.735e6, (2e3, 20e3)),)
    (match,) = match_bands(bands, 462.6e6, 12e3)
    assert match.grounded
    (match,) = match_bands(bands, 462.6e6, 200e3)  # center inside, OBW implausible
    assert not match.grounded


def test_band_edges_are_inclusive():
    """The whole occupied span must lie in the band; touching an edge does."""
    bands = (_entry("b", 100e6, 101e6),)
    assert match_bands(bands, 100e6 + 10e3, 20e3)[0].grounded
    assert match_bands(bands, 101e6 - 10e3, 20e3)[0].grounded


@pytest.mark.parametrize("center_hz", [2110e6 + 1.0, 2180e6 - 1.0])
def test_a_signal_centered_just_inside_an_edge_but_spilling_out_does_not_ground(center_hz):
    """The reviewer's case: a 20 MHz signal centered 1 Hz inside the band
    edge has half its energy outside the band."""
    bands = (_entry("downlink", 2110e6, 2180e6, (1e6, 20e6)),)
    (match,) = match_bands(bands, center_hz, 20e6)
    assert not match.grounded


@pytest.mark.parametrize("edge", ["low", "high"])
def test_the_span_may_overhang_an_edge_by_the_tolerance(edge):
    """One fine bin each side absorbs the OBW's quantization."""
    bands = (_entry("b", 100e6, 101e6),)
    center = 100e6 + 10e3 - 150.0 if edge == "low" else 101e6 - 10e3 + 150.0
    assert not match_bands(bands, center, 20e3)[0].grounded
    assert match_bands(bands, center, 20e3, tolerance_hz=200.0)[0].grounded


def test_signal_overlapping_an_edge_matches_ungrounded():
    """Center just outside, occupied span overlapping: listed, not grounded."""
    bands = (_entry("b", 100e6, 101e6),)
    (match,) = match_bands(bands, 101e6 + 5e3, 20e3)
    assert not match.grounded
    assert match_bands(bands, 101e6 + 11e3, 20e3) == []


def test_overlapping_entries_all_returned_lowest_start_first():
    bands = (_entry("wide", 900e6, 930e6, (5e3, 2e6)), _entry("narrow", 914e6, 916e6, (100e3, 300e3)))
    matches = match_bands(bands, 915e6, 125e3)
    assert [(m.entry.id, m.grounded) for m in matches] == [("wide", True), ("narrow", True)]
    matches = match_bands(bands, 915e6, 1e6)
    assert [(m.entry.id, m.grounded) for m in matches] == [("wide", True), ("narrow", False)]


@pytest.mark.parametrize("center_hz", [156.775e6, 161.975e6, 162.025e6])
def test_marine_vhf_covers_every_ais_channel(center_hz):
    """47 CFR 80.393: AIS 3/4 at 156.775/156.825 MHz and AIS 1/2 at
    161.975/162.025 MHz, each 25 kHz wide. AIS 2 lies above the 156-162 MHz
    band of 80.373(f), so the entry runs to 162.0375 MHz."""
    (match,) = match_bands(load_band_table().entries, center_hz, 14e3)
    assert (match.entry.id, match.grounded) == ("marine_vhf", True)
    assert match.entry.citation == "47 CFR 80.373(f); 80.393"


def test_an_expected_width_wider_than_its_band_is_rejected():
    """A signal wider than the band cannot lie inside it, so such a range
    could only ever ground by mistake."""
    with pytest.raises(ValidationError, match="wider than the band"):
        _entry("x", 100e6, 101e6, (10e3, 2e6))


def test_every_shipped_expected_width_fits_its_band():
    for entry in load_band_table().entries:
        assert entry.expected_obw_hz[1] <= entry.end_hz - entry.start_hz, entry.id
