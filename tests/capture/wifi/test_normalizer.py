import pytest

from capture.wifi.normalizer import normalize_kismet_device
from schema.records import Modality

SAMPLE_DEVICE = {
    "kismet.device.base.macaddr": "AA:BB:CC:DD:EE:FF",
    "kismet.device.base.channel": "6",
    "kismet.device.base.last_time": 1750000000,
    "kismet.device.base.signal": {"kismet.common.signal.last_signal": -55},
    "dot11.device": {
        "dot11.device.advertised_ssid_map": {
            "0": {
                "dot11.advertisedssid.ssid": "TestNet",
                "dot11.advertisedssid.crypt_string": "WPA2",
            }
        }
    },
}


def test_normalize_kismet_device_maps_fields():
    record = normalize_kismet_device(SAMPLE_DEVICE, survey_id="s1", operator_id="op1")
    assert record is not None
    assert record.modality is Modality.WIFI
    assert record.identifier.bssid == "AA:BB:CC:DD:EE:FF"
    assert record.identifier.ssid == "TestNet"
    assert record.identifier.channel == 6
    assert record.signal.rssi == -55.0
    assert record.metadata.encryption_type_if_broadcast_visible == "WPA2"
    assert record.survey_id == "s1"
    assert record.operator_id == "op1"


@pytest.mark.parametrize(
    "raw_channel, expected",
    [
        ("6", 6),
        ("6HT40", 6),
        ("36HT80", 36),
        ("157VHT80", 157),
        ("11HT20", 11),
        (6, 6),
        ("", None),
        ("unknown", None),
    ],
)
def test_normalize_kismet_device_parses_ht_vht_channel(raw_channel, expected):
    """Kismet reports HT/VHT channels as strings like "6HT40"; passing those
    straight into Identifier(channel=...) raised a pydantic ValidationError and
    killed the whole capture process on the first 802.11n/ac AP seen."""
    device = {**SAMPLE_DEVICE, "kismet.device.base.channel": raw_channel}
    record = normalize_kismet_device(device, survey_id="s1", operator_id="op1")
    assert record is not None
    assert record.identifier.channel == expected


def test_normalize_kismet_device_channel_missing_stays_none():
    device = {k: v for k, v in SAMPLE_DEVICE.items() if k != "kismet.device.base.channel"}
    record = normalize_kismet_device(device, survey_id="s1", operator_id="op1")
    assert record is not None
    assert record.identifier.channel is None


def test_normalize_kismet_device_returns_none_without_signal():
    device_without_signal = {"kismet.device.base.macaddr": "AA:BB:CC:DD:EE:FF"}
    assert normalize_kismet_device(device_without_signal, "s1", "op1") is None


def test_nul_characters_from_the_air_are_stripped():
    """Kismet reports raw SSID bytes; a NUL would make the record invalid."""
    device = {
        **SAMPLE_DEVICE,
        "dot11.device": {
            "dot11.device.advertised_ssid_map": {
                "0": {"dot11.advertisedssid.ssid": "Test\x00Net\x00", "dot11.advertisedssid.crypt_string": "WPA2\x00"}
            }
        },
    }
    record = normalize_kismet_device(device, survey_id="s1", operator_id="op1")
    assert record.identifier.ssid == "TestNet"
    assert record.metadata.encryption_type_if_broadcast_visible == "WPA2"
