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


def test_normalize_kismet_device_returns_none_without_signal():
    device_without_signal = {"kismet.device.base.macaddr": "AA:BB:CC:DD:EE:FF"}
    assert normalize_kismet_device(device_without_signal, "s1", "op1") is None
