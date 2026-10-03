from capture.bluetooth.normalizer import normalize_ble_advertisement
from schema.records import Modality


def test_normalize_ble_advertisement_maps_fields():
    record = normalize_ble_advertisement(
        address="11:22:33:44:55:66",
        device_name="Wearable-42",
        rssi=-62.0,
        survey_id="s1",
        operator_id="op1",
    )
    assert record.modality is Modality.BLUETOOTH
    assert record.identifier.bt_mac == "11:22:33:44:55:66"
    assert record.identifier.device_name == "Wearable-42"
    assert record.signal.rssi == -62.0
    assert record.gps_fix_quality is None


def test_normalize_ble_advertisement_allows_missing_name():
    record = normalize_ble_advertisement(
        address="11:22:33:44:55:66",
        device_name=None,
        rssi=-70.0,
        survey_id="s1",
        operator_id="op1",
    )
    assert record.identifier.device_name is None


def test_nul_characters_in_an_advertised_name_are_stripped():
    record = normalize_ble_advertisement(
        address="11:22:33:44:55:66", device_name="Tag\x00\x00", rssi=-70.0, survey_id="s1", operator_id="op1"
    )
    assert record.identifier.device_name == "Tag"
