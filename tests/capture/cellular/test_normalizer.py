from capture.cellular.normalizer import normalize_cellsearch_result
from schema.records import Modality

SAMPLE_RESULT = {
    "duplex": "FDD",
    "freq_mhz": 1815.3,
    "cell_id": 301,
    "pss_id": 1,
    "rx_power_db": -9.44976,
    "freq_offset_hz": 14302.6,
}


def test_normalize_cellsearch_result_maps_fields():
    record = normalize_cellsearch_result(SAMPLE_RESULT, survey_id="s1", operator_id="op1")
    assert record.modality is Modality.CELLULAR
    assert record.identifier.cell_id == "301"
    assert record.identifier.center_freq == 1815300000.0
    assert record.signal.rssi == -9.44976
    assert record.signal.rsrp is None
    assert record.survey_id == "s1"
    assert record.operator_id == "op1"
    assert record.lat == 0.0
    assert record.gps_fix_quality is None
