from __future__ import annotations

from datetime import datetime, timezone

from schema.records import Identifier, Modality, Signal, UnifiedRecord, without_nul


def normalize_ble_advertisement(
    address: str,
    device_name: str | None,
    rssi: float,
    survey_id: str,
    operator_id: str,
) -> UnifiedRecord:
    """lat/lon are left at 0.0 with gps_fix_quality=None — ingest.service
    attaches the real fix, since BLE capture has no GPS access of its own."""
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=0.0,
        lon=0.0,
        gps_fix_quality=None,
        survey_id=survey_id,
        operator_id=operator_id,
        modality=Modality.BLUETOOTH,
        identifier=Identifier(bt_mac=address, device_name=without_nul(device_name)),
        signal=Signal(rssi=rssi),
    )
