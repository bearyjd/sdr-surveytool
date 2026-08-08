from __future__ import annotations

from datetime import datetime, timezone

from schema.records import Identifier, Modality, Signal, UnifiedRecord


def normalize_kismet_device(
    device: dict, survey_id: str, operator_id: str
) -> UnifiedRecord | None:
    """Converts a single Kismet phy80211 device JSON object into a
    UnifiedRecord. Returns None if the device has no signal reading yet
    (Kismet reports devices before their first signal sample). lat/lon are
    left at 0.0 with gps_fix_quality=None — ingest.service attaches the real
    fix, since WiFi capture has no GPS access of its own."""
    signal_block = device.get("kismet.device.base.signal", {})
    signal_dbm = signal_block.get("kismet.common.signal.last_signal")
    if signal_dbm is None:
        return None

    ssid_map = device.get("dot11.device", {}).get(
        "dot11.device.advertised_ssid_map", {}
    )
    ssid = None
    encryption = None
    for entry in ssid_map.values():
        ssid = entry.get("dot11.advertisedssid.ssid")
        encryption = entry.get("dot11.advertisedssid.crypt_string")
        break

    last_seen = device.get("kismet.device.base.last_time")
    timestamp = (
        datetime.fromtimestamp(last_seen, tz=timezone.utc)
        if last_seen
        else datetime.now(timezone.utc)
    )

    return UnifiedRecord(
        timestamp=timestamp,
        lat=0.0,
        lon=0.0,
        gps_fix_quality=None,
        survey_id=survey_id,
        operator_id=operator_id,
        modality=Modality.WIFI,
        identifier=Identifier(
            bssid=device.get("kismet.device.base.macaddr"),
            ssid=ssid,
            channel=device.get("kismet.device.base.channel"),
        ),
        signal=Signal(rssi=float(signal_dbm)),
        metadata={"encryption_type_if_broadcast_visible": encryption} if encryption else {},
    )
