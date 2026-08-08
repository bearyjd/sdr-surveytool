from __future__ import annotations

import asyncio

from bleak import BleakScanner

from capture.bluetooth.normalizer import normalize_ble_advertisement
from capture.common.emitter import RecordEmitter


async def run(
    socket_path: str, survey_id: str, operator_id: str, scan_seconds: float = 5.0
) -> None:
    with RecordEmitter(socket_path) as emitter:
        while True:
            devices = await BleakScanner.discover(timeout=scan_seconds, return_adv=True)
            for device, adv in devices.values():
                record = normalize_ble_advertisement(
                    address=device.address,
                    device_name=device.name,
                    rssi=adv.rssi,
                    survey_id=survey_id,
                    operator_id=operator_id,
                )
                emitter.emit(record)


def main(socket_path: str, survey_id: str, operator_id: str) -> None:
    asyncio.run(run(socket_path, survey_id, operator_id))
