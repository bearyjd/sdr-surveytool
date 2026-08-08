from __future__ import annotations

import argparse
import asyncio
import logging

from bleak import BleakScanner

from capture.bluetooth.normalizer import normalize_ble_advertisement
from capture.common.emitter import RecordEmitter

logger = logging.getLogger(__name__)

# On consecutive failures the sleep doubles from scan_seconds up to this
# ceiling, so a wedged BlueZ adapter isn't retried in a tight loop.
_MAX_BACKOFF_SECONDS = 60.0


async def run(
    socket_path: str, survey_id: str, operator_id: str, scan_seconds: float = 5.0
) -> None:
    """Scan for BLE advertisements forever, emitting each one.

    A failing scan cycle (BlueZ adapter error, dead ingest socket) is logged
    and skipped rather than killing the capture process.
    """
    backoff = scan_seconds
    with RecordEmitter(socket_path) as emitter:
        while True:
            try:
                await _scan_once(emitter, survey_id, operator_id, scan_seconds)
            except Exception:
                logger.exception("BLE scan cycle failed; retrying in %.1fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                continue
            backoff = scan_seconds


async def _scan_once(
    emitter: RecordEmitter,
    survey_id: str,
    operator_id: str,
    scan_seconds: float,
) -> None:
    # PASSIVE-ONLY CONSTRAINT EXCEPTION (deliberate, not an oversight):
    # bleak defaults to ACTIVE BLE scanning, which transmits SCAN_REQ packets
    # to elicit scan responses from nearby devices. The plan's Global
    # Constraints call for passive-only capture. We keep active scanning here
    # as an accepted tradeoff: passive mode on BlueZ requires supplying
    # `or_patterns` filters and drops scan-response payloads, which is where
    # device names live — data this survey tool specifically wants. See the
    # amendment note in docs/superpowers/plans/2026-08-08-wifi-bluetooth-pipeline.md.
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


def main() -> None:
    """CLI entry point (`sdr-capture-bluetooth`)."""
    parser = argparse.ArgumentParser(description="BLE capture -> ingest queue")
    parser.add_argument("--socket-path", default="/tmp/sdr-ingest.sock")
    parser.add_argument("--survey-id", required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--scan-seconds", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        run(
            socket_path=args.socket_path,
            survey_id=args.survey_id,
            operator_id=args.operator_id,
            scan_seconds=args.scan_seconds,
        )
    )
