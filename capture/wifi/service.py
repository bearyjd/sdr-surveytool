from __future__ import annotations

import logging
import time

from capture.common.emitter import RecordEmitter
from capture.wifi.kismet_client import KismetClient
from capture.wifi.normalizer import normalize_kismet_device

logger = logging.getLogger(__name__)

# On consecutive failures the sleep doubles from poll_interval_seconds up to
# this ceiling, so a Kismet outage isn't hammered every poll interval.
_MAX_BACKOFF_SECONDS = 60.0


def run(
    kismet_base_url: str,
    socket_path: str,
    survey_id: str,
    operator_id: str,
    poll_interval_seconds: float = 2.0,
) -> None:
    """Poll Kismet forever, normalizing and emitting each device.

    A failing poll cycle (Kismet HTTP error, malformed JSON, dead ingest
    socket) is logged and skipped rather than killing the capture process:
    field surveys must survive transient faults unattended.
    """
    client = KismetClient(kismet_base_url)
    backoff = poll_interval_seconds
    with RecordEmitter(socket_path) as emitter:
        while True:
            try:
                _poll_once(client, emitter, survey_id, operator_id)
            except Exception:
                logger.exception(
                    "WiFi poll cycle failed; retrying in %.1fs", backoff
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                continue
            backoff = poll_interval_seconds
            time.sleep(poll_interval_seconds)


def _poll_once(
    client: KismetClient,
    emitter: RecordEmitter,
    survey_id: str,
    operator_id: str,
) -> None:
    for device in client.get_wifi_devices():
        record = normalize_kismet_device(device, survey_id, operator_id)
        if record is not None:
            emitter.emit(record)


def main() -> None:
    """CLI entry point (`sdr-capture-wifi`)."""
    import argparse

    parser = argparse.ArgumentParser(description="Kismet WiFi capture -> ingest queue")
    parser.add_argument("--kismet-url", default="http://localhost:2501")
    parser.add_argument("--socket-path", default="/tmp/sdr-ingest.sock")
    parser.add_argument("--survey-id", required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(
        kismet_base_url=args.kismet_url,
        socket_path=args.socket_path,
        survey_id=args.survey_id,
        operator_id=args.operator_id,
        poll_interval_seconds=args.poll_interval,
    )
