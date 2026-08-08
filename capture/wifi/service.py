from __future__ import annotations

import time

from capture.common.emitter import RecordEmitter
from capture.wifi.kismet_client import KismetClient
from capture.wifi.normalizer import normalize_kismet_device


def run(
    kismet_base_url: str,
    socket_path: str,
    survey_id: str,
    operator_id: str,
    poll_interval_seconds: float = 2.0,
) -> None:
    client = KismetClient(kismet_base_url)
    with RecordEmitter(socket_path) as emitter:
        while True:
            for device in client.get_wifi_devices():
                record = normalize_kismet_device(device, survey_id, operator_id)
                if record is not None:
                    emitter.emit(record)
            time.sleep(poll_interval_seconds)
