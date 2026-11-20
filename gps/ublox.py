"""u-blox M8N GPS fix provider fed by a background NMEA line reader."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Protocol

from gps.nmea import parse_nmea
from ingest.gps_fix import GpsFix

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE_S = 5.0


class LineSource(Protocol):
    def readline(self) -> bytes | str: ...


class UbloxGpsFixProvider:
    """Implements GpsFixProvider. current_fix() returns the latest fix seen,
    or None if there is none or it is older than max_age_s."""

    def __init__(
        self,
        source: LineSource,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._max_age_s = max_age_s
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: tuple[GpsFix, float] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ublox-gps", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def current_fix(self) -> GpsFix | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        fix, received = latest
        if self._clock() - received > self._max_age_s:
            return None
        return fix

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = self._source.readline()
            except Exception:
                logger.exception("GPS read failed; retrying")
                self._stop.wait(0.1)
                continue
            if isinstance(line, bytes):
                line = line.decode("ascii", errors="replace")
            fix = parse_nmea(line)
            if fix is not None:
                with self._lock:
                    self._latest = (fix, self._clock())


def open_ublox_provider(device: str, baud: int) -> UbloxGpsFixProvider:
    """Open a serial device with pyserial and return a (not yet started) provider."""
    import serial  # pyserial; imported lazily so static-fix deployments don't need it

    port = serial.Serial(device, baud, timeout=1.0)
    return UbloxGpsFixProvider(port)
