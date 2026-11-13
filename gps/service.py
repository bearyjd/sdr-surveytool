"""u-blox M8N reader: a background thread keeps the latest NMEA GGA fix and
serves it to ingest through the GpsFixProvider interface.

Stdlib only (termios for the serial line), so no extra dependency on the
Jetson. The source may also be a regular file or FIFO, which tests and replay
use."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Iterable

from gps.nmea import parse_gga
from ingest.gps_fix import GpsFix

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE_S = 5.0


def open_nmea_lines(device: str, baud: int = 9600) -> Iterable[str]:
    """Yield NMEA lines from a serial device (raw mode at `baud`) or any file.
    Raises OSError if the device cannot be opened."""
    fd = os.open(device, os.O_RDONLY | os.O_NOCTTY)
    try:
        if os.isatty(fd):
            import termios

            attrs = termios.tcgetattr(fd)
            speed = getattr(termios, f"B{baud}")
            attrs[0] = 0  # iflag
            attrs[1] = 0  # oflag
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[3] = 0  # lflag: raw
            attrs[4] = attrs[5] = speed
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        with os.fdopen(fd, "r", encoding="ascii", errors="replace", newline=None) as handle:
            fd = -1
            yield from handle
    finally:
        if fd >= 0:
            os.close(fd)


class NmeaGpsFixProvider:
    """Latest valid GGA fix, or None when none has arrived within `max_age_s`.

    Returning None (rather than a stale position) makes ingest leave records
    un-geotagged, which is honest: a vehicle that lost sky view must not keep
    stamping its last position on new samples."""

    def __init__(
        self,
        lines: Callable[[], Iterable[str]],
        max_age_s: float = DEFAULT_MAX_AGE_S,
        clock: Callable[[], float] = time.monotonic,
        reconnect_delay_s: float = 2.0,
    ) -> None:
        self._lines = lines
        self._max_age_s = max_age_s
        self._clock = clock
        self._reconnect_delay_s = reconnect_delay_s
        self._lock = threading.Lock()
        self._fix: GpsFix | None = None
        self._fix_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def feed(self, line: str) -> None:
        fix = parse_gga(line)
        if fix is not None:
            with self._lock:
                self._fix, self._fix_at = fix, self._clock()

    def current_fix(self) -> GpsFix | None:
        with self._lock:
            if self._fix is None or self._clock() - self._fix_at > self._max_age_s:
                return None
            return self._fix

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="nmea-gps", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for line in self._lines():
                    if self._stop.is_set():
                        return
                    self.feed(line)
                logger.warning("GPS source ended; reopening")
            except OSError as exc:
                logger.warning("GPS source unavailable (%s); retrying", exc)
            self._stop.wait(self._reconnect_delay_s)
