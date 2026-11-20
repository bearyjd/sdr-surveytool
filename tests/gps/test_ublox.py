import threading
import time

from gps.nmea import with_checksum
from gps.ublox import UbloxGpsFixProvider

GGA = with_checksum("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,")
GGA2 = with_checksum("GPGGA,123520,4808.000,N,01132.000,E,1,08,0.9,546.0,M,46.9,M,,")


class FakeLines:
    def __init__(self, lines):
        self._lines = list(lines)
        self.done = threading.Event()

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self.done.set()
        time.sleep(0.01)
        return ""


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_none_before_any_fix():
    p = UbloxGpsFixProvider(FakeLines([]))
    assert p.current_fix() is None


def test_latest_fix_kept():
    src = FakeLines(["junk\r\n", GGA + "\r\n", GGA2 + "\r\n"])
    p = UbloxGpsFixProvider(src)
    p.start()
    try:
        assert src.done.wait(2)
        assert abs(p.current_fix().lat - 48.133333) < 1e-5
    finally:
        p.stop()


def test_stale_fix_discarded():
    now = [100.0]
    src = FakeLines([GGA + "\r\n"])
    p = UbloxGpsFixProvider(src, max_age_s=5.0, clock=lambda: now[0])
    p.start()
    try:
        assert _wait(lambda: p.current_fix() is not None)
        now[0] = 110.0
        assert p.current_fix() is None
    finally:
        p.stop()


def test_bytes_lines_and_reader_errors_survive():
    class Flaky(FakeLines):
        def readline(self):
            if not getattr(self, "boom", False):
                self.boom = True
                raise OSError("device hiccup")
            return super().readline()

    src = Flaky([GGA.encode() + b"\r\n"])
    p = UbloxGpsFixProvider(src)
    p.start()
    try:
        assert _wait(lambda: p.current_fix() is not None)
    finally:
        p.stop()
