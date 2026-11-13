import threading

from gps.nmea import checksum_ok, parse_gga
from gps.service import NmeaGpsFixProvider, open_nmea_lines


def _nmea(body: str) -> str:
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    return f"${body}*{calc:02X}\r\n"


GGA = _nmea("GNGGA,123519.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,")


def test_parses_valid_gga():
    fix = parse_gga(GGA)
    assert fix is not None
    assert abs(fix.lat - 48.1173) < 1e-6
    assert abs(fix.lon - 11.516667) < 1e-5
    assert fix.altitude == 545.4
    assert fix.fix_quality == 1


def test_southern_western_hemispheres_negative():
    fix = parse_gga(_nmea("GPGGA,1,3351.000,S,15112.000,W,2,05,1,10.0,M,0,M,,"))
    assert fix.lat < 0 and fix.lon < 0 and fix.fix_quality == 2


def test_rejects_bad_checksum_no_fix_and_other_sentences():
    assert not checksum_ok(GGA.replace("4807", "4808"))
    assert parse_gga(GGA.replace("4807", "4808")) is None
    assert parse_gga(_nmea("GNGGA,1,,,,,0,00,99.99,,,,,,")) is None
    assert parse_gga(_nmea("GNRMC,1,A,4807.038,N,01131.000,E,0,0,230394,,")) is None
    assert parse_gga("$GNGGA,no,checksum") is None


def test_provider_serves_latest_and_expires():
    now = [0.0]
    p = NmeaGpsFixProvider(lambda: [], max_age_s=5, clock=lambda: now[0])
    assert p.current_fix() is None
    p.feed(GGA)
    assert p.current_fix().fix_quality == 1
    now[0] = 6.0
    assert p.current_fix() is None


def test_provider_reads_from_file_in_thread(tmp_path):
    dev = tmp_path / "nmea.txt"
    dev.write_text("garbage\n" + GGA)
    p = NmeaGpsFixProvider(lambda: open_nmea_lines(str(dev)), reconnect_delay_s=0.01)
    p.start()
    done = threading.Event()
    for _ in range(200):
        if p.current_fix() is not None:
            done.set()
            break
        threading.Event().wait(0.01)
    p.stop()
    assert done.is_set()
