from gps.nmea import parse_nmea, with_checksum

GGA = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
RMC = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
RMC_VOID = with_checksum("GPRMC,123519,V,,,,,,,230394,,")
GGA_NOFIX = with_checksum("GPGGA,123519,,,,,0,00,,,M,,M,,")


def test_gga_parsed():
    fix = parse_nmea(GGA)
    assert fix is not None
    assert abs(fix.lat - 48.1173) < 1e-6
    assert abs(fix.lon - 11.516667) < 1e-5
    assert fix.altitude == 545.4
    assert fix.fix_quality == 1


def test_rmc_parsed_without_altitude():
    fix = parse_nmea(RMC)
    assert fix is not None
    assert abs(fix.lat - 48.1173) < 1e-6
    assert fix.altitude is None
    assert fix.fix_quality == 1


def test_southern_western_hemisphere_negative():
    s = with_checksum("GNGGA,123519,3352.000,S,15112.000,W,2,08,0.9,10.0,M,0,M,,")
    fix = parse_nmea(s)
    assert fix.lat < 0 and fix.lon < 0
    assert fix.fix_quality == 2


def test_bad_checksum_rejected():
    assert parse_nmea(GGA.replace("*47", "*00")) is None


def test_missing_checksum_rejected():
    assert parse_nmea(GGA.split("*")[0]) is None


def test_no_fix_returns_none():
    assert parse_nmea(GGA_NOFIX) is None
    assert parse_nmea(RMC_VOID) is None


def test_other_sentences_and_garbage_ignored():
    assert parse_nmea(with_checksum("GPGSV,1,1,00")) is None
    assert parse_nmea("garbage") is None
    assert parse_nmea("") is None


def test_malformed_fields_return_none():
    assert parse_nmea(with_checksum("GPGGA,123519,xx,N,01131.000,E,1,08,0.9,5,M,,M,,")) is None
