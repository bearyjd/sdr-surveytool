# gps

u-blox M8N reader. A background thread parses NMEA GGA sentences from the receiver
(USB `/dev/ttyACM*` or any file/FIFO) and serves the latest fix to `ingest/` through
the `GpsFixProvider` interface.

- `nmea.py` — checksum validation and GGA parsing (no fix / bad checksum → ignored).
- `service.py` — `NmeaGpsFixProvider`; stdlib only (termios), reconnects if the
  device disappears. A fix older than `--gps-max-age` (default 5 s) is dropped, so
  records are left un-geotagged rather than stamped with a stale position.

Enable with `sdr-ingest --gps-device /dev/ttyACM0` (`--gps-baud`, `--gps-max-age`).
Without it, ingest uses the fixed `--gps-lat/--gps-lon/--gps-fix-quality` stand-in.

Not yet validated against a real receiver; the M8N must be configured to emit GGA
(default NMEA output does).
