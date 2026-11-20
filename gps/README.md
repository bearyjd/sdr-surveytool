# gps

u-blox M8N GPS fix provider. `ingest/` calls `current_fix()` to attach the
current fix (lat, lon, altitude, fix quality) to records.

- `nmea.py`: parses GGA and RMC sentences (checksum verified). Returns a
  `GpsFix`, or `None` when there is no fix or the sentence is invalid.
- `ublox.py`: `UbloxGpsFixProvider` reads lines from any object with
  `readline()` (pyserial by default) on a background thread and keeps the
  latest fix. Fixes older than `max_age_s` (default 5 s) are discarded, so
  `current_fix()` returns `None` after signal loss.

Enable in ingest with `--gps-device /dev/ttyACM0 [--gps-baud 9600]`. Without
`--gps-device`, ingest falls back to the static fix (`--gps-lat/--gps-lon/
--gps-altitude`, with `--gps-fix-quality` required).
