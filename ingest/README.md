# ingest

The normalizer/geotagger service. Consumes unified-schema JSON records from the local
capture queue, validates against `schema/`, attaches the nearest GPS fix + quality
flag, computes `sample_count_in_grid_cell`, and writes to `storage/`.

This is the only service that writes to storage — capture plugins never touch it
directly.

**Status**: not yet implemented.
