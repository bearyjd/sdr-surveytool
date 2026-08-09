from __future__ import annotations

import math


def grid_cell_key(lat: float, lon: float, cell_size_degrees: float = 0.0001) -> str:
    """Buckets a lat/lon into a stable grid-cell key for sample-density
    tracking. 0.0001 degrees is roughly 11m at the equator, a reasonable
    default survey grid resolution."""
    cell_lat = math.floor(lat / cell_size_degrees)
    cell_lon = math.floor(lon / cell_size_degrees)
    return f"{cell_lat}:{cell_lon}"
