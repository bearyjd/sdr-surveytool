from __future__ import annotations

from datetime import datetime, timezone

from schema.records import Identifier, Modality, Signal, UnifiedRecord


def normalize_cellsearch_result(
    result: dict, survey_id: str, operator_id: str
) -> UnifiedRecord:
    """Converts one parsed CellSearch detection dict (see
    capture.cellular.offline_scanner.parse_cellsearch_output) into a
    UnifiedRecord. lat/lon are left at 0.0 with gps_fix_quality=None —
    ingest.service attaches the real fix, matching capture.wifi/bluetooth.
    No RSRP/RSRQ/SINR: CellSearch doesn't report them. The parsed dict may
    carry CRC-verified MIB fields (n_rb_dl, PHICH config, ...), but they aren't
    mapped into the record yet; n_rb_dl/bandwidth are the natural first fields
    to wire. No PLMN or SIB fields: those come from SIB1, which this
    codebase can't decode."""
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=0.0,
        lon=0.0,
        gps_fix_quality=None,
        survey_id=survey_id,
        operator_id=operator_id,
        modality=Modality.CELLULAR,
        identifier=Identifier(
            cell_id=str(result["cell_id"]),
            center_freq=result["freq_mhz"] * 1_000_000.0,
        ),
        signal=Signal(rssi=result["rx_power_db"]),
        metadata={},
    )
