from __future__ import annotations

from datetime import timezone

from sqlalchemy.orm import Session

from schema.records import UnifiedRecord
from storage.models import SurveyRecord


def save_record(session: Session, record: UnifiedRecord) -> SurveyRecord:
    """
    Persist a UnifiedRecord to the database as a SurveyRecord.

    Normalizes timestamp to UTC-naive datetime for consistent backend handling.
    Stores all schema fields with explicit null values (no exclude_none) for
    consistent JSON shape across all records regardless of modality.

    Rolls back the session on any error to prevent poisoning for subsequent calls.
    """
    try:
        # Normalize timestamp to UTC-naive for consistent storage.
        # Naive inputs are treated as already UTC (just strip the marker).
        # Aware inputs are converted to UTC then stripped to naive.
        ts = record.timestamp
        normalized_ts = (
            ts.replace(tzinfo=None)
            if ts.tzinfo is None
            else ts.astimezone(timezone.utc).replace(tzinfo=None)
        )

        row = SurveyRecord(
            timestamp=normalized_ts,
            lat=record.lat,
            lon=record.lon,
            altitude=record.altitude,
            gps_fix_quality=record.gps_fix_quality,
            survey_id=record.survey_id,
            operator_id=record.operator_id,
            modality=record.modality.value,
            # Store all schema keys with explicit null values for consistent JSON shape.
            identifier=record.identifier.model_dump(),
            signal=record.signal.model_dump(),
            metadata_=record.metadata.model_dump(),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
    except Exception:
        session.rollback()
        raise
