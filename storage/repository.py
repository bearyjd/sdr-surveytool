from __future__ import annotations

from sqlalchemy.orm import Session

from schema.records import UnifiedRecord
from storage.models import SurveyRecord


def save_record(session: Session, record: UnifiedRecord) -> SurveyRecord:
    row = SurveyRecord(
        timestamp=record.timestamp,
        lat=record.lat,
        lon=record.lon,
        altitude=record.altitude,
        gps_fix_quality=record.gps_fix_quality,
        survey_id=record.survey_id,
        operator_id=record.operator_id,
        modality=record.modality.value,
        identifier=record.identifier.model_dump(exclude_none=True),
        signal=record.signal.model_dump(exclude_none=True),
        metadata_=record.metadata.model_dump(exclude_none=True),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row
