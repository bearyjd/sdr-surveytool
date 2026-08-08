from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SurveyRecord(Base):
    __tablename__ = "survey_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # All stored timestamps are UTC-naive datetime objects. Consumers must interpret as UTC.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    altitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_fix_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)
    survey_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    operator_id: Mapped[str] = mapped_column(String, nullable=False)
    modality: Mapped[str] = mapped_column(String, nullable=False, index=True)
    identifier: Mapped[dict] = mapped_column(JSON, nullable=False)
    signal: Mapped[dict] = mapped_column(JSON, nullable=False)
    # NOTE: use .metadata_, NOT .metadata — .metadata is reserved by SQLAlchemy's DeclarativeBase
    # and returns a SQLAlchemy MetaData object. This column stores the survey metadata dict.
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False)
