from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


def without_nul(value: str | None) -> str | None:
    """`value` with any NUL characters removed: radio-derived strings (an
    SSID, a BLE device name) may carry them, and a record may not."""
    return None if value is None else value.replace("\x00", "")


def _nul_at(value: Any, path: str) -> str | None:
    """The path of the first string (or dict key) holding a NUL, or None."""
    if isinstance(value, str):
        return path if "\x00" in value else None
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and "\x00" in key:
                return path
            found = _nul_at(item, f"{path}.{key}")
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _nul_at(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


class Modality(str, Enum):
    CELLULAR = "cellular"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    UNKNOWN = "unknown"


class ClassificationStatus(str, Enum):
    UNCLASSIFIED = "unclassified"
    MANUALLY_TAGGED = "manually_tagged"
    AUTO_CLASSIFIED = "auto_classified"
    # Part 4 agent: a judgment that a human must resolve (to manually_tagged).
    # Terminal for the agent, and never assigned for systemic faults.
    NEEDS_REVIEW = "needs_review"


class Identifier(BaseModel):
    cell_id: Optional[str] = None
    plmn: Optional[str] = None
    band: Optional[str] = None
    bssid: Optional[str] = None
    ssid: Optional[str] = None
    channel: Optional[int] = None
    bt_mac: Optional[str] = None
    device_name: Optional[str] = None
    center_freq: Optional[float] = None
    bandwidth_estimate: Optional[float] = None


class Signal(BaseModel):
    rssi: float
    rsrp: Optional[float] = None
    rsrq: Optional[float] = None
    snr: Optional[float] = None
    peak_power: Optional[float] = None


class Metadata(BaseModel):
    encryption_type_if_broadcast_visible: Optional[str] = None
    quality_flags: dict = Field(default_factory=dict)
    sample_count_in_grid_cell: int = 0
    iq_snippet_path: Optional[str] = None
    snippet_duration_ms: Optional[int] = None
    sample_rate: Optional[float] = None
    classification_status: Optional[ClassificationStatus] = None
    tag: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class UnifiedRecord(BaseModel):
    timestamp: datetime
    lat: float
    lon: float
    altitude: Optional[float] = None
    gps_fix_quality: Optional[int] = None
    survey_id: str
    operator_id: str
    modality: Modality
    identifier: Identifier
    signal: Signal
    metadata: Metadata = Field(default_factory=Metadata)

    @model_validator(mode="after")
    def _no_nul_characters(self) -> UnifiedRecord:
        """PostgreSQL text and jsonb cannot hold U+0000: one stored row with it
        fails every read of the agent's view and every jsonb cast."""
        for name, value in self.model_dump().items():
            found = _nul_at(value, name)
            if found is not None:
                raise ValueError(f"{found} holds a NUL character, which PostgreSQL cannot store")
        return self
