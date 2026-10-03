from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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
