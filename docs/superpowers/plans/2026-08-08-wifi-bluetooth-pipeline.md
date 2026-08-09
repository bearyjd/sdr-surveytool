# WiFi/Bluetooth Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first end-to-end, hardware-optional slice of the survey pipeline —
WiFi (Kismet) and Bluetooth (bleak) capture, normalized into the unified record schema,
flowing through a local queue into Postgres/PostGIS-compatible storage, visible on a
local map dashboard.

**Architecture:** Each capture source runs as an independent process and emits
unified-schema JSON records over a local Unix domain socket. A single `ingest` service
is the only consumer of that socket and the only writer to storage — it validates
records, attaches the nearest GPS fix when a capture plugin didn't supply one, and
tracks per-grid-cell sample density before persisting. A minimal FastAPI dashboard
reads directly from storage for field verification.

**Tech Stack:** Python 3.10+, pydantic v2 (schema), SQLAlchemy 2.0 (storage), stdlib
`socket` (local queue — no external broker), `requests` (Kismet REST polling), `bleak`
(BLE scanning), FastAPI + Leaflet (viz).

## Global Constraints

- Python >= 3.10 (per `pyproject.toml` in Task 1).
- WiFi/Bluetooth capture is passive scanning only — no association, deauth, injection,
  pairing, or payload capture (per `docs/superpowers/specs/2026-08-08-multimodality-survey-tool-design.md`, §1).
- Capture plugins never write to storage directly — only `ingest` does (design doc §6).
- The unified record schema (design doc §4) is the only contract between capture and
  everything downstream; do not add modality-specific fields anywhere outside
  `schema/records.py`.
- This plan targets a dev machine first (no bladeRF/Jetson/DGX dependency) — the Kismet
  and BLE adapter integration is real, but Postgres/PostGIS is optional for unit tests
  (SQLite in-memory is used) and required only for the docker-compose integration path.

> **Amendment (post-implementation):** BLE capture (Task 7) uses bleak's default *active*
> scanning mode rather than passive, as a deliberate accepted exception to the passive-only
> constraint above. Passive mode on BlueZ requires `or_patterns` filtering and loses
> scan-response data (device names) that this survey tool wants to capture. See the comment
> above the `BleakScanner.discover(...)` call in `capture/bluetooth/service.py` for the full
> rationale. No other part of the passive-only constraint is relaxed: there is still no
> association, deauth, injection, pairing, or payload capture.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `schema/__init__.py`, `capture/__init__.py`, `capture/common/__init__.py`, `capture/wifi/__init__.py`, `capture/bluetooth/__init__.py`, `ingest/__init__.py`, `storage/__init__.py`, `viz/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: an installable `sdr-surveytool` package covering the `schema`, `capture`,
  `capture.common`, `capture.wifi`, `capture.bluetooth`, `ingest`, `storage`, `viz`
  packages, and a working `pytest` setup, for every later task to build on.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "sdr-surveytool"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.6",
    "sqlalchemy>=2.0",
    "requests>=2.31",
    "bleak>=0.22",
    "fastapi>=0.110",
    "uvicorn>=0.29",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
]

[tool.setuptools]
packages = [
    "schema",
    "capture",
    "capture.common",
    "capture.wifi",
    "capture.bluetooth",
    "ingest",
    "storage",
    "viz",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create empty `__init__.py` files for every package listed above**

```bash
touch schema/__init__.py capture/__init__.py capture/common/__init__.py \
      capture/wifi/__init__.py capture/bluetooth/__init__.py \
      ingest/__init__.py storage/__init__.py viz/__init__.py
mkdir -p tests
```

- [ ] **Step 3: Install the package in editable mode with dev dependencies**

Run: `pip install -e ".[dev]"`
Expected: installs without error.

- [ ] **Step 4: Write a smoke test**

```python
# tests/test_smoke.py
def test_packages_import():
    import capture.common  # noqa: F401
    import ingest  # noqa: F401
    import schema  # noqa: F401
    import storage  # noqa: F401
    import viz  # noqa: F401

    assert True
```

- [ ] **Step 5: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml schema/__init__.py capture/__init__.py capture/common/__init__.py \
        capture/wifi/__init__.py capture/bluetooth/__init__.py ingest/__init__.py \
        storage/__init__.py viz/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold installable package structure"
```

---

### Task 2: Unified record schema

**Files:**
- Create: `schema/records.py`
- Test: `tests/schema/test_records.py`

**Interfaces:**
- Consumes: nothing (foundational).
- Produces: `UnifiedRecord`, `Modality` (enum: `CELLULAR`, `WIFI`, `BLUETOOTH`,
  `UNKNOWN`), `Identifier`, `Signal`, `Metadata`, `ClassificationStatus` (enum:
  `UNCLASSIFIED`, `MANUALLY_TAGGED`, `AUTO_CLASSIFIED`) from `schema.records` — every
  later task imports these exact names.

- [ ] **Step 1: Write the failing test**

```python
# tests/schema/test_records.py
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schema.records import Identifier, Modality, Signal, UnifiedRecord


def _make_record(**overrides):
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        lat=47.6062,
        lon=-122.3321,
        survey_id="survey-1",
        operator_id="op-1",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF", ssid="TestNet", channel=6),
        signal=Signal(rssi=-55.0),
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


def test_unified_record_round_trips_through_json():
    record = _make_record()
    restored = UnifiedRecord.model_validate_json(record.model_dump_json())
    assert restored.identifier.bssid == "AA:BB:CC:DD:EE:FF"
    assert restored.modality is Modality.WIFI


def test_metadata_defaults_to_zero_sample_count():
    record = _make_record()
    assert record.metadata.sample_count_in_grid_cell == 0
    assert record.metadata.classification_status is None


def test_signal_requires_rssi():
    with pytest.raises(ValidationError):
        Signal()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/schema/test_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'schema.records'`

- [ ] **Step 3: Write the implementation**

```python
# schema/records.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/schema/test_records.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add schema/records.py tests/schema/test_records.py
git commit -m "feat: add unified record schema"
```

---

### Task 3: Local queue — Unix domain socket emitter and server

**Files:**
- Create: `capture/common/emitter.py`
- Create: `ingest/queue_server.py`
- Test: `tests/capture/test_emitter.py`
- Test: `tests/ingest/test_queue_server.py`

**Interfaces:**
- Consumes: `schema.records.UnifiedRecord`.
- Produces: `capture.common.emitter.RecordEmitter` (methods: `connect()`, `emit(record:
  UnifiedRecord)`, `close()`, usable as a context manager) and
  `ingest.queue_server.QueueServer` (methods: `start()`, `get(timeout: float | None) ->
  UnifiedRecord`, `stop()`) — every capture module (Tasks 6, 7) uses `RecordEmitter`;
  `ingest.service` (Task 5) uses `QueueServer`.

- [ ] **Step 1: Write the failing test for `QueueServer`**

```python
# tests/ingest/test_queue_server.py
from datetime import datetime, timezone

from capture.common.emitter import RecordEmitter
from ingest.queue_server import QueueServer
from schema.records import Identifier, Modality, Signal, UnifiedRecord


def _record(bssid: str) -> UnifiedRecord:
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=1.0,
        lon=2.0,
        survey_id="s",
        operator_id="o",
        modality=Modality.WIFI,
        identifier=Identifier(bssid=bssid),
        signal=Signal(rssi=-40.0),
    )


def test_queue_server_delivers_records_sent_by_emitter(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_record("AA:BB:CC:DD:EE:01"))
            emitter.emit(_record("AA:BB:CC:DD:EE:02"))

        first = server.get(timeout=2)
        second = server.get(timeout=2)
        assert first.identifier.bssid == "AA:BB:CC:DD:EE:01"
        assert second.identifier.bssid == "AA:BB:CC:DD:EE:02"
    finally:
        server.stop()


def test_queue_server_drops_invalid_json_without_crashing(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()
    try:
        import socket as socket_module

        sock = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        sock.connect(socket_path)
        sock.sendall(b"not valid json\n")
        sock.close()

        with RecordEmitter(socket_path) as emitter:
            emitter.emit(_record("AA:BB:CC:DD:EE:03"))

        record = server.get(timeout=2)
        assert record.identifier.bssid == "AA:BB:CC:DD:EE:03"
    finally:
        server.stop()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/ingest/test_queue_server.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `capture/common/emitter.py`**

```python
# capture/common/emitter.py
from __future__ import annotations

import socket

from schema.records import UnifiedRecord


class RecordEmitter:
    """Connects to the ingest queue's Unix domain socket and sends unified
    records as newline-delimited JSON."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)

    def emit(self, record: UnifiedRecord) -> None:
        if self._sock is None:
            raise RuntimeError("RecordEmitter.connect() must be called before emit()")
        line = record.model_dump_json() + "\n"
        self._sock.sendall(line.encode("utf-8"))

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "RecordEmitter":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
```

- [ ] **Step 4: Write `ingest/queue_server.py`**

```python
# ingest/queue_server.py
from __future__ import annotations

import os
import socket
import threading
from queue import Queue

from pydantic import ValidationError

from schema.records import UnifiedRecord


class QueueServer:
    """Unix domain socket server. Capture processes connect and send
    newline-delimited unified-schema JSON records; validated records are
    made available via `get()`. Malformed lines are dropped, not raised."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._queue: Queue[UnifiedRecord] = Queue()
        self._server_sock: socket.socket | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if os.path.exists(self._socket_path):
            os.remove(self._socket_path)
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self._socket_path)
        self._server_sock.listen(8)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        self._server_sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket) -> None:
        buffer = b""
        with conn:
            while not self._stop.is_set():
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    self._ingest_line(line)

    def _ingest_line(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            record = UnifiedRecord.model_validate_json(line)
        except ValidationError:
            return
        self._queue.put(record)

    def get(self, timeout: float | None = None) -> UnifiedRecord:
        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._server_sock is not None:
            self._server_sock.close()
        if os.path.exists(self._socket_path):
            os.remove(self._socket_path)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/ingest/test_queue_server.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Write and run the emitter's own unit test**

```python
# tests/capture/test_emitter.py
import json
import socket
import threading

from datetime import datetime, timezone

from capture.common.emitter import RecordEmitter
from schema.records import Identifier, Modality, Signal, UnifiedRecord


def test_emit_sends_record_as_ndjson_line(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    received = {}

    def accept_one():
        conn, _ = server.accept()
        data = b""
        while not data.endswith(b"\n"):
            data += conn.recv(4096)
        received["line"] = data.decode("utf-8")
        conn.close()

    thread = threading.Thread(target=accept_one)
    thread.start()

    record = UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=1.0,
        lon=2.0,
        survey_id="s",
        operator_id="o",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
        signal=Signal(rssi=-40.0),
    )

    with RecordEmitter(socket_path) as emitter:
        emitter.emit(record)

    thread.join(timeout=2)
    server.close()

    assert received["line"].endswith("\n")
    parsed = json.loads(received["line"])
    assert parsed["identifier"]["bssid"] == "AA:BB:CC:DD:EE:FF"
```

Run: `pytest tests/capture/test_emitter.py tests/ingest/test_queue_server.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add capture/common/emitter.py ingest/queue_server.py tests/capture/test_emitter.py tests/ingest/test_queue_server.py
git commit -m "feat: add local Unix-socket record queue"
```

---

### Task 4: Storage layer

**Files:**
- Create: `storage/models.py`
- Create: `storage/db.py`
- Create: `storage/repository.py`
- Create: `docker-compose.yml`
- Test: `tests/storage/test_repository.py`

**Interfaces:**
- Consumes: `schema.records.UnifiedRecord`.
- Produces: `storage.models.SurveyRecord` (ORM row with columns: `id`, `timestamp`,
  `lat`, `lon`, `altitude`, `gps_fix_quality`, `survey_id`, `operator_id`, `modality`,
  `identifier` (JSON), `signal` (JSON), `metadata_` (JSON, mapped to DB column
  `metadata`)); `storage.db.make_engine(database_url: str)`,
  `storage.db.make_session_factory(engine) -> sessionmaker`, `storage.db.init_db(engine)
  -> None`; `storage.repository.save_record(session, record: UnifiedRecord) ->
  SurveyRecord` — used by `ingest.service` (Task 5) and `viz.app` (Task 8).

Note on scope: this plan uses plain `Float` lat/lon columns, not a PostGIS `Geometry`
column, because no task in this plan needs geospatial queries (e.g. coverage-gap
polygons) yet. `docker-compose.yml` still provisions Postgres/PostGIS so the schema can
grow a geometry column later without a storage-engine migration — that's a deliberate
YAGNI call, not a missing implementation of the design doc's storage section.

- [ ] **Step 1: Write the failing test**

```python
# tests/storage/test_repository.py
from datetime import datetime, timezone

from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.repository import save_record


def test_save_record_persists_all_fields():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    record = UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=47.6,
        lon=-122.3,
        survey_id="s1",
        operator_id="op1",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF", ssid="Net", channel=6),
        signal=Signal(rssi=-50.0),
    )

    with session_factory() as session:
        saved = save_record(session, record)
        assert saved.id is not None
        assert saved.survey_id == "s1"
        assert saved.modality == "wifi"
        assert saved.identifier["bssid"] == "AA:BB:CC:DD:EE:FF"
        assert saved.signal["rssi"] == -50.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/storage/test_repository.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `storage/models.py`**

```python
# storage/models.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SurveyRecord(Base):
    __tablename__ = "survey_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    altitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_fix_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)
    survey_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    operator_id: Mapped[str] = mapped_column(String, nullable=False)
    modality: Mapped[str] = mapped_column(String, nullable=False, index=True)
    identifier: Mapped[dict] = mapped_column(JSON, nullable=False)
    signal: Mapped[dict] = mapped_column(JSON, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False)
```

- [ ] **Step 4: Write `storage/db.py`**

```python
# storage/db.py
from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from storage.models import Base


def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
```

- [ ] **Step 5: Write `storage/repository.py`**

```python
# storage/repository.py
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pytest tests/storage/test_repository.py -v`
Expected: PASS

- [ ] **Step 7: Write `docker-compose.yml` for local Postgres/PostGIS**

```yaml
services:
  postgres:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_USER: surveytool
      POSTGRES_PASSWORD: surveytool
      POSTGRES_DB: surveytool
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

- [ ] **Step 8: Commit**

```bash
git add storage/models.py storage/db.py storage/repository.py docker-compose.yml tests/storage/test_repository.py
git commit -m "feat: add storage layer (SQLAlchemy models + repository)"
```

---

### Task 5: Ingest service

**Files:**
- Create: `ingest/grid.py`
- Create: `ingest/gps_fix.py`
- Create: `ingest/service.py`
- Test: `tests/ingest/test_grid.py`
- Test: `tests/ingest/test_service.py`

**Interfaces:**
- Consumes: `ingest.queue_server.QueueServer`, `capture.common.emitter.RecordEmitter`
  (Task 3); `storage.repository.save_record`, `storage.db.*` (Task 4);
  `schema.records.UnifiedRecord` (Task 2).
- Produces: `ingest.grid.grid_cell_key(lat: float, lon: float, cell_size_degrees: float
  = 0.0001) -> str`; `ingest.gps_fix.GpsFix` (dataclass: `lat`, `lon`, `altitude`,
  `fix_quality`), `ingest.gps_fix.GpsFixProvider` (protocol: `current_fix() -> GpsFix |
  None`), `ingest.gps_fix.StaticGpsFixProvider` (test/dev stand-in for the real `gps/`
  service, not yet built); `ingest.service.IngestService` (constructor:
  `(queue_server, session_factory, gps_provider)`; method: `process_one(timeout: float |
  None = None) -> UnifiedRecord`).

- [ ] **Step 1: Write the failing test for `grid_cell_key`**

```python
# tests/ingest/test_grid.py
from ingest.grid import grid_cell_key


def test_nearby_points_share_a_grid_cell():
    key_a = grid_cell_key(47.60620, -122.33210)
    key_b = grid_cell_key(47.60623, -122.33208)
    assert key_a == key_b


def test_distant_points_have_different_grid_cells():
    key_a = grid_cell_key(47.6062, -122.3321)
    key_b = grid_cell_key(48.0000, -121.0000)
    assert key_a != key_b
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/ingest/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `ingest/grid.py`**

```python
# ingest/grid.py
from __future__ import annotations

import math


def grid_cell_key(lat: float, lon: float, cell_size_degrees: float = 0.0001) -> str:
    """Buckets a lat/lon into a stable grid-cell key for sample-density
    tracking. 0.0001 degrees is roughly 11m at the equator, a reasonable
    default survey grid resolution."""
    cell_lat = math.floor(lat / cell_size_degrees)
    cell_lon = math.floor(lon / cell_size_degrees)
    return f"{cell_lat}:{cell_lon}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/ingest/test_grid.py -v`
Expected: PASS

- [ ] **Step 5: Write `ingest/gps_fix.py`**

```python
# ingest/gps_fix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GpsFix:
    lat: float
    lon: float
    altitude: float | None
    fix_quality: int


class GpsFixProvider(Protocol):
    def current_fix(self) -> GpsFix | None: ...


class StaticGpsFixProvider:
    """Test/dev stand-in until the real u-blox M8N reader service (gps/)
    exists. Always returns the fix it was constructed with."""

    def __init__(self, fix: GpsFix) -> None:
        self._fix = fix

    def current_fix(self) -> GpsFix | None:
        return self._fix
```

- [ ] **Step 6: Write the failing test for `IngestService`**

```python
# tests/ingest/test_service.py
from datetime import datetime, timezone

from capture.common.emitter import RecordEmitter
from ingest.gps_fix import GpsFix, StaticGpsFixProvider
from ingest.queue_server import QueueServer
from ingest.service import IngestService
from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.models import SurveyRecord


def test_process_one_attaches_gps_and_persists(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider)

    try:
        with RecordEmitter(socket_path) as emitter:
            record = UnifiedRecord(
                timestamp=datetime.now(timezone.utc),
                lat=0.0,
                lon=0.0,
                gps_fix_quality=None,
                survey_id="s",
                operator_id="o",
                modality=Modality.WIFI,
                identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
                signal=Signal(rssi=-40.0),
            )
            emitter.emit(record)

        processed = service.process_one(timeout=2)
        assert processed.lat == 47.6062
        assert processed.gps_fix_quality == 4
        assert processed.metadata.sample_count_in_grid_cell == 1

        with session_factory() as session:
            rows = session.query(SurveyRecord).all()
            assert len(rows) == 1
            assert rows[0].lat == 47.6062
    finally:
        server.stop()


def test_process_one_increments_grid_density_for_repeated_cell(tmp_path):
    socket_path = str(tmp_path / "ingest.sock")
    server = QueueServer(socket_path)
    server.start()

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    gps_provider = StaticGpsFixProvider(
        GpsFix(lat=47.6062, lon=-122.3321, altitude=15.0, fix_quality=4)
    )
    service = IngestService(server, session_factory, gps_provider)

    try:
        with RecordEmitter(socket_path) as emitter:
            for _ in range(2):
                emitter.emit(
                    UnifiedRecord(
                        timestamp=datetime.now(timezone.utc),
                        lat=0.0,
                        lon=0.0,
                        gps_fix_quality=None,
                        survey_id="s",
                        operator_id="o",
                        modality=Modality.WIFI,
                        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
                        signal=Signal(rssi=-40.0),
                    )
                )

        first = service.process_one(timeout=2)
        second = service.process_one(timeout=2)
        assert first.metadata.sample_count_in_grid_cell == 1
        assert second.metadata.sample_count_in_grid_cell == 2
    finally:
        server.stop()
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `pytest tests/ingest/test_service.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 8: Write `ingest/service.py`**

```python
# ingest/service.py
from __future__ import annotations

from collections import Counter

from sqlalchemy.orm import sessionmaker

from ingest.gps_fix import GpsFixProvider
from ingest.grid import grid_cell_key
from ingest.queue_server import QueueServer
from schema.records import UnifiedRecord
from storage.repository import save_record


class IngestService:
    """Consumes validated records from a QueueServer, attaches the nearest
    GPS fix when the record didn't already carry one, tracks per-grid-cell
    sample density, and persists via storage.repository.save_record."""

    def __init__(
        self,
        queue_server: QueueServer,
        session_factory: sessionmaker,
        gps_provider: GpsFixProvider,
    ) -> None:
        self._queue_server = queue_server
        self._session_factory = session_factory
        self._gps_provider = gps_provider
        self._grid_counts: Counter[str] = Counter()

    def process_one(self, timeout: float | None = None) -> UnifiedRecord:
        record = self._queue_server.get(timeout=timeout)
        record = self._attach_gps_if_missing(record)
        record = self._attach_grid_density(record)
        with self._session_factory() as session:
            save_record(session, record)
        return record

    def _attach_gps_if_missing(self, record: UnifiedRecord) -> UnifiedRecord:
        if record.gps_fix_quality is not None:
            return record
        fix = self._gps_provider.current_fix()
        if fix is None:
            return record
        return record.model_copy(
            update={
                "lat": fix.lat,
                "lon": fix.lon,
                "altitude": fix.altitude,
                "gps_fix_quality": fix.fix_quality,
            }
        )

    def _attach_grid_density(self, record: UnifiedRecord) -> UnifiedRecord:
        key = grid_cell_key(record.lat, record.lon)
        self._grid_counts[key] += 1
        updated_metadata = record.metadata.model_copy(
            update={"sample_count_in_grid_cell": self._grid_counts[key]}
        )
        return record.model_copy(update={"metadata": updated_metadata})
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `pytest tests/ingest/ -v`
Expected: PASS (6 tests total across grid/queue_server/service)

- [ ] **Step 10: Commit**

```bash
git add ingest/grid.py ingest/gps_fix.py ingest/service.py tests/ingest/test_grid.py tests/ingest/test_service.py
git commit -m "feat: add ingest service (gps attach, grid density, persistence)"
```

---

### Task 6: WiFi capture (Kismet)

**Files:**
- Create: `capture/wifi/kismet_client.py`
- Create: `capture/wifi/normalizer.py`
- Create: `capture/wifi/service.py`
- Test: `tests/capture/wifi/test_normalizer.py`

**Interfaces:**
- Consumes: `schema.records.{Identifier, Modality, Signal, UnifiedRecord}` (Task 2),
  `capture.common.emitter.RecordEmitter` (Task 3).
- Produces: `capture.wifi.kismet_client.KismetClient` (constructor: `(base_url: str,
  api_key: str | None = None)`; method: `get_wifi_devices() -> list[dict]`);
  `capture.wifi.normalizer.normalize_kismet_device(device: dict, survey_id: str,
  operator_id: str) -> UnifiedRecord | None`; `capture.wifi.service.run(kismet_base_url,
  socket_path, survey_id, operator_id, poll_interval_seconds=2.0) -> None`.

Note: `KismetClient` and `service.run` are thin I/O wrappers around a real running
Kismet instance and are not unit-tested here — they have no branching logic of their
own. All the actual field-mapping logic lives in `normalize_kismet_device`, which is
a pure function and gets full TDD coverage below.

- [ ] **Step 1: Write the failing test**

```python
# tests/capture/wifi/test_normalizer.py
from capture.wifi.normalizer import normalize_kismet_device
from schema.records import Modality

SAMPLE_DEVICE = {
    "kismet.device.base.macaddr": "AA:BB:CC:DD:EE:FF",
    "kismet.device.base.channel": "6",
    "kismet.device.base.last_time": 1750000000,
    "kismet.device.base.signal": {"kismet.common.signal.last_signal": -55},
    "dot11.device": {
        "dot11.device.advertised_ssid_map": {
            "0": {
                "dot11.advertisedssid.ssid": "TestNet",
                "dot11.advertisedssid.crypt_string": "WPA2",
            }
        }
    },
}


def test_normalize_kismet_device_maps_fields():
    record = normalize_kismet_device(SAMPLE_DEVICE, survey_id="s1", operator_id="op1")
    assert record is not None
    assert record.modality is Modality.WIFI
    assert record.identifier.bssid == "AA:BB:CC:DD:EE:FF"
    assert record.identifier.ssid == "TestNet"
    assert record.identifier.channel == "6"
    assert record.signal.rssi == -55.0
    assert record.metadata.encryption_type_if_broadcast_visible == "WPA2"
    assert record.survey_id == "s1"
    assert record.operator_id == "op1"


def test_normalize_kismet_device_returns_none_without_signal():
    device_without_signal = {"kismet.device.base.macaddr": "AA:BB:CC:DD:EE:FF"}
    assert normalize_kismet_device(device_without_signal, "s1", "op1") is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/capture/wifi/test_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `capture/wifi/normalizer.py`**

```python
# capture/wifi/normalizer.py
from __future__ import annotations

from datetime import datetime, timezone

from schema.records import Identifier, Modality, Signal, UnifiedRecord


def normalize_kismet_device(
    device: dict, survey_id: str, operator_id: str
) -> UnifiedRecord | None:
    """Converts a single Kismet phy80211 device JSON object into a
    UnifiedRecord. Returns None if the device has no signal reading yet
    (Kismet reports devices before their first signal sample). lat/lon are
    left at 0.0 with gps_fix_quality=None — ingest.service attaches the real
    fix, since WiFi capture has no GPS access of its own."""
    signal_block = device.get("kismet.device.base.signal", {})
    signal_dbm = signal_block.get("kismet.common.signal.last_signal")
    if signal_dbm is None:
        return None

    ssid_map = device.get("dot11.device", {}).get(
        "dot11.device.advertised_ssid_map", {}
    )
    ssid = None
    encryption = None
    for entry in ssid_map.values():
        ssid = entry.get("dot11.advertisedssid.ssid")
        encryption = entry.get("dot11.advertisedssid.crypt_string")
        break

    last_seen = device.get("kismet.device.base.last_time")
    timestamp = (
        datetime.fromtimestamp(last_seen, tz=timezone.utc)
        if last_seen
        else datetime.now(timezone.utc)
    )

    return UnifiedRecord(
        timestamp=timestamp,
        lat=0.0,
        lon=0.0,
        gps_fix_quality=None,
        survey_id=survey_id,
        operator_id=operator_id,
        modality=Modality.WIFI,
        identifier=Identifier(
            bssid=device.get("kismet.device.base.macaddr"),
            ssid=ssid,
            channel=device.get("kismet.device.base.channel"),
        ),
        signal=Signal(rssi=float(signal_dbm)),
        metadata={"encryption_type_if_broadcast_visible": encryption} if encryption else {},
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/capture/wifi/test_normalizer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write `capture/wifi/kismet_client.py`**

```python
# capture/wifi/kismet_client.py
from __future__ import annotations

import requests


class KismetClient:
    """Thin client for polling Kismet's REST API for known 802.11 devices."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def get_wifi_devices(self) -> list[dict]:
        params = {"KISMET": self._api_key} if self._api_key else {}
        response = requests.get(
            f"{self._base_url}/phy/phy80211/devices/all_devices.json",
            params=params,
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
```

- [ ] **Step 6: Write `capture/wifi/service.py`**

```python
# capture/wifi/service.py
from __future__ import annotations

import time

from capture.common.emitter import RecordEmitter
from capture.wifi.kismet_client import KismetClient
from capture.wifi.normalizer import normalize_kismet_device


def run(
    kismet_base_url: str,
    socket_path: str,
    survey_id: str,
    operator_id: str,
    poll_interval_seconds: float = 2.0,
) -> None:
    client = KismetClient(kismet_base_url)
    with RecordEmitter(socket_path) as emitter:
        while True:
            for device in client.get_wifi_devices():
                record = normalize_kismet_device(device, survey_id, operator_id)
                if record is not None:
                    emitter.emit(record)
            time.sleep(poll_interval_seconds)
```

- [ ] **Step 7: Commit**

```bash
git add capture/wifi/kismet_client.py capture/wifi/normalizer.py capture/wifi/service.py \
        tests/capture/wifi/test_normalizer.py
git commit -m "feat: add WiFi capture via Kismet"
```

---

### Task 7: Bluetooth capture (bleak)

**Files:**
- Create: `capture/bluetooth/normalizer.py`
- Create: `capture/bluetooth/service.py`
- Test: `tests/capture/bluetooth/test_normalizer.py`

**Interfaces:**
- Consumes: `schema.records.{Identifier, Modality, Signal, UnifiedRecord}` (Task 2),
  `capture.common.emitter.RecordEmitter` (Task 3).
- Produces: `capture.bluetooth.normalizer.normalize_ble_advertisement(address: str,
  device_name: str | None, rssi: float, survey_id: str, operator_id: str) ->
  UnifiedRecord`; `capture.bluetooth.service.main(socket_path, survey_id, operator_id)
  -> None`.

Note: as with Kismet in Task 6, `service.py` is a thin `bleak`-dependent I/O wrapper
with no branching logic — it is not unit-tested here (BLE scanning requires real
hardware). `normalize_ble_advertisement` is the pure function carrying the actual
logic, and gets full TDD coverage.

- [ ] **Step 1: Write the failing test**

```python
# tests/capture/bluetooth/test_normalizer.py
from capture.bluetooth.normalizer import normalize_ble_advertisement
from schema.records import Modality


def test_normalize_ble_advertisement_maps_fields():
    record = normalize_ble_advertisement(
        address="11:22:33:44:55:66",
        device_name="Wearable-42",
        rssi=-62.0,
        survey_id="s1",
        operator_id="op1",
    )
    assert record.modality is Modality.BLUETOOTH
    assert record.identifier.bt_mac == "11:22:33:44:55:66"
    assert record.identifier.device_name == "Wearable-42"
    assert record.signal.rssi == -62.0
    assert record.gps_fix_quality is None


def test_normalize_ble_advertisement_allows_missing_name():
    record = normalize_ble_advertisement(
        address="11:22:33:44:55:66",
        device_name=None,
        rssi=-70.0,
        survey_id="s1",
        operator_id="op1",
    )
    assert record.identifier.device_name is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/capture/bluetooth/test_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `capture/bluetooth/normalizer.py`**

```python
# capture/bluetooth/normalizer.py
from __future__ import annotations

from datetime import datetime, timezone

from schema.records import Identifier, Modality, Signal, UnifiedRecord


def normalize_ble_advertisement(
    address: str,
    device_name: str | None,
    rssi: float,
    survey_id: str,
    operator_id: str,
) -> UnifiedRecord:
    """lat/lon are left at 0.0 with gps_fix_quality=None — ingest.service
    attaches the real fix, since BLE capture has no GPS access of its own."""
    return UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=0.0,
        lon=0.0,
        gps_fix_quality=None,
        survey_id=survey_id,
        operator_id=operator_id,
        modality=Modality.BLUETOOTH,
        identifier=Identifier(bt_mac=address, device_name=device_name),
        signal=Signal(rssi=rssi),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/capture/bluetooth/test_normalizer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write `capture/bluetooth/service.py`**

```python
# capture/bluetooth/service.py
from __future__ import annotations

import asyncio

from bleak import BleakScanner

from capture.bluetooth.normalizer import normalize_ble_advertisement
from capture.common.emitter import RecordEmitter


async def run(
    socket_path: str, survey_id: str, operator_id: str, scan_seconds: float = 5.0
) -> None:
    with RecordEmitter(socket_path) as emitter:
        while True:
            devices = await BleakScanner.discover(timeout=scan_seconds, return_adv=True)
            for device, adv in devices.values():
                record = normalize_ble_advertisement(
                    address=device.address,
                    device_name=device.name,
                    rssi=adv.rssi,
                    survey_id=survey_id,
                    operator_id=operator_id,
                )
                emitter.emit(record)


def main(socket_path: str, survey_id: str, operator_id: str) -> None:
    asyncio.run(run(socket_path, survey_id, operator_id))
```

- [ ] **Step 6: Commit**

```bash
git add capture/bluetooth/normalizer.py capture/bluetooth/service.py \
        tests/capture/bluetooth/test_normalizer.py
git commit -m "feat: add Bluetooth capture via bleak"
```

---

### Task 8: Field-verification viz dashboard

**Files:**
- Create: `viz/app.py`
- Test: `tests/viz/test_app.py`

**Interfaces:**
- Consumes: `storage.models.SurveyRecord`, `storage.db.*`,
  `storage.repository.save_record` (Task 4).
- Produces: `viz.app.create_app(session_factory: sessionmaker) -> FastAPI` — exposes
  `GET /api/records?modality=&limit=` (JSON list of records) and `GET /` (Leaflet map
  page that fetches `/api/records`).

- [ ] **Step 1: Write the failing test**

```python
# tests/viz/test_app.py
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_engine, make_session_factory
from storage.repository import save_record
from viz.app import create_app


def test_list_records_returns_seeded_record():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    record = UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=1.0,
        lon=2.0,
        survey_id="s",
        operator_id="o",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
        signal=Signal(rssi=-40.0),
    )
    with session_factory() as session:
        save_record(session, record)

    app = create_app(session_factory)
    client = TestClient(app)

    response = client.get("/api/records")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["identifier"]["bssid"] == "AA:BB:CC:DD:EE:FF"


def test_list_records_filters_by_modality():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = make_session_factory(engine)

    wifi_record = UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=1.0,
        lon=2.0,
        survey_id="s",
        operator_id="o",
        modality=Modality.WIFI,
        identifier=Identifier(bssid="AA:BB:CC:DD:EE:FF"),
        signal=Signal(rssi=-40.0),
    )
    bt_record = UnifiedRecord(
        timestamp=datetime.now(timezone.utc),
        lat=1.0,
        lon=2.0,
        survey_id="s",
        operator_id="o",
        modality=Modality.BLUETOOTH,
        identifier=Identifier(bt_mac="11:22:33:44:55:66"),
        signal=Signal(rssi=-60.0),
    )
    with session_factory() as session:
        save_record(session, wifi_record)
        save_record(session, bt_record)

    app = create_app(session_factory)
    client = TestClient(app)

    response = client.get("/api/records", params={"modality": "bluetooth"})
    body = response.json()
    assert len(body) == 1
    assert body[0]["modality"] == "bluetooth"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/viz/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `viz/app.py`**

```python
# viz/app.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from storage.models import SurveyRecord

_MAP_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Survey Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>#map { height: 100vh; }</style>
</head>
<body>
  <div id="map"></div>
  <script>
    const map = L.map('map').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    fetch('/api/records').then(r => r.json()).then(records => {
      records.forEach(r => {
        L.circleMarker([r.lat, r.lon], {radius: 4})
          .bindPopup(JSON.stringify(r.identifier))
          .addTo(map);
      });
    });
  </script>
</body>
</html>"""


def create_app(session_factory: sessionmaker) -> FastAPI:
    app = FastAPI()

    @app.get("/api/records")
    def list_records(modality: str | None = None, limit: int = 500) -> list[dict]:
        with session_factory() as session:
            stmt = select(SurveyRecord).order_by(SurveyRecord.id.desc()).limit(limit)
            if modality:
                stmt = stmt.where(SurveyRecord.modality == modality)
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "lat": row.lat,
                    "lon": row.lon,
                    "modality": row.modality,
                    "signal": row.signal,
                    "identifier": row.identifier,
                }
                for row in rows
            ]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _MAP_HTML

    return app
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/viz/test_app.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests across all tasks — smoke, schema, emitter, queue_server,
storage, grid, service, wifi normalizer, bluetooth normalizer, viz app)

- [ ] **Step 6: Commit**

```bash
git add viz/app.py tests/viz/test_app.py
git commit -m "feat: add field-verification map dashboard"
```

---

## What this plan deliberately does not cover

- **Cellular capture** (LTE-Cell-Scanner port) — separate plan, gated on the hardware
  spike described in the design doc §10 steps 1 and 3.
- **Unknown-signal capture** (GNU Radio/gr-soapy flowgraph) — separate plan, per design
  doc §10 step 4.
- **Part 4 agent** (signal characterization) — separate plan, depends on unknown-signal
  records existing first, per design doc §10 step 5.
- **PostGIS geometry column / geospatial queries** — add when a task actually needs
  coverage-gap polygon queries; `docker-compose.yml` already provisions PostGIS so this
  is additive, not a migration.
- **Kismet's exact BT device JSON field mapping** — this plan uses bleak directly for
  Bluetooth (per the design doc's recommendation) and does not integrate Kismet's BT
  PHY at all.
