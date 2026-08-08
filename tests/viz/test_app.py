from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from schema.records import Identifier, Modality, Signal, UnifiedRecord
from storage.db import init_db, make_session_factory
from storage.repository import save_record
from viz.app import create_app


def _make_memory_engine():
    # FastAPI's TestClient runs request handlers via a portal in a worker
    # thread distinct from the thread that seeds data below. SQLite's
    # `:memory:` database is per-connection, and storage.db.make_engine's
    # default pooling (SingletonThreadPool) ties a connection to the
    # thread that first uses it, so the request thread would otherwise see
    # an empty, tableless database. StaticPool + check_same_thread=False
    # shares a single connection across threads, which is safe here since
    # TestClient issues requests sequentially, not concurrently.
    return create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_list_records_returns_seeded_record():
    engine = _make_memory_engine()
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
    engine = _make_memory_engine()
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
