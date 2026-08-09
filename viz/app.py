from __future__ import annotations

from fastapi import FastAPI, Query
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
    def list_records(
        modality: str | None = None,
        # Bounded so a caller can neither materialize the whole table
        # (?limit=999999999) nor pass a negative value, which SQLite treats as
        # "no limit" at all. Out-of-range values are rejected by FastAPI with 422.
        limit: int = Query(500, ge=1, le=5000),
    ) -> list[dict]:
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
