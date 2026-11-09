from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

import uvicorn
from fastapi import Depends, FastAPI, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ingest.main import DATABASE_URL_ENV
from storage.db import make_engine, make_session_factory
from storage.models import SurveyRecord
from viz.auth import ApiTokenError, bearer_dependency, load_api_token, validate_api_token

logger = logging.getLogger(__name__)

# The map shell is a static string with no survey data; its script fetches
# /api/records with the operator-entered token. Every other route stays closed.
_PUBLIC_PATHS = frozenset({"/"})
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000
_DEFAULT_DATABASE_URL = "sqlite:///survey.db"

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
  <form id="login" style="display:none;position:absolute;z-index:1000;background:#fff;padding:8px;left:50px;top:10px">
    <label>API token <input id="token" type="password" autocomplete="off"></label>
    <button type="submit">Load</button>
    <div id="msg" role="alert"></div>
  </form>
  <div id="map"></div>
  <script>
    const map = L.map('map').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    const KEY = 'surveytoken';
    const msg = document.getElementById('msg');
    function load(token) {
      fetch('/api/records', {headers: {'Authorization': 'Bearer ' + token}}).then(r => {
        if (r.status === 401) {
          sessionStorage.removeItem(KEY);
          msg.textContent = 'Invalid or missing token. Enter the API token.';
          document.getElementById('login').style.display = 'block';
          return [];
        }
        if (!r.ok) { msg.textContent = 'Request failed (' + r.status + ').'; return []; }
        document.getElementById('login').style.display = 'none';
        msg.textContent = '';
        return r.json();
      }).then(records => {
        records.forEach(r => {
          L.circleMarker([r.lat, r.lon], {radius: 4})
            .bindPopup(JSON.stringify(r.identifier))
            .addTo(map);
        });
      });
    }
    document.getElementById('login').addEventListener('submit', e => {
      e.preventDefault();
      const input = document.getElementById('token');
      sessionStorage.setItem(KEY, input.value);
      input.value = '';
      load(sessionStorage.getItem(KEY));
    });
    const saved = sessionStorage.getItem(KEY);
    if (saved) { load(saved); } else { document.getElementById('login').style.display = 'block'; }
  </script>
</body>
</html>"""


def create_app(session_factory: sessionmaker, api_token: str | None = None) -> FastAPI:
    """Every route except the static map shell sits behind the bearer token through an app-level
    dependency, so a route added later is protected by default. Raises
    ApiTokenError when no strong token is configured. The generated
    /docs, /redoc and /openapi.json routes would bypass app-level
    dependencies, so they are turned off."""
    token = load_api_token() if api_token is None else validate_api_token(api_token)
    app = FastAPI(
        dependencies=[Depends(bearer_dependency(token, _PUBLIC_PATHS))],
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/api/records")
    def list_records(
        modality: str | None = None,
        # Bounded so a caller can neither materialize the whole table
        # (?limit=999999999) nor pass a negative value, which SQLite treats as
        # "no limit" at all. Out-of-range values are rejected by FastAPI with 422.
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0, le=2**31 - 1),
    ) -> list[dict]:
        with session_factory() as session:
            stmt = (
                select(SurveyRecord)
                .order_by(SurveyRecord.id.desc())
                .limit(limit)
                .offset(offset)
            )
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Survey map and records API (bearer-token protected).")
    parser.add_argument("--host", default=_DEFAULT_HOST, help=f"bind address (default: {_DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args(argv)
    try:
        token = load_api_token()
    except ApiTokenError as exc:
        parser.error(str(exc))
    engine = make_engine(os.environ.get(DATABASE_URL_ENV) or _DEFAULT_DATABASE_URL)
    app = create_app(make_session_factory(engine), api_token=token)
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        logging.basicConfig()
        logger.warning("binding %s: the API is reachable beyond this machine, and plain HTTP exposes the token", args.host)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main(sys.argv[1:])
