# viz

Lightweight local web dashboard (live map + signal heatmap) reading from
Postgres/PostGIS, for field verification only. Separate from the eventual
carrier-facing product dashboard.

**Status**: not yet implemented.

## Running and authentication

`sdr-viz` (or `python -m viz.app`) serves the map and `/api/records`. Every route
requires `Authorization: Bearer <token>`; there is no unauthenticated mode and the
server refuses to start without a token.

- `SURVEYTOOL_API_TOKEN` (or the systemd credential `api_token`): at least 32
  characters, e.g. `openssl rand -hex 32`.
- Binds `127.0.0.1:8000` by default (`--host`/`--port`); the connection is plain HTTP,
  so put TLS in front before binding anything else.
- `SURVEYTOOL_DATABASE_URL` selects the database (default `sqlite:///survey.db`).
- `/api/records` takes `limit` (default 100, max 1000) and `offset`.
- `/` serves the map shell without authentication: it is a static page with no survey
  data. It asks for the token in a password field, keeps it in `sessionStorage` only
  (cleared when the tab closes, never in the URL or `localStorage`), and sends it as
  `Authorization: Bearer` on every fetch. A 401 re-prompts. Every other route stays closed.
- Tokens need 32+ characters with 16+ distinct ones; `offset` is capped at 2**31-1.
- The first `Authorization` header wins: behind a reverse proxy, make sure it sets
  exactly one. The token is a single shared secret sent over plain HTTP, so use TLS
  whenever the server is reachable beyond loopback.

## Page security

- Leaflet 1.9.4 (BSD-2-Clause, (c) Vladimir Agafonkin) is vendored in `viz/static/`
  and served by the app; the page loads no third-party script. Tests pin its sha256.
- `viz/static/map.js` renders record fields (SSIDs, BLE names, ...) as DOM text only;
  never `innerHTML`/string popups. A test scans for this and a node-based test renders
  hostile records.
- Every response carries a CSP (no `unsafe-inline` for scripts), `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, `nosniff` and `Cache-Control: no-store`.
- Public paths are exactly `/` and `/static/{map.js,leaflet.js,leaflet.css}`.
