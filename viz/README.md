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
- The built-in map page cannot attach the header from a plain browser navigation; use
  a header-injecting client or reverse proxy until the page gains a token prompt.
