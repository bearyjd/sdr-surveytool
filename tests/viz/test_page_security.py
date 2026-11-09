import hashlib
import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.viz.test_auth import TOKEN, _session_factory
from viz.app import create_app

STATIC = Path(__file__).resolve().parents[2] / "viz" / "static"
PUBLIC = ("/", "/static/map.js", "/static/leaflet.js", "/static/leaflet.css")
HOSTILE = [
    "<img src=x onerror=alert(1)>",
    "</script><script>alert(1)</script>",
    '"><svg onload=alert(1)>',
]


def _client() -> TestClient:
    return TestClient(create_app(_session_factory(), api_token=TOKEN))


def test_map_script_never_turns_data_into_html():
    js = (STATIC / "map.js").read_text()
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "divIcon", "bindTooltip", "eval(", "new Function"):
        assert banned not in js, banned
    # Every bindPopup argument must be a DOM node built by popupFor, never a string.
    assert re.findall(r"bindPopup\(([^)]*\)?)\)", js) == ["popupFor(r)"]
    assert "JSON.stringify" not in js
    assert "textContent" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_map_script_renders_hostile_records_as_text_only():
    harness = r"""
const fs = require('fs');
const made = [];
function el(tag) {
  const e = {tag, children: [], style: {}, _text: '', appendChild(c) { this.children.push(c); },
    addEventListener() {}, set textContent(v) { this._text = v; }, get textContent() { return this._text; }};
  made.push(e);
  return e;
}
const popups = [];
global.document = {createElement: el, getElementById: () => el('x')};
global.sessionStorage = {getItem: () => 'tok', setItem() {}, removeItem() {}};
const marker = {bindPopup(c) { popups.push(c); return this; }, addTo() { return this; }};
global.L = {map: () => ({setView() { return this; }}), tileLayer: () => ({addTo() {}}), circleMarker: () => marker};
const records = JSON.parse(process.argv[2]);
global.fetch = () => Promise.resolve({status: 200, ok: true, json: () => Promise.resolve(records)});
eval(fs.readFileSync(process.argv[1], 'utf8'));
setTimeout(() => {
  const flat = c => [c._text].concat(c.children.flatMap(flat));
  console.log(JSON.stringify({n: popups.length, types: popups.map(p => typeof p), text: popups.map(flat)}));
}, 50);
"""
    records = [
        {"lat": 1, "lon": 2, "modality": "wifi", "signal": {}, "identifier": {"ssid": h, "bssid": h}}
        for h in HOSTILE
    ]
    out = subprocess.run(
        ["node", "-e", harness, str(STATIC / "map.js"), json.dumps(records)],
        capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["n"] == 3
    assert set(result["types"]) == {"object"}
    joined = " ".join(" ".join(t) for t in result["text"])
    for h in HOSTILE:
        assert h in joined  # shown verbatim, as text


def test_page_loads_only_same_origin_scripts_and_no_inline_script():
    html = _client().get("/").text
    scripts = re.findall(r"<script\b[^>]*>", html)
    assert scripts and all('src="/static/' in s for s in scripts)
    assert not re.search(r"<script\b(?![^>]*\bsrc=)[^>]*>", html)
    assert not re.findall(r'(?:src|href)="https?://', html)


def test_vendored_leaflet_matches_the_pinned_release():
    # sha256 values published for leaflet@1.9.4 dist files.
    expected = {
        "leaflet.js": "20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=",
        "leaflet.css": "p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=",
    }
    for name, digest in expected.items():
        actual = base64.b64encode(hashlib.sha256((STATIC / name).read_bytes()).digest()).decode()
        assert actual == digest, name


def test_public_static_routes_are_served_with_the_right_type():
    client = _client()
    assert client.get("/static/map.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/leaflet.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/leaflet.css").headers["content-type"].startswith("text/css")
    assert client.get("/static/other.js").status_code in (401, 404)
    assert client.get("/static/../app.py").status_code in (401, 404)


def test_only_the_exact_shell_routes_are_public():
    client = _client()
    public = set()
    for path in sorted({r.path for r in client.app.routes if hasattr(r, "methods")}):
        if client.get(path).status_code != 401:
            public.add(path)
    assert public == set(PUBLIC)


def test_security_headers_on_every_response_including_errors():
    client = _client()
    for path in (*PUBLIC, "/api/records", "/nope"):
        r = client.get(path)
        csp = r.headers["content-security-policy"]
        assert "script-src 'self'" in csp
        script_src = re.search(r"script-src ([^;]*)", csp).group(1)
        assert "unsafe-inline" not in script_src and "unpkg" not in script_src
        for directive in ("default-src 'self'", "connect-src 'self'", "frame-ancestors 'none'",
                          "base-uri 'none'", "form-action 'none'", "https://*.tile.openstreetmap.org"):
            assert directive in csp, directive
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["x-content-type-options"] == "nosniff"
    ok = client.get("/api/records", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.headers["cache-control"] == "no-store"
    assert client.get("/api/records").headers["cache-control"] == "no-store"
