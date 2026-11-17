from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from capture.wifi.kismet_client import KismetClient


def _call(api_key):
    response = MagicMock()
    response.json.return_value = []
    with patch("capture.wifi.kismet_client.requests.get", return_value=response) as get:
        KismetClient("http://kismet:2501", api_key).get_wifi_devices()
    return get.call_args


def test_api_key_not_sent_in_url_params():
    call = _call("secret")
    assert not call.kwargs.get("params")
    assert "secret" not in call.args[0]


def test_api_key_sent_as_cookie_header():
    call = _call("secret")
    assert call.kwargs["headers"]["Cookie"] == "KISMET=secret"


def test_no_credentials_without_api_key():
    call = _call(None)
    assert not call.kwargs.get("headers")
    assert not call.kwargs.get("params")


@pytest.mark.parametrize(
    "base_url", ["http://kismet:2501", "http://192.168.1.20:2501", "HTTP://Kismet.lan"]
)
def test_warns_when_api_key_sent_over_remote_cleartext_http(base_url, caplog):
    with caplog.at_level(logging.WARNING, logger="capture.wifi.kismet_client"):
        KismetClient(base_url, "secret")
    assert "cleartext" in caplog.text
    assert "secret" not in caplog.text


@pytest.mark.parametrize(
    ("base_url", "api_key"),
    [
        ("https://kismet:2501", "secret"),
        ("http://localhost:2501", "secret"),
        ("http://127.0.0.1:2501", "secret"),
        ("http://[::1]:2501", "secret"),
        ("http://kismet:2501", None),
    ],
)
def test_no_warning_for_https_loopback_or_keyless(base_url, api_key, caplog):
    with caplog.at_level(logging.WARNING, logger="capture.wifi.kismet_client"):
        KismetClient(base_url, api_key)
    assert caplog.records == []
