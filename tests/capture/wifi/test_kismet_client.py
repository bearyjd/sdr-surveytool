from __future__ import annotations

from unittest.mock import MagicMock, patch

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
