from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)


def _is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


class KismetClient:
    """Thin client for polling Kismet's REST API for known 802.11 devices."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        url = urlsplit(self._base_url)
        if api_key and url.scheme.lower() == "http" and not _is_loopback(url.hostname):
            logger.warning(
                "Kismet API key will be sent over cleartext http to %s; "
                "use an https:// Kismet URL for non-localhost hosts",
                url.hostname,
            )

    def get_wifi_devices(self) -> list[dict]:
        headers = {"Cookie": f"KISMET={self._api_key}"} if self._api_key else {}
        response = requests.get(
            f"{self._base_url}/phy/phy80211/devices/all_devices.json",
            headers=headers,
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
