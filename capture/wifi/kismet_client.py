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
