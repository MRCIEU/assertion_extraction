"""Minimal OncoKB REST client with Bearer auth from environment."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from .config import ONCOKB_BASE_URL, REQUEST_PAUSE, REQUEST_TIMEOUT


class OncoKBClient:
    """Authenticated OncoKB API client."""

    def __init__(self, base_url: str = ONCOKB_BASE_URL, pause_seconds: float = REQUEST_PAUSE):
        self.base_url = base_url.rstrip("/")
        self.pause_seconds = pause_seconds
        self.session = requests.Session()
        self.token = self._load_token()
        self.session.headers.update(
            {
                "accept": "application/json",
                "Authorization": f"Bearer {self.token}",
            }
        )

    @staticmethod
    def _load_token() -> str:
        for name in ("ONCOKB_API_TOKEN", "ONCOKB_TOKEN"):
            value = os.environ.get(name, "").strip()
            if value:
                return value
        raise RuntimeError(
            "OncoKB API token not found. Export ONCOKB_API_TOKEN (or ONCOKB_TOKEN) before running step 06."
        )

    @property
    def access_mode(self) -> str:
        return "Bearer token via Authorization header (token from environment; value not logged)"

    def get(self, path: str, params: dict[str, Any] | None = None) -> tuple[int, Any, str]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if response.headers.get("content-type", "").startswith("application/json"):
                    body = response.json()
                else:
                    body = response.text
                time.sleep(self.pause_seconds)
                return response.status_code, body, url
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise last_error or RuntimeError(f"GET failed for {url}")

    def post_json(self, path: str, payload: list[dict[str, Any]] | dict[str, Any]) -> tuple[int, Any, str]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        headers = dict(self.session.headers)
        headers["Content-Type"] = "application/json"
        for attempt in range(5):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
                body = response.json() if response.content else None
                time.sleep(self.pause_seconds)
                return response.status_code, body, url
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise last_error or RuntimeError(f"POST failed for {url}")
