"""Minimal GraphQL client for the CIViC API."""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import CIVIC_GRAPHQL_URL, REQUEST_TIMEOUT


class CivicGraphQLClient:
    """Paginated GraphQL client with basic retry logic."""

    def __init__(self, url: str = CIVIC_GRAPHQL_URL, pause_seconds: float = 0.15):
        self.url = url
        self.pause_seconds = pause_seconds
        self.session = requests.Session()

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"query": query, "variables": variables or {}}
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                body = response.json()
                if "errors" not in body:
                    return body["data"]
                message = body["errors"][0].get("message", "Unknown GraphQL error")
                last_error = RuntimeError(f"GraphQL error: {message}")
            except (requests.HTTPError, requests.RequestException) as exc:
                last_error = exc
            time.sleep(1.5 * (attempt + 1))
        raise last_error or RuntimeError("GraphQL request failed")

    def paginate_connection(
        self,
        query: str,
        connection_path: list[str],
        variables: dict[str, Any],
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Fetch all nodes from a Relay-style connection."""
        all_nodes: list[dict[str, Any]] = []
        total_count = 0
        after = None

        while True:
            page_vars = dict(variables)
            page_vars["first"] = page_size
            page_vars["after"] = after
            data = self.execute(query, page_vars)
            connection = data
            for key in connection_path:
                connection = connection[key]

            nodes = connection.get("nodes") or []
            all_nodes.extend(nodes)
            total_count = connection.get("totalCount", len(all_nodes))

            page_info = connection.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")
            time.sleep(self.pause_seconds)

        return all_nodes, total_count
