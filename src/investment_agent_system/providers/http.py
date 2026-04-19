from __future__ import annotations

import logging
from typing import Any, Optional, cast

import httpx

from investment_agent_system.config import get_settings

logger = logging.getLogger(__name__)


def build_http_client(headers: Optional[dict[str, str]] = None) -> httpx.AsyncClient:
    settings = get_settings()
    base_headers = {"Accept": "application/json, text/plain, */*"}
    if headers:
        base_headers.update(headers)
    return httpx.AsyncClient(
        timeout=settings.request_timeout_seconds,
        headers=base_headers,
        follow_redirects=True,
    )


async def get_json(
    client: httpx.AsyncClient, url: str, params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return cast(dict[str, Any], response.json())


async def get_text(
    client: httpx.AsyncClient, url: str, params: Optional[dict[str, Any]] = None
) -> str:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.text
