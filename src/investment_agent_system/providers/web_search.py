from __future__ import annotations

import asyncio
import importlib
import warnings
from datetime import datetime, timezone
from typing import Any

from investment_agent_system.config import get_settings
from investment_agent_system.providers.base import ProviderResult, ResearchProvider

warnings.filterwarnings(
    "ignore",
    message=r".*renamed to `ddgs`!.*",
    category=RuntimeWarning,
)


class WebSearchProvider(ResearchProvider):
    source_type = "web_search"
    source_name = "DuckDuckGo Search"

    async def fetch(self, ticker: str) -> list[ProviderResult]:
        return await asyncio.to_thread(self._fetch_sync, ticker)

    def _fetch_sync(self, ticker: str) -> list[ProviderResult]:
        settings = get_settings()
        queries = [
            f"{ticker} earnings guidance",
            f"{ticker} competitive risks",
            f"{ticker} valuation analysis",
            f"{ticker} regulation legal issues",
            f"{ticker} management commentary strategy",
            f"{ticker} customer demand trends",
        ]
        rows: list[ProviderResult] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ddgs_client = _resolve_ddgs_client()

            with ddgs_client() as ddgs:
                for query in queries:
                    items = list(
                        ddgs.text(
                            query, max_results=max(3, settings.max_web_results // len(queries))
                        )
                    )
                    for item in items:
                        title = str(item.get("title", "")).strip()
                        href = str(item.get("href", "")).strip()
                        body = str(item.get("body", "")).strip()
                        if not title or not href:
                            continue
                        rows.append(
                            ProviderResult(
                                source_type=self.source_type,
                                source_name=self.source_name,
                                title=title[:500],
                                url=href,
                                content=body[:2500],
                                published_at=datetime.now(timezone.utc),
                                metadata={"query": query},
                            )
                        )
        dedup: dict[str, ProviderResult] = {}
        for row in rows:
            dedup[row.url] = row
        return list(dedup.values())[: min(settings.max_web_results, 24)]


def _resolve_ddgs_client() -> Any:
    try:
        module = importlib.import_module("ddgs")
    except ImportError:
        module = importlib.import_module("duckduckgo_search")
    return module.DDGS
