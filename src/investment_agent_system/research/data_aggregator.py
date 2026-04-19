from __future__ import annotations

import asyncio
import logging

from investment_agent_system.models.schemas import EvidenceItem
from investment_agent_system.providers.base import ProviderResult, ResearchProvider
from investment_agent_system.providers.fmp import FMPProvider
from investment_agent_system.providers.macro import MacroProvider
from investment_agent_system.providers.market import MarketDataProvider
from investment_agent_system.providers.news import NewsProvider
from investment_agent_system.providers.sec import SECProvider
from investment_agent_system.providers.web_search import WebSearchProvider
from investment_agent_system.storage.repository import ResearchRepository

logger = logging.getLogger(__name__)


class ResearchDataAggregator:
    def __init__(self, repository: ResearchRepository):
        self.repository = repository
        self.providers: list[ResearchProvider] = [
            MarketDataProvider(),
            SECProvider(),
            NewsProvider(),
            MacroProvider(),
            WebSearchProvider(),
            FMPProvider(),
        ]

    async def collect_and_persist(self, run_id: str, ticker: str) -> list[EvidenceItem]:
        tasks = [self._safe_fetch(provider, ticker) for provider in self.providers]
        batches = await asyncio.gather(*tasks)
        flat_results = [item for batch in batches for item in batch]
        deduped = self._dedupe(flat_results)
        evidence_items: list[EvidenceItem] = []
        for item in deduped:
            evidence_id = self.repository.save_evidence(
                run_id=run_id,
                source_type=item.source_type,
                source_name=item.source_name,
                title=item.title,
                url=item.url,
                content=item.content,
                published_at=item.published_at,
                metadata_json=item.metadata,
            )
            evidence_items.append(
                EvidenceItem(
                    id=evidence_id,
                    source_type=item.source_type,
                    source_name=item.source_name,
                    title=item.title,
                    url=item.url,
                    content=item.content,
                    published_at=item.published_at,
                    metadata=item.metadata,
                )
            )
        return evidence_items

    async def _safe_fetch(self, provider: ResearchProvider, ticker: str) -> list[ProviderResult]:
        try:
            return await provider.fetch(ticker)
        except Exception as exc:
            logger.warning("Provider %s failed: %s", provider.source_name, exc)
            return []

    def _dedupe(self, items: list[ProviderResult]) -> list[ProviderResult]:
        seen: dict[str, ProviderResult] = {}
        for item in items:
            key = f"{item.source_type}:{item.url}:{item.title[:120]}"
            seen[key] = item
        return list(seen.values())
