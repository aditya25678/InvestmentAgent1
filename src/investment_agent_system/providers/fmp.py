from __future__ import annotations

from datetime import datetime, timezone

from investment_agent_system.config import get_settings
from investment_agent_system.providers.base import ProviderResult, ResearchProvider
from investment_agent_system.providers.http import build_http_client, get_json


class FMPProvider(ResearchProvider):
    source_type = "estimates"
    source_name = "Financial Modeling Prep"

    async def fetch(self, ticker: str) -> list[ProviderResult]:
        settings = get_settings()
        if not settings.fmp_api_key:
            return []

        base = "https://financialmodelingprep.com/api/v3"
        params = {"apikey": settings.fmp_api_key}
        async with build_http_client() as client:
            estimates = await get_json(client, f"{base}/analyst-estimates/{ticker}", params=params)
            earnings_surprises = await get_json(
                client, f"{base}/earnings-surprises/{ticker}", params=params
            )

        rows: list[ProviderResult] = []
        if isinstance(estimates, list) and estimates:
            top = estimates[:6]
            content = " ; ".join(
                [
                    (
                        "period={period}, estimatedEpsAvg={estimatedEpsAvg}, "
                        "estimatedRevenueAvg={estimatedRevenueAvg}"
                    ).format(
                        period=item.get("date"),
                        estimatedEpsAvg=item.get("estimatedEpsAvg"),
                        estimatedRevenueAvg=item.get("estimatedRevenueAvg"),
                    )
                    for item in top
                ]
            )
            rows.append(
                ProviderResult(
                    source_type=self.source_type,
                    source_name=self.source_name,
                    title=f"{ticker} analyst estimates trend",
                    url="https://site.financialmodelingprep.com/developer/docs/stable/analyst-estimates",
                    content=content,
                    published_at=datetime.now(timezone.utc),
                    metadata={"rows": top},
                )
            )

        if isinstance(earnings_surprises, list) and earnings_surprises:
            top_surprises = earnings_surprises[:8]
            content = " ; ".join(
                [
                    f"date={item.get('date')}, actual={item.get('actualEarningResult')}, "
                    f"estimated={item.get('estimatedEarning')}, "
                    f"surprise={item.get('earningSurprise')}"
                    for item in top_surprises
                ]
            )
            rows.append(
                ProviderResult(
                    source_type=self.source_type,
                    source_name=self.source_name,
                    title=f"{ticker} earnings surprise history",
                    url="https://site.financialmodelingprep.com/developer/docs/stable/earnings-surprises",
                    content=content,
                    published_at=datetime.now(timezone.utc),
                    metadata={"rows": top_surprises},
                )
            )
        return rows
