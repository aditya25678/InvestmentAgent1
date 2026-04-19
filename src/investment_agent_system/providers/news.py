from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote_plus

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from investment_agent_system.config import get_settings
from investment_agent_system.providers.base import ProviderResult, ResearchProvider
from investment_agent_system.providers.http import build_http_client, get_json

logger = logging.getLogger(__name__)


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None


class NewsProvider(ResearchProvider):
    source_type = "news"
    source_name = "News Feed"

    async def fetch(self, ticker: str) -> list[ProviderResult]:
        settings = get_settings()
        combined: list[ProviderResult] = []

        if settings.newsapi_key:
            try:
                combined.extend(await self._fetch_newsapi(ticker, settings.newsapi_key))
            except Exception as exc:
                logger.warning("NewsAPI failed, using RSS fallback: %s", exc)

        try:
            combined.extend(await self._fetch_web_news(ticker))
        except Exception as exc:
            logger.warning("Web-search news fallback failed: %s", exc)

        if len(combined) < max(5, settings.max_news_articles // 2):
            combined.extend(await self._fetch_rss(ticker))

        deduped = self._dedupe_news(combined)
        if not deduped:
            return []
        deduped = deduped[: settings.max_news_articles]
        deduped.append(self._build_sentiment_summary(ticker=ticker, news_rows=deduped))
        return deduped

    async def _fetch_newsapi(self, ticker: str, api_key: str) -> list[ProviderResult]:
        settings = get_settings()
        query = (
            f"{ticker} stock OR {ticker} earnings OR {ticker} guidance OR "
            f"{ticker} margin OR {ticker} regulation OR {ticker} demand"
        )
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": settings.max_news_articles,
            "apiKey": api_key,
        }
        async with build_http_client() as client:
            payload = await get_json(client, url=url, params=params)

        articles = payload.get("articles", [])
        analyzer = SentimentIntensityAnalyzer()
        results: list[ProviderResult] = []
        for article in articles:
            title = article.get("title", "").strip()
            description = article.get("description", "").strip()
            if not title:
                continue
            combined = f"{title}. {description}".strip()
            score = analyzer.polarity_scores(combined)["compound"]
            results.append(
                ProviderResult(
                    source_type=self.source_type,
                    source_name="NewsAPI",
                    title=title[:500],
                    url=article.get("url", ""),
                    content=combined[:2500],
                    published_at=_parse_date(article.get("publishedAt")),
                    metadata={
                        "sentiment_compound": score,
                        "source": article.get("source", {}).get("name"),
                    },
                )
            )
        if not results:
            return []
        return results

    async def _fetch_rss(self, ticker: str) -> list[ProviderResult]:
        settings = get_settings()
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US"
        parsed = feedparser.parse(url)
        analyzer = SentimentIntensityAnalyzer()

        results: list[ProviderResult] = []
        for entry in parsed.entries[: settings.max_news_articles]:
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            link = getattr(entry, "link", "")
            published = getattr(entry, "published", None)
            content = f"{title}. {summary}".strip()
            score = analyzer.polarity_scores(content)["compound"]
            results.append(
                ProviderResult(
                    source_type=self.source_type,
                    source_name="Yahoo Finance RSS",
                    title=title[:500],
                    url=link,
                    content=content[:2500],
                    published_at=_parse_date(published),
                    metadata={"sentiment_compound": score},
                )
            )
        return results

    async def _fetch_web_news(self, ticker: str) -> list[ProviderResult]:
        settings = get_settings()
        analyzer = SentimentIntensityAnalyzer()
        results: list[ProviderResult] = []
        queries = [
            f"{ticker} stock news",
            f"{ticker} earnings preview",
            f"{ticker} valuation debate",
            f"{ticker} legal risk",
            f"{ticker} supply chain",
            f"{ticker} analyst revisions",
        ]
        per_query = max(4, settings.max_news_articles // max(1, len(queries)))
        for raw_query in queries:
            query = quote_plus(raw_query)
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:per_query]:
                title = str(getattr(entry, "title", "")).strip()
                body = str(getattr(entry, "summary", "")).strip()
                link = str(getattr(entry, "link", "")).strip()
                published = str(getattr(entry, "published", "")).strip()
                if not title or not link:
                    continue
                text = f"{title}. {body}".strip()
                score = analyzer.polarity_scores(text)["compound"]
                results.append(
                    ProviderResult(
                        source_type=self.source_type,
                        source_name="Google News RSS Search",
                        title=title[:500],
                        url=link,
                        content=text[:2500],
                        published_at=_parse_date(published),
                        metadata={"sentiment_compound": score, "query": raw_query},
                    )
                )
        return results

    def _dedupe_news(self, rows: list[ProviderResult]) -> list[ProviderResult]:
        dedup: dict[str, ProviderResult] = {}
        for row in rows:
            dedup[row.url] = row
        return list(dedup.values())

    def _build_sentiment_summary(
        self, ticker: str, news_rows: list[ProviderResult]
    ) -> ProviderResult:
        aggregate = round(
            sum(float(item.metadata.get("sentiment_compound", 0.0)) for item in news_rows)
            / len(news_rows),
            4,
        )
        return ProviderResult(
            source_type="sentiment",
            source_name="VADER",
            title=f"{ticker} aggregate headline sentiment",
            url=f"https://news.google.com/search?q={quote_plus(ticker + ' stock news')}",
            content=f"Aggregate headline sentiment compound score: {aggregate}",
            published_at=datetime.now(timezone.utc),
            metadata={"aggregate_sentiment_compound": aggregate},
        )
