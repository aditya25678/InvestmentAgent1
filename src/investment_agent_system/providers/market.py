from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any, Optional

import httpx

from investment_agent_system.providers.base import ProviderResult, ResearchProvider
from investment_agent_system.providers.http import build_http_client, get_json, get_text


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annualized_vol(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return stdev(daily_returns) * math.sqrt(252)


def _max_drawdown(prices: list[float]) -> float:
    if not prices:
        return 0.0
    peak = prices[0]
    min_drawdown = 0.0
    for price in prices:
        peak = max(peak, price)
        drawdown = price / peak - 1.0
        min_drawdown = min(min_drawdown, drawdown)
    return min_drawdown


def _percent_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current / previous - 1.0) * 100.0


def _simple_moving_average(prices: list[float], window: int) -> Optional[float]:
    if len(prices) < window:
        return None
    subset = prices[-window:]
    return round(sum(subset) / len(subset), 4)


def _rsi(prices: list[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for idx in range(len(prices) - period, len(prices)):
        change = prices[idx] - prices[idx - 1]
        if change >= 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


class MarketDataProvider(ResearchProvider):
    source_type = "market_data"
    source_name = "Yahoo Finance API"

    async def fetch(self, ticker: str) -> list[ProviderResult]:
        chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        quote_url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        options_url = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}"

        try:
            async with build_http_client() as client:
                chart = await get_json(
                    client,
                    chart_url,
                    params={"range": "2y", "interval": "1d", "events": "div,splits"},
                )
                quote = await get_json(
                    client,
                    quote_url,
                    params={"modules": "price,defaultKeyStatistics,financialData,summaryDetail"},
                )
                options = await get_json(client, options_url)
            return self._build_results(
                ticker=ticker,
                chart_payload=chart,
                quote_payload=quote,
                options_payload=options,
            )
        except httpx.HTTPError:
            # Yahoo can rate-limit; fallback to Stooq for price history.
            return await self._fetch_from_stooq(ticker)

    async def _fetch_from_stooq(self, ticker: str) -> list[ProviderResult]:
        symbol = f"{ticker.lower()}.us"
        stooq_url = f"https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"
        async with build_http_client() as client:
            csv_text = await get_text(client, stooq_url)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        if not rows:
            return []
        last_row = rows[-1]
        last_price = _safe_float(last_row.get("Close"))
        if last_price is None:
            return []
        last_date_raw = last_row.get("Date")
        published_at = (
            datetime.strptime(last_date_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if last_date_raw
            else datetime.now(timezone.utc)
        )
        open_price = _safe_float(last_row.get("Open"))
        high_price = _safe_float(last_row.get("High"))
        low_price = _safe_float(last_row.get("Low"))
        volume = _safe_float(last_row.get("Volume"))
        summary = (
            f"Source=Stooq fallback; Close={last_price:.2f}; Open={open_price}; "
            f"High={high_price}; Low={low_price}; Volume={volume}. "
            "Historical series metrics unavailable from anonymous endpoint."
        )
        return [
            ProviderResult(
                source_type=self.source_type,
                source_name="Stooq",
                title=f"{ticker} market snapshot (fallback)",
                url=stooq_url,
                content=summary,
                published_at=published_at,
                metadata={
                    "last_price": last_price,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "volume": volume,
                    "historical_metrics_available": False,
                    "fallback": True,
                },
            )
        ]

    def _build_results(
        self, ticker: str, chart_payload: dict, quote_payload: dict, options_payload: dict
    ) -> list[ProviderResult]:
        chart_result = chart_payload.get("chart", {}).get("result", [])
        if not chart_result:
            return []
        row = chart_result[0]
        timestamps = row.get("timestamp", [])
        closes_raw = row.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        close_prices_raw = [_safe_float(v) for v in closes_raw]
        close_prices = [value for value in close_prices_raw if value is not None]
        if len(close_prices) < 30:
            return []

        last_price = close_prices[-1]
        one_month_price = close_prices[-21] if len(close_prices) > 21 else close_prices[0]
        three_month_price = close_prices[-63] if len(close_prices) > 63 else close_prices[0]
        six_month_price = close_prices[-126] if len(close_prices) > 126 else close_prices[0]
        one_year_price = close_prices[-252] if len(close_prices) > 252 else close_prices[0]

        daily_returns = [
            close_prices[i] / close_prices[i - 1] - 1.0
            for i in range(1, len(close_prices))
            if close_prices[i - 1]
        ]
        perf = {
            "return_1m": round(_percent_change(last_price, one_month_price), 2),
            "return_3m": round(_percent_change(last_price, three_month_price), 2),
            "return_6m": round(_percent_change(last_price, six_month_price), 2),
            "return_1y": round(_percent_change(last_price, one_year_price), 2),
        }
        vol = round(_annualized_vol(daily_returns) * 100.0, 2)
        max_dd = round(_max_drawdown(close_prices) * 100.0, 2)
        sma_20 = _simple_moving_average(close_prices, 20)
        sma_50 = _simple_moving_average(close_prices, 50)
        sma_200 = _simple_moving_average(close_prices, 200)
        rsi_14 = _rsi(close_prices, period=14)
        last_timestamp = timestamps[-1] if timestamps else None
        published_at = (
            datetime.fromtimestamp(int(last_timestamp), tz=timezone.utc)
            if last_timestamp
            else datetime.now(timezone.utc)
        )

        quote_result = quote_payload.get("quoteSummary", {}).get("result", [{}])[0]
        price = quote_result.get("price", {})
        stats = quote_result.get("defaultKeyStatistics", {})
        financial = quote_result.get("financialData", {})
        detail = quote_result.get("summaryDetail", {})

        valuation_fields = {
            "market_cap": price.get("marketCap", {}).get("raw"),
            "trailing_pe": stats.get("trailingPE", {}).get("raw"),
            "forward_pe": stats.get("forwardPE", {}).get("raw"),
            "price_to_book": stats.get("priceToBook", {}).get("raw"),
            "shares_outstanding": stats.get("sharesOutstanding", {}).get("raw"),
            "enterprise_value": stats.get("enterpriseValue", {}).get("raw"),
            "beta": stats.get("beta", {}).get("raw"),
            "52w_high": detail.get("fiftyTwoWeekHigh", {}).get("raw"),
            "52w_low": detail.get("fiftyTwoWeekLow", {}).get("raw"),
            "target_mean_price": financial.get("targetMeanPrice", {}).get("raw"),
        }

        summary = (
            f"Price={last_price:.2f}; Perf={perf}; "
            f"AnnualizedVol={vol}% ; MaxDrawdown={max_dd}% ; "
            f"SMA20={sma_20}; SMA50={sma_50}; SMA200={sma_200}; RSI14={rsi_14}; "
            f"Valuation={valuation_fields}"
        )
        rows = [
            ProviderResult(
                source_type=self.source_type,
                source_name=self.source_name,
                title=f"{ticker} market snapshot",
                url=f"https://finance.yahoo.com/quote/{ticker}",
                content=summary,
                published_at=published_at,
                metadata={
                    "last_price": last_price,
                    "performance_pct": perf,
                    "annualized_vol_pct": vol,
                    "max_drawdown_pct": max_dd,
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "rsi_14": rsi_14,
                    "valuation_fields": valuation_fields,
                },
            )
        ]

        option_result = options_payload.get("optionChain", {}).get("result", [])
        if option_result:
            option_row = option_result[0]
            expiries = option_row.get("expirationDates", [])
            options = option_row.get("options", [])
            nearest = expiries[0] if expiries else None
            if options:
                chain = options[0]
                calls = chain.get("calls", [])
                puts = chain.get("puts", [])
                call_oi = sum(_safe_float(c.get("openInterest")) or 0.0 for c in calls)
                put_oi = sum(_safe_float(p.get("openInterest")) or 0.0 for p in puts)
                call_iv_values: list[float] = []
                for call in calls:
                    iv = _safe_float(call.get("impliedVolatility"))
                    if iv is not None:
                        call_iv_values.append(iv)
                avg_call_iv = round(mean(call_iv_values), 4) if call_iv_values else None
                put_call_ratio = round(put_oi / call_oi, 3) if call_oi else None
                expiry_text = (
                    datetime.fromtimestamp(int(nearest), tz=timezone.utc).strftime("%Y-%m-%d")
                    if nearest
                    else "unknown"
                )
                options_content = (
                    f"NearestExpiry={expiry_text}; PutCallOI={put_call_ratio}; "
                    f"AvgCallIV={avg_call_iv}; CallOI={int(call_oi)}; PutOI={int(put_oi)}"
                )
                rows.append(
                    ProviderResult(
                        source_type=self.source_type,
                        source_name=self.source_name,
                        title=f"{ticker} options positioning snapshot",
                        url=f"https://finance.yahoo.com/quote/{ticker}/options",
                        content=options_content,
                        published_at=datetime.now(timezone.utc),
                        metadata={
                            "nearest_expiry_ts": nearest,
                            "put_call_oi_ratio": put_call_ratio,
                            "avg_call_iv": avg_call_iv,
                            "call_oi": call_oi,
                            "put_oi": put_oi,
                        },
                    )
                )
        return rows
