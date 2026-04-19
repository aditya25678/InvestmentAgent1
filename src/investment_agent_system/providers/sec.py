from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Union

from investment_agent_system.config import get_settings
from investment_agent_system.providers.base import ProviderResult, ResearchProvider
from investment_agent_system.providers.http import build_http_client, get_json

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _format_cik(cik: Union[int, str]) -> str:
    cik_int = int(cik)
    return f"{cik_int:010d}"


def _safe_parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return None


class SECProvider(ResearchProvider):
    source_type = "filings"
    source_name = "SEC EDGAR"

    async def fetch(self, ticker: str) -> list[ProviderResult]:
        settings = get_settings()
        headers = {"User-Agent": settings.sec_user_agent}
        async with build_http_client(headers=headers) as client:
            ticker_map = await get_json(client, SEC_TICKERS_URL)
            cik = self._lookup_cik(ticker_map, ticker)
            if cik is None:
                logger.warning("Ticker %s not found in SEC map", ticker)
                return []

            submissions = await get_json(client, SEC_SUBMISSIONS_URL.format(cik=_format_cik(cik)))
            company_facts = await get_json(
                client, SEC_COMPANY_FACTS_URL.format(cik=_format_cik(cik))
            )

        filing_results = self._extract_filing_results(
            ticker=ticker, cik=cik, submissions=submissions
        )
        facts_results = self._extract_company_facts(ticker=ticker, company_facts=company_facts)
        return filing_results + facts_results

    def _lookup_cik(self, ticker_map: dict, ticker: str) -> Optional[int]:
        normalized = ticker.upper()
        for entry in ticker_map.values():
            if str(entry.get("ticker", "")).upper() == normalized:
                cik_value = entry.get("cik_str")
                if cik_value is not None:
                    return int(cik_value)
        return None

    def _extract_filing_results(
        self,
        ticker: str,
        cik: int,
        submissions: dict,
    ) -> list[ProviderResult]:
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        preferred_forms = {"10-K", "10-Q", "8-K", "DEF 14A"}
        picked: list[ProviderResult] = []
        for idx, form in enumerate(forms):
            if form not in preferred_forms:
                continue
            accession = accession_numbers[idx].replace("-", "")
            filing_date = filing_dates[idx]
            primary_doc = primary_docs[idx]
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
            )
            title = f"{ticker} {form} filed {filing_date}"
            content = (
                f"Filing form {form} for {ticker}, filed on {filing_date}. "
                f"Primary document: {primary_doc}. Review full filing for management disclosures."
            )
            picked.append(
                ProviderResult(
                    source_type=self.source_type,
                    source_name=self.source_name,
                    title=title,
                    url=filing_url,
                    content=content,
                    published_at=_safe_parse_date(filing_date),
                    metadata={
                        "form": form,
                        "filing_date": filing_date,
                        "primary_document": primary_doc,
                        "cik": cik,
                    },
                )
            )
            if len(picked) >= 8:
                break
        return picked

    def _extract_company_facts(self, ticker: str, company_facts: dict) -> list[ProviderResult]:
        us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
        wanted_tags = {
            "Revenues": "Revenue",
            "GrossProfit": "Gross Profit",
            "NetIncomeLoss": "Net Income",
            "OperatingIncomeLoss": "Operating Income",
            "Assets": "Total Assets",
            "Liabilities": "Total Liabilities",
            "StockholdersEquity": "Stockholders Equity",
            "NetCashProvidedByUsedInOperatingActivities": "Operating Cash Flow",
        }

        snippets: list[str] = []
        metadata: dict[str, list[dict]] = {}
        for tag, label in wanted_tags.items():
            tag_obj = us_gaap.get(tag, {})
            units = tag_obj.get("units", {})
            unit_name = "USD" if "USD" in units else next(iter(units), None)
            if not unit_name:
                continue
            values = units[unit_name]
            values = sorted(values, key=lambda x: x.get("end", ""), reverse=True)
            latest = values[:4]
            extracted = []
            for item in latest:
                if "val" not in item:
                    continue
                extracted.append(
                    {
                        "value": item.get("val"),
                        "period_end": item.get("end"),
                        "form": item.get("form"),
                    }
                )
            if not extracted:
                continue
            metadata[tag] = extracted
            latest_str = ", ".join(
                [
                    f"{entry['period_end']}: {entry['value']} ({entry['form']})"
                    for entry in extracted
                    if entry.get("period_end")
                ]
            )
            snippets.append(f"{label}: {latest_str}")

        if not snippets:
            return []

        return [
            ProviderResult(
                source_type="fundamentals",
                source_name=self.source_name,
                title=f"{ticker} SEC XBRL fundamentals snapshot",
                url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{company_facts.get('cik')}.json",
                content=" ; ".join(snippets),
                metadata=metadata,
            )
        ]
