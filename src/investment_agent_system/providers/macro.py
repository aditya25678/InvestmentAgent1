from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Optional

from investment_agent_system.providers.base import ProviderResult, ResearchProvider
from investment_agent_system.providers.http import build_http_client, get_text

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _parse_float(value: str) -> Optional[float]:
    if value in {"", ".", "NaN", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class MacroProvider(ResearchProvider):
    source_type = "macro"
    source_name = "FRED"

    async def fetch(self, ticker: str) -> list[ProviderResult]:
        _ = ticker
        series_map = {
            "DGS10": "10Y Treasury Yield",
            "DGS2": "2Y Treasury Yield",
            "T10Y2Y": "10Y-2Y Curve",
            "VIXCLS": "VIX",
            "BAMLH0A0HYM2": "US High Yield OAS",
        }
        results: list[ProviderResult] = []

        async with build_http_client() as client:
            for series_id, label in series_map.items():
                csv_text = await get_text(client, FRED_CSV_URL, params={"id": series_id})
                reader = csv.DictReader(io.StringIO(csv_text))
                rows = list(reader)
                values: list[tuple[str, float]] = []
                for row in rows:
                    raw = row.get(series_id, "")
                    parsed = _parse_float(str(raw))
                    if parsed is None:
                        continue
                    values.append((str(row.get("DATE", "")), parsed))
                if len(values) < 2:
                    continue
                tail = values[-90:]
                first_date, first_val = tail[0]
                last_date, last_val = tail[-1]
                delta = round(last_val - first_val, 4)
                content = (
                    f"{label} ({series_id}) latest={last_val}, 90d_change={delta}, "
                    f"start={first_date}, last_observation_date={last_date}"
                )
                results.append(
                    ProviderResult(
                        source_type=self.source_type,
                        source_name=self.source_name,
                        title=f"Macro series snapshot: {label}",
                        url=f"https://fred.stlouisfed.org/series/{series_id}",
                        content=content,
                        published_at=datetime.now(timezone.utc),
                        metadata={
                            "series_id": series_id,
                            "label": label,
                            "latest": last_val,
                            "90d_change": delta,
                            "start_date": first_date,
                            "last_observation_date": last_date,
                        },
                    )
                )
        return results
