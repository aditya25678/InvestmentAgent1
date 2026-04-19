from __future__ import annotations

from statistics import mean, pstdev

from investment_agent_system.models.schemas import AgentMemo


def aggregate_scorecards(memos: list[AgentMemo]) -> dict[str, float]:
    if not memos:
        return {}

    fields = [
        "business_quality",
        "valuation_attractiveness",
        "earnings_durability",
        "balance_sheet_strength",
        "management_credibility",
        "catalyst_strength",
        "technical_setup",
        "macro_sensitivity",
        "downside_asymmetry",
    ]
    aggregated: dict[str, float] = {}
    for field in fields:
        values = [float(getattr(memo.scorecard, field)) for memo in memos]
        aggregated[field] = round(mean(values), 2)
    totals = [memo.scorecard.weighted_total for memo in memos]
    aggregated["weighted_total_mean"] = round(mean(totals), 2)
    aggregated["weighted_total_dispersion"] = round(pstdev(totals), 2) if len(totals) > 1 else 0.0
    return aggregated
