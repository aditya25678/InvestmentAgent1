from __future__ import annotations

from collections import defaultdict

from investment_agent_system.models.schemas import AgentRole, EvidenceItem

ROLE_SOURCE_ACCESS: dict[AgentRole, set[str]] = {
    AgentRole.FUNDAMENTAL: {"filings", "fundamentals", "news", "web_search"},
    AgentRole.QUANT: {"market_data", "macro", "estimates"},
    AgentRole.MACRO: {"macro", "market_data", "news"},
    AgentRole.SENTIMENT: {"news", "sentiment", "market_data"},
    AgentRole.SKEPTIC: {"filings", "fundamentals", "news", "macro"},
    AgentRole.CATALYST: {"news", "filings", "web_search"},
    AgentRole.PORTFOLIO: {"market_data", "macro", "fundamentals", "news", "sentiment"},
    AgentRole.COMMITTEE: {
        "market_data",
        "macro",
        "fundamentals",
        "filings",
        "news",
        "sentiment",
        "web_search",
        "estimates",
    },
}


def _sort_key(item: EvidenceItem) -> float:
    if not item.published_at:
        return 0.0
    return item.published_at.timestamp()


def select_evidence_for_role(
    role: AgentRole, evidence: list[EvidenceItem], max_items: int = 30
) -> list[EvidenceItem]:
    allowed = ROLE_SOURCE_ACCESS.get(role, set())
    filtered = [item for item in evidence if item.source_type in allowed]
    filtered.sort(key=_sort_key, reverse=True)
    return filtered[:max_items]


def render_research_packet(
    ticker: str,
    horizon: str,
    role: AgentRole,
    evidence: list[EvidenceItem],
) -> str:
    selected = select_evidence_for_role(role, evidence=evidence)
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in selected:
        grouped[item.source_type].append(item)

    lines: list[str] = [
        f"Ticker: {ticker}",
        f"Horizon: {horizon}",
        f"Role: {role.value}",
        "",
        "Evidence IDs are mandatory for citations and should map into claim citations.",
        "Every non-obvious claim must cite one or more evidence IDs and URLs.",
        "If evidence conflicts, surface both sides and lower confidence.",
        "",
    ]
    for source_type, rows in grouped.items():
        lines.append(f"## Source type: {source_type}")
        for row in rows:
            published = row.published_at.isoformat() if row.published_at else "unknown_date"
            excerpt = row.content.replace("\n", " ").strip()
            if len(excerpt) > 800:
                excerpt = excerpt[:800] + "..."
            lines.append(
                f"[EVIDENCE_ID={row.id}] title={row.title} | source={row.source_name} | "
                f"published={published} | url={row.url} | content={excerpt}"
            )
        lines.append("")

    if len(lines) <= 8:
        lines.append("No source-specific evidence available for this role.")
    return "\n".join(lines)


def render_global_packet(
    ticker: str, horizon: str, evidence: list[EvidenceItem], max_items: int = 50
) -> str:
    selected = sorted(evidence, key=_sort_key, reverse=True)[:max_items]
    lines = [
        f"Ticker: {ticker}",
        f"Horizon: {horizon}",
        "Cross-agent evidence packet:",
    ]
    for item in selected:
        excerpt = item.content.replace("\n", " ").strip()
        if len(excerpt) > 500:
            excerpt = excerpt[:500] + "..."
        lines.append(
            f"[EVIDENCE_ID={item.id}] {item.source_type} | {item.title} | {item.url} | {excerpt}"
        )
    return "\n".join(lines)
